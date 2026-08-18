"""Benchmark-specific configuration — GPU warmup, fixtures, markers.

Sets OMP_NUM_THREADS=1 before any torch/faiss import to prevent
thread contention. Creates a temporary TWIN_IMAGES_DIR so the
startup sync in the TestClient lifespan doesn't touch real data.

Auto-memory-tracking: a session-scoped monkeypatch wraps every
BenchmarkFixture call to record peak RSS delta and peak GPU VRAM
in benchmark.extra_info, requiring zero changes to individual tests.
"""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# ── Module-level env setup (runs at collection time, before any imports) ──────


def pytest_configure(config: pytest.Config) -> None:
    """Set environment variables and temp dirs before test collection."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Use a temp directory for TWIN_IMAGES_DIR so startup sync doesn't
    # try to index real data/images/ (which may have thousands of files).
    if "TWIN_IMAGES_DIR" not in os.environ:
        tmpdir = tempfile.mkdtemp(prefix="twin_bench_images_")
        os.environ["TWIN_IMAGES_DIR"] = tmpdir
        config._twin_bench_images_dir = tmpdir  # type: ignore[attr-defined]

    # Register custom markers
    config.addinivalue_line("markers", "gpu: Benchmark requires GPU hardware")
    config.addinivalue_line("markers", "slow: Large-scale benchmark (long runtime)")
    config.addinivalue_line("markers", "scaling: Benchmark that sweeps parameter sizes")
    config.addinivalue_line("markers", "smoke: Quick sanity check (fast, high level)")


def pytest_unconfigure(config: pytest.Config) -> None:
    """Clean up temp directories created during pytest_configure."""
    tmpdir = getattr(config, "_twin_bench_images_dir", None)
    if tmpdir and os.path.isdir(tmpdir):
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Session-scoped fixtures (expensive, created once) ─────────────────────────


@pytest.fixture(scope="session")
def gpu_info() -> dict:
    """Detect GPU availability. Safe to call before torch is imported."""
    info: dict = {"device": "cpu", "cuda_available": False, "device_name": "CPU"}

    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            info["device"] = "mps"
            info["device_name"] = "Apple MPS"
    except ImportError:
        pass

    return info


@pytest.fixture(scope="session")
def clip_model(gpu_info: dict) -> bool:
    """Load CLIP model once for the benchmark session with GPU warmup.

    Returns True when the model is loaded and ready.
    Benchmarks that need CLIP should depend on this fixture.
    """
    from twin.models.clip_model import load as load_model

    load_model()

    # GPU warmup: run 5 dummy forward passes to stabilize CUDA kernel cache
    if gpu_info["device"] == "cuda":
        import torch

        from tests.benchmarks.fixtures.synthetic import random_image
        from twin.services.embedding import compute_embedding

        for _ in range(5):
            img = random_image((224, 224))
            _ = compute_embedding(img)
            torch.cuda.synchronize()

    return True


# ── Function-scoped fixtures (fresh per benchmark) ────────────────────────────


@pytest.fixture
def bench_image_224() -> "Image.Image":
    """A single 224x224 random image — the default CLIP input size."""
    from tests.benchmarks.fixtures.synthetic import random_image

    return random_image((224, 224))


@pytest.fixture
def bench_image_512() -> "Image.Image":
    """A single 512x512 random image — typical high-res input."""
    from tests.benchmarks.fixtures.synthetic import random_image

    return random_image((512, 512))


@pytest.fixture
def bench_images_32() -> "list[Image.Image]":
    """A batch of 32 random 224x224 images (default CLIP batch size)."""
    from tests.benchmarks.fixtures.synthetic import image_batch

    return image_batch(32, (224, 224))


@pytest.fixture
def bench_gradient_256() -> "Image.Image":
    """A 256x256 gradient image — deterministic, stable for hash benchmarks."""
    from tests.benchmarks.fixtures.synthetic import gradient_image

    return gradient_image((256, 256))


@pytest.fixture
def bench_checkerboard_256() -> "Image.Image":
    """A 256x256 checkerboard — deterministic structure for SSIM benchmarks."""
    from tests.benchmarks.fixtures.synthetic import checkerboard_image

    return checkerboard_image(squares=8, size=(256, 256))


@pytest.fixture
def temp_image_dir() -> Path:
    """Temporary directory with pre-written synthetic images for I/O benchmarks."""
    import shutil

    from tests.benchmarks.fixtures.synthetic import save_images_to_dir

    tmp = Path(tempfile.mkdtemp(prefix="twin_bench_io_"))
    save_images_to_dir(tmp, count=20, sizes=[(224, 224), (512, 512), (1024, 1024)],
                       formats=["png", "jpg"])
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ── Faiss index fixtures (use raw faiss, not the Indexer singleton) ────────────


def _make_flat_index(dim: int = 512) -> "faiss.IndexFlatL2":
    """Create a bare IndexFlatL2 — no lock, no metadata, no disk I/O."""
    import faiss
    return faiss.IndexFlatL2(dim)


def _populate_index(
    idx: "faiss.Index",
    n: int,
    dim: int = 512,
) -> "np.ndarray":
    """Add N synthetic L2-normalized vectors to the index.

    Returns the vectors added (for recall validation).
    """
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    vecs = random_normalized_vectors(n, dim)
    idx.add(vecs)
    return vecs


@pytest.fixture
def bench_faiss_empty() -> "faiss.IndexFlatL2":
    """An empty IndexFlatL2 (512-dim)."""
    return _make_flat_index(512)


@pytest.fixture
def bench_faiss_1k() -> "faiss.IndexFlatL2":
    """IndexFlatL2 pre-populated with 1,000 synthetic vectors."""
    idx = _make_flat_index(512)
    _populate_index(idx, 1_000)
    return idx


@pytest.fixture
def bench_faiss_10k() -> "faiss.IndexFlatL2":
    """IndexFlatL2 pre-populated with 10,000 synthetic vectors."""
    idx = _make_flat_index(512)
    _populate_index(idx, 10_000)
    return idx


@pytest.fixture
def bench_faiss_100k() -> "faiss.IndexFlatL2":
    """IndexFlatL2 pre-populated with 100,000 synthetic vectors (~200 MB)."""
    idx = _make_flat_index(512)
    _populate_index(idx, 100_000)
    return idx


@pytest.fixture
def bench_query_vec() -> "np.ndarray":
    """A single L2-normalized query vector (1, 512)."""
    from tests.benchmarks.fixtures.synthetic import query_vector

    return query_vector(512)


# ── HNSW fixtures ──────────────────────────────────────────────────────────────


def _make_hnsw_index(dim: int = 512, m: int = 32) -> "faiss.IndexHNSWFlat":
    """Create a bare IndexHNSWFlat with efConstruction=200."""
    import faiss

    idx = faiss.IndexHNSWFlat(dim, m)
    idx.hnsw.efConstruction = 200
    return idx


@pytest.fixture
def bench_hnsw_1k() -> "faiss.IndexHNSWFlat":
    """IndexHNSWFlat (M=32) pre-populated with 1,000 synthetic vectors."""
    idx = _make_hnsw_index(512, m=32)
    _populate_index(idx, 1_000)
    idx.hnsw.efSearch = 64
    return idx


@pytest.fixture
def bench_hnsw_10k() -> "faiss.IndexHNSWFlat":
    """IndexHNSWFlat (M=32) pre-populated with 10,000 synthetic vectors."""
    idx = _make_hnsw_index(512, m=32)
    _populate_index(idx, 10_000)
    idx.hnsw.efSearch = 64
    return idx


@pytest.fixture
def bench_hnsw_100k() -> "faiss.IndexHNSWFlat":
    """IndexHNSWFlat (M=32) pre-populated with 100,000 synthetic vectors (~200 MB)."""
    idx = _make_hnsw_index(512, m=32)
    _populate_index(idx, 100_000)
    idx.hnsw.efSearch = 64
    return idx


# ── IVFPQ fixtures ────────────────────────────────────────────────────────────


def _make_ivfpq_index(
    dim: int = 512,
    nlist: int = 40,
    m: int = 64,
    nbits: int = 8,
) -> "faiss.IndexIVFPQ":
    """Create a bare IndexIVFPQ with a FlatL2 coarse quantizer."""
    import faiss

    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    idx.nprobe = 8
    return idx


def _populate_ivfpq_index(
    idx: "faiss.IndexIVFPQ",
    n: int,
    dim: int = 512,
) -> "np.ndarray":
    """Train and populate an IVFPQ index with N synthetic vectors.

    Returns the vectors added (for recall validation).
    """
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    vecs = random_normalized_vectors(n, dim)
    idx.train(vecs)
    idx.add(vecs)
    return vecs


@pytest.fixture
def bench_ivfpq_1k() -> "faiss.IndexIVFPQ":
    """IndexIVFPQ (nlist=16, M=64, nbits=8) with 1,000 vectors."""
    idx = _make_ivfpq_index(512, nlist=16, m=64, nbits=8)
    _populate_ivfpq_index(idx, 1_000)
    return idx


@pytest.fixture
def bench_ivfpq_10k() -> "faiss.IndexIVFPQ":
    """IndexIVFPQ (nlist=40, M=64, nbits=8) with 10,000 vectors."""
    idx = _make_ivfpq_index(512, nlist=40, m=64, nbits=8)
    _populate_ivfpq_index(idx, 10_000)
    return idx


@pytest.fixture
def bench_ivfpq_100k() -> "faiss.IndexIVFPQ":
    """IndexIVFPQ (nlist=100, M=64, nbits=8) with 100,000 vectors."""
    idx = _make_ivfpq_index(512, nlist=100, m=64, nbits=8)
    _populate_ivfpq_index(idx, 100_000)
    return idx


# ── System info collection ───────────────────────────────────────────────────


def _collect_memory_info() -> dict:
    """Collect detailed memory info from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            raw: dict[str, str] = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    raw[parts[0].strip()] = parts[1].strip()

        def _kb(key: str) -> int:
            return int(raw.get(key, "0 kB").split()[0])

        total_kb = _kb("MemTotal")
        available_kb = _kb("MemAvailable")
        free_kb = _kb("MemFree")
        swap_total_kb = _kb("SwapTotal")
        swap_free_kb = _kb("SwapFree")

        return {
            "total_gb": round(total_kb / (1024 * 1024), 1),
            "available_gb": round(available_kb / (1024 * 1024), 1),
            "used_gb": round((total_kb - available_kb) / (1024 * 1024), 1),
            "swap_total_gb": round(swap_total_kb / (1024 * 1024), 1),
            "swap_used_gb": round((swap_total_kb - swap_free_kb) / (1024 * 1024), 1),
        }
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def _collect_gpu_info() -> dict | None:
    """Collect GPU details via torch + nvidia-smi if available."""
    info: dict = {}
    try:
        import torch

        if torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            )
            info["compute_capability"] = "{}.{}".format(
                *torch.cuda.get_device_capability(0)
            )
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["torch_version"] = torch.__version__
        else:
            return None
    except ImportError:
        return None

    # Try nvidia-smi for driver version (more precise)
    import shutil

    if shutil.which("nvidia-smi"):
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["driver_version"] = result.stdout.strip()
        except Exception:
            pass

    return info


def _collect_os_info() -> dict:
    """Collect OS/distribution details."""
    import platform

    info: dict = {
        "kernel": platform.release(),
        "system": platform.system(),
        "machine": platform.machine(),
    }

    # Try to get distro name
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    info["distro"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except FileNotFoundError:
        pass

    return info


def _collect_package_versions() -> dict:
    """Collect versions of key Python packages."""
    packages = [
        "torch", "faiss", "numpy", "PIL", "open_clip",
        "imagehash", "skimage", "fastapi", "uvicorn",
    ]
    versions: dict = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            for attr in ["__version__", "VERSION", "version"]:
                v = getattr(mod, attr, None)
                if v:
                    if isinstance(v, tuple):
                        v = ".".join(str(x) for x in v)
                    versions[pkg] = str(v)
                    break
        except ImportError:
            pass

    # faiss-cpu vs faiss-gpu
    if "faiss" in versions:
        try:
            import faiss
            versions["faiss_gpu"] = str(hasattr(faiss, "StandardGpuResources"))
        except Exception:
            pass

    return versions


def _collect_disk_info() -> dict:
    """Collect disk info for the project data directory."""
    import shutil
    import os as _os

    info: dict = {}
    project_root = _os.environ.get(
        "TWIN_PROJECT_ROOT",
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    )

    # Disk usage on project partition
    try:
        import shutil
        usage = shutil.disk_usage(project_root)
        info["project_root"] = project_root
        info["disk_total_gb"] = round(usage.total / (1024**3), 1)
        info["disk_free_gb"] = round(usage.free / (1024**3), 1)
    except Exception:
        pass

    # Filesystem type
    try:
        import subprocess
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", project_root],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["filesystem"] = result.stdout.strip()
    except Exception:
        pass

    return info


@pytest.hookimpl(trylast=True, optionalhook=True)
def pytest_benchmark_update_machine_info(config, machine_info):
    """Augment benchmark machine_info with detailed system data.

    Adds: memory (detailed), gpu, os, packages, disk.
    Called once per session by pytest-benchmark.
    """
    machine_info.setdefault("memory", _collect_memory_info())
    machine_info.setdefault("os", _collect_os_info())

    gpu = _collect_gpu_info()
    if gpu:
        machine_info.setdefault("gpu", gpu)

    machine_info.setdefault("packages", _collect_package_versions())
    machine_info.setdefault("disk", _collect_disk_info())


@pytest.fixture(scope="session", autouse=True)
def _auto_memory_tracking():
    """Monkey-patch BenchmarkFixture to auto-record peak memory in extra_info.

    Wraps both __call__ and pedantic so every benchmark — without any
    test-code changes — gets::

        extra_info["peak_tracemalloc_mb"]  — Python heap peak (tracemalloc)
        extra_info["peak_rss_mb"]          — OS RSS net peak (1 kHz sampler)
        extra_info["peak_gpu_mb"]          — GPU VRAM net peak (baseline subtracted)
    """
    from tests.benchmarks.memory import wrap_with_memory

    try:
        from pytest_benchmark.fixture import BenchmarkFixture
    except ImportError:
        yield  # pytest-benchmark not installed; nothing to patch
        return

    _orig_call = BenchmarkFixture.__call__
    _orig_pedantic = BenchmarkFixture.pedantic

    @functools.wraps(_orig_call)
    def _call_with_memory(self, function_to_benchmark, *args, **kwargs):
        wrapped_fn, peaks = wrap_with_memory(function_to_benchmark)
        result = _orig_call(self, wrapped_fn, *args, **kwargs)
        # Store all three peak metrics
        for key in ("peak_tracemalloc_mb", "peak_rss_mb", "peak_gpu_mb"):
            if peaks.get(key, 0) > 0:
                self.extra_info.setdefault(key, round(peaks[key], 2))
        return result

    @functools.wraps(_orig_pedantic)
    def _pedantic_with_memory(self, target, args=(), kwargs=None, setup=None,
                               teardown=None, rounds=1, warmup_rounds=0, iterations=1):
        wrapped_target, peaks = wrap_with_memory(target)
        result = _orig_pedantic(
            self, wrapped_target, args=args, kwargs=kwargs,
            setup=setup, teardown=teardown, rounds=rounds,
            warmup_rounds=warmup_rounds, iterations=iterations,
        )
        # Store all three peak metrics
        for key in ("peak_tracemalloc_mb", "peak_rss_mb", "peak_gpu_mb"):
            if peaks.get(key, 0) > 0:
                self.extra_info.setdefault(key, round(peaks[key], 2))
        return result

    BenchmarkFixture.__call__ = _call_with_memory
    BenchmarkFixture.pedantic = _pedantic_with_memory

    yield

    BenchmarkFixture.__call__ = _orig_call
    BenchmarkFixture.pedantic = _orig_pedantic
