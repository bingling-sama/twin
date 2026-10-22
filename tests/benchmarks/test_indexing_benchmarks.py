"""Benchmarks for image indexing pipeline (single and batch).

Measures the throughput of the indexing pipeline: CLIP encode + dHash
+ pHash + Faiss insert. These are the operations that run when a user
uploads images via POST /api/v1/index or POST /api/v1/index/batch.
"""

from __future__ import annotations

import pytest

# ── index_single throughput ───────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_index_single(benchmark, clip_model):
    """Benchmark: full index_single() pipeline (embed + double-hash + add).

    Uses benchmark.pedantic with setup to clear the indexer before each
    round, preventing dedup hits from skewing the measurement.
    """
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.index_service import index_single
    from twin.services.indexer import indexer

    img = random_image((224, 224), seed=42)

    def _index():
        return index_single(img, "bench_test.png", b"dummy")

    benchmark.pedantic(_index, setup=lambda: indexer.clear(), rounds=10)

    indexer.clear()


# ── index_batch throughput ────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
def test_bench_index_batch_throughput(benchmark, clip_model, tmp_path, batch_size):
    """Benchmark: index_batch() at various batch sizes.

    Writes synthetic images to a temp directory, then indexes them.
    Measures the full pipeline: disk I/O + CLIP + hash + Faiss insert.
    """
    from tests.benchmarks.fixtures.synthetic import save_images_to_dir
    from twin.services.index_service import index_batch

    save_images_to_dir(tmp_path, count=batch_size, sizes=[(224, 224)], formats=["png"])

    benchmark(index_batch, str(tmp_path))


# ── Multi-batch pipelined throughput (double-buffering preloader gate) ─────────


@pytest.mark.smoke
def test_bench_index_batch_pipelined_multi_batch(benchmark, clip_model, tmp_path):
    """Benchmark: end-to-end multi-batch index_batch() throughput with preloader.

    Writes 64 images across 4 batches (batch_size=16) to exercise the
    ThreadPoolExecutor double-buffering image preloader pipeline.
    Serves as an empirical regression gate for pipeline throughput gains.
    """
    from tests.benchmarks.fixtures.synthetic import save_images_to_dir
    from twin.core.config import settings
    from twin.services.index_service import index_batch
    from twin.services.indexer import indexer

    save_images_to_dir(tmp_path, count=64, sizes=[(224, 224)], formats=["png"])

    old_batch_size = settings.batch_size
    settings.batch_size = 16

    def _run():
        indexer.clear()
        return index_batch(str(tmp_path))

    try:
        benchmark(_run)
    finally:
        settings.batch_size = old_batch_size
        indexer.clear()


# ── _index_single_from_disk (internal fallback) ───────────────────────────────


@pytest.mark.smoke
def test_bench_index_single_from_disk(benchmark, clip_model, tmp_path):
    """Benchmark: _index_single_from_disk() — the single-image fallback path.

    This path is used when batch indexing fails and the system falls
    back to indexing images one at a time.
    """
    from PIL import Image

    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.index_service import _index_single_from_disk
    from twin.services.indexer import indexer

    # Ensure a clean IndexFlatL2 before benchmarking — prior tests
    # (e.g., index_batch_throughput) may have auto-upgraded to IVF.
    indexer.clear()

    img = random_image((224, 224), seed=42)
    path = tmp_path / "bench_single.png"
    img.save(path)

    def _index():
        reloaded = Image.open(path).convert("RGB")
        return _index_single_from_disk(reloaded, path)

    benchmark(_index)

    indexer.clear()


# ── Dedup check overhead ──────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_index_dedup_check(benchmark, clip_model):
    """Benchmark: deduplication check for an already-indexed filename.

    When the same filename is uploaded twice, the system should detect
    it and return 'already_exists' without re-indexing.
    """
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.index_service import index_single
    from twin.services.indexer import indexer

    # Ensure a clean IndexFlatL2 — prior tests may have left an
    # untrained IVF index that can't accept add() calls.
    indexer.clear()

    img = random_image((224, 224), seed=42)
    fname = "bench_dedup_test.png"

    # First index
    index_single(img, fname, b"dummy")

    # Benchmark the dedup check (second upload of same filename)
    def _dedup():
        return index_single(img, fname, b"dummy")

    benchmark(_dedup)

    # Clean up
    indexer.clear()
