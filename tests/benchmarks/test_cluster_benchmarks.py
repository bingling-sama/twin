"""Benchmarks for seedless texture discovery and clustering via Greedy Leader Elimination.

Measures the end-to-end performance and clustering quality of discover_similar_batches():
  1. Faiss binary pHash SIMD recall with adaptive window expansion
  2. Parallel image I/O with path deduplication and thread pool decoding
  3. DiffMask pixel verification and strict elimination (anti-transitivity collapse)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from twin.services.cluster import discover_similar_batches
from twin.services.hasher import compute_ahash, compute_dhash, compute_phash
from twin.services.indexer import indexer


def _create_synthetic_skin(seed: int, recolor_torso: bool = False) -> Image.Image:
    """Create a 64x64 standard RGBA skin for benchmark testing."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((64, 64, 4), dtype=np.uint8)

    r = int(rng.randint(50, 200))
    g = int(rng.randint(50, 200))
    b = int(rng.randint(50, 200))

    # Base head and torso
    arr[8:16, 8:24] = [r, g, b, 255]
    if not recolor_torso:
        arr[20:32, 20:36] = [(r + 40) % 255, (g + 40) % 255, (b + 40) % 255, 255]
    else:
        arr[20:32, 20:36] = [(r + 140) % 255, (g + 140) % 255, (b + 140) % 255, 255]
    arr[32:48, 0:64] = [r // 2, g // 2, b // 2, 255]
    arr[0:48, 0:64, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def _populate_cluster_benchmark_index(
    tmp_path: Path,
    num_duplicates: int = 30,
    num_variants: int = 15,
    num_noise: int = 25,
) -> None:
    """Populate indexer with physical files for benchmark testing."""
    indexer.clear()
    dummy_vec = np.zeros(512, dtype=np.float32)

    # 1. Base anchor for duplicates
    base_dup = _create_synthetic_skin(seed=42)
    for i in range(num_duplicates):
        fname = f"dup_{i:04d}.png"
        fpath = tmp_path / fname
        base_dup.save(fpath)
        meta = {
            "filename": fname,
            "path": str(fpath),
            "datasource": "skin",
            "texture_type": "skin_double",
            "ahash": compute_ahash(base_dup),
            "dhash": compute_dhash(base_dup),
            "phash": compute_phash(base_dup),
        }
        indexer.add_item(dummy_vec, meta, datasource="skin")

    # 2. Base anchor and variants
    for i in range(num_variants):
        var_img = _create_synthetic_skin(seed=100, recolor_torso=(i > 0))
        fname = f"var_{i:04d}.png"
        fpath = tmp_path / fname
        var_img.save(fpath)
        meta = {
            "filename": fname,
            "path": str(fpath),
            "datasource": "skin",
            "texture_type": "skin_double",
            "ahash": compute_ahash(var_img),
            "dhash": compute_dhash(var_img),
            "phash": compute_phash(var_img),
        }
        indexer.add_item(dummy_vec, meta, datasource="skin")

    # 3. Random noise items
    for i in range(num_noise):
        noise_img = _create_synthetic_skin(seed=1000 + i * 17)
        fname = f"noise_{i:04d}.png"
        fpath = tmp_path / fname
        noise_img.save(fpath)
        meta = {
            "filename": fname,
            "path": str(fpath),
            "datasource": "skin",
            "texture_type": "skin_double",
            "ahash": compute_ahash(noise_img),
            "dhash": compute_dhash(noise_img),
            "phash": compute_phash(noise_img),
        }
        indexer.add_item(dummy_vec, meta, datasource="skin")


@pytest.mark.smoke
def test_bench_cluster_confirmed_duplicates(tmp_path: Path) -> None:
    """Benchmark confirmed mode: pure duplicates discovery with zero diff."""
    _populate_cluster_benchmark_index(tmp_path, num_duplicates=25, num_variants=0, num_noise=20)

    t0 = time.perf_counter()
    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=5,
        mode="confirmed",
        max_probes=100,
        seed_strategy="random",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    batches = res["batches"]
    assert len(batches) >= 4
    for b in batches:
        assert b["mode"] == "confirmed"
        assert b["max_diff_ratio"] <= 0.05
        # All members must be identical duplicates
        for m in b["members"]:
            assert m["diff_ratio"] == 0.0

    assert elapsed_ms < 500  # Latency budget under 500ms for 45 items


@pytest.mark.smoke
def test_bench_cluster_adaptive_recall_scaling(tmp_path: Path) -> None:
    """Benchmark adaptive recall expansion on a large swarm of 50 identical duplicates."""
    _populate_cluster_benchmark_index(tmp_path, num_duplicates=50, num_variants=0, num_noise=10)

    t0 = time.perf_counter()
    res = discover_similar_batches(
        datasource="skin",
        max_batches=20,
        batch_size=5,
        mode="confirmed",
        max_probes=200,
        seed_strategy="sequential",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    batches = res["batches"]
    # 50 duplicates in batches of 5 = exactly 10 batches
    assert len(batches) == 10
    total_items_in_batches = sum(b["size"] for b in batches)
    assert total_items_in_batches == 50

    # Ensure stages telemetry accurately records adaptive expansions
    stages = res["stages"]
    assert stages["recall"]["probes_evaluated"] >= 10
    assert stages["verify"]["batches_formed"] == 10
    assert elapsed_ms < 1000


@pytest.mark.smoke
def test_bench_cluster_variants_mode(tmp_path: Path) -> None:
    """Benchmark variants mode on recolored derivative textures."""
    _populate_cluster_benchmark_index(tmp_path, num_duplicates=10, num_variants=20, num_noise=20)

    t0 = time.perf_counter()
    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=5,
        mode="variants",
        max_probes=100,
        seed_strategy="random",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    batches = res["batches"]
    assert len(batches) >= 2
    for b in batches:
        assert b["mode"] == "variants"
        assert b["max_diff_ratio"] <= 0.35

    assert elapsed_ms < 500
