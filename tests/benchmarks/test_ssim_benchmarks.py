"""Benchmarks for SSIM (Structural Similarity) computation.

Measures pairwise SSIM latency at various image sizes and for
identical vs. different image pairs. This is the most expensive
per-image filter in the search pipeline.
"""

from __future__ import annotations

import pytest

from twin.services.hasher import compute_ssim


# ── Identical image pairs ─────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_ssim_identical(benchmark, bench_checkerboard_256):
    """Benchmark: SSIM on identical 256x256 images (the default SSIM resize size)."""
    img = bench_checkerboard_256
    benchmark(compute_ssim, img, img)


@pytest.mark.scaling
@pytest.mark.parametrize("size", [(64, 64), (128, 128), (256, 256), (512, 512)])
def test_bench_ssim_identical_size(benchmark, size):
    """Benchmark: SSIM on identical images at various resolutions.

    Note: compute_ssim internally resizes everything to 256x256 grayscale,
    so the input resolution should have minimal impact on timing.
    """
    from tests.benchmarks.fixtures.synthetic import checkerboard_image

    img = checkerboard_image(squares=8, size=size, seed=42)
    benchmark(compute_ssim, img, img)


# ── Different image pairs ─────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_ssim_different(benchmark, bench_checkerboard_256):
    """Benchmark: SSIM on different images (worst-case, no early exit)."""
    from tests.benchmarks.fixtures.synthetic import random_image

    img1 = bench_checkerboard_256
    img2 = random_image((256, 256), seed=99)
    benchmark(compute_ssim, img1, img2)


# ── SSIM in pipeline context ──────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_ssim_batch_10_pairs(benchmark):
    """Benchmark: SSIM on 10 pairs (simulates post-pHash survivor batch)."""
    from tests.benchmarks.fixtures.synthetic import (
        checkerboard_image,
        near_duplicate_pair,
        random_image,
    )

    base = checkerboard_image(squares=8, size=(256, 256), seed=42)
    # Create 10 pairs: 5 near-duplicates, 5 random
    pairs = []
    for i in range(5):
        _, dup = near_duplicate_pair((256, 256), noise_level=3.0, seed=100 + i)
        pairs.append((base, dup))
    for i in range(5):
        pairs.append((base, random_image((256, 256), seed=200 + i)))

    def _run_all():
        for a, b in pairs:
            compute_ssim(a, b)

    benchmark(_run_all)
