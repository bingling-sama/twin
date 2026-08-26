"""Benchmarks for perceptual hashing (dHash and pHash).

Measures single-image and batch throughput for dHash and pHash computation.
These are pure CPU benchmarks — no CLIP or GPU needed.
"""

from __future__ import annotations

import pytest

from twin.services.hasher import compute_dhash, compute_dhashes, compute_phash, compute_phashes

# ── dHash benchmarks ──────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_dhash_single(benchmark, bench_gradient_256):
    """Benchmark: single dHash computation on a 256x256 image."""
    benchmark(compute_dhash, bench_gradient_256)


@pytest.mark.scaling
@pytest.mark.parametrize("n", [1, 8, 32, 128])
def test_bench_dhash_batch(benchmark, n):
    """Benchmark: batch dHash computation at various batch sizes."""
    from tests.benchmarks.fixtures.synthetic import image_batch

    images = image_batch(n, (256, 256), seed=42)
    benchmark(compute_dhashes, images)


@pytest.mark.scaling
@pytest.mark.parametrize("size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_bench_dhash_resolution(benchmark, size):
    """Benchmark: single dHash at various image resolutions."""
    from tests.benchmarks.fixtures.synthetic import gradient_image

    img = gradient_image(size, seed=42)
    benchmark(compute_dhash, img)


# ── pHash benchmarks ──────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_phash_single(benchmark, bench_gradient_256):
    """Benchmark: single pHash computation on a 256x256 image."""
    benchmark(compute_phash, bench_gradient_256)


@pytest.mark.scaling
@pytest.mark.parametrize("n", [1, 8, 32, 128])
def test_bench_phash_batch(benchmark, n):
    """Benchmark: batch pHash computation at various batch sizes."""
    from tests.benchmarks.fixtures.synthetic import image_batch

    images = image_batch(n, (256, 256), seed=42)
    benchmark(compute_phashes, images)


@pytest.mark.scaling
@pytest.mark.parametrize("size", [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_bench_phash_resolution(benchmark, size):
    """Benchmark: single pHash at various image resolutions."""
    from tests.benchmarks.fixtures.synthetic import gradient_image

    img = gradient_image(size, seed=42)
    benchmark(compute_phash, img)


# ── dHash vs pHash comparison ─────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_dhash_vs_phash_single(benchmark, bench_gradient_256):
    """Benchmark: dHash latency — use alongside phash_single to compare.

    Run both dhash_single and phash_single to see side-by-side comparison.
    dHash is typically slightly faster than pHash (fewer FFT operations).
    """
    from twin.services.hasher import compute_dhash as dhash

    benchmark(dhash, bench_gradient_256)
