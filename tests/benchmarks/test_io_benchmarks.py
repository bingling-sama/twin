"""Benchmarks for image I/O — loading from disk, decoding from bytes.

These benchmarks measure the raw I/O throughput that underlies
both the indexing pipeline and the SSIM stage (which loads candidate
images from disk).
"""

from __future__ import annotations

import pytest

from twin.utils.image import load_image, load_images


# ── Single image load from disk ───────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_load_image_png_224(benchmark, temp_image_dir):
    """Benchmark: load a single 224x224 PNG from disk."""
    path = next(p for p in temp_image_dir.iterdir() if p.suffix == ".png")
    benchmark(load_image, str(path))


@pytest.mark.smoke
def test_bench_load_image_jpg_224(benchmark, temp_image_dir):
    """Benchmark: load a single 224x224 JPEG from disk."""
    path = next(p for p in temp_image_dir.iterdir() if p.suffix == ".jpg")
    benchmark(load_image, str(path))


@pytest.mark.scaling
@pytest.mark.parametrize("resolution", ["224x224", "512x512", "1024x1024"])
def test_bench_load_image_resolution(benchmark, temp_image_dir, resolution):
    """Benchmark: load images at various resolutions (PNG)."""
    path = next(
        p for p in sorted(temp_image_dir.iterdir())
        if resolution in p.name and p.suffix == ".png"
    )
    benchmark(load_image, str(path))


# ── Batch image load ──────────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("n", [1, 10, 50])
def test_bench_load_images_batch(benchmark, temp_image_dir, n):
    """Benchmark: batch-load N images from disk."""
    paths = sorted(temp_image_dir.iterdir())[:n]
    benchmark(load_images, [str(p) for p in paths])


# ── BytesIO decode (simulates HTTP upload) ────────────────────────────────────


@pytest.mark.smoke
def test_bench_image_decode_from_bytes(benchmark, temp_image_dir):
    """Benchmark: decode an image from bytes (simulates POST /search upload)."""
    from io import BytesIO

    from PIL import Image

    path = next(p for p in temp_image_dir.iterdir() if p.suffix == ".jpg")
    content = path.read_bytes()

    def _decode():
        img = Image.open(BytesIO(content))
        img.load()
        return img.convert("RGB")

    benchmark(_decode)


@pytest.mark.scaling
@pytest.mark.parametrize("resolution", ["224x224", "512x512", "1024x1024"])
def test_bench_image_decode_resolution(benchmark, temp_image_dir, resolution):
    """Benchmark: decode bytes at various resolutions."""
    from io import BytesIO

    from PIL import Image

    path = next(
        p for p in sorted(temp_image_dir.iterdir())
        if resolution in p.name and p.suffix == ".jpg"
    )
    content = path.read_bytes()

    def _decode():
        img = Image.open(BytesIO(content))
        img.load()
        return img.convert("RGB")

    benchmark(_decode)
