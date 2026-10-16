"""Benchmarks for DINOv2 vision transformer encoding (single and batch).

Requires the session-scoped dinov2_model fixture. On first run this downloads
the DINOv2 weights (~85 MB for vit_small_patch14_dinov2) if not cached.
"""

from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_bench_dinov2_encode_single(benchmark, dinov2_model, bench_image_224):
    """Benchmark: encode a single 224x224 image through DINOv2."""
    from twin.models import dinov2_model as model

    benchmark(model.encode_image, bench_image_224)


@pytest.mark.smoke
def test_bench_dinov2_encode_single_512(benchmark, dinov2_model, bench_image_512):
    """Benchmark: encode a single 512x512 image through DINOv2."""
    from twin.models import dinov2_model as model

    benchmark(model.encode_image, bench_image_512)


@pytest.mark.smoke
def test_bench_dinov2_encode_batch_32(benchmark, dinov2_model, bench_images_32):
    """Benchmark: batch-encode 32 images through DINOv2."""
    from twin.models import dinov2_model as model

    benchmark(model.encode_images, bench_images_32)


@pytest.mark.scaling
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_bench_dinov2_encode_batch_size(benchmark, dinov2_model, batch_size):
    """Benchmark: DINOv2 throughput scaling across various batch sizes."""
    from tests.benchmarks.fixtures.synthetic import image_batch
    from twin.models import dinov2_model as model

    images = image_batch(batch_size, (224, 224), seed=42)
    benchmark(model.encode_images, images)
