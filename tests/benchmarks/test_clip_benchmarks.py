"""Benchmarks for CLIP model encoding (single and batch).

Requires the session-scoped clip_model fixture. On first run this
downloads the CLIP ViT-B-32 weights (~350 MB) if not cached.

GPU warmup: the clip_model fixture runs 5 CUDA warmup passes before
any benchmark runs, so these measurements reflect steady-state performance.
"""

from __future__ import annotations

import pytest

# ── Single image encoding ─────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_clip_encode_single(benchmark, clip_model, bench_image_224):
    """Benchmark: encode a single 224x224 image through CLIP.

    Expected: ~5-20ms on GPU, ~20-100ms on CPU depending on hardware.
    """
    from twin.services.embedding import compute_embedding

    benchmark(compute_embedding, bench_image_224)


@pytest.mark.smoke
def test_bench_clip_encode_single_512(benchmark, clip_model, bench_image_512):
    """Benchmark: encode a single 512x512 image (higher preprocess cost)."""
    from twin.services.embedding import compute_embedding

    benchmark(compute_embedding, bench_image_512)


# ── Batch encoding ────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_clip_encode_batch_32(benchmark, clip_model, bench_images_32):
    """Benchmark: batch-encode 32 images (the default batch size).

    Batch encoding should be significantly faster per-image than
    encoding images one at a time.
    """
    from twin.services.embedding import compute_embeddings

    benchmark(compute_embeddings, bench_images_32)


@pytest.mark.scaling
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_bench_clip_encode_batch_size(benchmark, clip_model, batch_size):
    """Benchmark: batch-encode at various batch sizes to find the throughput sweet spot."""
    from tests.benchmarks.fixtures.synthetic import image_batch
    from twin.services.embedding import compute_embeddings

    images = image_batch(batch_size, (224, 224), seed=42)
    benchmark(compute_embeddings, images)


# ── Resolution sweep ──────────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("size", [(128, 128), (224, 224), (384, 384), (512, 512)])
def test_bench_clip_encode_resolution(benchmark, clip_model, size):
    """Benchmark: how does input resolution affect CLIP encoding time?

    CLIP's preprocessing resizes everything to 224x224, so larger inputs
    primarily affect the preprocessing cost, not the forward pass.
    """
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.embedding import compute_embedding

    img = random_image(size, seed=42)
    benchmark(compute_embedding, img)


# ── Cold vs warm start ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_bench_clip_first_encode(benchmark):
    """Benchmark: very first CLIP encode after model load (includes GPU kernel compilation).

    This is NOT a warm benchmark — it loads the model fresh and measures
    the cold-start latency. Useful for understanding the user experience
    of the first search after server restart.
    """
    import torch

    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.models.clip_model import load as load_model
    from twin.services.embedding import compute_embedding

    img = random_image((224, 224), seed=42)

    def _cold_encode():
        load_model()  # idempotent if already loaded in this process
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return compute_embedding(img)

    benchmark(_cold_encode)
