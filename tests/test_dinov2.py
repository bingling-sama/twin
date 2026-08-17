"""Tests for DINOv2 embedding model and service dispatch."""

from pathlib import Path

import numpy as np
from PIL import Image

from twin.core.config import settings
from twin.models import dinov2_model
from twin.services.embedding import compute_embedding, compute_embeddings

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Image.Image:
    return Image.open(FIXTURES / name).convert("RGB")


def test_dinov2_model_load_and_encode():
    """DINOv2 loads and extracts normalized vectors with matching dimensionality."""
    dinov2_model.load(device="cpu", model_name="vit_small_patch14_dinov2")
    assert dinov2_model.is_loaded()
    assert dinov2_model.get_embedding_dim() == 384

    img = _load("red.png")
    feat = dinov2_model.encode_image(img)
    assert feat.shape == (1, 384)
    # Check L2 normalization
    norm = feat.norm(dim=-1).item()
    assert abs(norm - 1.0) < 1e-4

    # Batch encode
    imgs = [_load("red.png"), _load("blue.png")]
    batch_feats = dinov2_model.encode_images(imgs)
    assert batch_feats.shape == (2, 384)


def test_dinov2_service_dispatch(monkeypatch):
    """Embedding service delegates to DINOv2 when model_type is configured."""
    dinov2_model.load(device="cpu", model_name="vit_small_patch14_dinov2")
    monkeypatch.setattr(settings, "model_type", "dinov2")
    monkeypatch.setattr(settings, "embedding_dim", 384)

    img = _load("red.png")
    vec = compute_embedding(img)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    # Batch compute
    imgs = [_load("red.png"), _load("blue.png")]
    vecs = compute_embeddings(imgs)
    assert vecs.shape == (2, 384)
