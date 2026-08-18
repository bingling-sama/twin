"""Tests for embedding service."""

from pathlib import Path

import numpy as np
from PIL import Image

from twin.core.config import settings
from twin.services.embedding import (
    compute_embedding,
    compute_embeddings,
    compute_text_embedding,
    compute_text_embeddings,
    load_model,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _ensure_model():
    """Load CLIP model if not already loaded. load() is idempotent."""
    settings.model_type = "clip"
    settings.embedding_dim = 512
    settings.model_name = "ViT-B-32"
    load_model(device="cpu", model_name="ViT-B-32", pretrained="openai")


def test_embedding_shape():
    """Embedding should be 512-dim float32."""
    _ensure_model()
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    vec = compute_embedding(img)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert vec.dtype == np.float32


def test_embedding_normalized():
    """CLIP embeddings should be L2-normalized (norm ≈ 1.0)."""
    _ensure_model()
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    vec = compute_embedding(img)

    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4, f"Expected norm ≈ 1.0, got {norm}"


def test_embedding_deterministic():
    """Same input → same output (inference mode)."""
    _ensure_model()
    img_a = Image.open(FIXTURES / "red.png").convert("RGB")
    img_b = Image.open(FIXTURES / "red.png").convert("RGB")

    v1 = compute_embedding(img_a)
    v2 = compute_embedding(img_b)

    assert np.allclose(v1, v2, atol=1e-6)


def test_embedding_grayscale_handled():
    """Grayscale images should be convertible (Pillow handles this)."""
    _ensure_model()
    img = Image.new("L", (224, 224), 128)
    img = img.convert("RGB")
    vec = compute_embedding(img)
    assert vec.shape == (512,)


def test_similar_images_have_similar_embeddings():
    """Red and red_variant should be closer in embedding space than red and blue."""
    _ensure_model()
    red = Image.open(FIXTURES / "red.png").convert("RGB")
    variant = Image.open(FIXTURES / "red_variant.png").convert("RGB")
    blue = Image.open(FIXTURES / "blue.png").convert("RGB")

    v_red = compute_embedding(red)
    v_var = compute_embedding(variant)
    v_blue = compute_embedding(blue)

    dist_similar = float(np.linalg.norm(v_red - v_var))
    dist_different = float(np.linalg.norm(v_red - v_blue))

    assert dist_similar < dist_different, (
        f"Similar={dist_similar:.4f} should be < Different={dist_different:.4f}"
    )


# ---------------------------------------------------------------------------
# Batch embedding — edge cases
# ---------------------------------------------------------------------------
def test_compute_embeddings_empty():
    """Empty image list returns (0, 512) array."""
    result = compute_embeddings([])
    assert result.shape == (0, 512)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Text embedding tests
# ---------------------------------------------------------------------------
def test_text_embedding_shape_and_norm():
    """Text embedding is 512-dim float32 and L2 normalized."""
    _ensure_model()
    vec = compute_text_embedding("a red apple")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_batch_text_embeddings():
    """compute_text_embeddings returns (N, 512) array."""
    _ensure_model()
    assert compute_text_embeddings([]).shape == (0, 512)
    texts = ["a red square", "a blue ocean"]
    vecs = compute_text_embeddings(texts)
    assert vecs.shape == (2, 512)
    assert vecs.dtype == np.float32

