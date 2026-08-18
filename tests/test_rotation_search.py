"""Tests for rotation-invariant image search and dHash funnel filtering."""

from pathlib import Path

import pytest
from PIL import Image

from twin.models.clip_model import load as load_model
from twin.services.hasher import compute_dhash, compute_phash
from twin.services.indexer import indexer
from twin.services.search import search

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _setup_test():
    """Ensure model is loaded and clear index."""
    load_model()
    indexer.clear()
    yield
    indexer.clear()


def test_rotation_invariant_search_90_180_270():
    """Rotated variants (90°, 180°, 270°) match with dHash distance 0 and pass Stage 2."""
    from twin.services.embedding import compute_embedding

    original = Image.open(FIXTURES / "red.png").convert("RGB")
    vec = compute_embedding(original)

    # Index the original image
    meta = {
        "filename": "red.png",
        "path": str(FIXTURES / "red.png"),
        "dhash": compute_dhash(original),
        "phash": compute_phash(original),
    }
    indexer.add_item(vec, meta)

    # Query with 90-degree rotated image
    rot_90 = original.transpose(Image.Transpose.ROTATE_90)
    res_90 = search(rot_90, rotation_invariant=True)
    assert res_90["count"] == 1
    item_90 = res_90["results"][0]
    assert item_90["filename"] == "red.png"
    assert item_90["dhash_distance"] == 0
    assert item_90["stages_passed"] >= 1

    # Query with 180-degree rotated image
    rot_180 = original.transpose(Image.Transpose.ROTATE_180)
    res_180 = search(rot_180, rotation_invariant=True)
    assert res_180["count"] == 1
    item_180 = res_180["results"][0]
    assert item_180["filename"] == "red.png"
    assert item_180["dhash_distance"] == 0

    # Query with 270-degree rotated image
    rot_270 = original.transpose(Image.Transpose.ROTATE_270)
    res_270 = search(rot_270, rotation_invariant=True)
    assert res_270["count"] == 1
    item_270 = res_270["results"][0]
    assert item_270["filename"] == "red.png"
    assert item_270["dhash_distance"] == 0
