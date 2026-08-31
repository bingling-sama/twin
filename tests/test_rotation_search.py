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


def test_asymmetric_image_rotation_full_funnel(tmp_path):
    """Asymmetric image rotated by 90° passes all 4 stages to reach 'confirmed' status."""
    import numpy as np

    from twin.services.embedding import compute_embedding

    # Create asymmetric test image (gradient + pattern)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]  # Top-left red
    arr[50:, :50] = [0, 255, 0]  # Bottom-left green
    arr[:50, 50:] = [0, 0, 255]  # Top-right blue
    arr[50:, 50:] = [255, 255, 0]  # Bottom-right yellow
    img = Image.fromarray(arr, mode="RGB")

    saved_path = tmp_path / "asym.png"
    img.save(saved_path)

    vec = compute_embedding(img)
    meta = {
        "filename": "asym.png",
        "path": str(saved_path),
        "dhash": compute_dhash(img),
        "phash": compute_phash(img),
    }
    indexer.add_item(vec, meta)

    # 90° clockwise rotation
    rot_90 = img.transpose(Image.Transpose.ROTATE_90)

    # Without rotation invariance: pHash and SSIM fail because shapes are unaligned
    res_strict = search(rot_90, rotation_invariant=False)
    assert res_strict["count"] == 1
    assert res_strict["results"][0]["match_level"] != "confirmed"

    # With rotation invariance: Stage 2 detects 90° rotation, Stage 3 and Stage 4 evaluate at 90°
    res_rot = search(rot_90, rotation_invariant=True)
    assert res_rot["count"] == 1
    item = res_rot["results"][0]
    assert item["match_level"] == "confirmed"
    assert item["stages_passed"] == 3
    assert item["dhash_distance"] == 0
    assert item["phash_distance"] == 0
    assert item["ssim_score"] >= 0.95


def test_asymmetric_image_rotation_stage1_recall(tmp_path):
    """Rotated asymmetric query is recalled in Stage 1 via orthogonal batch embeddings
    even among multiple distractor images.
    """
    import numpy as np

    from twin.services.embedding import compute_embedding

    # Create target asymmetric test image
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]
    arr[50:, :50] = [0, 255, 0]
    arr[:50, 50:] = [0, 0, 255]
    arr[50:, 50:] = [255, 255, 0]
    img = Image.fromarray(arr, mode="RGB")
    target_path = tmp_path / "target.png"
    img.save(target_path)
    indexer.add_item(
        compute_embedding(img),
        {
            "filename": "target.png",
            "path": str(target_path),
            "dhash": compute_dhash(img),
            "phash": compute_phash(img),
        },
    )

    # Add 5 distractor images
    for i in range(5):
        d_arr = np.full((100, 100, 3), fill_value=i * 40, dtype=np.uint8)
        d_img = Image.fromarray(d_arr, mode="RGB")
        d_path = tmp_path / f"distractor_{i}.png"
        d_img.save(d_path)
        indexer.add_item(
            compute_embedding(d_img),
            {
                "filename": f"distractor_{i}.png",
                "path": str(d_path),
                "dhash": compute_dhash(d_img),
                "phash": compute_phash(d_img),
            },
        )

    # Query with 270° rotated image
    rot_270 = img.transpose(Image.Transpose.ROTATE_270)
    res = search(rot_270, top_k=5, rotation_invariant=True)
    assert res["count"] >= 1
    top_result = res["results"][0]
    assert top_result["filename"] == "target.png"
    assert top_result["match_level"] == "confirmed"
    assert top_result["dhash_distance"] == 0
    assert top_result["phash_distance"] == 0
    assert top_result["ssim_score"] >= 0.95

