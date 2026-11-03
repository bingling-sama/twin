"""Unit tests for seedless similar texture discovery (cluster.py and API)."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from twin.main import app
from twin.services.cluster import discover_similar_batches
from twin.services.hasher import compute_ahash, compute_dhash, compute_phash
from twin.services.indexer import indexer


@pytest.fixture(autouse=True)
def clean_index():
    """Ensure clean indexer state before and after each test."""
    indexer.clear()
    yield
    indexer.clear()


@pytest.fixture
def sample_skins():
    """Create a set of distinct and duplicate synthetic skins."""
    # Base skin A
    arr_a = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_a[8:16, 8:24] = [200, 100, 50, 255]
    arr_a[20:32, 20:36] = [50, 100, 200, 255]
    arr_a[:, :, 3] = 255
    img_a1 = Image.fromarray(arr_a, mode="RGBA")
    img_a2 = Image.fromarray(arr_a.copy(), mode="RGBA")

    # Base skin B
    arr_b = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_b[8:16, 8:24] = [20, 220, 40, 255]
    arr_b[20:32, 20:36] = [220, 20, 40, 255]
    arr_b[:, :, 3] = 255
    img_b1 = Image.fromarray(arr_b, mode="RGBA")
    img_b2 = Image.fromarray(arr_b.copy(), mode="RGBA")

    # Base skin C (singleton, no duplicates)
    arr_c = np.full((64, 64, 4), 180, dtype=np.uint8)
    arr_c[:, :, 3] = 255
    img_c = Image.fromarray(arr_c, mode="RGBA")

    return {
        "a1.png": img_a1,
        "a2.png": img_a2,
        "b1.png": img_b1,
        "b2.png": img_b2,
        "c.png": img_c,
    }


def _index_skin(img: Image.Image, filename: str, tmp_path: Path, datasource: str = "skin") -> Path:
    """Save image to real disk location and index into FAISS with metadata."""
    file_path = tmp_path / filename
    img.save(file_path, format="PNG")
    vec = np.random.randn(512).astype(np.float32)
    meta = {
        "filename": filename,
        "path": str(file_path),
        "datasource": datasource,
        "ahash": compute_ahash(img),
        "dhash": compute_dhash(img),
        "phash": compute_phash(img),
    }
    indexer.add_item(vec, meta, datasource=datasource)
    return file_path


def test_discover_empty_index(tmp_path):
    """Empty or single-item index must return count=0 gracefully with valid telemetry."""
    res = discover_similar_batches(datasource="skin")
    assert res["count"] == 0
    assert res["batches"] == []
    assert "stages" in res
    assert "recall" in res["stages"]

    # Single item
    dummy_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    _index_skin(dummy_img, "single.png", tmp_path)
    res_single = discover_similar_batches(datasource="skin")
    assert res_single["count"] == 0
    assert res_single["batches"] == []


def test_discover_identical_duplicates(tmp_path, sample_skins):
    """Accurately detect duplicate pairs using real disk files without artificial scores."""
    for fname, img in sample_skins.items():
        _index_skin(img, fname, tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=5,
        batch_size=5,
        mode="confirmed",
    )

    assert res["count"] == 2  # exactly two duplicate pairs (A pair and B pair)
    assert len(res["batches"]) == 2

    seen_ids = set()
    for batch in res["batches"]:
        assert batch["size"] == 2
        assert batch["mode"] == "confirmed"
        seed_id = batch["seed"]["id"]
        member_id = batch["members"][0]["id"]

        assert seed_id not in seen_ids
        assert member_id not in seen_ids
        seen_ids.add(seed_id)
        seen_ids.add(member_id)

        # Verified similarity from disk I/O
        assert batch["members"][0]["diff_ratio"] == 0.0
        assert batch["members"][0]["verifier_score"] == 1.0
        assert batch["members"][0]["match_level"] == "confirmed"

    # Telemetry verification (Show your work)
    assert "stages" in res
    assert res["stages"]["verify"]["pairs_verified"] >= 2
    assert res["stages"]["io_load"]["images_loaded"] >= 4


def test_discover_variants_mode(tmp_path):
    """Variants mode should detect edits/recolors and exit immediately without probe inflation."""
    # Base structured skin
    arr_base = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_base[8:16, 8:24] = [200, 100, 50, 255]
    arr_base[20:32, 20:36] = [50, 100, 200, 255]
    arr_base[:, :, 3] = 255
    base = Image.fromarray(arr_base, mode="RGBA")

    # Derivative recolor with diff_ratio in [0.05, 0.35]
    arr_variant = arr_base.copy()
    arr_variant[8:16, 8:24, :3] = [220, 120, 70]
    arr_variant[20:32, 20:36, :3] = [70, 120, 220]
    variant = Image.fromarray(arr_variant, mode="RGBA")

    _index_skin(base, "base.png", tmp_path)
    _index_skin(variant, "variant.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=1,
        batch_size=5,
        mode="variants",
    )

    assert res["count"] == 1
    # Verify no probe inflation: exactly 1 probe evaluated to yield max_batches=1
    assert res["total_probes"] == 1
    assert res["stages"]["recall"]["probes_evaluated"] == 1

    batch = res["batches"][0]
    assert batch["size"] == 2
    assert batch["mode"] == "variants"
    assert batch["members"][0]["match_level"] == "suspected"
    assert 0.05 < batch["members"][0]["diff_ratio"] <= 0.35


def test_discover_variants_no_shortcut_on_high_pixel_diff(tmp_path):
    """Ensure or cand_phash_dist <= 2 shortcut is eliminated.

    Textures with small pHash distance (dist <= 2) but diff_ratio > 0.35 MUST NOT match.
    """
    arr_base = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_base[8:16, 8:24] = [200, 100, 50, 255]
    arr_base[20:32, 20:36] = [50, 100, 200, 255]
    arr_base[:, :, 3] = 255
    base = Image.fromarray(arr_base, mode="RGBA")

    # split_y = 31 yields pHash distance = 2 (<= 2) but diff_ratio = 0.5156 (> 0.35)
    arr_diff = arr_base.copy()
    arr_diff[31:, :, :3] = [30, 30, 30]
    img_diff = Image.fromarray(arr_diff, mode="RGBA")

    _index_skin(base, "base.png", tmp_path)
    _index_skin(img_diff, "diff.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=5,
        batch_size=5,
        mode="variants",
    )

    # Must NOT form any cluster because diff_ratio (0.5156) exceeds variants threshold (0.35)
    assert res["count"] == 0
    assert len(res["batches"]) == 0


def test_discover_elimination_guarantee(tmp_path):
    """A cluster member must be eliminated and never appear in another batch."""
    # 3 identical copies: copy1, copy2, copy3
    img = Image.new("RGBA", (64, 64), (50, 150, 250, 255))
    images = {"c1.png": img, "c2.png": img, "c3.png": img}
    for name, pic in images.items():
        _index_skin(pic, name, tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=10,
        mode="confirmed",
    )

    # All 3 must be swallowed into 1 single batch of size 3
    assert res["count"] == 1
    batch = res["batches"][0]
    assert batch["size"] == 3
    all_members = [batch["seed"]["filename"]] + [m["filename"] for m in batch["members"]]
    assert set(all_members) == {"c1.png", "c2.png", "c3.png"}


def test_discover_cape_datasource(tmp_path):
    """Verify multi-datasource isolation: cape queries only discover within cape partition."""
    # Cape textures (standard 64x32 layout)
    arr_cape = np.zeros((32, 64, 4), dtype=np.uint8)
    arr_cape[:17, :22] = [200, 30, 30, 255]
    arr_cape[:, :, 3] = 255
    cape1 = Image.fromarray(arr_cape, mode="RGBA")
    cape2 = Image.fromarray(arr_cape.copy(), mode="RGBA")

    # Skin textures (64x64)
    arr_skin = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_skin[8:16, 8:24] = [30, 200, 30, 255]
    arr_skin[:, :, 3] = 255
    skin1 = Image.fromarray(arr_skin, mode="RGBA")
    skin2 = Image.fromarray(arr_skin.copy(), mode="RGBA")

    _index_skin(cape1, "cape1.png", tmp_path, datasource="cape")
    _index_skin(cape2, "cape2.png", tmp_path, datasource="cape")
    _index_skin(skin1, "skin1.png", tmp_path, datasource="skin")
    _index_skin(skin2, "skin2.png", tmp_path, datasource="skin")

    # Run discovery on 'cape'
    res_cape = discover_similar_batches(
        datasource="cape",
        max_batches=5,
        mode="confirmed",
    )

    assert res_cape["count"] == 1
    assert res_cape["datasource"] == "cape"
    batch_cape = res_cape["batches"][0]
    assert batch_cape["datasource"] == "cape"
    assert batch_cape["seed"]["datasource"] == "cape"
    assert batch_cape["members"][0]["datasource"] == "cape"
    cape_fnames = {batch_cape["seed"]["filename"], batch_cape["members"][0]["filename"]}
    assert cape_fnames == {"cape1.png", "cape2.png"}

    # Run discovery on 'skin'
    res_skin = discover_similar_batches(
        datasource="skin",
        max_batches=5,
        mode="confirmed",
    )
    assert res_skin["count"] == 1
    assert res_skin["datasource"] == "skin"
    batch_skin = res_skin["batches"][0]
    skin_fnames = {batch_skin["seed"]["filename"], batch_skin["members"][0]["filename"]}
    assert skin_fnames == {"skin1.png", "skin2.png"}


def test_discover_batches_api_endpoint(tmp_path, sample_skins):
    """GET /api/v1/discover/batches must return valid DiscoverResponse reusing global app."""
    for fname, img in sample_skins.items():
        _index_skin(img, fname, tmp_path)

    client = TestClient(app)
    response = client.get("/api/v1/discover/batches?datasource=skin&max_batches=2&mode=confirmed")
    assert response.status_code == 200
    data = response.json()

    assert "batches" in data
    assert "count" in data
    assert "query_time_ms" in data
    assert "stages" in data
    assert "recall" in data["stages"]
    assert "verify" in data["stages"]
    assert data["datasource"] == "skin"
    assert data["mode"] == "confirmed"
    assert data["count"] <= 2
