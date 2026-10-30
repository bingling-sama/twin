"""Unit tests for seedless similar texture discovery (cluster.py and API)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from twin.api.routes import router
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


def _index_skin(img: Image.Image, filename: str, datasource: str = "skin"):
    vec = np.random.randn(512).astype(np.float32)
    meta = {
        "filename": filename,
        "path": f"/tmp/virtual/{filename}",
        "datasource": datasource,
        "ahash": compute_ahash(img),
        "dhash": compute_dhash(img),
        "phash": compute_phash(img),
    }
    indexer.add_item(vec, meta, datasource=datasource)


def test_discover_empty_index():
    """Empty or single-item index must return count=0 gracefully."""
    res = discover_similar_batches(datasource="skin")
    assert res["count"] == 0
    assert res["batches"] == []

    # Single item
    dummy_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    _index_skin(dummy_img, "single.png")
    res_single = discover_similar_batches(datasource="skin")
    assert res_single["count"] == 0
    assert res_single["batches"] == []


def test_discover_identical_duplicates(sample_skins):
    """Must accurately detect confirmed duplicate pairs and eliminate them from candidate pool."""
    for fname, img in sample_skins.items():
        _index_skin(img, fname)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=5,
        batch_size=5,
        mode="confirmed",
        images_map=sample_skins,
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

        # Verified similarity
        assert batch["members"][0]["diff_ratio"] == 0.0
        assert batch["members"][0]["verifier_score"] == 1.0
        assert batch["members"][0]["match_level"] == "confirmed"


def test_discover_variants_mode():
    """Variants mode should detect edits/recolors with match_level='suspected'."""
    # Base skin
    base = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
    # Subtle recolor with small pHash shift
    variant = Image.new("RGBA", (64, 64), (105, 145, 195, 255))

    images = {"base.png": base, "variant.png": variant}
    _index_skin(base, "base.png")
    _index_skin(variant, "variant.png")

    res = discover_similar_batches(
        datasource="skin",
        max_batches=5,
        batch_size=5,
        mode="variants",
        images_map=images,
    )

    assert res["count"] == 1
    batch = res["batches"][0]
    assert batch["size"] == 2
    assert batch["mode"] == "variants"
    assert batch["members"][0]["match_level"] == "suspected"


def test_discover_elimination_guarantee():
    """A cluster member must be eliminated and never appear in another batch."""
    # 3 identical copies: copy1, copy2, copy3
    img = Image.new("RGBA", (64, 64), (50, 150, 250, 255))
    images = {"c1.png": img, "c2.png": img, "c3.png": img}
    for name, pic in images.items():
        _index_skin(pic, name)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=10,
        mode="confirmed",
        images_map=images,
    )

    # All 3 must be swallowed into 1 single batch of size 3
    assert res["count"] == 1
    batch = res["batches"][0]
    assert batch["size"] == 3
    all_members = [batch["seed"]["filename"]] + [m["filename"] for m in batch["members"]]
    assert set(all_members) == {"c1.png", "c2.png", "c3.png"}


def test_discover_batches_api_endpoint(sample_skins):
    """GET /api/v1/discover/batches must return valid DiscoverResponse schema."""
    from fastapi import FastAPI

    for fname, img in sample_skins.items():
        _index_skin(img, fname)

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    response = client.get("/api/v1/discover/batches?datasource=skin&max_batches=2&mode=confirmed")
    assert response.status_code == 200
    data = response.json()

    assert "batches" in data
    assert "count" in data
    assert "query_time_ms" in data
    assert data["datasource"] == "skin"
    assert data["mode"] == "confirmed"
    assert data["count"] <= 2
