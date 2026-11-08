"""Unit tests for seedless similar texture discovery (cluster.py and API)."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from twin.api.schemas import (
    DiscoveredMemberItem,
    DiscoverResponse,
    DiscoverStages,
    SimilarBatchItem,
)
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


def test_discover_recall_starvation_large_duplicates(tmp_path):
    """[P1 Bugfix Test] Large cluster (> initial recall_k) must not suffer recall starvation.

    When 30 identical textures exist and batch_size=5, initial recall_k=15 will exhaust
    fresh candidates after batch 3. Adaptive recall expansion must kick in to successfully
    discover all 6 full batches (30 items total).
    """
    img = Image.new("RGBA", (64, 64), (120, 80, 200, 255))
    total_imgs = 30
    for i in range(total_imgs):
        _index_skin(img, f"dup_{i:02d}.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=50,
        batch_size=5,
        mode="confirmed",
        seed_strategy="sequential",
        max_probes=200,
    )

    # 30 items divided by batch_size 5 = exactly 6 batches
    assert res["count"] == 6
    assert len(res["batches"]) == 6

    discovered_files = set()
    for batch in res["batches"]:
        assert batch["size"] == 5
        discovered_files.add(batch["seed"]["filename"])
        for member in batch["members"]:
            discovered_files.add(member["filename"])

    # All 30 files must be discovered without dropping any into starvation
    assert len(discovered_files) == total_imgs
    expected_files = {f"dup_{i:02d}.png" for i in range(total_imgs)}
    assert discovered_files == expected_files


def test_discover_probes_budget_exhaustion(tmp_path):
    """[P2 Bugfix Test] Probes counter must strictly equal max_probes when budget exhausts.

    When max_probes is reached, the probe count and telemetry must NOT report max_probes + 1.
    """
    # Index 20 distinct textures with no duplicate peers
    for i in range(20):
        arr = np.full((64, 64, 4), i * 10, dtype=np.uint8)
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, mode="RGBA")
        _index_skin(img, f"solo_{i:02d}.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=50,
        batch_size=5,
        mode="confirmed",
        seed_strategy="sequential",
        max_probes=10,
    )

    # Exactly 10 probes evaluated, NO phantom 11th probe recorded
    assert res["total_probes"] == 10
    assert res["stages"]["recall"]["probes_evaluated"] == 10
    assert res["count"] == 0


def test_discover_transitivity_collapse_prevention(tmp_path):
    """[Spec Anti-Collapse Test] Star topology must prevent transitive small-world avalanche.

    Given A ≈ B (diff 0.20), B ≈ C (diff 0.31), but A ≉ C (diff 0.52):
    Traditional union-find / graph clustering causes transitivity collapse {A, B, C}.
    Greedy leader elimination forms {A, B}, eliminates {A, B}, and isolates C cleanly.
    """
    arr_a = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_a[:, :, :3] = [100, 150, 200]
    arr_a[:, :, 3] = 255

    # B has 20% diff from A
    arr_b = arr_a.copy()
    arr_b[:13, :, :3] = [200, 80, 50]

    # C has 51.5% diff from A, but 31.25% diff from B
    arr_c = arr_a.copy()
    arr_c[:33, :, :3] = [200, 80, 50]

    _index_skin(Image.fromarray(arr_a, mode="RGBA"), "skin_a.png", tmp_path)
    _index_skin(Image.fromarray(arr_b, mode="RGBA"), "skin_b.png", tmp_path)
    _index_skin(Image.fromarray(arr_c, mode="RGBA"), "skin_c.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=5,
        mode="variants",
        seed_strategy="sequential",
    )

    # Only 1 cluster formed: {skin_a, skin_b}. skin_c must NEVER be dragged in!
    assert res["count"] == 1
    batch = res["batches"][0]
    cluster_members = {batch["seed"]["filename"]} | {m["filename"] for m in batch["members"]}
    assert cluster_members == {"skin_a.png", "skin_b.png"}
    assert "skin_c.png" not in cluster_members


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


def test_discover_adaptive_recall_with_eliminated_holes(tmp_path):
    """Verify adaptive recall penetrates interleaved clusters with eliminated holes.

    Constructs two distinct duplicate clusters (Skin A and Skin B) interleaved in the index.
    After the first 2 batches consume and eliminate 10 items (5 of A, 5 of B), the remaining
    10 items of Skin A reside behind these eliminated nodes.
    The adaptive recall must expand its search window to bypass eliminated holes and
    successfully discover all remaining clusters with zero omissions.
    """
    arr_a = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_a[8:16, 8:24] = [200, 100, 50, 255]
    arr_a[:, :, 3] = 255
    img_a = Image.fromarray(arr_a, mode="RGBA")

    arr_b = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_b[20:32, 20:36] = [50, 200, 100, 255]
    arr_b[:, :, 3] = 255
    img_b = Image.fromarray(arr_b, mode="RGBA")

    # Interleaved index layout: A1 (5) -> B (5) -> A2 (10)
    for i in range(5):
        _index_skin(img_a, f"cluster_a1_{i:02d}.png", tmp_path)
    for i in range(5):
        _index_skin(img_b, f"cluster_b_{i:02d}.png", tmp_path)
    for i in range(10):
        _index_skin(img_a, f"cluster_a2_{i:02d}.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=5,
        mode="confirmed",
        seed_strategy="sequential",
    )
    # Expected: 3 batches of A (15 items) and 1 batch of B (5 items) = 4 batches total
    assert res["count"] == 4
    discovered_fnames = set()
    for batch in res["batches"]:
        assert batch["size"] == 5
        discovered_fnames.add(batch["seed"]["filename"])
        for member in batch["members"]:
            discovered_fnames.add(member["filename"])

    assert len(discovered_fnames) == 20
    assert res["stages"]["recall"]["candidates_recalled"] >= 20


def test_discover_hamming_boundary_and_ties(tmp_path):
    """Verify Hamming threshold boundaries, tied distances, and K-window scaling.

    - Candidates with distance <= threshold (including exact boundary) are evaluated.
    - Multiple candidates with tied distances are deterministically handled.
    - Candidates exceeding the Hamming threshold (dist=8 > 4) are strictly pruned out.
    - Verifies execution across a non-trivial dataset (> discover_min_recall_k = 15).
    """
    base = np.zeros((64, 64, 4), dtype=np.uint8)
    base[8:16, 8:24] = [200, 100, 50, 255]
    base[20:32, 20:36] = [50, 100, 200, 255]
    base[:, :, 3] = 255
    base_img = Image.fromarray(base, mode="RGBA")
    _index_skin(base_img, "seed_base.png", tmp_path)

    # 1. Tied candidates at Hamming distance 2
    cand_dist2_a = base.copy()
    cand_dist2_a[40:41, 40:41] = [255, 255, 255, 255]
    _index_skin(Image.fromarray(cand_dist2_a, mode="RGBA"), "cand_dist2_a.png", tmp_path)

    cand_dist2_b = base.copy()
    cand_dist2_b[40:42, 40:42] = [255, 255, 255, 255]
    _index_skin(Image.fromarray(cand_dist2_b, mode="RGBA"), "cand_dist2_b.png", tmp_path)

    # 2. Tied candidates at exact confirmed boundary (Hamming distance 4)
    cand_dist4_a = base.copy()
    cand_dist4_a[40:43, 40:43] = [255, 255, 255, 255]
    _index_skin(Image.fromarray(cand_dist4_a, mode="RGBA"), "cand_dist4_a.png", tmp_path)

    cand_dist4_b = base.copy()
    cand_dist4_b[40:44, 40:44] = [255, 255, 255, 255]
    _index_skin(Image.fromarray(cand_dist4_b, mode="RGBA"), "cand_dist4_b.png", tmp_path)

    # 3. Candidate strictly exceeding confirmed boundary (Hamming distance 8 > 4)
    cand_dist8_out = base.copy()
    cand_dist8_out[40:45, 40:45] = [255, 255, 255, 255]
    _index_skin(Image.fromarray(cand_dist8_out, mode="RGBA"), "cand_dist8_out.png", tmp_path)

    # 4. Add 15 filler noise images so total_items = 21 > discover_min_recall_k = 15
    for i in range(15):
        noise = np.full((64, 64, 4), 10 + i * 10, dtype=np.uint8)
        noise[:, :, 3] = 255
        _index_skin(Image.fromarray(noise, mode="RGBA"), f"noise_{i:02d}.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=1,
        batch_size=5,
        mode="confirmed",
        seed_strategy="sequential",
    )
    assert res["count"] == 1
    batch = res["batches"][0]
    members = {batch["seed"]["filename"]} | {m["filename"] for m in batch["members"]}

    # Boundary and tied candidates must be included
    assert "cand_dist2_a.png" in members
    assert "cand_dist2_b.png" in members
    assert "cand_dist4_a.png" in members
    assert "cand_dist4_b.png" in members
    # Exceeded candidate and noise skins must be pruned
    assert "cand_dist8_out.png" not in members
    for i in range(15):
        assert f"noise_{i:02d}.png" not in members


def test_discover_batches_api_minimal_probes(tmp_path, sample_skins):
    """API must accept max_probes=1 for minimal budget and reject max_probes=0."""
    for fname, img in sample_skins.items():
        _index_skin(img, fname, tmp_path)

    client = TestClient(app)

    # Valid: max_probes=1 (ge=1)
    res_valid = client.get("/api/v1/discover/batches?datasource=skin&max_probes=1")
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["total_probes"] == 1
    assert data["stages"]["recall"]["probes_evaluated"] == 1

    # Invalid: max_probes=0 (violates ge=1)
    res_invalid = client.get("/api/v1/discover/batches?datasource=skin&max_probes=0")
    assert res_invalid.status_code == 422


def test_discover_response_schema_validation():
    """Verify strong typing and validation guards of discovery Pydantic models."""
    member = DiscoveredMemberItem(
        id=1,
        filename="skin_1.png",
        datasource="skin",
        match_level="confirmed",
    )
    assert member.datasource == "skin"
    assert member.match_level == "confirmed"

    batch = SimilarBatchItem(
        batch_id=1,
        size=2,
        mode="confirmed",
        datasource="skin",
        seed=member,
        members=[member],
    )
    assert batch.mode == "confirmed"

    response = DiscoverResponse(
        batches=[batch],
        count=1,
        total_probes=1,
        query_time_ms=2.5,
        datasource="skin",
        mode="confirmed",
        stages={
            "recall": {"probes_evaluated": 1, "candidates_recalled": 5, "elapsed_ms": 0.8},
            "io_load": {"images_loaded": 2, "elapsed_ms": 0.4},
            "verify": {
                "verifier": "DiffMaskVerifier",
                "pairs_verified": 1,
                "batches_formed": 1,
                "elapsed_ms": 1.3,
            },
        },
    )
    assert isinstance(response.stages, DiscoverStages)
    assert response.stages.recall.probes_evaluated == 1
    assert response.stages.verify.verifier == "DiffMaskVerifier"

    # Strict Literal domain validation
    with pytest.raises(ValidationError):
        DiscoveredMemberItem(id=1, filename="x.png", datasource="invalid_source")

    with pytest.raises(ValidationError):
        SimilarBatchItem(batch_id=1, size=1, mode="invalid_mode", seed=member, members=[])

    with pytest.raises(ValidationError):
        DiscoveredMemberItem(id=1, filename="x.png", match_level="unknown_level")

    # Nested stages validation guards
    with pytest.raises(ValidationError):
        DiscoverResponse(
            batches=[],
            count=0,
            total_probes=0,
            query_time_ms=0.0,
            datasource="skin",
            mode="confirmed",
            stages={
                "verify": {"verifier": "UnknownVerifier"},
            },
        )

    with pytest.raises(ValidationError):
        DiscoverResponse(
            batches=[],
            count=0,
            total_probes=0,
            query_time_ms=0.0,
            datasource="skin",
            mode="confirmed",
            stages={
                "recall": {"probes_evaluated": "not_an_int"},
            },
        )
