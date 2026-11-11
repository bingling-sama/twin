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
from twin.core.config import settings
from twin.main import app
from twin.services.cluster import discover_similar_batches
from twin.services.embedding import get_embedding_dim
from twin.services.hasher import (
    compute_ahash,
    compute_dhash,
    compute_phash,
    hamming_distance,
)
from twin.services.indexer import indexer
from twin.services.verifiers import DiffMaskVerifier


@pytest.fixture(autouse=True)
def clean_index(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Ensure clean indexer state before and after each test inside an isolated temp dir."""
    idx_dir = tmp_path_factory.mktemp("test_discover_index")
    monkeypatch.setattr(settings, "index_path", str(idx_dir))
    monkeypatch.setenv("TWIN_INDEX_PATH", str(idx_dir))
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
    vec = np.random.randn(get_embedding_dim()).astype(np.float32)
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


def test_discover_adaptive_recall_with_eliminated_holes(tmp_path, monkeypatch):
    """Verify adaptive recall expands k > discover_min_recall_k across >= 15 eliminated holes."""
    arr_a = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_a[8:16, 8:24] = [200, 100, 50, 255]
    arr_a[:, :, 3] = 255
    img_a = Image.fromarray(arr_a, mode="RGBA")

    arr_b = np.zeros((64, 64, 4), dtype=np.uint8)
    arr_b[20:32, 20:36] = [50, 200, 100, 255]
    arr_b[:, :, 3] = 255
    img_b = Image.fromarray(arr_b, mode="RGBA")

    # Interleaved index layout: A1 (5) -> B (5) -> A2 (25); total A = 30 > min_recall_k (15).
    # Once the first 15 A items (IDs 0..4, 10..19) are eliminated, top-15 recall holds 15 holes
    # and k must expand past discover_min_recall_k=15 to recover the remaining A items.
    for i in range(5):
        _index_skin(img_a, f"cluster_a1_{i:02d}.png", tmp_path)
    for i in range(5):
        _index_skin(img_b, f"cluster_b_{i:02d}.png", tmp_path)
    for i in range(25):
        _index_skin(img_a, f"cluster_a2_{i:02d}.png", tmp_path)

    seen_ks: list[int] = []
    orig_search_binary = indexer.search_binary

    def _tracking_search_binary(hex_hash: str, k: int = 10, datasource: str = "skin"):
        seen_ks.append(k)
        return orig_search_binary(hex_hash, k=k, datasource=datasource)

    monkeypatch.setattr(indexer, "search_binary", _tracking_search_binary)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=10,
        batch_size=5,
        mode="confirmed",
        seed_strategy="sequential",
    )
    # Expected: 6 batches of A (30 items) and 1 batch of B (5 items) = 7 batches total
    assert res["count"] == 7
    discovered_fnames = set()
    for batch in res["batches"]:
        assert batch["size"] == 5
        discovered_fnames.add(batch["seed"]["filename"])
        for member in batch["members"]:
            discovered_fnames.add(member["filename"])

    assert len(discovered_fnames) == 35
    assert any(k > settings.discover_min_recall_k for k in seen_ks), (
        f"Expected adaptive recall k > {settings.discover_min_recall_k}, got {seen_ks}"
    )


def _find_hamming_variants(
    base_arr: np.ndarray,
    base_img: Image.Image,
    targets: dict[int, int],
    max_diff_ratio: float = 0.05,
    seed: int = 7,
) -> dict[int, list[Image.Image]]:
    """Deterministically generate small-patch RGBA variants hitting exact pHash Hamming targets."""
    verifier = DiffMaskVerifier()
    base_phash = compute_phash(base_img)
    base_int = int(base_phash, 16)
    rng = np.random.default_rng(seed)
    found: dict[int, list[Image.Image]] = {t: [] for t in targets}

    for _ in range(10000):
        cand = base_arr.copy()
        for _ in range(int(rng.integers(1, 6))):
            y, x = int(rng.integers(0, 44)), int(rng.integers(0, 60))
            h, w = int(rng.integers(1, 6)), int(rng.integers(1, 6))
            cand[y : y + h, x : x + w] = [
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
                255,
            ]
        img = Image.fromarray(cand, mode="RGBA")
        hd = (base_int ^ int(compute_phash(img), 16)).bit_count()
        if hd in targets and len(found[hd]) < targets[hd]:
            dr = verifier.verify(base_img, img).diff_ratio
            if dr <= max_diff_ratio:
                found[hd].append(img)
        if all(len(found[t]) >= targets[t] for t in targets):
            return found

    raise AssertionError(f"Could not find all exact Hamming targets {targets}: found {found}")


def test_discover_hamming_boundary_and_ties(tmp_path):
    """Verify Hamming threshold boundaries, tied distances, and full-batch pruning.

    - Candidates with verified Hamming distance <= 4 (including exact boundary 4) are included.
    - Multiple candidates with tied distances are deterministically handled.
    - Candidate with diff_ratio <= 0.05 but Hamming distance 8 > 4 is strictly pruned out
      even when batch_size=8 allows more peers (no early break masking the exclusion).
    - Verifies execution across a non-trivial dataset (> discover_min_recall_k = 15).
    """
    base = np.zeros((64, 64, 4), dtype=np.uint8)
    base[8:16, 8:24] = [200, 100, 50, 255]
    base[20:32, 20:36] = [50, 100, 200, 255]
    base[:, :, 3] = 255
    base_img = Image.fromarray(base, mode="RGBA")
    _index_skin(base_img, "seed_base.png", tmp_path)

    variants = _find_hamming_variants(
        base, base_img, targets={2: 2, 4: 2, 8: 1}, max_diff_ratio=0.05
    )

    # 1. Tied candidates at verified Hamming distance 2
    _index_skin(variants[2][0], "cand_dist2_a.png", tmp_path)
    _index_skin(variants[2][1], "cand_dist2_b.png", tmp_path)

    # 2. Tied candidates at exact confirmed boundary (verified Hamming distance 4)
    _index_skin(variants[4][0], "cand_dist4_a.png", tmp_path)
    _index_skin(variants[4][1], "cand_dist4_b.png", tmp_path)

    # 3. Candidate with diff_ratio <= 0.05 yet verified Hamming distance 8 > 4
    _index_skin(variants[8][0], "cand_dist8_out.png", tmp_path)

    # 4. Add 15 filler noise images so total_items = 21 > discover_min_recall_k = 15
    for i in range(15):
        noise = np.full((64, 64, 4), 10 + i * 10, dtype=np.uint8)
        noise[:, :, 3] = 255
        _index_skin(Image.fromarray(noise, mode="RGBA"), f"noise_{i:02d}.png", tmp_path)

    res = discover_similar_batches(
        datasource="skin",
        max_batches=1,
        batch_size=8,
        mode="confirmed",
        seed_strategy="sequential",
    )
    assert res["count"] == 1
    batch = res["batches"][0]
    assert batch["seed"]["filename"] == "seed_base.png"
    assert batch["size"] == 5
    members = {batch["seed"]["filename"]} | {m["filename"] for m in batch["members"]}

    # Boundary and tied candidates must be included
    assert "cand_dist2_a.png" in members
    assert "cand_dist2_b.png" in members
    assert "cand_dist4_a.png" in members
    assert "cand_dist4_b.png" in members
    # Exceeded candidate and noise skins must be pruned even with batch_size=8
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


@pytest.mark.parametrize(
    ("kwargs", "err_fragment"),
    [
        ({"datasource": "invalid"}, "Unsupported datasource"),
        ({"mode": "invalid"}, "Unsupported discovery mode"),
        ({"seed_strategy": "invalid"}, "Unsupported seed strategy"),
        ({"batch_size": 1}, "batch_size must be >= 2"),
        ({"batch_size": 0}, "batch_size must be >= 2"),
        ({"max_batches": 0}, "max_batches must be >= 1"),
        ({"max_probes": 0}, "max_probes must be >= 1"),
    ],
)
def test_discover_invalid_parameters(kwargs, err_fragment):
    """Service entry point must reject out-of-domain parameters loudly."""
    with pytest.raises(ValueError, match=err_fragment):
        discover_similar_batches(**kwargs)


def test_discover_expands_past_rejected_phash_false_positives(tmp_path):
    """Adaptive recall must expand curr_k when initial candidates fail DiffMask verification."""
    verifier = DiffMaskVerifier()

    # Build rich spatially textured seed so DCT pHash has non-trivial structure
    rng = np.random.default_rng(42)
    base_rgb = np.zeros((64, 64, 4), dtype=np.uint8)
    for by in range(0, 64, 8):
        for bx in range(0, 64, 8):
            g_val = int(rng.integers(40, 210))
            base_rgb[by : by + 8, bx : bx + 8] = [100, g_val, 100, 255]
    seed_img = Image.fromarray(base_rgb, mode="RGBA")
    seed_ph = compute_phash(seed_img)

    _index_skin(seed_img, "seed.png", tmp_path)

    # Create >= discover_min_recall_k luma-preserving RGB shifts (pHash dist=0, diff_ratio > 0.05)
    min_k = max(2 * settings.discover_recall_multiplier, settings.discover_min_recall_k)
    false_count = min_k + 2
    for i in range(false_count):
        arr = base_rgb.copy()
        row_start = (i * 3) % 32
        arr[row_start : row_start + 32, :, 0] = arr[row_start : row_start + 32, :, 0] + 20
        arr[row_start : row_start + 32, :, 2] = arr[row_start : row_start + 32, :, 2] - 52
        false_img = Image.fromarray(arr, mode="RGBA")
        assert hamming_distance(seed_ph, compute_phash(false_img)) == 0
        assert verifier.verify(seed_img, false_img).diff_ratio > 0.05
        _index_skin(false_img, f"false_{i:02d}.png", tmp_path)

    # Create a real confirmed match at pHash distance 2..4 so it sits strictly behind dist=0 items
    match_img = None
    for attempt in range(500):
        cand = base_rgb.copy()
        py = int(rng.integers(0, 56))
        px = int(rng.integers(0, 56))
        cand[py : py + 4, px : px + 4, :3] = [
            int(rng.integers(0, 255)),
            int(rng.integers(0, 255)),
            int(rng.integers(0, 255)),
        ]
        cand_img = Image.fromarray(cand, mode="RGBA")
        dist = hamming_distance(seed_ph, compute_phash(cand_img))
        if 1 <= dist <= settings.discover_confirmed_hamming and (
            verifier.verify(seed_img, cand_img).diff_ratio <= settings.discover_confirmed_diff_ratio
        ):
            match_img = cand_img
            break
    assert match_img is not None, f"Failed to craft test match after {attempt + 1} attempts"
    _index_skin(match_img, "true_match.png", tmp_path)

    # Single probe on id=0 must expand curr_k past min_k rejected dist=0 items to reach true_match
    res = discover_similar_batches(
        datasource="skin",
        max_batches=1,
        batch_size=2,
        mode="confirmed",
        max_probes=1,
        seed_strategy="sequential",
    )

    assert res["count"] == 1
    batch = res["batches"][0]
    assert batch["seed"]["filename"] == "seed.png"
    assert [m["filename"] for m in batch["members"]] == ["true_match.png"]
