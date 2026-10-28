"""Tests for search pipeline — helpers, edge cases, and full flow."""

from pathlib import Path

import pytest
from PIL import Image

from twin.models.clip_model import load as load_model
from twin.services.indexer import indexer
from twin.services.search import (
    SearchCandidate,
    _assign_match_level,
    _build_response,
    _empty,
    _final_sort,
    search,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _ensure_model():
    """Load CLIP model if not already loaded."""
    load_model()


@pytest.fixture(autouse=True)
def _clear_index():
    """Reset indexer state before and after each test."""
    indexer.clear()
    yield
    indexer.clear()


# ---------------------------------------------------------------------------
# _assign_match_level
# ---------------------------------------------------------------------------
def test_assign_match_level_confirmed():
    assert _assign_match_level(3) == "confirmed"
    assert _assign_match_level(4) == "confirmed"


def test_assign_match_level_suspected():
    assert _assign_match_level(2) == "suspected"


def test_assign_match_level_none():
    assert _assign_match_level(1) == "none"
    assert _assign_match_level(0) == "none"


# ---------------------------------------------------------------------------
# _final_sort
# ---------------------------------------------------------------------------
def test_final_sort():
    """Results sorted by: stages_passed DESC, dhash_distance ASC, distance ASC."""
    # Test with dicts
    results = [
        {"stages_passed": 1, "dhash_distance": 5, "distance": 0.5, "id": 1},
        {"stages_passed": 3, "dhash_distance": 10, "distance": 1.0, "id": 2},
        {"stages_passed": 3, "dhash_distance": 2, "distance": 1.5, "id": 3},
        {"stages_passed": 0, "dhash_distance": 0, "distance": 0.1, "id": 4},
        {"stages_passed": 1, "dhash_distance": 3, "distance": 0.5, "id": 5},
    ]
    _final_sort(results)
    ids = [r["id"] for r in results]
    assert ids == [3, 2, 5, 1, 4]

    # Test with typed SearchCandidate instances
    candidate_results = [
        SearchCandidate(
            id=1, filename="1.png", distance=0.5, meta={}, dhash_distance=5, stages_passed=1
        ),
        SearchCandidate(
            id=2, filename="2.png", distance=1.0, meta={}, dhash_distance=10, stages_passed=3
        ),
        SearchCandidate(
            id=3, filename="3.png", distance=1.5, meta={}, dhash_distance=2, stages_passed=3
        ),
        SearchCandidate(
            id=4, filename="4.png", distance=0.1, meta={}, dhash_distance=0, stages_passed=0
        ),
        SearchCandidate(
            id=5, filename="5.png", distance=0.5, meta={}, dhash_distance=3, stages_passed=1
        ),
    ]
    _final_sort(candidate_results)
    c_ids = [c.id for c in candidate_results]
    assert c_ids == [3, 2, 5, 1, 4]


# ---------------------------------------------------------------------------
# _empty / _build_response
# ---------------------------------------------------------------------------
def test_empty_response():
    import time

    t0 = time.perf_counter()
    stages = {"faiss": {"in": 0}}
    result = _empty(t0, stages)
    assert result["count"] == 0
    assert result["results"] == []
    assert result["query_time_ms"] >= 0
    assert stages["faiss"]["elapsed_ms"] == 0


def test_build_response():
    import time

    t0 = time.perf_counter()
    stages = {"faiss": {"in": 1, "out": 1, "elapsed_ms": 5.0}}
    results = [
        {
            "id": 0,
            "filename": "test.png",
            "distance": 0.123,
            "meta": {"dhash": "abc", "phash": "def", "path": "/tmp/test.png"},
            "dhash_distance": 3,
            "phash_distance": 5,
            "ssim_score": 0.95,
            "stages_passed": 3,
        },
    ]
    resp = _build_response(results, stages, t0)
    assert resp["count"] == 1
    item = resp["results"][0]
    assert item["match_level"] == "confirmed"
    assert item["dhash_hex"] == "abc"
    assert item["phash_hex"] == "def"
    assert item["path"] == "/tmp/test.png"
    assert "meta" not in item  # internal key stripped

    # Test with SearchCandidate
    cand = SearchCandidate(
        id=1,
        filename="cand.png",
        distance=0.456,
        meta={"dhash": "111", "phash": "222", "path": "/tmp/cand.png"},
        dhash_distance=2,
        phash_distance=4,
        verifier_score=0.98,
        verifier_type="diffmask",
        datasource="skin",
        ssim_score=0.98,
        stages_passed=3,
    )
    resp_cand = _build_response([cand], stages, t0)
    assert resp_cand["count"] == 1
    cand_item = resp_cand["results"][0]
    assert cand_item["id"] == 1
    assert cand_item["match_level"] == "confirmed"
    assert cand_item["verifier_score"] == 0.98
    assert cand_item["verifier_type"] == "diffmask"
    assert cand_item["datasource"] == "skin"
    assert cand_item["dhash_hex"] == "111"
    assert cand_item["phash_hex"] == "222"
    assert cand_item["path"] == "/tmp/cand.png"


def test_final_sort_verifier_score_priority():
    """Candidates in the same stage (confirmed) are sorted by verifier_score DESC."""
    cands = [
        SearchCandidate(
            id=1, filename="1.png", distance=0.1, meta={},
            stages_passed=3, verifier_score=0.92, phash_distance=2
        ),
        SearchCandidate(
            id=2, filename="2.png", distance=0.2, meta={},
            stages_passed=3, verifier_score=1.00, phash_distance=4
        ),
        SearchCandidate(
            id=3, filename="3.png", distance=0.0, meta={},
            stages_passed=3, verifier_score=0.95, phash_distance=0
        ),
    ]
    _final_sort(cands)
    ids = [c.id for c in cands]
    # id=2 (score 1.0) must come first, followed by id=3 (0.95), then id=1 (0.92)
    assert ids == [2, 3, 1]


# ---------------------------------------------------------------------------
# Full search — edge cases
# ---------------------------------------------------------------------------
def test_search_empty_index():
    """Search against empty index returns empty results."""
    _ensure_model()
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    result = search(img)
    assert result["count"] == 0
    assert result["results"] == []


def test_search_explicit_params():
    """Search respects explicit threshold parameters."""
    _ensure_model()
    img = Image.open(FIXTURES / "red.png").convert("RGB")

    # First index red.png
    from twin.services.embedding import compute_embedding
    from twin.services.hasher import compute_dhash, compute_phash

    v = compute_embedding(img)
    indexer.add_item(
        v,
        {
            "filename": "red.png",
            "path": str(FIXTURES / "red.png"),
            "dhash": compute_dhash(img),
            "phash": compute_phash(img),
        },
    )

    # Search with custom params
    result = search(img, top_k=10, dhash_threshold=5, phash_threshold=5, ssim_threshold=0.80)
    assert result["count"] == 1
    assert result["results"][0]["filename"] == "red.png"


def test_search_unknown_texture_returns_empty():
    """Searching with an unrecognized texture dimension returns empty and datasource='unknown'."""
    _ensure_model()
    img = Image.new("RGBA", (100, 50), (128, 128, 128, 255))
    result = search(img)
    assert result["count"] == 0
    assert result["results"] == []
    assert result["datasource"] == "unknown"
    assert "warning" in result["stages"]["classifier"]

