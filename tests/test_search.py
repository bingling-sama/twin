"""Tests for search pipeline — helpers, edge cases, and full flow."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from twin.models.clip_model import load as load_model
from twin.services.indexer import indexer
from twin.services.search import (
    _assign_match_level,
    _build_response,
    _empty,
    _final_sort,
    _passes_hash,
    _passes_ssim,
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
# _passes_hash
# ---------------------------------------------------------------------------
def test_passes_hash_both_valid():
    ok, dist = _passes_hash("a1b2c3d4e5f67890", "a1b2c3d4e5f67890", 10)
    assert ok is True
    assert dist == 0


def test_passes_hash_empty_query():
    """Empty query hash → False, 999."""
    ok, dist = _passes_hash("", "a1b2c3d4e5f67890", 10)
    assert ok is False
    assert dist == 999


def test_passes_hash_empty_candidate():
    """Empty candidate hash → False, 999."""
    ok, dist = _passes_hash("a1b2c3d4e5f67890", "", 10)
    assert ok is False
    assert dist == 999


def test_passes_hash_both_empty():
    ok, dist = _passes_hash("", "", 10)
    assert ok is False
    assert dist == 999


def test_passes_hash_beyond_threshold():
    """Very different hashes exceed threshold."""
    # Hash 1: all zeros, Hash 2: all ones → max distance
    ok, dist = _passes_hash("0000000000000000", "ffffffffffffffff", 10)
    assert ok is False
    assert dist == 64  # all 64 bits differ


# ---------------------------------------------------------------------------
# _passes_ssim
# ---------------------------------------------------------------------------
def test_passes_ssim_missing_path():
    """Non-existent file path → False, 0.0."""
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    ok, score = _passes_ssim(img, "/tmp/nonexistent_ssim_test.png", 0.9)
    assert ok is False
    assert score == 0.0


def test_passes_ssim_empty_path():
    """Empty path → False."""
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    ok, score = _passes_ssim(img, "", 0.9)
    assert ok is False
    assert score == 0.0


def test_passes_ssim_corrupt_file():
    """Path exists but isn't a valid image → False."""
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"not an image")
        corrupt_path = f.name

    try:
        ok, score = _passes_ssim(img, corrupt_path, 0.9)
        assert ok is False
        assert score == 0.0
    finally:
        Path(corrupt_path).unlink(missing_ok=True)


def test_passes_ssim_identical():
    """Same image compared to itself → high SSIM, passes."""
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    ok, score = _passes_ssim(img, str(FIXTURES / "red.png"), 0.9)
    assert ok is True
    assert score >= 0.9


# ---------------------------------------------------------------------------
# _assign_match_level
# ---------------------------------------------------------------------------
def test_assign_match_level_confirmed():
    assert _assign_match_level(3) == "confirmed"
    assert _assign_match_level(4) == "confirmed"


def test_assign_match_level_suspected():
    assert _assign_match_level(1) == "suspected"
    assert _assign_match_level(2) == "suspected"


def test_assign_match_level_none():
    assert _assign_match_level(0) == "none"


# ---------------------------------------------------------------------------
# _final_sort
# ---------------------------------------------------------------------------
def test_final_sort():
    """Results sorted by: stages_passed DESC, dhash_distance ASC, distance ASC."""
    results = [
        {"stages_passed": 1, "dhash_distance": 5, "distance": 0.5, "id": 1},
        {"stages_passed": 3, "dhash_distance": 10, "distance": 1.0, "id": 2},
        {"stages_passed": 3, "dhash_distance": 2, "distance": 1.5, "id": 3},
        {"stages_passed": 0, "dhash_distance": 0, "distance": 0.1, "id": 4},
        {"stages_passed": 1, "dhash_distance": 3, "distance": 0.5, "id": 5},
    ]
    _final_sort(results)
    ids = [r["id"] for r in results]
    # id=3 (3, 2, 1.5) > id=2 (3, 10, 1.0) > id=5 (1, 3, 0.5) > id=1 (1, 5, 0.5) > id=4 (0, ...)
    assert ids == [3, 2, 5, 1, 4]


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
