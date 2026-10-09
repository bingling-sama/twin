"""Integration tests for FastAPI endpoints.

Run with: uv run pytest tests/test_api.py -v

Requires the CLIP model to be downloaded (first run will download ~340MB).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twin.main import app
from twin.services.indexer import indexer

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient — model loads once."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_index():
    """Reset index state between tests using public API."""
    indexer.clear()
    yield
    indexer.clear()


def _upload(client: TestClient, path: Path):
    """Helper: upload a fixture image for indexing."""
    with open(path, "rb") as f:
        return client.post("/api/v1/index", files={"file": (path.name, f, "image/png")})


def _search(client: TestClient, path: Path):
    """Helper: search with a fixture image."""
    with open(path, "rb") as f:
        return client.post("/api/v1/search", files={"file": (path.name, f, "image/png")})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "indexed_count" in data
    assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Sync status
# ---------------------------------------------------------------------------
def test_sync_status(client):
    """Sync status endpoint returns valid structure when no sync is running."""
    resp = client.get("/api/v1/sync/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert data["total_files"] == 0
    assert data["indexed_files"] == 0
    assert data["skipped_files"] == 0
    assert data["failed_files"] == 0
    assert data["progress_pct"] == 0.0
    assert data["elapsed_ms"] == 0.0
    assert data["eta_ms"] == 0.0


# ---------------------------------------------------------------------------
# Index + Search
# ---------------------------------------------------------------------------
def test_index_single_image(client):
    resp = _upload(client, FIXTURES / "red.png")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["filename"] == "red.png"
    assert isinstance(data["id"], int)


def test_index_duplicate_file(client):
    """Uploading same file twice returns already_exists."""
    _upload(client, FIXTURES / "red.png")
    resp = _upload(client, FIXTURES / "red.png")
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_exists"


def test_index_invalid_file(client):
    """Uploading a non-image returns 400."""
    resp = client.post(
        "/api/v1/index",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_search_empty_index(client):
    """Searching an empty index returns empty results."""
    resp = _search(client, FIXTURES / "red.png")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["results"] == []


def test_full_search_flow(client):
    """Index several images, then search and verify results."""
    # Index three files
    _upload(client, FIXTURES / "red.png")
    _upload(client, FIXTURES / "red_variant.png")
    _upload(client, FIXTURES / "blue.png")

    # Health check
    resp = client.get("/api/v1/health")
    assert resp.json()["indexed_count"] == 3

    # Search with red → red should be top, blue should be last
    resp = _search(client, FIXTURES / "red.png")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3

    filenames = [r["filename"] for r in data["results"]]

    # red.png or red_variant.png should be first (duplicates)
    assert filenames[0] in ("red.png", "red_variant.png")
    assert filenames[0] != "blue.png"

    # blue.png should be last (most different)
    assert filenames[-1] == "blue.png"


def test_clear_index(client):
    """Clearing the index resets count to 0."""
    _upload(client, FIXTURES / "red.png")
    resp = client.delete("/api/v1/index")
    assert resp.status_code == 200

    resp = client.get("/api/v1/health")
    assert resp.json()["indexed_count"] == 0
