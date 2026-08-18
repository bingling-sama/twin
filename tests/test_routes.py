"""Tests for API routes — thin HTTP handlers with edge cases."""

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

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
    """Reset index state between tests."""
    indexer.clear()
    yield
    indexer.clear()


# ---------------------------------------------------------------------------
# serve_image
# ---------------------------------------------------------------------------
def test_serve_image_not_found(client):
    """Requesting a non-existent image returns 404."""
    resp = client.get("/api/v1/images/nonexistent_file.png")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Image not found"


def test_serve_image_found(client):
    """An uploaded image can be served back."""
    with open(FIXTURES / "red.png", "rb") as f:
        client.post(
            "/api/v1/index",
            files={"file": ("red.png", f, "image/png")},
        )

    resp = client.get("/api/v1/images/red.png")
    assert resp.status_code == 200
    assert int(resp.headers.get("content-length", "0")) > 0


# ---------------------------------------------------------------------------
# index/batch — ValueError handling
# ---------------------------------------------------------------------------
def test_index_batch_invalid_directory(client):
    """Batch indexing a non-existent directory returns 400."""
    resp = client.post(
        "/api/v1/index/batch",
        json={"directory": "/tmp/does_not_exist_xyz_batch"},
    )
    assert resp.status_code == 400
    assert "Not a directory" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# list_index — pagination
# ---------------------------------------------------------------------------
def test_list_index_pagination(client):
    """Paginated listing returns correct page/page_size/total."""
    # Upload 3 images
    for name in ("red.png", "blue.png", "red_variant.png"):
        with open(FIXTURES / name, "rb") as f:
            client.post(
                "/api/v1/index",
                files={"file": (name, f, "image/png")},
            )

    # Page 1, size 2
    resp = client.get("/api/v1/index?page=1&page_size=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["page"] == 1
    assert data["page_size"] == 2
    assert len(data["items"]) == 2

    # Page 2 should have 1 remaining item
    resp = client.get("/api/v1/index?page=2&page_size=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1


def test_list_index_empty(client):
    """Listing an empty index returns total=0 and empty items."""
    resp = client.get("/api/v1/index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


# ---------------------------------------------------------------------------
# _validate_image — corrupt content with valid extension
# ---------------------------------------------------------------------------
def test_validate_corrupt_image(client):
    """Upload a file with .png extension but invalid content returns 400."""
    resp = client.post(
        "/api/v1/index",
        files={"file": ("corrupt.png", b"this is not a PNG file", "image/png")},
    )
    assert resp.status_code == 400
    assert "Invalid or corrupt image" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _validate_image — unsupported extension
# ---------------------------------------------------------------------------
def test_validate_unsupported_extension(client):
    """Upload a .gif file returns 400 (gif not in allowed list)."""
    # Create a minimal valid GIF
    buf = BytesIO()
    img = Image.new("RGB", (10, 10), color="red")
    img.save(buf, format="GIF")
    buf.seek(0)

    resp = client.post(
        "/api/v1/index",
        files={"file": ("anim.gif", buf.read(), "image/gif")},
    )
    assert resp.status_code == 400
    assert "Unsupported image format" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Text Search endpoint
# ---------------------------------------------------------------------------
def test_text_search_endpoint(client):
    """search/text returns ranked candidate images."""
    # Index red and blue
    with open(FIXTURES / "red.png", "rb") as f:
        client.post("/api/v1/index", files={"file": ("red.png", f, "image/png")})
    with open(FIXTURES / "blue.png", "rb") as f:
        client.post("/api/v1/index", files={"file": ("blue.png", f, "image/png")})

    resp = client.post(
        "/api/v1/search/text",
        json={"query": "a red square", "k": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "a red square"
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] in ("red.png", "blue.png")


# ---------------------------------------------------------------------------
# Async Batch Indexing endpoint
# ---------------------------------------------------------------------------
def test_async_batch_index_endpoint(client, tmp_path):
    """POST /index/batch with async_mode=True returns task_id and tracking status."""
    import time
    # Copy fixture image to tmp dir
    img = Image.open(FIXTURES / "red.png")
    img.save(tmp_path / "test_async_red.png")

    resp = client.post(
        "/api/v1/index/batch",
        json={"directory": str(tmp_path), "async_mode": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("started", "running", "completed")
    task_id = data["task_id"]
    assert task_id

    # Poll status
    for _ in range(30):
        s_resp = client.get(f"/api/v1/index/batch/status/{task_id}")
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        if s_data["status"] == "completed":
            break
        time.sleep(0.1)

    assert s_data["status"] == "completed"
    assert s_data["indexed"] == 1

