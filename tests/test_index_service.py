"""Tests for index_service — single and batch indexing workflows."""

import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from twin.models.clip_model import load as load_model
from twin.services.index_service import _index_single_from_disk, index_batch, index_single
from twin.services.indexer import indexer

FIXTURES = Path(__file__).parent / "fixtures"


def _ensure_model():
    """Load CLIP model if not already loaded. load() is idempotent."""
    load_model()


@pytest.fixture(autouse=True)
def _clear_index():
    """Reset indexer state before and after each test."""
    indexer.clear()
    yield
    indexer.clear()


# ---------------------------------------------------------------------------
# index_single
# ---------------------------------------------------------------------------


def test_index_single_happy_path():
    """Direct call to index_single stores image + returns indexed status."""
    _ensure_model()
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    content = (FIXTURES / "red.png").read_bytes()

    result = index_single(img, "red.png", content)

    assert result["status"] == "indexed"
    assert result["filename"] == "red.png"
    assert isinstance(result["id"], int)
    assert result["id"] >= 0

    # Verify it's actually in the index
    assert "red.png" in indexer.get_indexed_filenames()
    assert indexer.count == 1


def test_index_single_already_exists():
    """Second call with same filename returns already_exists."""
    _ensure_model()
    img = Image.open(FIXTURES / "red.png").convert("RGB")
    content = (FIXTURES / "red.png").read_bytes()

    index_single(img, "red.png", content)
    result = index_single(img, "red.png", content)

    assert result["status"] == "already_exists"
    assert result["id"] == -1
    assert indexer.count == 1  # still only one


# ---------------------------------------------------------------------------
# _index_single_from_disk
# ---------------------------------------------------------------------------


def test_index_single_from_disk():
    """Internal helper indexes an image that's already on disk."""
    _ensure_model()
    img = Image.open(FIXTURES / "blue.png").convert("RGB")
    path = FIXTURES / "blue.png"

    idx = _index_single_from_disk(img, path)

    assert isinstance(idx, int)
    assert idx >= 0
    assert indexer.count == 1

    meta = indexer.get_metadata(idx)
    assert meta is not None
    assert meta["filename"] == "blue.png"
    assert str(path) in meta["path"]


# ---------------------------------------------------------------------------
# index_batch
# ---------------------------------------------------------------------------


def test_index_batch_happy_path():
    """Batch-index images from a directory."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Copy fixture images
        for name in ["red.png", "red_variant.png", "blue.png"]:
            shutil.copy(FIXTURES / name, tmp / name)

        result = index_batch(tmp)

        assert result["status"] == "completed"
        assert result["total"] == 3
        assert result["indexed"] == 3
        assert result["failed"] == 0
        assert result["time_ms"] >= 0

        assert indexer.count == 3
        assert indexer.get_indexed_filenames() == {"red.png", "red_variant.png", "blue.png"}


def test_index_batch_nonexistent_directory():
    """Passing a non-existent path raises ValueError."""
    with pytest.raises(ValueError, match="Not a directory"):
        index_batch("/tmp/does_not_exist_xyz_123")


def test_index_batch_empty_directory():
    """Empty directory returns zero results without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = index_batch(tmpdir)

        assert result["status"] == "completed"
        assert result["total"] == 0
        assert result["indexed"] == 0
        assert result["failed"] == 0


def test_index_batch_directory_with_no_images():
    """Directory with only non-image files returns zero results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "notes.txt").write_text("hello")
        (tmp / "data.csv").write_text("a,b,c")

        result = index_batch(tmp)

        assert result["status"] == "completed"
        assert result["total"] == 0
        assert result["indexed"] == 0


def test_index_batch_respects_settings_batch_size(monkeypatch):
    """Batch processes in chunks according to settings.batch_size."""
    _ensure_model()

    # Use a small batch size to force multiple iterations
    monkeypatch.setattr("twin.services.index_service.settings.batch_size", 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in ["red.png", "red_variant.png", "blue.png"]:
            shutil.copy(FIXTURES / name, tmp / name)

        result = index_batch(tmp)

        assert result["status"] == "completed"
        assert result["total"] == 3
        assert result["indexed"] == 3
        assert result["failed"] == 0
        assert indexer.count == 3


def test_index_batch_skips_already_indexed():
    """Batch does NOT skip already-indexed files — it re-indexes everything in dir."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")

        # First batch
        r1 = index_batch(tmp)
        assert r1["indexed"] == 1

        # Add another file, re-run batch on same dir
        shutil.copy(FIXTURES / "blue.png", tmp / "blue.png")
        r2 = index_batch(tmp)

        # index_batch indexes everything in the directory (no dedup by filename)
        assert r2["total"] == 2
        assert r2["indexed"] == 2  # both are re-indexed


def test_index_batch_fallback_on_corrupt_image():
    """Batch with a corrupt file falls back to individual indexing for the batch."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")
        # Create a fake "image" file that will fail to load
        (tmp / "bad.png").write_bytes(b"not a real png")

        result = index_batch(tmp)

        assert result["status"] == "completed"
        assert result["total"] == 2
        # red.png should succeed, bad.png should fail
        assert result["indexed"] == 1
        assert result["failed"] == 1

        assert "red.png" in indexer.get_indexed_filenames()
        assert "bad.png" not in indexer.get_indexed_filenames()


def test_index_batch_all_images_corrupt():
    """Batch where ALL images fail to load (imgs empty) → continue."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create only corrupt files
        (tmp / "bad1.png").write_bytes(b"not a real image")
        (tmp / "bad2.png").write_bytes(b"also garbage")

        result = index_batch(tmp)

        assert result["status"] == "completed"
        assert result["total"] == 2
        assert result["indexed"] == 0
        assert result["failed"] == 2


def test_index_single_rejects_unknown():
    """index_single must raise ValueError when texture is UNKNOWN."""
    _ensure_model()
    # Create an arbitrary non-standard dimension image (e.g. 50x50)
    img = Image.new("RGBA", (50, 50), (128, 128, 128, 255))
    with pytest.raises(ValueError, match="unrecognized Minecraft texture format"):
        index_single(img, "invalid.png", b"fake_content")


def test_index_single_from_disk_skips_unknown():
    """_index_single_from_disk returns -1 for UNKNOWN textures."""
    _ensure_model()
    img = Image.new("RGBA", (50, 50), (128, 128, 128, 255))
    fake_path = Path("fake_dir/invalid.png")
    idx = _index_single_from_disk(img, fake_path)
    assert idx == -1


def test_index_batch_skips_unknown_textures():
    """index_batch should skip unknown texture dimensions without failing the whole batch."""
    _ensure_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 1 valid skin
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")
        # 1 unknown dimension image
        invalid_img = Image.new("RGBA", (100, 100), (200, 200, 200, 255))
        invalid_img.save(tmp / "unknown.png")

        result = index_batch(tmp)
        assert result["status"] == "completed"
        assert result["total"] == 2
        assert result["indexed"] == 1
        assert "red.png" in indexer.get_indexed_filenames()
        assert "unknown.png" not in indexer.get_indexed_filenames()

