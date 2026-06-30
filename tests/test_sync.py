"""Tests for startup sync — scans images dir and indexes new files."""

import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from twin.models.clip_model import load as load_model
from twin.services.indexer import indexer
from twin.services.sync import sync_images_dir

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


def _patch_images_dir(monkeypatch, path: Path) -> None:
    """Redirect settings.images_dir to a temp path.

    We patch images_dir (the raw string field) rather than images_path
    (a read-only @property) so monkeypatch can restore it on teardown.
    """
    monkeypatch.setattr(
        "twin.services.sync.settings.images_dir",
        str(path),
    )


# ---------------------------------------------------------------------------
# sync_images_dir
# ---------------------------------------------------------------------------

def test_sync_nonexistent_directory(monkeypatch):
    """When images_path doesn't exist, sync returns zeros gracefully."""
    _patch_images_dir(monkeypatch, Path("/tmp/does_not_exist_sync_test_xyz"))
    result = sync_images_dir()

    assert result["total"] == 0
    assert result["indexed"] == 0
    assert result["skipped"] == 0
    assert result["failed"] == 0


def test_sync_empty_directory(monkeypatch):
    """When images dir exists but is empty, sync returns zeros."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _patch_images_dir(monkeypatch, Path(tmpdir))
        result = sync_images_dir()

        assert result["total"] == 0
        assert result["indexed"] == 0
        assert result["skipped"] == 0
        assert result["failed"] == 0


def test_sync_directory_with_no_images(monkeypatch):
    """Directory with non-image files only — nothing to sync."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "readme.txt").write_text("hello")
        (tmp / "notes.md").write_text("# notes")

        _patch_images_dir(monkeypatch, tmp)
        result = sync_images_dir()

        assert result["total"] == 0
        assert result["indexed"] == 0


def test_sync_indexes_new_files(monkeypatch):
    """Sync should index files not yet in the index."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")
        shutil.copy(FIXTURES / "blue.png", tmp / "blue.png")

        _patch_images_dir(monkeypatch, tmp)
        result = sync_images_dir()

        assert result["total"] == 2
        assert result["indexed"] == 2
        assert result["skipped"] == 0
        assert result["failed"] == 0

        assert indexer.count == 2
        assert indexer.get_indexed_filenames() == {"red.png", "blue.png"}


def test_sync_skips_already_indexed(monkeypatch):
    """Files already in the index are skipped on subsequent sync."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")
        shutil.copy(FIXTURES / "blue.png", tmp / "blue.png")

        _patch_images_dir(monkeypatch, tmp)

        # First sync — indexes both
        r1 = sync_images_dir()
        assert r1["indexed"] == 2

        # Second sync — nothing new
        r2 = sync_images_dir()
        assert r2["total"] == 2
        assert r2["indexed"] == 0
        assert r2["skipped"] == 2

        # Index should still have 2 items (no duplicates)
        assert indexer.count == 2


def test_sync_partial_new_files(monkeypatch):
    """Some files already indexed, some new — only new ones indexed."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")
        shutil.copy(FIXTURES / "blue.png", tmp / "blue.png")
        shutil.copy(FIXTURES / "red_variant.png", tmp / "red_variant.png")

        _patch_images_dir(monkeypatch, tmp)

        # First sync — indexes all 3
        r1 = sync_images_dir()
        assert r1["indexed"] == 3

        # Clear index, manually index just one file
        indexer.clear()
        img = Image.open(FIXTURES / "red.png").convert("RGB")
        from twin.services.embedding import compute_embedding
        from twin.services.hasher import compute_dhash, compute_phash

        v = compute_embedding(img)
        indexer.add_item(v, {
            "filename": "red.png",
            "path": str(tmp / "red.png"),
            "dhash": compute_dhash(img),
            "phash": compute_phash(img),
        })

        # Second sync — red.png already indexed, blue + variant are new
        r2 = sync_images_dir()
        assert r2["total"] == 3
        assert r2["indexed"] == 2  # blue.png + red_variant.png
        assert r2["skipped"] == 1  # red.png
        assert r2["failed"] == 0

        assert indexer.count == 3


def test_sync_with_corrupt_file(monkeypatch):
    """Sync handles corrupt image files gracefully via failed count."""
    _ensure_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(FIXTURES / "red.png", tmp / "red.png")
        # Create a corrupt "image"
        (tmp / "bad.png").write_bytes(b"this is not a valid PNG file")

        _patch_images_dir(monkeypatch, tmp)
        result = sync_images_dir()

        assert result["total"] == 2
        assert result["indexed"] == 1  # only red.png
        assert result["failed"] == 1  # bad.png
        assert "red.png" in indexer.get_indexed_filenames()
        assert "bad.png" not in indexer.get_indexed_filenames()
