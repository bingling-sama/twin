"""Tests for shared image utilities."""

import tempfile
from pathlib import Path

from twin.utils.image import IMAGE_EXTENSIONS, load_image, load_images

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_image_happy_path():
    """load_image returns a PIL Image for valid files."""
    img = load_image(FIXTURES / "red.png")
    assert img is not None
    assert img.mode == "RGB"


def test_load_image_missing_file():
    """load_image returns None for non-existent files."""
    img = load_image("/tmp/does_not_exist_image_test.png")
    assert img is None


def test_load_image_corrupt_file():
    """load_image returns None for corrupt files."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"this is not a valid PNG image file")
        corrupt_path = f.name

    try:
        img = load_image(corrupt_path)
        assert img is None
    finally:
        Path(corrupt_path).unlink(missing_ok=True)


def test_load_images_mixed():
    """load_images separates valid and invalid paths."""
    imgs, ok, failed = load_images(
        [
            FIXTURES / "red.png",
            Path("/tmp/does_not_exist_xyz.png"),
        ]
    )
    assert len(imgs) == 1
    assert len(ok) == 1
    assert len(failed) == 1
    assert ok[0].name == "red.png"
    assert failed[0].name == "does_not_exist_xyz.png"


def test_image_extensions():
    """IMAGE_EXTENSIONS includes common formats."""
    assert ".png" in IMAGE_EXTENSIONS
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".webp" in IMAGE_EXTENSIONS
