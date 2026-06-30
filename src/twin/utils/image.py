import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def load_image(path: str | Path) -> Image.Image | None:
    """Safely load an image from path. Returns None if file missing/corrupt."""
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception:
        logger.warning("Failed to load image: %s", path, exc_info=True)
        return None


def load_images(paths: list) -> tuple[list[Image.Image], list, list]:
    """Load a batch of images. Returns (valid_images, valid_paths, failed_paths)."""
    imgs = []
    ok = []
    failed = []
    for p in paths:
        img = load_image(p)
        if img is not None:
            imgs.append(img)
            ok.append(p)
        else:
            failed.append(p)
    return imgs, ok, failed


def iter_image_files(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
