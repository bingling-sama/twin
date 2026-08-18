"""Perceptual hashing + structural similarity service.

Provides three comparison methods:
  - dHash (difference hash): gradient-based, fast, robust to scaling/compression
  - pHash (perceptual hash): DCT-based, robust to filters/color adjustments
  - SSIM  (structural similarity): luminance + contrast + structure, closest to human vision
"""

from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# dHash
# ---------------------------------------------------------------------------
def compute_dhash(image: Image.Image) -> str:
    """Compute dHash. Returns hex string (16 chars, 64-bit)."""
    return str(imagehash.dhash(image))


def compute_dhashes(images: list[Image.Image], max_workers: int = 8) -> list[str]:
    """Compute dHash for multiple images in parallel."""
    if not images:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda img: str(imagehash.dhash(img)), images))


# ---------------------------------------------------------------------------
# pHash
# ---------------------------------------------------------------------------
def compute_phash(image: Image.Image) -> str:
    """Compute pHash (DCT-based). Returns hex string (16 chars, 64-bit)."""
    return str(imagehash.phash(image))


def compute_phashes(images: list[Image.Image], max_workers: int = 8) -> list[str]:
    """Compute pHash for multiple images in parallel."""
    if not images:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda img: str(imagehash.phash(img)), images))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def compute_ssim(img1: Image.Image, img2: Image.Image, size: tuple[int, int] | None = None) -> float:
    """
    Compute SSIM between two images. Returns float in [-1, 1], where 1 = identical.

    Images are resized to a common size for comparison. Uses float32 (8-bit
    input doesn't need double precision) and defaults to the configured size
    (128×128, ~9× faster than the old 256×256 float64 path with no accuracy
    regression for duplicate detection).
    """
    from twin.core.config import settings

    if size is None:
        size = (settings.ssim_size, settings.ssim_size)
    a = np.array(img1.resize(size).convert("L"), dtype=np.float32)
    b = np.array(img2.resize(size).convert("L"), dtype=np.float32)
    return float(ssim(a, b, data_range=255))


# ---------------------------------------------------------------------------
# Hamming distance helpers (shared by dHash and pHash)
# ---------------------------------------------------------------------------
def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    n1 = int(hash1, 16)
    n2 = int(hash2, 16)
    return (n1 ^ n2).bit_count()


def is_duplicate(hash1: str, hash2: str, threshold: int = 10) -> bool:
    """Return True if two hashes are within the threshold Hamming distance."""
    return hamming_distance(hash1, hash2) <= threshold
