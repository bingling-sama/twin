"""Perceptual hashing + structural similarity service.

Provides four comparison methods:
  - aHash (average hash): mean intensity comparison, ultra-fast pre-filter
  - dHash (difference hash): gradient-based, fast, robust to scaling/compression
  - pHash (perceptual hash): DCT-based, robust to filters/color adjustments
  - SSIM  (structural similarity): luminance + contrast + structure, closest to human vision
"""

from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# aHash
# ---------------------------------------------------------------------------
def compute_ahash(image: Image.Image) -> str:
    """Compute aHash (average hash). Returns hex string (16 chars, 64-bit)."""
    return str(imagehash.average_hash(image))


def compute_ahashes(images: list[Image.Image], max_workers: int = 8) -> list[str]:
    """Compute aHash for multiple images in parallel."""
    if not images:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda img: str(imagehash.average_hash(img)), images))


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


_gaussian_window_cache: dict[tuple[str, int], torch.Tensor] = {}


def _get_gaussian_window(
    window_size: int = 11, channel: int = 1, sigma: float = 1.5, device: str = "cuda"
) -> torch.Tensor:
    key = (device, window_size)
    if key not in _gaussian_window_cache:
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ], dtype=torch.float32)
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(device)
        _gaussian_window_cache[key] = window
    return _gaussian_window_cache[key]


def compute_ssim_torch(
    img1: Image.Image,
    img2: Image.Image,
    size: tuple[int, int] | None = None,
    device: str = "cuda",
) -> float:
    """Compute SSIM using PyTorch on GPU or CPU."""
    from twin.core.config import settings

    if size is None:
        size = (settings.ssim_size, settings.ssim_size)

    a = np.array(img1.resize(size).convert("L"), dtype=np.float32) / 255.0
    b = np.array(img2.resize(size).convert("L"), dtype=np.float32) / 255.0

    t1 = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).to(device)
    window = _get_gaussian_window(11, 1, device=device)

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    with torch.inference_mode():
        mu1 = F.conv2d(t1, window, padding=window.shape[-1] // 2, groups=1)
        mu2 = F.conv2d(t2, window, padding=window.shape[-1] // 2, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(t1 * t1, window, padding=window.shape[-1] // 2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(t2 * t2, window, padding=window.shape[-1] // 2, groups=1) - mu2_sq
        sigma12 = F.conv2d(t1 * t2, window, padding=window.shape[-1] // 2, groups=1) - mu1_mu2

        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den
        return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def compute_ssim(
    img1: Image.Image,
    img2: Image.Image,
    size: tuple[int, int] | None = None,
    use_gpu: bool | None = None,
) -> float:
    """
    Compute SSIM between two images. Returns float in [-1, 1], where 1 = identical.

    Supports automatic GPU acceleration via PyTorch when CUDA is available.
    """
    from twin.core.config import settings

    if use_gpu is None:
        use_gpu = (
            settings.ssim_device == "cuda"
            or (settings.ssim_device == "auto" and torch.cuda.is_available())
        )

    if use_gpu and torch.cuda.is_available():
        try:
            return compute_ssim_torch(img1, img2, size=size, device="cuda")
        except Exception:
            pass  # Fallback to skimage

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


# ---------------------------------------------------------------------------
# Rotation invariant helpers (0°, 90°, 180°, 270°)
# ---------------------------------------------------------------------------
def compute_rotated_dhashes(image: Image.Image) -> list[str]:
    """Compute dHash for image in 4 orthogonal orientations (0°, 90°, 180°, 270°)."""
    return [
        str(imagehash.dhash(image)),
        str(imagehash.dhash(image.transpose(Image.Transpose.ROTATE_90))),
        str(imagehash.dhash(image.transpose(Image.Transpose.ROTATE_180))),
        str(imagehash.dhash(image.transpose(Image.Transpose.ROTATE_270))),
    ]


def min_rotated_hamming_distance(hash_hex: str, target_image: Image.Image) -> int:
    """Compute minimum Hamming distance across 4 orthogonal rotations of target image."""
    target_hashes = compute_rotated_dhashes(target_image)
    return min(hamming_distance(hash_hex, th) for th in target_hashes)

