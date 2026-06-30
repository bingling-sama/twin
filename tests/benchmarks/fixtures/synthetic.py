"""Deterministic synthetic image and vector generators for benchmarks.

All generators use a seeded RNG for reproducibility.
Images are generated in-memory (PIL.Image); vectors as numpy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

# Seeded RNG for deterministic benchmark data
_SEED = 42
_rng = np.random.RandomState(_SEED)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB mode if not already."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ── Image generators ──────────────────────────────────────────────────────────


def random_image(size: tuple[int, int] = (224, 224), seed: int | None = None) -> Image.Image:
    """Generate a random-noise RGB image of the given size."""
    rng = np.random.RandomState(seed) if seed is not None else _rng
    arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def gradient_image(size: tuple[int, int] = (224, 224), seed: int | None = None) -> Image.Image:
    """Generate a horizontal linear gradient RGB image.

    Gradient images produce stable perceptual hashes and are useful
    for deterministic hash and SSIM benchmarks.
    """
    rng = np.random.RandomState(seed) if seed is not None else _rng
    w, h = size
    # Horizontal gradient: R and G vary left→right, B is constant
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = ramp  # R: horizontal ramp
    arr[:, :, 1] = ramp[::-1]  # G: reverse ramp
    arr[:, :, 2] = 128  # B: constant mid-gray
    return Image.fromarray(arr, mode="RGB")


def checkerboard_image(
    squares: int = 8, size: tuple[int, int] = (256, 256), seed: int | None = None
) -> Image.Image:
    """Generate a checkerboard pattern RGB image.

    Strong structural content — ideal for SSIM benchmarks where
    identical images should score 1.0 and modified ones < 1.0.
    """
    rng = np.random.RandomState(seed) if seed is not None else _rng
    w, h = size
    square_w, square_h = w // squares, h // squares
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(squares):
        for x in range(squares):
            val = 255 if (x + y) % 2 == 0 else 0
            y0, y1 = y * square_h, min((y + 1) * square_h, h)
            x0, x1 = x * square_w, min((x + 1) * square_w, w)
            arr[y0:y1, x0:x1, :] = val
    return Image.fromarray(arr, mode="RGB")


def solid_image(
    color: tuple[int, int, int] = (128, 128, 128),
    size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """Generate a solid-color RGB image."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def image_batch(
    n: int, size: tuple[int, int] = (224, 224), seed: int | None = None
) -> list[Image.Image]:
    """Generate a batch of N random images."""
    rng = np.random.RandomState(seed) if seed is not None else _rng
    return [random_image(size, seed=rng.randint(0, 2**31)) for _ in range(n)]


def near_duplicate_pair(
    size: tuple[int, int] = (224, 224),
    noise_level: float = 5.0,
    seed: int | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Generate a pair of images that are nearly identical (small pixel noise).

    The second image is the first image plus weak Gaussian noise.
    Useful for search pipeline benchmarks where we expect a confirmed match.
    """
    rng = np.random.RandomState(seed) if seed is not None else _rng
    base = random_image(size, seed=rng.randint(0, 2**31))
    arr = np.array(base, dtype=np.float32)
    noise = rng.randn(*size, 3).astype(np.float32) * noise_level
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return base, Image.fromarray(noisy, mode="RGB")


# ── Vector generators (for Faiss benchmarks, bypass CLIP) ─────────────────────


def random_normalized_vectors(n: int, dim: int = 512, seed: int | None = None) -> np.ndarray:
    """Generate N random L2-normalized float32 vectors.

    Distribution matches the output of CLIP encode (± L2 normalization),
    so these are realistic Faiss index entries without needing the model.
    """
    rng = np.random.RandomState(seed) if seed is not None else _rng
    vecs = rng.randn(n, dim).astype(np.float32)
    # L2-normalize in-place
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid div-by-zero
    vecs /= norms
    return vecs


def query_vector(dim: int = 512, seed: int | None = None) -> np.ndarray:
    """Generate a single query vector (1, dim) for Faiss search."""
    return random_normalized_vectors(1, dim, seed=seed)


# ── Disk I/O helpers ──────────────────────────────────────────────────────────


def save_images_to_dir(
    directory: Path,
    count: int = 10,
    sizes: list[tuple[int, int]] | None = None,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save synthetic images to a directory for I/O benchmarks.

    Returns list of saved file paths.
    """
    if sizes is None:
        sizes = [(224, 224)]
    if formats is None:
        formats = ["png", "jpg"]

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(count):
        size = sizes[i % len(sizes)]
        fmt = formats[i % len(formats)]
        img = random_image(size, seed=_SEED + i)
        fname = f"bench_{i:04d}_{size[0]}x{size[1]}.{fmt}"
        fpath = directory / fname
        img.save(fpath)
        paths.append(fpath)

    return sorted(paths)
