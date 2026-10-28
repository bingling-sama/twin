"""Deterministic image perturbation operators for Minecraft texture robustness benchmarks.

All functions are pure, idempotent for fixed seeds, and preserve Alpha channels
to prevent corrupting Minecraft skin/cape transparency masks.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance


def _split_alpha(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    """Split image into RGB base and optional Alpha mask."""
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        return Image.merge("RGB", (r, g, b)), a
    return image.convert("RGB"), None


def _merge_alpha(rgb_image: Image.Image, alpha: Image.Image | None) -> Image.Image:
    """Recombine RGB image with optional Alpha mask."""
    if alpha is not None:
        r, g, b = rgb_image.split()
        return Image.merge("RGBA", (r, g, b, alpha))
    return rgb_image


def apply_grayscale(image: Image.Image) -> Image.Image:
    """Convert RGB channels to grayscale while preserving alpha transparency."""
    rgb, alpha = _split_alpha(image)
    gray = rgb.convert("L")
    gray_rgb = Image.merge("RGB", (gray, gray, gray))
    return _merge_alpha(gray_rgb, alpha)


def apply_color_tint(
    image: Image.Image,
    tint_rgb: tuple[int, int, int] = (255, 190, 150),  # Warm amber tint
    strength: float = 0.25,
) -> Image.Image:
    """Blend image RGB towards a tint color by strength [0.0..1.0]."""
    rgb, alpha = _split_alpha(image)
    arr = np.array(rgb, dtype=np.float32)
    tint = np.array(tint_rgb, dtype=np.float32).reshape(1, 1, 3)
    blended = (1.0 - strength) * arr + strength * tint
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return _merge_alpha(Image.fromarray(blended, mode="RGB"), alpha)


def apply_hue_shift(image: Image.Image, shift_ratio: float = 0.15) -> Image.Image:
    """Rotate hue by shift_ratio in [0.0..1.0] cycle in HSV space."""
    rgb, alpha = _split_alpha(image)
    hsv = rgb.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h, dtype=np.uint16)
    delta = int(shift_ratio * 256)
    h_arr = (h_arr + delta) % 256
    new_h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    shifted_rgb = Image.merge("HSV", (new_h, s, v)).convert("RGB")
    return _merge_alpha(shifted_rgb, alpha)


def apply_brightness_contrast(
    image: Image.Image,
    brightness: float = 1.15,
    contrast: float = 1.15,
) -> Image.Image:
    """Scale brightness and contrast simultaneously."""
    rgb, alpha = _split_alpha(image)
    b_enhancer = ImageEnhance.Brightness(rgb)
    boosted = b_enhancer.enhance(brightness)
    c_enhancer = ImageEnhance.Contrast(boosted)
    boosted = c_enhancer.enhance(contrast)
    return _merge_alpha(boosted, alpha)


def apply_jpeg_compression(image: Image.Image, quality: int = 75) -> Image.Image:
    """Simulate web re-upload JPEG compression artifacts."""
    rgb, alpha = _split_alpha(image)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed_rgb = Image.open(buf).convert("RGB")
    return _merge_alpha(compressed_rgb, alpha)


def apply_pixel_noise(
    image: Image.Image,
    sigma: float = 6.0,
    seed: int | None = 42,
) -> Image.Image:
    """Add subtle Gaussian noise to visible foreground pixels."""
    rgb, alpha = _split_alpha(image)
    rng = np.random.RandomState(seed)
    arr = np.array(rgb, dtype=np.float32)
    noise = rng.normal(0.0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return _merge_alpha(Image.fromarray(noisy_arr, mode="RGB"), alpha)


def apply_identity(image: Image.Image) -> Image.Image:
    """Return unmodified clone as ground truth baseline duplicate."""
    return image.copy()


def apply_composite(image: Image.Image, seed: int | None = 42) -> Image.Image:
    """Apply realistic composite degradation: color tint + brightness + JPEG artifacts."""
    out = apply_color_tint(image, tint_rgb=(255, 200, 170), strength=0.20)
    out = apply_brightness_contrast(out, brightness=1.10, contrast=1.08)
    out = apply_jpeg_compression(out, quality=80)
    return out


ALL_PERTURBATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "identity": apply_identity,
    "grayscale": apply_grayscale,
    "warm_tint": lambda img: apply_color_tint(img, tint_rgb=(255, 190, 150), strength=0.25),
    "cool_tint": lambda img: apply_color_tint(img, tint_rgb=(150, 190, 255), strength=0.25),
    "hue_shift_15pct": lambda img: apply_hue_shift(img, shift_ratio=0.15),
    "bright_contrast": lambda img: apply_brightness_contrast(img, brightness=1.15, contrast=1.15),
    "jpeg_q75": lambda img: apply_jpeg_compression(img, quality=75),
    "pixel_noise": lambda img: apply_pixel_noise(img, sigma=6.0, seed=42),
    "composite": lambda img: apply_composite(img, seed=42),
}
