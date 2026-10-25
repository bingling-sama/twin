"""Minecraft texture classifier for skins, capes, and elytra.

Ports the heuristic classification rules from Eidolon (src/classifier.rs) to
distinguish double-layer skins, legacy single-layer skins, capes, and elytra atlases.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image


class TextureKind(str, Enum):
    """Categorized Minecraft texture types."""

    SKIN_DOUBLE = "skin_double"
    SKIN_SINGLE = "skin_single"
    CAPE = "cape"
    ELYTRA = "elytra"
    UNKNOWN = "unknown"


TextureType = TextureKind


@dataclass(frozen=True)
class TextureReport:
    """Detailed structural classification report for a texture."""

    kind: TextureKind
    width: int
    height: int
    hd_scale: int
    is_hd: bool
    can_auto_upgrade: bool
    description: str

    @property
    def texture_type(self) -> TextureKind:
        """Alias for kind."""
        return self.kind


def _region_has_opaque_pixels(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    alpha_threshold: int = 10,
) -> bool:
    """Return True if any pixel in rect [x0..x1, y0..y1] has Alpha > alpha_threshold."""
    h, w, c = arr.shape
    if c < 4:
        return True  # If RGB without alpha, consider all pixels opaque

    x_start = max(0, min(x0, w))
    x_end = max(0, min(x1, w))
    y_start = max(0, min(y0, h))
    y_end = max(0, min(y1, h))

    if x_start >= x_end or y_start >= y_end:
        return False

    alpha_slice = arr[y_start:y_end, x_start:x_end, 3]
    return bool(np.any(alpha_slice > alpha_threshold))


def _region_has_foreground_opaque(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    bg_color: np.ndarray,
    color_threshold: float = 15.0,
) -> bool:
    """Check if region contains pixels distinct from background in an opaque texture."""
    h, w, _ = arr.shape
    x_start = max(0, min(x0, w))
    x_end = max(0, min(x1, w))
    y_start = max(0, min(y0, h))
    y_end = max(0, min(y1, h))

    if x_start >= x_end or y_start >= y_end:
        return False

    region_rgb = arr[y_start:y_end, x_start:x_end, :3].astype(np.int16)
    diff = np.max(np.abs(region_rgb - bg_color.astype(np.int16)), axis=-1)
    return bool(np.any(diff > color_threshold))


def _classify_2_to_1_texture(image: Image.Image, scale: int) -> TextureKind:
    """Heuristic analyzer for 2:1 textures (64x32 base layout)."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)

    # Check whether the image has any transparent pixels
    has_transparency = bool(np.any(arr[..., 3] <= 10))

    if has_transparency:
        # Standard alpha mask evaluation
        skin_right_arm_has_pixels = _region_has_opaque_pixels(
            arr, 46 * scale, 18 * scale, 64 * scale, 32 * scale
        )
        skin_torso_has_pixels = _region_has_opaque_pixels(
            arr, 20 * scale, 20 * scale, 36 * scale, 32 * scale
        )
        elytra_region_has_pixels = _region_has_opaque_pixels(
            arr, 23 * scale, 2 * scale, 45 * scale, 22 * scale
        )
        cape_body_has_pixels = _region_has_opaque_pixels(
            arr, 1 * scale, 1 * scale, 21 * scale, 16 * scale
        )
    else:
        # Fallback for opaque images (e.g. RGB JPG/PNG with solid background).
        # In Minecraft 64x32 layouts, the bottom-right corner [56..64, 24..32] is
        # geometrically guaranteed to be unused across skins, capes, and elytra.
        bg_sample = arr[24 * scale : 32 * scale, 56 * scale : 64 * scale, :3]
        bg_color = np.median(bg_sample, axis=(0, 1))

        skin_right_arm_has_pixels = _region_has_foreground_opaque(
            arr, 46 * scale, 18 * scale, 64 * scale, 32 * scale, bg_color
        )
        skin_torso_has_pixels = _region_has_foreground_opaque(
            arr, 20 * scale, 20 * scale, 36 * scale, 32 * scale, bg_color
        )
        elytra_region_has_pixels = _region_has_foreground_opaque(
            arr, 23 * scale, 2 * scale, 45 * scale, 22 * scale, bg_color
        )
        cape_body_has_pixels = _region_has_foreground_opaque(
            arr, 1 * scale, 1 * scale, 21 * scale, 16 * scale, bg_color
        )

    if skin_right_arm_has_pixels and skin_torso_has_pixels:
        return TextureKind.SKIN_SINGLE
    elif skin_right_arm_has_pixels:
        return TextureKind.SKIN_SINGLE
    elif elytra_region_has_pixels and cape_body_has_pixels:
        # Modern Minecraft cape atlas contains BOTH cape and matching elytra
        return TextureKind.CAPE
    elif elytra_region_has_pixels:
        return TextureKind.ELYTRA
    elif cape_body_has_pixels:
        return TextureKind.CAPE
    else:
        return TextureKind.UNKNOWN


def inspect_texture(image: Image.Image) -> TextureReport:
    """Inspect and classify texture structure from a PIL Image.

    Returns TextureReport containing classification kind, dimensions, HD scale,
    and upgrade eligibility.
    """
    width, height = image.size

    # 1. 1:1 Aspect Ratio (Square Double-Layer Skin)
    if width == height and width >= 64 and width % 64 == 0:
        hd_scale = width // 64
        is_hd = hd_scale > 1
        desc = (
            f"HD {width}x{height} double-layer skin ({hd_scale}x)"
            if is_hd
            else "Standard 64x64 double-layer skin"
        )
        return TextureReport(
            kind=TextureKind.SKIN_DOUBLE,
            width=width,
            height=height,
            hd_scale=hd_scale,
            is_hd=is_hd,
            can_auto_upgrade=False,
            description=desc,
        )

    # 2. 2:1 Aspect Ratio (64x32 Base: Single-Layer Skin, Cape, or Elytra)
    if width == 2 * height and width >= 64 and width % 64 == 0:
        hd_scale = width // 64
        is_hd = hd_scale > 1
        kind = _classify_2_to_1_texture(image, hd_scale)

        if kind == TextureKind.SKIN_SINGLE:
            desc = (
                f"HD {width}x{height} legacy skin ({hd_scale}x)"
                if is_hd
                else "Standard 64x32 legacy skin (convertible to 64x64)"
            )
            can_upgrade = True
        elif kind == TextureKind.CAPE:
            desc = (
                f"HD {width}x{height} cape atlas ({hd_scale}x)"
                if is_hd
                else "Standard 64x32 cape atlas"
            )
            can_upgrade = False
        elif kind == TextureKind.ELYTRA:
            desc = (
                f"HD {width}x{height} elytra atlas ({hd_scale}x)"
                if is_hd
                else "Standard 64x32 elytra atlas"
            )
            can_upgrade = False
        else:
            desc = f"Unrecognized 2:1 texture ({width}x{height})"
            can_upgrade = False

        return TextureReport(
            kind=kind,
            width=width,
            height=height,
            hd_scale=hd_scale,
            is_hd=is_hd,
            can_auto_upgrade=can_upgrade,
            description=desc,
        )

    # 3. Fallback: Unknown dimensions
    return TextureReport(
        kind=TextureKind.UNKNOWN,
        width=width,
        height=height,
        hd_scale=0,
        is_hd=False,
        can_auto_upgrade=False,
        description=f"Non-standard texture dimensions ({width}x{height})",
    )
