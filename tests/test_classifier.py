"""Tests for Minecraft texture classifier service."""

import numpy as np
from PIL import Image

from twin.services.classifier import (
    TextureKind,
    inspect_texture,
)


def test_square_double_layer_skin():
    """64x64 standard double-layer skin."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    report = inspect_texture(img)
    assert report.kind == TextureKind.SKIN_DOUBLE
    assert report.width == 64
    assert report.height == 64
    assert report.hd_scale == 1
    assert not report.is_hd
    assert not report.can_auto_upgrade


def test_hd_square_double_layer_skin():
    """128x128 HD double-layer skin."""
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    report = inspect_texture(img)
    assert report.kind == TextureKind.SKIN_DOUBLE
    assert report.width == 128
    assert report.height == 128
    assert report.hd_scale == 2
    assert report.is_hd
    assert not report.can_auto_upgrade


def test_legacy_single_layer_skin():
    """64x32 legacy skin with pixels in arm and torso areas."""
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    # Draw opaque pixels in right arm region: x in [46..64], y in [18..32]
    # and torso region: x in [20..36], y in [20..32]
    arr = np.array(img)
    arr[20:30, 48:54] = [255, 0, 0, 255]
    arr[22:28, 22:30] = [0, 255, 0, 255]
    test_img = Image.fromarray(arr, mode="RGBA")

    report = inspect_texture(test_img)
    assert report.kind == TextureKind.SKIN_SINGLE
    assert report.width == 64
    assert report.height == 32
    assert report.hd_scale == 1
    assert not report.is_hd
    assert report.can_auto_upgrade


def test_cape_texture():
    """64x32 cape texture with empty arm area and pixels in cape body."""
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    arr = np.array(img)
    # Cape body is at x in [1..21], y in [1..16]
    arr[2:10, 2:15] = [0, 0, 255, 255]
    test_img = Image.fromarray(arr, mode="RGBA")

    report = inspect_texture(test_img)
    assert report.kind == TextureKind.CAPE
    assert report.width == 64
    assert report.height == 32
    assert report.hd_scale == 1
    assert not report.can_auto_upgrade


def test_elytra_texture():
    """64x32 elytra texture with pixels in elytra region."""
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    arr = np.array(img)
    # Elytra is at x in [23..45], y in [2..22]
    arr[5:15, 25:35] = [255, 255, 0, 255]
    test_img = Image.fromarray(arr, mode="RGBA")

    report = inspect_texture(test_img)
    assert report.kind == TextureKind.ELYTRA
    assert report.width == 64
    assert report.height == 32
    assert report.hd_scale == 1
    assert not report.can_auto_upgrade


def test_unknown_dimensions():
    """Non-standard image dimensions."""
    img = Image.new("RGBA", (100, 50), (255, 255, 255, 255))
    report = inspect_texture(img)
    assert report.kind == TextureKind.UNKNOWN
    assert report.width == 100
    assert report.height == 50
    assert not report.can_auto_upgrade


def test_opaque_rgb_cape():
    """64x32 opaque RGB cape with solid black background must not be misclassified as skin."""
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[2:15, 2:20] = [0, 100, 255]  # Cape body
    img = Image.fromarray(arr, mode="RGB")
    report = inspect_texture(img)
    assert report.kind == TextureKind.CAPE
    assert not report.can_auto_upgrade


def test_opaque_white_bg_cape():
    """64x32 opaque RGB cape with solid white background."""
    arr = np.full((32, 64, 3), 255, dtype=np.uint8)
    arr[2:15, 2:20] = [10, 80, 240]  # Cape body
    img = Image.fromarray(arr, mode="RGB")
    report = inspect_texture(img)
    assert report.kind == TextureKind.CAPE
    assert not report.can_auto_upgrade


def test_opaque_rgb_legacy_skin():
    """64x32 opaque RGB legacy skin with solid black background and arm/torso pixels."""
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[2:15, 2:20] = [200, 150, 100]  # Head
    arr[20:30, 22:30] = [0, 255, 0]    # Torso
    arr[20:30, 48:54] = [255, 0, 0]    # Arm
    img = Image.fromarray(arr, mode="RGB")
    report = inspect_texture(img)
    assert report.kind == TextureKind.SKIN_SINGLE
    assert report.can_auto_upgrade


def test_opaque_rgb_elytra():
    """64x32 opaque RGB elytra with solid black background."""
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[5:15, 25:35] = [255, 255, 0]  # Elytra
    img = Image.fromarray(arr, mode="RGB")
    report = inspect_texture(img)
    assert report.kind == TextureKind.ELYTRA
    assert not report.can_auto_upgrade
