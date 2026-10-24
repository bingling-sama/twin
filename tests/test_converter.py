"""Tests for single2double skin layout converter."""

import numpy as np
import pytest
from PIL import Image

from twin.services.converter import single2double


def test_single2double_dimensions():
    """Converts 64x32 to 64x64."""
    img = Image.new("RGBA", (64, 32), (100, 150, 200, 255))
    out = single2double(img)
    assert out.size == (64, 64)
    assert out.mode == "RGBA"
    # Upper half should match original image
    assert np.array_equal(np.array(out)[:32, :, :], np.array(img))


def test_single2double_hd_dimensions():
    """Converts 128x64 HD to 128x128."""
    img = Image.new("RGBA", (128, 64), (50, 100, 150, 255))
    out = single2double(img)
    assert out.size == (128, 128)
    assert out.mode == "RGBA"


def test_single2double_invalid_dimensions():
    """Rejects square or non-2:1 images."""
    img_square = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    with pytest.raises(ValueError, match="Input image is not a single-layer skin"):
        single2double(img_square)

    img_bad_ratio = Image.new("RGBA", (64, 30), (0, 0, 0, 255))
    with pytest.raises(ValueError, match="Input image is not a single-layer skin"):
        single2double(img_bad_ratio)


def test_single2double_limb_mirroring():
    """Verify right limb parts are mirrored to lower left limb locations."""
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    # Right arm front: (x: 44..48, y: 20..32)
    # Paint a gradient on the right arm front so horizontal flipping is detectable
    arr = np.array(img)
    for y in range(20, 32):
        for x in range(44, 48):
            arr[y, x] = [x - 44, y - 20, 100, 255]

    source_img = Image.fromarray(arr, mode="RGBA")
    converted = single2double(source_img)
    out_arr = np.array(converted)

    # Left arm front is at (x: 36..40, y: 52..64)
    # Original x=44 (val 0) should flip to right side of left arm (x=39)
    # Original x=47 (val 3) should flip to left side of left arm (x=36)
    assert out_arr[52, 36, 0] == 3  # Flipped left edge
    assert out_arr[52, 39, 0] == 0  # Flipped right edge
