"""Tests for DiffMaskVerifier and pluggable verification architecture (Phase 3)."""

import time

import numpy as np
from PIL import Image

from twin.services.verifiers import DiffMaskVerifier, SSIMVerifier, get_verifier


def test_diffmask_exact_identical():
    """Identical images produce diff_ratio=0.0 and score=1.0."""
    img = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
    v = DiffMaskVerifier()
    res = v.verify(img, img)
    assert res.passed is True
    assert res.score == 1.0
    assert res.diff_ratio == 0.0
    assert res.details["diff_pixels"] == 0


def test_diffmask_alpha_noise_filtering():
    """Transparent background noise with alpha <= 10 is ignored."""
    img1 = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    # Add opaque face
    for x in range(20, 30):
        for y in range(20, 30):
            img1.putpixel((x, y), (200, 100, 50, 255))

    img2 = img1.copy()
    # Add low-alpha noise in previously transparent areas
    for x in range(0, 10):
        for y in range(0, 10):
            img2.putpixel((x, y), (255, 255, 255, 5))  # alpha <= 10

    v = DiffMaskVerifier(alpha_threshold=10)
    res = v.verify(img1, img2)
    assert res.passed is True
    assert res.diff_ratio == 0.0
    assert res.score == 1.0


def test_diffmask_slight_eye_edit_detection():
    """Modifying 4 eye pixels is captured accurately and recognized as confirmed variant."""
    img1 = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    img2 = img1.copy()

    # Modify 4 pixels to bright blue eyes
    for x in range(12, 14):
        for y in range(12, 14):
            img2.putpixel((x, y), (0, 120, 255, 255))

    v = DiffMaskVerifier(tolerance=5, threshold=0.05)
    res = v.verify(img1, img2)
    assert res.passed is True
    assert res.details["diff_pixels"] == 4
    # 4 / 4096 = ~0.00098
    assert res.diff_ratio < 0.002
    assert res.score > 0.99


def test_diffmask_completely_different_textures():
    """Vastly different images fail verification."""
    img1 = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    img2 = Image.new("RGBA", (64, 64), (255, 255, 255, 255))

    v = DiffMaskVerifier(tolerance=5, threshold=0.05)
    res = v.verify(img1, img2)
    assert res.passed is False
    assert res.diff_ratio == 1.0
    assert res.score == 0.0


def test_diffmask_auto_upgrade_parity():
    """A 64x32 single skin auto-upgrades and compares identically against
    its 64x64 double version.
    """
    from twin.services.converter import single2double

    # Create 64x32 single skin
    rng = np.random.RandomState(42)
    raw = rng.randint(0, 256, (32, 64, 4), dtype=np.uint8)
    raw[:, :, 3] = 255
    single_img = Image.fromarray(raw, mode="RGBA")

    # Double skin version via canonical converter
    double_img = single2double(single_img)

    v = DiffMaskVerifier()
    res = v.verify(single_img, double_img)
    assert res.passed is True
    assert res.diff_ratio == 0.0
    assert res.score == 1.0


def test_diffmask_performance_under_1ms():
    """DiffMask verification executes in under 1ms (target ~0.1ms)."""
    img1 = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    img2 = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    v = DiffMaskVerifier()

    # Warmup
    v.verify(img1, img2)

    # 100 iterations
    t0 = time.perf_counter()
    for _ in range(100):
        v.verify(img1, img2)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / 100) * 1000

    # Ensure average latency < 1.0ms
    assert avg_ms < 1.0, f"DiffMask took {avg_ms:.3f}ms, expected < 1.0ms"


def test_verifier_factory():
    """Verifier factory resolves configured classes correctly."""
    v_diff = get_verifier("diffmask")
    assert isinstance(v_diff, DiffMaskVerifier)

    v_ssim = get_verifier("ssim")
    assert isinstance(v_ssim, SSIMVerifier)


def test_diffmask_hd_texture_nearest_scaling():
    """64x64 skin vs 128x128 HD skin scales with NEAREST and matches perfectly."""
    rng = np.random.RandomState(42)
    raw = rng.randint(0, 256, (64, 64, 4), dtype=np.uint8)
    raw[:, :, 3] = 255
    skin64 = Image.fromarray(raw, mode="RGBA")

    # 128x128 HD double-sampled skin via nearest neighbor
    skin128 = skin64.resize((128, 128), resample=Image.Resampling.NEAREST)

    v = DiffMaskVerifier()
    res = v.verify(skin64, skin128)
    assert res.passed is True
    assert res.diff_ratio == 0.0
    assert res.score == 1.0

    # Incompatible aspect ratio test (1:1 skin vs 64x20 non-standard image)
    weird_img = Image.new("RGBA", (64, 20), (255, 0, 0, 255))
    res_weird = v.verify(skin64, weird_img)
    assert res_weird.passed is False
    assert res_weird.details.get("reason") == "aspect_ratio_mismatch"

