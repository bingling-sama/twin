"""Tests for perceptual hashing service."""

from pathlib import Path

from PIL import Image

from twin.services.hasher import (
    compute_ahash,
    compute_ahashes,
    compute_dhash,
    compute_dhashes,
    compute_phash,
    compute_phashes,
    compute_rotated_dhashes,
    compute_ssim,
    compute_ssim_torch,
    hamming_distance,
    is_duplicate,
    min_rotated_hamming_distance,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Image.Image:
    return Image.open(FIXTURES / name).convert("RGB")


def test_identical_images_yield_same_hash():
    """Same image loaded twice produces identical dHash."""
    img_a = _load("red.png")
    img_b = _load("red.png")

    h1 = compute_dhash(img_a)
    h2 = compute_dhash(img_b)

    assert h1 == h2
    assert hamming_distance(h1, h2) == 0


def test_different_images_yield_different_hash():
    """Two different images have Hamming distance > threshold."""
    red = _load("red.png")
    blue = _load("blue.png")

    h_red = compute_dhash(red)
    h_blue = compute_dhash(blue)

    d = hamming_distance(h_red, h_blue)
    assert d > 10, f"Expected distance > 10 for different images, got {d}"


def test_near_duplicates_have_small_distance():
    """Near-identical images (slightly tinted) have very small distance."""
    red = _load("red.png")
    variant = _load("red_variant.png")

    h_red = compute_dhash(red)
    h_var = compute_dhash(variant)

    d = hamming_distance(h_red, h_var)
    # Near-duplicates should have much smaller distance than different images
    assert d <= 14, f"Expected distance <= 14 for near-duplicates, got {d}"


def test_compressed_variant_still_close():
    """JPEG compression shouldn't radically change dHash."""
    red = _load("red.png")
    compressed = _load("red_compressed.jpg")

    h1 = compute_dhash(red)
    h2 = compute_dhash(compressed)

    d = hamming_distance(h1, h2)
    # dHash is robust to JPEG compression
    assert d <= 10, f"Compressed variant should be close, got {d}"


def test_is_duplicate_threshold():
    """is_duplicate respects the threshold boundary."""
    red = _load("red.png")
    blue = _load("blue.png")

    h_red = compute_dhash(red)
    h_blue = compute_dhash(blue)

    assert is_duplicate(h_red, h_red, threshold=10) is True
    assert is_duplicate(h_red, h_blue, threshold=10) is False


def test_hamming_distance_symmetric():
    """Hamming distance is symmetric."""
    red = _load("red.png")
    blue = _load("blue.png")

    h1 = compute_dhash(red)
    h2 = compute_dhash(blue)

    assert hamming_distance(h1, h2) == hamming_distance(h2, h1)


def test_hash_is_string():
    """dHash returns a hex string."""
    img = _load("red.png")
    h = compute_dhash(img)
    assert isinstance(h, str)
    # dHash with hash_size=8 produces 64-bit → 16 hex chars
    assert len(h) == 16


# ---------------------------------------------------------------------------
# Batch hash — empty list
# ---------------------------------------------------------------------------
def test_compute_dhashes_empty():
    """Empty image list returns empty hash list."""
    assert compute_dhashes([]) == []


def test_compute_phashes_empty():
    """Empty image list returns empty hash list."""
    assert compute_phashes([]) == []


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def test_compute_ssim_identical():
    """SSIM of two identical images is ~1.0."""
    img1 = _load("red.png")
    img2 = _load("red.png")
    score = compute_ssim(img1, img2)
    assert score >= 0.99, f"Expected SSIM ≈ 1.0 for identical images, got {score}"


def test_compute_ssim_different():
    """SSIM of very different images is lower."""
    red = _load("red.png")
    blue = _load("blue.png")
    score = compute_ssim(red, blue)
    assert score < 0.99, f"Expected lower SSIM for different images, got {score}"


# ---------------------------------------------------------------------------
# aHash Tests
# ---------------------------------------------------------------------------
def test_ahash_identical():
    """Identical images yield the same aHash."""
    img_a = _load("red.png")
    img_b = _load("red.png")
    h1 = compute_ahash(img_a)
    h2 = compute_ahash(img_b)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0
    assert len(h1) == 16


def test_ahash_batch_and_empty():
    """compute_ahashes works on empty and non-empty lists."""
    assert compute_ahashes([]) == []
    imgs = [_load("red.png"), _load("blue.png")]
    hashes = compute_ahashes(imgs)
    assert len(hashes) == 2
    assert hashes[0] == compute_ahash(imgs[0])
    assert hashes[1] == compute_ahash(imgs[1])


# ---------------------------------------------------------------------------
# Rotation Invariant Tests
# ---------------------------------------------------------------------------
def test_rotation_invariant_hamming_distance():
    """Rotated image matches 0 Hamming distance under rotation-aware check."""
    red = _load("red.png")
    h_red = compute_dhash(red)

    # 90-degree rotated
    red_90 = red.transpose(Image.Transpose.ROTATE_90)
    min_dist_90 = min_rotated_hamming_distance(h_red, red_90)
    assert min_dist_90 == 0

    # 180-degree rotated
    red_180 = red.transpose(Image.Transpose.ROTATE_180)
    min_dist_180 = min_rotated_hamming_distance(h_red, red_180)
    assert min_dist_180 == 0

    # 270-degree rotated
    red_270 = red.transpose(Image.Transpose.ROTATE_270)
    min_dist_270 = min_rotated_hamming_distance(h_red, red_270)
    assert min_dist_270 == 0

    # compute_rotated_dhashes returns 4 orientations
    rot_hashes = compute_rotated_dhashes(red)
    assert len(rot_hashes) == 4
    assert all(len(h) == 16 for h in rot_hashes)

    # compute_phash
    ph = compute_phash(red)
    assert len(ph) == 16


# ---------------------------------------------------------------------------
# PyTorch / CUDA SSIM Tests
# ---------------------------------------------------------------------------
def test_compute_ssim_torch_cpu_and_gpu():
    """compute_ssim_torch accurately detects identical and different images."""
    red1 = _load("red.png")
    red2 = _load("red.png")
    blue = _load("blue.png")

    score_same = compute_ssim_torch(red1, red2, device="cpu")
    assert score_same >= 0.99

    score_diff = compute_ssim_torch(red1, blue, device="cpu")
    assert score_diff < 0.99

    # Test via top-level compute_ssim dispatch
    assert compute_ssim(red1, red2, use_gpu=False) >= 0.99
    assert compute_ssim(red1, red2, use_gpu=True) >= 0.99

