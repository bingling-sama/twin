"""Tests for robustness benchmark perturbation operators and dataset sampling."""

import numpy as np
import pytest
from PIL import Image

from benchmarks.robustness.dataset import (
    AnchorSample,
    build_evaluation_queries,
    sample_anchor_dataset,
)
from benchmarks.robustness.evaluator import populate_benchmark_index
from benchmarks.robustness.perturbations import (
    ALL_PERTURBATIONS,
    apply_color_tint,
    apply_grayscale,
    apply_hue_shift,
    apply_identity,
)
from twin.services.classifier import TextureKind
from twin.services.verifiers import DiffMaskVerifier
from twin.utils.image import load_image


@pytest.fixture
def sample_rgba_skin() -> Image.Image:
    """Create a 64x64 RGBA skin with transparency in unused areas."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    # Head rect [8..24, 8..16]
    arr[8:16, 8:24] = [200, 150, 100, 255]
    # Body rect [20..32, 20..36]
    arr[20:32, 20:36] = [0, 100, 255, 255]
    # Keep rest alpha = 0
    return Image.fromarray(arr, mode="RGBA")


def test_perturbation_operators_preserve_dimensions_and_alpha(sample_rgba_skin):
    """All operators must maintain 64x64 size and keep alpha channel intact."""
    w, h = sample_rgba_skin.size
    orig_alpha = np.array(sample_rgba_skin.split()[3])

    for name, op in ALL_PERTURBATIONS.items():
        out = op(sample_rgba_skin)
        assert out.size == (w, h), f"Operator {name} altered image dimensions"
        assert out.mode == "RGBA", f"Operator {name} dropped RGBA mode"
        out_alpha = np.array(out.split()[3])
        np.testing.assert_array_equal(
            out_alpha, orig_alpha, err_msg=f"Operator {name} mutated alpha channel"
        )


def test_grayscale_rgb_consistency():
    """RGB image converted with apply_grayscale has identical R, G, B channels."""
    img = Image.new("RGB", (64, 64), (255, 128, 64))
    out = apply_grayscale(img)
    arr = np.array(out)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    np.testing.assert_array_equal(r, g)
    np.testing.assert_array_equal(g, b)


def test_color_tint_shifts_mean():
    """Warm tint should increase red channel relative to blue."""
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    tinted = apply_color_tint(img, tint_rgb=(255, 0, 0), strength=0.5)
    arr = np.array(tinted)
    assert arr[:, :, 0].mean() > 100
    assert arr[:, :, 2].mean() < 100


def test_hue_shift_alters_colors():
    """Hue shift should alter pure green without breaking shape."""
    img = Image.new("RGB", (64, 64), (0, 255, 0))
    shifted = apply_hue_shift(img, shift_ratio=0.33)
    arr = np.array(shifted)
    assert not np.array_equal(arr, np.array(img))


def test_sample_anchor_dataset_size_and_types():
    """Sampler should yield requested count and valid Minecraft textures."""
    anchors = sample_anchor_dataset(sample_size=5, seed=123)
    assert len(anchors) == 5
    for a in anchors:
        assert isinstance(a, AnchorSample)
        assert a.image.size[0] >= 64 and a.image.size[0] % 64 == 0
        assert a.texture_type in (
            TextureKind.SKIN_DOUBLE,
            TextureKind.SKIN_SINGLE,
            TextureKind.CAPE,
        )
        assert a.datasource in ("skin", "cape")


def test_build_evaluation_queries():
    """Query generator builds query pairs with all perturbation operators."""
    anchors = sample_anchor_dataset(sample_size=2, seed=999)
    queries = build_evaluation_queries(anchors)
    expected_count = len(anchors) * len(ALL_PERTURBATIONS)
    assert len(queries) == expected_count
    for q in queries:
        assert q.target_filename in [a.filename for a in anchors]
        assert q.perturbation_type in ALL_PERTURBATIONS
        assert q.image.size[0] >= 64 and q.image.size[0] % 64 == 0


def test_identity_operator(sample_rgba_skin):
    """Identity operator must return an identical duplicate image."""
    out = apply_identity(sample_rgba_skin)
    np.testing.assert_array_equal(np.array(out), np.array(sample_rgba_skin))


def test_load_image_preserves_rgba(tmp_path):
    """load_image must preserve RGBA transparency for Minecraft textures and RGB for opaque."""
    rgba_path = tmp_path / "test_rgba.png"
    img_rgba = Image.new("RGBA", (64, 64), (10, 20, 30, 128))
    img_rgba.save(rgba_path)

    loaded_rgba = load_image(rgba_path)
    assert loaded_rgba is not None
    assert loaded_rgba.mode == "RGBA"
    assert loaded_rgba.getpixel((0, 0)) == (10, 20, 30, 128)

    rgb_path = tmp_path / "test_rgb.jpg"
    img_rgb = Image.new("RGB", (64, 64), (100, 150, 200))
    img_rgb.save(rgb_path)

    loaded_rgb = load_image(rgb_path)
    assert loaded_rgb is not None
    assert loaded_rgb.mode == "RGB"


def test_populate_benchmark_index_with_temp_dir(tmp_path):
    """populate_benchmark_index with tmp_dir must persist files and enable DiffMask verification."""
    from twin.models.clip_model import load as load_clip
    load_clip()

    anchors = sample_anchor_dataset(sample_size=2, seed=42)
    populate_benchmark_index(anchors, tmp_dir=tmp_path)

    verifier = DiffMaskVerifier()
    for a in anchors:
        saved_file = tmp_path / a.filename
        assert saved_file.exists(), f"Anchor file {a.filename} was not saved to tmp_dir"

        # Verify against saved path directly
        res = verifier.verify_path(a.image, str(saved_file))
        assert res.passed is True
        assert res.diff_ratio == 0.0
        assert res.score == 1.0
