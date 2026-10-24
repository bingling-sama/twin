"""Tests for Flat SIMD Recall and Pluggable Verifiers (Issue #5)."""

import tempfile

from PIL import Image

from twin.core.config import Settings
from twin.models.clip_model import load as load_model
from twin.services.index_service import _index_single_from_disk
from twin.services.indexer import indexer
from twin.services.search import search


def test_simd_recall_saves_minor_edit_skin(monkeypatch):
    """A slightly edited derivative skin has tiny Hamming distance (<= 2) and is accurately
    retrieved by IndexBinaryFlat and confirmed by DiffMaskVerifier (0% false negative).
    """
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
            verifier_type="diffmask",
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.clear()

        # 1. Base skin
        img_base = Image.new("RGBA", (64, 64), (100, 120, 140, 255))
        p_base = s.images_path / "base.png"
        img_base.save(p_base)
        _index_single_from_disk(img_base, p_base)

        # 2. Derivative skin: only 4 pixels changed
        img_deriv = img_base.copy()
        for x in range(12, 14):
            for y in range(12, 14):
                img_deriv.putpixel((x, y), (255, 0, 0, 255))
        p_deriv = s.images_path / "derivative.png"
        img_deriv.save(p_deriv)
        _index_single_from_disk(img_deriv, p_deriv)

        # Search query with derivative image
        res = search(img_deriv, top_k=5)

        # Both base and derivative should be in top_k
        assert res["count"] >= 2
        found_filenames = [r["filename"] for r in res["results"]]
        assert "derivative.png" in found_filenames
        assert "base.png" in found_filenames

        # Derivative must be top-1 confirmed
        deriv_res = res["results"][0]
        assert deriv_res["filename"] == "derivative.png"
        assert deriv_res["match_level"] == "confirmed"
        assert deriv_res["ssim_score"] == 1.0


def test_single_pipeline_topk_ranking(monkeypatch):
    """Single pipeline ranks candidates cleanly without compensatory patches."""
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.clear()

        img = Image.new("RGBA", (64, 64), (80, 100, 120, 255))
        p = s.images_path / "sample.png"
        img.save(p)
        _index_single_from_disk(img, p)

        res = search(img, top_k=5)
        assert res["count"] == 1
        assert res["results"][0]["filename"] == "sample.png"
        assert res["results"][0]["match_level"] == "confirmed"


def test_dynamic_verifier_type_switching(monkeypatch):
    """Runtime verifier_type parameter overrides configuration smoothly."""
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
            verifier_type="diffmask",
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.clear()

        img = Image.new("RGBA", (64, 64), (50, 60, 70, 255))
        p = s.images_path / "skin.png"
        img.save(p)
        _index_single_from_disk(img, p)

        # 1. Default config: DiffMaskVerifier
        res_default = search(img, top_k=1)
        assert res_default["stages"]["verify"]["verifier"] == "DiffMaskVerifier"

        # 2. Dynamic override to SSIMVerifier
        res_ssim = search(img, top_k=1, verifier_type="ssim")
        assert res_ssim["stages"]["verify"]["verifier"] == "SSIMVerifier"
