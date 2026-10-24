"""Tests for Dual-Recall Union and Pluggable Verifiers (Phase 3)."""

import tempfile

from PIL import Image

from twin.core.config import Settings
from twin.models.clip_model import load as load_model
from twin.services.index_service import _index_single_from_disk
from twin.services.indexer import indexer
from twin.services.search import search


def test_dual_recall_saves_minor_edit_skin(monkeypatch):
    """A slightly edited derivative skin (4 eye pixels changed) excluded by vector search
    is saved and recalled via the secondary IndexBinaryFlat hash branch.
    """
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
            enable_fast_path=False,  # force pipeline to run Stage 1 dual-recall
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

        # 3. Simulate vector search returning only base.png (derivative excluded from vector top-k)
        def mock_vector_search(query_vec, k=100, datasource="skin"):
            # Exclude derivative (id=1) from vector recall to simulate vector false-negative
            return [0.0], [0]

        monkeypatch.setattr(indexer, "search", mock_vector_search)

        # Search query with derivative image
        res = search(img_deriv, top_k=5)

        # Assert dual-recall saved derivative.png via the binary index branch
        assert "dual_recall" in res["stages"]
        d_stage = res["stages"]["dual_recall"]
        assert d_stage["vector_count"] == 1
        assert d_stage["binary_count"] >= 1
        assert d_stage["union_count"] >= 2

        # Derivative must be confirmed by DiffMaskVerifier
        found_filenames = [r["filename"] for r in res["results"]]
        assert "derivative.png" in found_filenames
        deriv_res = next(r for r in res["results"] if r["filename"] == "derivative.png")
        assert deriv_res["match_level"] == "confirmed"
        assert deriv_res["ssim_score"] > 0.99  # 1.0 - diff_ratio


def test_dual_recall_deduplication(monkeypatch):
    """Candidates recalled by both vector and binary branches are deduplicated."""
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
            enable_fast_path=False,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.clear()

        img = Image.new("RGBA", (64, 64), (80, 100, 120, 255))
        p = s.images_path / "sample.png"
        img.save(p)
        _index_single_from_disk(img, p)

        res = search(img, top_k=5)
        d_stage = res["stages"]["dual_recall"]
        assert d_stage["vector_count"] == 1
        assert d_stage["binary_count"] == 1
        assert d_stage["union_count"] == 1  # Deduplicated!
        assert len(res["results"]) == 1


def test_dynamic_verifier_type_switching(monkeypatch):
    """Runtime verifier_type parameter overrides configuration smoothly."""
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
            enable_fast_path=False,
            verifier_type="diffmask",
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.clear()

        img = Image.new("RGBA", (64, 64), (50, 60, 70, 255))
        p = s.images_path / "skin.png"
        img.save(p)
        _index_single_from_disk(img, p)

        # 1. Default (diffmask)
        res_default = search(img, top_k=1)
        assert res_default["stages"]["verify"]["verifier"] == "DiffMaskVerifier"

        # 2. Explicit SSIM switch
        res_ssim = search(img, top_k=1, verifier_type="ssim")
        assert res_ssim["stages"]["verify"]["verifier"] == "SSIMVerifier"
