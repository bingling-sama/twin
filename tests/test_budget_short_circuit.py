"""Tests for zero-GPU fast-path and budget short-circuit verification (Phase 2)."""

import tempfile
from unittest.mock import MagicMock

from PIL import Image

import twin.services.search as search_mod
from twin.core.config import Settings
from twin.models.clip_model import load as load_model
from twin.services.index_service import _index_single_from_disk
from twin.services.indexer import indexer
from twin.services.search import search


def test_exact_duplicate_zero_gpu_fast_path(monkeypatch):
    """When top_k=1 and exact match is in index, fast-path exits in < 3ms
    without calling compute_embedding.
    """
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

        # Create and index an image
        img = Image.new("RGBA", (64, 64), (120, 80, 200, 255))
        img_path = s.images_path / "original.png"
        img.save(img_path)
        _index_single_from_disk(img, img_path)

        # Mock compute_embedding to ensure GPU forward pass is NEVER invoked
        mock_embed = MagicMock(side_effect=RuntimeError("GPU inference should not be called!"))
        monkeypatch.setattr(search_mod, "compute_embedding", mock_embed)

        # Query exact duplicate with top_k=1
        res = search(img, top_k=1)

        # Assertions
        assert mock_embed.call_count == 0
        assert res["count"] == 1
        assert res["results"][0]["match_level"] == "confirmed"
        assert res["results"][0]["ssim_score"] == 1.0
        assert res["stages"]["fast_path"]["hit"] is True
        assert res["stages"]["fast_path"]["candidates"] == 1


def test_top_k_exceeds_exact_matches_falls_back_to_vector(monkeypatch):
    """When requested top_k=5 exceeds available exact duplicates (only 1 exists),
    fallback to vector recall.
    """
    import numpy as np

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

        # Add 1 base image with distinct noise pattern
        rng0 = np.random.RandomState(42)
        arr0 = rng0.randint(0, 256, (64, 64, 4), dtype=np.uint8)
        arr0[:, :, 3] = 255
        img = Image.fromarray(arr0, mode="RGBA")
        p0 = s.images_path / "skin_0.png"
        img.save(p0)
        _index_single_from_disk(img, p0)

        # Add 3 completely distinct random skins so their hashes diverge
        for i in range(1, 4):
            rng_i = np.random.RandomState(100 + i)
            arr_i = rng_i.randint(0, 256, (64, 64, 4), dtype=np.uint8)
            arr_i[:, :, 3] = 255
            v_img = Image.fromarray(arr_i, mode="RGBA")
            p_i = s.images_path / f"skin_{i}.png"
            v_img.save(p_i)
            _index_single_from_disk(v_img, p_i)

        embed_calls = []
        orig_embed = search_mod.compute_embedding

        def track_embed(im):
            embed_calls.append(im)
            return orig_embed(im)

        monkeypatch.setattr(search_mod, "compute_embedding", track_embed)

        # Request top_k=3 -> fast-path cannot satisfy 3 exact matches, so it falls back to vector
        res = search(img, top_k=3)

        assert len(embed_calls) == 1  # Called vector embedding once
        assert "fast_path" not in res["stages"] or res["stages"]["fast_path"].get("hit") is not True
        assert res["count"] <= 3
        # Original exact match must still be ranked at #1
        assert res["results"][0]["filename"] == "skin_0.png"
        assert res["results"][0]["match_level"] == "confirmed"


def test_low_entropy_black_skin_collision_protection(monkeypatch):
    """Even if pHash matches, SSIM verification prevents false positive fast-path exit."""
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            rotation_invariant=False,
            ssim_threshold=0.95,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.clear()

        # Base dark image
        img1 = Image.new("RGBA", (64, 64), (10, 10, 10, 255))
        p1 = s.images_path / "dark1.png"
        img1.save(p1)
        _index_single_from_disk(img1, p1)

        # Second image with distinct white block (low SSIM, but might have close hash)
        img2 = Image.new("RGBA", (64, 64), (10, 10, 10, 255))
        for x in range(20, 44):
            for y in range(20, 44):
                img2.putpixel((x, y), (255, 255, 255, 255))

        # Force binary search to return index 0 with hamming distance 0 to simulate collision
        monkeypatch.setattr(indexer, "search_binary", lambda q, k, datasource="skin": ([0], [0]))

        # Search with img2
        res = search(img2, top_k=1)

        # Since img2 vs dark1 SSIM is very low (< 0.95), fast-path MUST NOT accept it as confirmed
        assert "fast_path" not in res["stages"] or res["stages"]["fast_path"].get("hit") is not True
