"""Tests for pure-CPU zero-GPU image search and Verifier protection (Issue #5)."""

import tempfile
from unittest.mock import MagicMock

from PIL import Image

from twin.core.config import Settings
from twin.models.clip_model import load as load_model
from twin.services.index_service import _index_single_from_disk
from twin.services.indexer import indexer
from twin.services.search import search


def test_exact_duplicate_zero_gpu_search(monkeypatch):
    """Image search is completely decoupled from GPU and NEVER invokes compute_embedding
    regardless of top_k.
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
        mock_embed = MagicMock(side_effect=RuntimeError("GPU inference should NEVER be called!"))
        monkeypatch.setattr("twin.services.embedding.compute_embedding", mock_embed)

        # Query exact duplicate with top_k=1
        res = search(img, top_k=1)

        # Assertions
        assert mock_embed.call_count == 0  # 100% Zero GPU
        assert res["count"] == 1
        assert res["results"][0]["match_level"] == "confirmed"
        assert res["results"][0]["phash_distance"] == 0
        assert res["stages"]["simd_recall"]["in"] == 1
        assert res["stages"]["simd_recall"]["out"] == 1


def test_top_k_exact_recall_without_gpu(monkeypatch):
    """When requested top_k=3, all candidates are recalled via CPU SIMD and evaluated
    via Unified Verifier without any GPU dependency.
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

        # Add 3 distinct random skins
        for i in range(1, 4):
            rng_i = np.random.RandomState(100 + i)
            arr_i = rng_i.randint(0, 256, (64, 64, 4), dtype=np.uint8)
            arr_i[:, :, 3] = 255
            v_img = Image.fromarray(arr_i, mode="RGBA")
            p_i = s.images_path / f"skin_{i}.png"
            v_img.save(p_i)
            _index_single_from_disk(v_img, p_i)

        mock_embed = MagicMock(side_effect=RuntimeError("GPU inference should NEVER be called!"))
        monkeypatch.setattr("twin.services.embedding.compute_embedding", mock_embed)

        # Request top_k=3
        res = search(img, top_k=3)

        assert mock_embed.call_count == 0  # Zero GPU
        assert res["count"] == 3
        # Original exact match must still be ranked at #1
        assert res["results"][0]["filename"] == "skin_0.png"
        assert res["results"][0]["match_level"] == "confirmed"
        assert res["results"][0]["phash_distance"] == 0
        assert res["stages"]["verify"]["evaluated"] == 3


def test_low_entropy_black_skin_collision_protection(monkeypatch):
    """Even if binary hash is retrieved in top-K, Verifier (DiffMask) prevents false positives."""
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

        # Base dark image
        img1 = Image.new("RGBA", (64, 64), (10, 10, 10, 255))
        p1 = s.images_path / "dark1.png"
        img1.save(p1)
        _index_single_from_disk(img1, p1)

        # Distinct image with large white patch
        img2 = Image.new("RGBA", (64, 64), (10, 10, 10, 255))
        for x in range(20, 44):
            for y in range(20, 44):
                img2.putpixel((x, y), (255, 255, 255, 255))

        res = search(img2, top_k=1)
        assert res["count"] == 1
        # Recalled but not confirmed due to pixel diff
        assert res["results"][0]["match_level"] != "confirmed"
