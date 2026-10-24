"""Tests for faiss.IndexBinaryFlat pHash indexing and dynamic short-circuiting (Phase 2)."""

import tempfile

import numpy as np
from PIL import Image

from twin.core.config import Settings
from twin.models.clip_model import load as load_model
from twin.services.indexer import Indexer, phash_to_binary_bytes
from twin.services.search import search


def _make_vector(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_phash_to_binary_bytes_conversion():
    """Verify 16-hex to 8-byte conversion correctness and fault tolerance."""
    raw = phash_to_binary_bytes("0000000000000000")
    assert raw.shape == (8,)
    assert raw.dtype == np.uint8
    assert list(raw) == [0] * 8

    raw_ff = phash_to_binary_bytes("ffffffffffffffff")
    assert list(raw_ff) == [255] * 8

    # Fault tolerance on invalid or empty string
    empty_raw = phash_to_binary_bytes("")
    assert list(empty_raw) == [0] * 8
    corrupt_raw = phash_to_binary_bytes("not_hex")
    assert list(corrupt_raw) == [0] * 8


def test_binary_index_search_and_isolation(monkeypatch):
    """Test search_binary Hamming distance precision and datasource isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx.load()

        # Add 3 skins with varying Hamming distances
        v = _make_vector(1)
        idx.add_item(v, {"filename": "skin0.png", "phash": "0000000000000000"}, datasource="skin")
        idx.add_item(v, {"filename": "skin1.png", "phash": "000000000000000f"}, datasource="skin")
        idx.add_item(v, {"filename": "skin2.png", "phash": "ffffffffffffffff"}, datasource="skin")

        # Add cape
        idx.add_item(v, {"filename": "cape0.png", "phash": "0000000000000000"}, datasource="cape")

        # Search binary in skin
        dists, ids = idx.search_binary("0000000000000000", k=10, datasource="skin")
        assert len(ids) == 3
        assert ids == [0, 1, 2]
        assert dists[0] == 0  # identical
        assert dists[1] == 4  # 0x0f has 4 bits set
        assert dists[2] == 64  # all 64 bits inverted

        # Search binary in cape
        dists_c, ids_c = idx.search_binary("0000000000000000", k=10, datasource="cape")
        assert len(ids_c) == 1
        assert ids_c == [0]
        meta_c = idx.get_metadata(ids_c[0], datasource="cape")
        assert meta_c["filename"] == "cape0.png"


def test_binary_index_persistence_and_auto_rebuild(monkeypatch):
    """Binary index is persisted to disk and automatically rebuilt from metadata if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx1 = Indexer()
        idx1.load()
        idx1.add_item(
            _make_vector(1),
            {"filename": "s1.png", "phash": "0123456789abcdef"},
            datasource="skin",
        )
        idx1.save()

        bin_file = s._binary_faiss_path_for(datasource="skin")
        assert bin_file.exists()

        # Reload into fresh Indexer
        idx2 = Indexer()
        idx2.load()
        dists, ids = idx2.search_binary("0123456789abcdef", k=1, datasource="skin")
        assert ids == [0]
        assert dists == [0]

        # Delete binary index file to simulate legacy migration
        bin_file.unlink()
        assert not bin_file.exists()

        idx3 = Indexer()
        idx3.load()
        # Verify self-healing: automatically rebuilt from metadata
        dists3, ids3 = idx3.search_binary("0123456789abcdef", k=1, datasource="skin")
        assert ids3 == [0]
        assert dists3 == [0]


def test_dynamic_short_circuiting(monkeypatch):
    """Stage 4 short-circuits when confirmed count hits top_k or budget is exhausted."""
    load_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            verification_budget=1,
            rotation_invariant=False,
            enable_fast_path=False,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        # Clear and populate with multiple identical images
        from twin.services.indexer import indexer
        indexer.clear()

        # Simple 64x64 skin
        img = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
        img_path = s.images_path / "sample.png"
        img.save(img_path)

        from twin.services.index_service import _index_single_from_disk
        for i in range(5):
            p = s.images_path / f"sample_{i}.png"
            img.save(p)
            _index_single_from_disk(img, p)

        # 1. Test budget exhaustion short-circuit
        res_budget = search(img, top_k=5)
        ssim_stage = res_budget["stages"]["ssim"]
        assert ssim_stage["short_circuited"] is True
        assert ssim_stage["evaluated"] == 1  # budget was 1

        # 2. Test top_k hit short-circuit with larger budget
        s2 = Settings(
            index_path=tmpdir,
            images_dir=tmpdir,
            faiss_index_type="flat",
            verification_budget=50,
            rotation_invariant=False,
            enable_fast_path=False,
        )
        monkeypatch.setattr("twin.services.search.settings", s2)
        res_topk = search(img, top_k=2)
        ssim_topk = res_topk["stages"]["ssim"]
        assert ssim_topk["short_circuited"] is True
        assert ssim_topk["evaluated"] == 2
        assert ssim_topk["out"] == 2
