"""Tests for Indexer multi-datasource isolation (skin vs cape)."""

import tempfile

import numpy as np

from twin.core.config import Settings
from twin.services.indexer import Indexer


def _make_vector(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_datasource_isolation_and_search(monkeypatch):
    """Vectors added to skin or cape must not collide or leak in searches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx.load()

        v_skin1 = _make_vector(1)
        v_skin2 = _make_vector(2)
        v_cape1 = _make_vector(10)
        v_cape2 = _make_vector(20)
        v_cape3 = _make_vector(30)

        idx.add_item(v_skin1, {"filename": "skin1.png", "type": "skin"}, datasource="skin")
        idx.add_item(v_skin2, {"filename": "skin2.png", "type": "skin"}, datasource="skin")

        idx.add_item(v_cape1, {"filename": "cape1.png", "type": "cape"}, datasource="cape")
        idx.add_item(v_cape2, {"filename": "cape2.png", "type": "cape"}, datasource="cape")
        idx.add_item(v_cape3, {"filename": "cape3.png", "type": "cape"}, datasource="cape")

        # Counts
        assert idx.get_count("skin") == 2
        assert idx.get_count("cape") == 3
        assert idx.count == 5

        # Search skin datasource
        dists, ids = idx.search(v_skin1, k=10, datasource="skin")
        assert len(ids) == 2
        meta0 = idx.get_metadata(ids[0], datasource="skin")
        assert meta0["filename"] == "skin1.png"
        assert meta0["datasource"] == "skin"

        # Search cape datasource
        dists_c, ids_c = idx.search(v_cape1, k=10, datasource="cape")
        assert len(ids_c) == 3
        meta_c0 = idx.get_metadata(ids_c[0], datasource="cape")
        assert meta_c0["filename"] == "cape1.png"
        assert meta_c0["datasource"] == "cape"

        # Cross-search returns 0 cape items when searching skin
        skin_filenames = {idx.get_metadata(i, datasource="skin")["filename"] for i in ids}
        assert "cape1.png" not in skin_filenames


def test_datasource_persistence_and_reload(monkeypatch):
    """Datasources are persisted to separate directories and restored properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx.load()

        idx.add_item(_make_vector(1), {"filename": "skin.png"}, datasource="skin")
        idx.add_item(_make_vector(2), {"filename": "cape.png"}, datasource="cape")
        idx.save()

        # Check physical paths
        skin_faiss = s._faiss_path_for("flat", datasource="skin")
        cape_faiss = s._faiss_path_for("flat", datasource="cape")
        assert skin_faiss.exists()
        assert cape_faiss.exists()
        assert skin_faiss != cape_faiss

        # Reload into new Indexer
        idx2 = Indexer()
        idx2.load()

        assert idx2.get_count("skin") == 1
        assert idx2.get_count("cape") == 1
        assert idx2.get_metadata(0, datasource="skin")["filename"] == "skin.png"
        assert idx2.get_metadata(0, datasource="cape")["filename"] == "cape.png"


def test_datasource_selective_clear(monkeypatch):
    """Clearing one datasource leaves the other intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx.load()

        idx.add_item(_make_vector(1), {"filename": "skin.png"}, datasource="skin")
        idx.add_item(_make_vector(2), {"filename": "cape.png"}, datasource="cape")
        idx.save()

        # Clear only cape
        idx.clear(datasource="cape")
        assert idx.get_count("skin") == 1
        assert idx.get_count("cape") == 0
        assert not s._faiss_path_for("flat", datasource="cape").exists()
        assert s._faiss_path_for("flat", datasource="skin").exists()

        # Clear all
        idx.clear()
        assert idx.get_count("skin") == 0
        assert idx.get_count("cape") == 0
        assert not s._faiss_path_for("flat", datasource="skin").exists()
