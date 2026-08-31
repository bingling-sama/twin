"""Tests for Faiss indexer."""

import tempfile
import threading

import faiss
import numpy as np
import pytest

from twin.core.config import Settings, settings
from twin.services.indexer import Indexer


@pytest.fixture(autouse=True)
def _reset_indexer_settings():
    orig_dim = settings.embedding_dim
    orig_type = settings.model_type
    settings.embedding_dim = 512
    settings.model_type = "clip"
    yield
    settings.embedding_dim = orig_dim
    settings.model_type = orig_type


def _make_settings(tmpdir: str) -> Settings:
    return Settings(index_path=tmpdir, embedding_dim=settings.embedding_dim)


def _make_vector(dim: int | None = None) -> np.ndarray:
    d = dim if dim is not None else settings.embedding_dim
    return np.random.randn(d).astype(np.float32)


def test_add_and_search():
    """Adding a vector and searching returns it with ~0 distance."""
    with tempfile.TemporaryDirectory():
        idx = Indexer()
        # Monkey-patch singleton access for isolated testing
        idx._index = idx._create_index()
        idx._metadata = []

        v1 = _make_vector()
        v2 = _make_vector()

        id1 = idx.add_item(v1.copy(), {"filename": "a.jpg"})
        id2 = idx.add_item(v2.copy(), {"filename": "b.jpg"})

        assert id1 == 0
        assert id2 == 1

        dists, ids = idx.search(v1.copy(), k=2)
        assert ids[0] == 0  # closest should be itself
        assert dists[0] < 1e-4

        assert ids[1] == 1


def test_empty_index_search_returns_empty():
    """Search on empty index should return empty lists, not crash."""
    idx = Indexer()
    idx._index = idx._create_index()
    idx._metadata = []

    dists, ids = idx.search(_make_vector(), k=10)
    assert dists == []
    assert ids == []


def test_metadata_roundtrip():
    """Metadata is stored and retrievable."""
    with tempfile.TemporaryDirectory():
        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        v = _make_vector()
        meta = {"filename": "test.png", "path": "/tmp/test.png", "dhash": "abc123"}

        assigned = idx.add_item(v, meta)

        stored = idx.get_metadata(assigned)
        assert stored is not None
        assert stored["filename"] == "test.png"
        assert stored["dhash"] == "abc123"


def test_thread_safety():
    """Concurrent adds from multiple threads don't corrupt state."""
    idx = Indexer()
    idx._index = idx._create_index()
    idx._metadata = []

    errors = []

    def add_one():
        try:
            v = _make_vector()
            idx.add_item(v, {"filename": "thread_test.png"})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_one) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert idx.count == 10


def test_batch_add():
    """Batch add should work correctly."""
    idx = Indexer()
    idx._index = idx._create_index()
    idx._metadata = []

    vectors = np.random.randn(5, settings.embedding_dim).astype(np.float32)
    metas = [{"filename": f"{i}.png"} for i in range(5)]

    ids = idx.add_items(vectors, metas)
    assert ids == [0, 1, 2, 3, 4]
    assert idx.count == 5

    # Search for first vector
    dists, found_ids = idx.search(vectors[0], k=1)
    assert found_ids[0] == 0
    assert dists[0] < 1e-4


def test_persistence_roundtrip(monkeypatch):
    """Save then load preserves vectors and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _make_settings(tmpdir)

        # Redirect module-level settings to temp dir
        monkeypatch.setattr("twin.services.indexer.settings", s)

        # Create and populate
        idx1 = Indexer()
        idx1._index = idx1._create_index()
        idx1._metadata = []

        v = _make_vector()
        meta = {"filename": "persist.png", "dhash": "fe01"}
        idx1.add_item(v.copy(), meta)

        idx1.save()

        # Load into fresh indexer
        idx2 = Indexer()
        idx2.load()

        assert idx2.count == 1
        m = idx2.get_metadata(0)
        assert m["filename"] == "persist.png"
        assert m["dhash"] == "fe01"

        dists, ids = idx2.search(v.copy(), k=1)
        assert ids[0] == 0
        assert dists[0] < 1e-4


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------
def test_clear(monkeypatch):
    """clear() resets index and deletes disk files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _make_settings(tmpdir)
        # Indexer uses its own module-level import of settings
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        v = _make_vector()
        idx.add_item(v.copy(), {"filename": "test.png"})
        idx.save()

        # Verify files exist
        assert s.faiss_path.exists()
        assert s.metadata_path.exists()

        # Clear
        idx.clear()

        assert idx.count == 0
        assert not s.faiss_path.exists()
        assert not s.metadata_path.exists()


# ---------------------------------------------------------------------------
# list_items
# ---------------------------------------------------------------------------
def test_list_items_pagination():
    """list_items returns paginated results."""
    idx = Indexer()
    idx._index = idx._create_index()
    idx._metadata = []

    for i in range(5):
        v = _make_vector()
        idx.add_item(v, {"filename": f"{i}.png"})

    result = idx.list_items(page=1, page_size=2)
    assert result["total"] == 5
    assert result["page"] == 1
    assert result["page_size"] == 2
    assert len(result["items"]) == 2
    assert result["items"][0]["filename"] == "0.png"

    # Page 3 should have 1 item
    result = idx.list_items(page=3, page_size=2)
    assert len(result["items"]) == 1
    assert result["items"][0]["filename"] == "4.png"


def test_list_items_empty():
    """list_items on empty index returns total=0."""
    idx = Indexer()
    idx._index = idx._create_index()
    idx._metadata = []

    result = idx.list_items()
    assert result["total"] == 0
    assert result["items"] == []


# ---------------------------------------------------------------------------
# get_metadata out of range
# ---------------------------------------------------------------------------
def test_get_metadata_out_of_range():
    """Negative or too-large index returns None."""
    idx = Indexer()
    idx._index = idx._create_index()
    idx._metadata = []

    assert idx.get_metadata(-1) is None
    assert idx.get_metadata(0) is None
    assert idx.get_metadata(100) is None


# ---------------------------------------------------------------------------
# save empty index
# ---------------------------------------------------------------------------
def test_save_empty_index(monkeypatch):
    """Saving an empty index is a no-op."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _make_settings(tmpdir)
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []
        idx._dirty = True

        idx.save()  # should not write files

        assert not s.faiss_path.exists()
        assert not s.metadata_path.exists()


# ---------------------------------------------------------------------------
# Load — corrupt index
# ---------------------------------------------------------------------------
def test_load_corrupt_index(monkeypatch):
    """Corrupt index files trigger a fresh start."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _make_settings(tmpdir)
        monkeypatch.setattr("twin.services.indexer.settings", s)

        # Write corrupt data
        s.faiss_path.write_text("not a faiss index")
        s.metadata_path.write_text("not valid json")

        idx = Indexer()
        idx.load()

        # Should have created a fresh empty index
        assert idx.count == 0


# ---------------------------------------------------------------------------
# Load — empty metadata (missing files)
# ---------------------------------------------------------------------------
def test_load_missing_files(monkeypatch):
    """When index files don't exist, load creates a fresh index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _make_settings(tmpdir)
        monkeypatch.setattr("twin.services.indexer.settings", s)

        # Don't create any files

        idx = Indexer()
        idx.load()

        assert idx.count == 0


# ---------------------------------------------------------------------------
# Auto-save — disabled
# ---------------------------------------------------------------------------
def test_auto_save_disabled(monkeypatch):
    """When interval ≤ 0, auto-save is not started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _make_settings(tmpdir)
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr(s, "auto_save_interval", 0)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        idx.start_auto_save()
        assert idx._auto_save_thread is None

        idx.stop_auto_save()
        assert idx._auto_save_thread is None


# ============================================================================
# HNSW index tests
# ============================================================================


def test_create_hnsw_index(monkeypatch):
    """When faiss_index_type='hnsw', _create_index returns IndexHNSWFlat."""
    s = Settings(index_path="/tmp", faiss_index_type="hnsw")
    monkeypatch.setattr("twin.services.indexer.settings", s)

    idx = Indexer()
    idx._index = None  # simulate empty state before load
    idx._metadata = []

    result = idx._create_index()
    assert isinstance(result, faiss.IndexHNSWFlat)
    assert hasattr(result, "hnsw")
    # efConstruction should be set before any add
    assert result.hnsw.efConstruction == s.faiss_hnsw_ef_construction


def test_hnsw_add_and_search(monkeypatch):
    """Adding vectors to HNSW and searching returns them correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="hnsw")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        v = _make_vector()
        idx.add_item(v.copy(), {"filename": "hnsw_test.png"})

        assert idx.count == 1
        assert idx.index_type_name == "IndexHNSWFlat"

        dists, ids = idx.search(v.copy(), k=1)
        assert ids[0] == 0
        assert dists[0] < 1e-4


def test_hnsw_save_load_roundtrip(monkeypatch):
    """HNSW index survives save/load and preserves search results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="hnsw")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        # Create and populate
        idx1 = Indexer()
        idx1._index = idx1._create_index()
        idx1._metadata = []

        v = _make_vector()
        idx1.add_item(v.copy(), {"filename": "hnsw_persist.png"})
        idx1.save()

        # Load into fresh indexer
        idx2 = Indexer()
        idx2.load()

        assert idx2.count == 1
        assert idx2.index_type_name == "IndexHNSWFlat"
        assert idx2.get_metadata(0)["filename"] == "hnsw_persist.png"

        # efSearch should be restored on load
        assert idx2._index.hnsw.efSearch == s.faiss_hnsw_ef_search

        # Search should still work
        dists, ids = idx2.search(v.copy(), k=1)
        assert ids[0] == 0
        assert dists[0] < 1e-4


def test_hnsw_ef_search_applied(monkeypatch):
    """Search sets efSearch on the HNSW index before querying."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="hnsw",
            faiss_hnsw_ef_search=42,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        # Add a few vectors so we can search
        for i in range(5):
            idx.add_item(_make_vector(), {"filename": f"{i}.png"})

        # Call search — it should set efSearch to 42
        idx.search(_make_vector(), k=3)
        assert idx._index.hnsw.efSearch == 42


def test_hnsw_train_is_noop(monkeypatch):
    """train_index returns 'skipped' for HNSW with a descriptive reason."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="hnsw")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        # Add some vectors — train should still be no-op for HNSW
        for i in range(10):
            idx.add_item(_make_vector(), {"filename": f"{i}.png"})

        result = idx.train_index()
        assert result["status"] == "skipped"
        assert "hnsw" in result["reason"].lower()


def test_hnsw_batch_add(monkeypatch):
    """Batch add works with HNSW index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="hnsw")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        vectors = np.random.randn(5, 512).astype(np.float32)
        metas = [{"filename": f"hnsw_batch_{i}.png"} for i in range(5)]

        ids = idx.add_items(vectors, metas)
        assert ids == [0, 1, 2, 3, 4]
        assert idx.count == 5

        # Search for first vector
        dists, found_ids = idx.search(vectors[0], k=1)
        assert found_ids[0] == 0
        assert dists[0] < 1e-4


# ============================================================================
# IVFPQ index tests
# ============================================================================


def test_create_ivfpq_index(monkeypatch):
    """When faiss_index_type='ivf_pq' and index has existing vectors,
    _create_index returns IndexIVFPQ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_pq_m=64,
            faiss_pq_nbits=8,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        # Simulate a pre-existing Flat index with some vectors
        idx._index = faiss.IndexFlatL2(512)
        idx._index.add(np.random.randn(10, 512).astype(np.float32))
        idx._metadata = [{"filename": f"{i}.png"} for i in range(10)]

        result = idx._create_index()
        assert isinstance(result, faiss.IndexIVFPQ)
        assert result.nprobe == s.faiss_nprobe


def test_ivfpq_empty_index_starts_flat(monkeypatch):
    """When faiss_index_type='ivf_pq' and no existing vectors,
    _create_index starts with IndexFlatL2 (PQ needs training)."""
    s = Settings(index_path="/tmp", faiss_index_type="ivf_pq")
    monkeypatch.setattr("twin.services.indexer.settings", s)

    idx = Indexer()
    idx._index = None
    idx._metadata = []

    result = idx._create_index()
    # Should start with Flat since PQ requires training
    assert isinstance(result, faiss.IndexFlatL2)


def test_ivfpq_add_and_search(monkeypatch):
    """Adding vectors to IVFPQ (pre-trained) and searching returns them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # nbits=3 → 8 centroids/sub-space (needs 39*8=312 vectors for good training)
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,  # small nlist → IVF training needs 4*39=156
            faiss_pq_m=16,  # 32-dimensional sub-vectors
            faiss_pq_nbits=3,  # 2^3=8 centroids/sub-space
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = idx._create_index()
        idx._metadata = []

        # Use 1000 vectors for reliable PQ training
        n_train = 1000
        rng = np.random.RandomState(42)
        train_vecs = rng.randn(n_train, 512).astype(np.float32)
        train_metas = [{"filename": f"train_{i}.png"} for i in range(n_train)]

        # Start with Flat, add training data
        idx._index = faiss.IndexFlatL2(512)
        idx._index.add(train_vecs)
        idx._metadata = train_metas

        # Train to IVFPQ
        result = idx.train_index()
        assert result["status"] == "trained"
        assert result["index_type"] == "IndexIVFPQ"

        # Now add a new vector and search for it
        v = _make_vector()
        idx.add_item(v.copy(), {"filename": "ivfpq_test.png"})

        assert idx.count == n_train + 1
        assert "IVFPQ" in idx.index_type_name

        dists, ids = idx.search(v.copy(), k=1)
        # PQ is approximate — self-distance won't be zero, but the vector
        # should still find itself as its own nearest neighbor
        assert ids[0] == n_train


def test_ivfpq_train_from_flat(monkeypatch):
    """train_index converts IndexFlatL2 → IndexIVFPQ when configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,  # small nlist → training needs 4*39=156 vectors
            faiss_pq_m=16,  # 32-dimensional sub-vectors
            faiss_pq_nbits=3,  # 2^3=8 centroids/sub-space
            faiss_auto_upgrade=False,  # disable so we test explicit train_index
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        # Add enough vectors to train (need >= 39*8=312 for PQ)
        n = 1000
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        metas = [{"filename": f"train_{i}.png"} for i in range(n)]
        ids = idx.add_items(vectors, metas)
        assert len(ids) == n

        # Explicitly train (not via auto-upgrade)
        result = idx.train_index()
        assert result["status"] == "trained"
        assert result["index_type"] == "IndexIVFPQ"
        assert "nlist" in result
        assert result["n_vectors"] == n

        # Search should still work after training
        dists, found_ids = idx.search(vectors[0], k=1)
        # PQ is approximate — self-distance may not be ~0, but self should be closest
        assert found_ids[0] == 0


def test_ivfpq_save_load_roundtrip(monkeypatch):
    """IVFPQ index survives save/load and preserves search results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,  # small nlist → training needs 4*39=156 vectors
            faiss_pq_m=16,  # 32-dimensional sub-vectors
            faiss_pq_nbits=3,  # 2^3=8 centroids/sub-space
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        # Create and train an IVFPQ index
        idx1 = Indexer()
        idx1._index = faiss.IndexFlatL2(512)
        idx1._metadata = []

        n = 1000
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx1._index.add(vectors[i : i + 1])
            idx1._metadata.append({"filename": f"pq_{i}.png"})

        idx1.train_index()
        idx1.save()

        # Verify files exist
        assert s.faiss_path.exists()
        assert s.metadata_path.exists()

        # Load into fresh indexer
        idx2 = Indexer()
        idx2.load()

        assert idx2.count == n
        assert "IVFPQ" in idx2.index_type_name
        assert idx2.get_metadata(0)["filename"] == "pq_0.png"

        # nprobe should be restored
        cpu_idx = idx2._maybe_to_cpu(idx2._index)
        assert cpu_idx.nprobe == s.faiss_nprobe

        # Search should still work
        dists, ids = idx2.search(vectors[0], k=1)
        # PQ is approximate — self should still be the closest match
        assert ids[0] == 0


def test_ivfpq_already_trained_is_skipped(monkeypatch):
    """train_index returns 'skipped' for an already-trained IVFPQ index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,  # small nlist → training needs 4*39=156 vectors
            faiss_pq_m=16,  # 32-dimensional sub-vectors
            faiss_pq_nbits=3,  # 2^3=8 centroids/sub-space
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        n = 1000
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"pq_{i}.png"})

        # First train
        result1 = idx.train_index()
        assert result1["status"] == "trained"

        # Second train should be skipped (already IVF)
        result2 = idx.train_index()
        assert result2["status"] == "skipped"
        assert "already an IndexIVFPQ" in result2["reason"]


def test_ivfpq_upgrade_from_ivfflat(monkeypatch):
    """train_index upgrades IVFFlat → IVFPQ when target is ivf_pq."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,
            faiss_pq_m=16,
            faiss_pq_nbits=3,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()

        # Phase 1: build an IVFFlat index (simulating the typical Flat→IVF path)
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []
        n = 1000
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"img_{i}.png"})

        # Train Flat → IVFFlat first (simulating what happens when target was ivf_flat)
        result1 = idx.train_index()
        assert result1["status"] == "trained"
        assert "IndexIVFPQ" in result1["index_type"]
        # It goes straight to IVFPQ because target is ivf_pq, not ivf_flat

        # Test the actual IVFFlat → IVFPQ path:
        # Build a fresh IVFFlat to simulate the intermediate state
        idx2 = Indexer()
        idx2._metadata = []
        n2 = 500
        rng2 = np.random.RandomState(99)
        vecs2 = rng2.randn(n2, 512).astype(np.float32)
        dim = 512
        nlist = 4
        quantizer = faiss.IndexFlatL2(dim)
        idx2._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        idx2._index.nprobe = 8
        idx2._index.train(vecs2)
        idx2._index.add(vecs2)
        for i in range(n2):
            idx2._metadata.append({"filename": f"ivf_{i}.png"})

        # Now upgrade to IVFPQ
        result2 = idx2.train_index()
        assert result2["status"] == "upgraded"
        assert "IndexIVFPQ" in result2["index_type"]
        assert idx2._index is not None
        assert "IVFPQ" in type(idx2._index).__name__

        # Search still works
        q = rng2.randn(1, 512).astype(np.float32)
        dists, ids = idx2.search(q, k=5)
        assert len(ids) == 5

        # Second train skips (already IVFPQ)
        result3 = idx2.train_index()
        assert result3["status"] == "skipped"
        assert "already an IndexIVFPQ" in result3["reason"]


def test_ivfpq_not_enough_vectors(monkeypatch):
    """train_index is skipped when fewer than nlist*39 vectors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=20,  # needs 20*39=780 vectors
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        # Add only 50 vectors (far fewer than 780)
        for i in range(50):
            idx._index.add(_make_vector().reshape(1, -1))
            idx._metadata.append({"filename": f"few_{i}.png"})

        result = idx.train_index()
        assert result["status"] == "skipped"
        assert "need >=" in result["reason"]


def test_resolve_pq_m(monkeypatch):
    """_resolve_pq_m returns configured value or auto-computed default."""
    idx = Indexer()

    # Default: dim // 8
    assert idx._resolve_pq_m(512) == 64  # 512 // 8
    assert idx._resolve_pq_m(768) == 96  # 768 // 8

    # Explicit override
    s = Settings(faiss_pq_m=32)
    monkeypatch.setattr("twin.services.indexer.settings", s)
    assert idx._resolve_pq_m(512) == 32
    assert idx._resolve_pq_m(768) == 32  # override always wins


def test_ivfpq_batch_add(monkeypatch):
    """Batch add works with pre-trained IVFPQ index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,  # small nlist → training needs 4*39=156 vectors
            faiss_pq_m=16,  # 32-dimensional sub-vectors
            faiss_pq_nbits=3,  # 2^3=8 centroids/sub-space
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        # Train first
        n_train = 1000
        rng = np.random.RandomState(42)
        train_vecs = rng.randn(n_train, 512).astype(np.float32)
        for i in range(n_train):
            idx._index.add(train_vecs[i : i + 1])
            idx._metadata.append({"filename": f"train_{i}.png"})
        result = idx.train_index()
        assert result["status"] == "trained"

        # Batch add
        batch_vecs = np.random.randn(5, 512).astype(np.float32)
        batch_metas = [{"filename": f"batch_{i}.png"} for i in range(5)]
        ids = idx.add_items(batch_vecs, batch_metas)

        assert ids == [1000, 1001, 1002, 1003, 1004]
        assert idx.count == 1005

        # Search for first batch vector
        dists, found_ids = idx.search(batch_vecs[0], k=1)
        # PQ is approximate — self should still be closest
        assert found_ids[0] == 1000


def test_ivfpq_search_distances_are_l2(monkeypatch):
    """IVFPQ search returns L2 distances in non-decreasing order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_pq",
            faiss_nlist=4,  # small nlist → training needs 4*39=156 vectors
            faiss_pq_m=16,  # 32-dimensional sub-vectors
            faiss_pq_nbits=3,  # 2^3=8 centroids/sub-space
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        n = 1000
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"d_{i}.png"})
        result = idx.train_index()
        assert result["status"] == "trained"

        # Search should return increasing distances
        dists, ids = idx.search(vectors[0], k=5)
        assert ids[0] == 0  # itself first
        # Distances should be non-decreasing (monotonic)
        for i in range(len(dists) - 1):
            assert dists[i] <= dists[i + 1] + 1e-5  # small epsilon for PQ approximation


def test_rebuild_preserves_data_flat_to_hnsw(monkeypatch):
    """rebuild() preserves all vectors when switching index types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="flat",
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"img_{i}.png"})

        orig_count = idx.count
        assert orig_count == n

        # Search works before rebuild
        q = vectors[0]
        dists_before, ids_before = idx.search(q, k=3)
        assert ids_before[0] == 0

        # Switch to HNSW and rebuild
        s.faiss_index_type = "hnsw"
        result = idx.rebuild()
        assert result["status"] == "rebuilt"
        assert result["n_vectors"] == n
        assert "HNSW" in result["index_type"]
        assert idx.count == n
        assert len(idx._metadata) == n

        # Search still works with correct results
        dists_after, ids_after = idx.search(q, k=3)
        assert ids_after[0] == 0  # self-match preserved


def test_rebuild_empty_index(monkeypatch):
    """rebuild() on an empty index does not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, faiss_index_type="hnsw")
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = None
        idx._metadata = []

        result = idx.rebuild()
        assert result["status"] == "rebuilt"
        assert result["n_vectors"] == 0
        assert idx.count == 0


def test_rebuild_flat_to_ivf_with_auto_train(monkeypatch):
    """rebuild() auto-trains when target is IVF and enough vectors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="ivf_flat",
            faiss_auto_upgrade=True,
            faiss_nlist=4,  # needs 4*39=156 vectors
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        n = 500
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"img_{i}.png"})

        # Rebuild: flat → ivf_flat (should auto-train)
        result = idx.rebuild()
        assert result["status"] == "rebuilt"
        assert result["n_vectors"] == n
        assert "IVF" in result["index_type"]

        # Search works
        q = vectors[0]
        dists, ids = idx.search(q, k=3)
        assert ids[0] == 0


def test_rebuild_fast_switch_uses_cache(monkeypatch):
    """Switching back to a previously-built type loads from disk (instant)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="flat",
            faiss_auto_upgrade=False,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"img_{i}.png"})

        # Switch flat → hnsw (slow path — builds from scratch)
        s.faiss_index_type = "hnsw"
        r1 = idx.rebuild()
        assert r1["status"] == "rebuilt"
        assert r1["cached"] is False

        # Switch hnsw → flat (fast path — loads cached flat index)
        s.faiss_index_type = "flat"
        r2 = idx.rebuild()
        assert r2["status"] == "switched"
        assert r2["cached"] is True
        assert idx.count == n

        # Switch flat → hnsw again (fast path — loads cached hnsw)
        s.faiss_index_type = "hnsw"
        r3 = idx.rebuild()
        assert r3["status"] == "switched"
        assert r3["cached"] is True
        assert idx.count == n

        # Search works after each switch
        q = vectors[0]
        dists, ids = idx.search(q, k=3)
        assert ids[0] == 0


def test_rebuild_detects_stale_cache_by_count(monkeypatch):
    """Cached index with wrong vector count is detected as stale and rebuilt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(
            index_path=tmpdir,
            faiss_index_type="flat",
            faiss_auto_upgrade=False,
        )
        monkeypatch.setattr("twin.services.indexer.settings", s)

        idx = Indexer()
        idx._index = faiss.IndexFlatL2(512)
        idx._metadata = []

        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, 512).astype(np.float32)
        for i in range(n):
            idx._index.add(vectors[i : i + 1])
            idx._metadata.append({"filename": f"img_{i}.png"})

        # Build and cache hnsw (saves .count = 200)
        s.faiss_index_type = "hnsw"
        idx.rebuild()

        # Switch back to flat
        s.faiss_index_type = "flat"
        idx.rebuild()

        # Verify hnsw cache + count file exist
        assert s._faiss_path_for("hnsw").exists()
        assert idx._count_path_for("hnsw").exists()

        # Add new vectors — cache files remain (not deleted), but .count is now stale
        new_vec = rng.randn(1, 512).astype(np.float32)
        idx.add_item(new_vec, {"filename": "new.png"})
        assert s._faiss_path_for("hnsw").exists()  # still there

        # Switch to hnsw — count mismatch (200 vs 201) → incremental add (fast!)
        s.faiss_index_type = "hnsw"
        r = idx.rebuild()
        assert r["status"] == "switched"
        assert r["cached"] is True
        assert r["incremental"] is True
        assert r["added"] == 1
        assert idx.count == n + 1  # includes the new vector

        # HNSW cache is now at 201 — next switch will be pure fast path
        s.faiss_index_type = "flat"
        idx.rebuild()
        s.faiss_index_type = "hnsw"
        r2 = idx.rebuild()
        assert r2["status"] == "switched"
        assert r2["cached"] is True
        assert r2.get("incremental") is not True  # nothing new to add
