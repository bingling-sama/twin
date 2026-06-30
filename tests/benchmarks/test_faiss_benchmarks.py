"""Benchmarks for Faiss index operations (search, add, train).

Uses synthetic 512-dim L2-normalized vectors directly — no CLIP model
needed. This isolates Faiss performance from CLIP encoding variance.

Tests both IndexFlatL2 (exhaustive) and IndexIVFFlat (clustered) at
various scales.
"""

from __future__ import annotations

import pytest

# faiss may not be importable on Python 3.13 (no cp313 wheel yet).
# Benchmarks gracefully skip if faiss is not available.
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore

import numpy as np

from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

pytestmark = pytest.mark.skipif(not HAS_FAISS, reason="faiss not available on this Python version")


# ── Flat (exhaustive) search scaling ──────────────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_flat_search_1k(benchmark, bench_faiss_1k, bench_query_vec):
    """Benchmark: Flat search over 1K vectors, top-50."""
    idx = bench_faiss_1k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
def test_bench_faiss_flat_search_10k(benchmark, bench_faiss_10k, bench_query_vec):
    """Benchmark: Flat search over 10K vectors, top-50."""
    idx = bench_faiss_10k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
@pytest.mark.slow
def test_bench_faiss_flat_search_100k(benchmark, bench_faiss_100k, bench_query_vec):
    """Benchmark: Flat search over 100K vectors, top-50.

    At 100K with d=512, IndexFlatL2 performs 100K × 512 ≈ 51M float ops
    per query (plus memory bandwidth). Expect ~5-20ms on modern CPU.
    """
    idx = bench_faiss_100k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ── IVF (clustered) search ────────────────────────────────────────────────────


@pytest.fixture
def bench_faiss_ivf_10k() -> faiss.IndexIVFFlat:
    """A trained IndexIVFFlat with 10K vectors."""
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    dim = 512
    nlist = 40  # 4 * sqrt(10000) ≈ 400, using smaller for benchmark
    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    idx.nprobe = 8

    vecs = random_normalized_vectors(10_000, dim)
    idx.train(vecs[:5_000])  # train on first half
    idx.add(vecs)  # add all
    return idx


@pytest.mark.smoke
@pytest.mark.gpu
def test_bench_faiss_ivf_search_10k(benchmark, bench_faiss_ivf_10k, bench_query_vec):
    """Benchmark: IVF search over 10K vectors (nprobe=8)."""
    idx = bench_faiss_ivf_10k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
@pytest.mark.parametrize("nprobe", [1, 4, 8, 16, 32])
def test_bench_faiss_ivf_nprobe_sweep(benchmark, bench_faiss_ivf_10k, bench_query_vec, nprobe):
    """Benchmark: IVF search latency vs nprobe (accuracy-speed tradeoff).

    Higher nprobe = more clusters searched = better recall but slower.
    """
    idx = bench_faiss_ivf_10k
    idx.nprobe = nprobe
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ── Add (insert) throughput ───────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_add_batch_32(benchmark, bench_faiss_empty):
    """Benchmark: add 32 vectors to an empty Flat index (simulates batch indexing)."""
    idx = bench_faiss_empty
    vecs = random_normalized_vectors(32, 512, seed=42)

    def _add():
        idx.add(vecs)

    benchmark(_add)


@pytest.mark.scaling
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 1024])
def test_bench_faiss_add_batch_size(benchmark, batch_size):
    """Benchmark: add vectors at various batch sizes.

    Creates a fresh IndexFlatL2 per call to avoid cumulative growth
    across benchmark iterations — critical at batch_size=1024 where
    repeated adds to the same index cause std::bad_alloc.
    """
    import faiss as faiss_lib

    vecs = random_normalized_vectors(batch_size, 512, seed=42)

    def _add():
        idx = faiss_lib.IndexFlatL2(512)
        idx.add(vecs)

    benchmark(_add)


# ── IVF training time ─────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.gpu
def test_bench_faiss_ivf_train(benchmark):
    """Benchmark: train an IVF index from 50K vectors.

    IVF training runs k-means clustering, which is O(nlist × n × d × iterations).
    """
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    dim = 512
    nlist = 100
    vecs = random_normalized_vectors(50_000, dim, seed=42)

    def _train():
        quantizer = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        idx.train(vecs)

    benchmark(_train)


# ── Batch query throughput ────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("n_queries", [1, 10, 50])
def test_bench_faiss_multi_query(benchmark, bench_faiss_10k, n_queries):
    """Benchmark: search with multiple query vectors (batch query)."""
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    idx = bench_faiss_10k
    queries = random_normalized_vectors(n_queries, 512, seed=42)

    def _search_batch():
        return idx.search(queries, 50)

    benchmark(_search_batch)


# ── HNSW (graph-based) search scaling ──────────────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_hnsw_search_1k(benchmark, bench_hnsw_1k, bench_query_vec):
    """Benchmark: HNSW search over 1K vectors (M=32, efSearch=64), top-50."""
    idx = bench_hnsw_1k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
def test_bench_faiss_hnsw_search_10k(benchmark, bench_hnsw_10k, bench_query_vec):
    """Benchmark: HNSW search over 10K vectors, top-50."""
    idx = bench_hnsw_10k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
@pytest.mark.slow
def test_bench_faiss_hnsw_search_100k(benchmark, bench_hnsw_100k, bench_query_vec):
    """Benchmark: HNSW search over 100K vectors (M=32, efSearch=64), top-50.

    HNSW uses a graph-based greedy search, O(log N) traversals per query.
    Should be significantly faster than Flat and competitive with IVF.
    """
    idx = bench_hnsw_100k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ── HNSW efSearch sweep (accuracy–speed trade-off) ─────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("ef_search", [16, 32, 64, 128, 256])
def test_bench_faiss_hnsw_efsearch_sweep(
    benchmark, bench_hnsw_10k, bench_query_vec, ef_search
):
    """Benchmark: HNSW search latency vs efSearch.

    Higher efSearch = deeper graph exploration = better recall but slower.
    Analogous to IVF nprobe sweep.
    """
    idx = bench_hnsw_10k
    idx.hnsw.efSearch = ef_search
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ── HNSW add throughput ────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_hnsw_add_batch_32(benchmark):
    """Benchmark: add 32 vectors to an empty HNSW index."""
    idx = faiss.IndexHNSWFlat(512, 32)
    idx.hnsw.efConstruction = 200
    vecs = random_normalized_vectors(32, 512, seed=42)

    def _add():
        idx.add(vecs)

    benchmark(_add)


@pytest.mark.scaling
@pytest.mark.parametrize("m", [16, 32, 64])
def test_bench_faiss_hnsw_construction_m(benchmark, m):
    """Benchmark: build an HNSW index at various M values.

    Higher M = denser graph = faster search but slower build + more memory.
    """
    idx = faiss.IndexHNSWFlat(512, m)
    idx.hnsw.efConstruction = 200
    vecs = random_normalized_vectors(10_000, 512, seed=42)

    def _build():
        idx.add(vecs)

    benchmark(_build)


# ── HNSW batch query ───────────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("n_queries", [1, 10, 50])
def test_bench_faiss_hnsw_multi_query(benchmark, bench_hnsw_10k, n_queries):
    """Benchmark: HNSW batch query with multiple vectors."""
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    idx = bench_hnsw_10k
    queries = random_normalized_vectors(n_queries, 512, seed=42)

    def _search_batch():
        return idx.search(queries, 50)

    benchmark(_search_batch)


# ── IVFPQ (Product Quantization) search scaling ────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_ivfpq_search_1k(benchmark, bench_ivfpq_1k, bench_query_vec):
    """Benchmark: IVFPQ search over 1K vectors (nprobe=8), top-50.

    IVFPQ uses asymmetric distance computation (ADC) — the query vector
    is not compressed, only the database vectors are.  This gives better
    accuracy than symmetric PQ while remaining memory-efficient.
    """
    idx = bench_ivfpq_1k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
def test_bench_faiss_ivfpq_search_10k(benchmark, bench_ivfpq_10k, bench_query_vec):
    """Benchmark: IVFPQ search over 10K vectors (nprobe=8), top-50."""
    idx = bench_ivfpq_10k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


@pytest.mark.scaling
@pytest.mark.slow
def test_bench_faiss_ivfpq_search_100k(benchmark, bench_ivfpq_100k, bench_query_vec):
    """Benchmark: IVFPQ search over 100K vectors (nprobe=8), top-50.

    At 100K vectors with PQ (M=64, nbits=8), the index stores only
    100K × 64 bytes ≈ 6.4 MB of compressed codes, compared to
    100K × 2048 bytes ≈ 205 MB for Flat/IVF.  Search is slightly
    slower than IVFFlat due to ADC decoding, but dramatically more
    memory-efficient.
    """
    idx = bench_ivfpq_100k
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ── IVFPQ nprobe sweep (accuracy–speed trade-off) ──────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("nprobe", [1, 4, 8, 16, 32])
def test_bench_faiss_ivfpq_nprobe_sweep(
    benchmark, bench_ivfpq_10k, bench_query_vec, nprobe
):
    """Benchmark: IVFPQ search latency vs nprobe.

    Higher nprobe = more inverted lists visited = more PQ codes decoded
    = better recall but proportionally slower.  The same trade-off as
    IndexIVFFlat, but each probe is more expensive due to PQ decode.
    """
    idx = bench_ivfpq_10k
    idx.nprobe = nprobe
    q = bench_query_vec

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ── IVFPQ training time ────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.gpu
def test_bench_faiss_ivfpq_train(benchmark):
    """Benchmark: train an IVFPQ index from 50K vectors.

    IVFPQ training runs two k-means passes:
      1. Coarse quantizer: nlist × n × d (same as IVFFlat)
      2. PQ encoder: m × (2^nbits) × (d/m) × n sub-space k-means

    This is more expensive than plain IVF training but only needs to
    be done once (or when rebuilding the index).
    """
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    dim = 512
    nlist = 100
    m = 64
    nbits = 8
    vecs = random_normalized_vectors(50_000, dim, seed=42)

    def _train():
        quantizer = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        idx.train(vecs)

    benchmark(_train)


# ── IVFPQ add throughput ───────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_ivfpq_add_batch_32(benchmark):
    """Benchmark: add 32 vectors to an empty IVFPQ index (after training).

    IVFPQ add() encodes each vector's residual via PQ, which is more
    expensive than a raw memcpy in IndexFlatL2/IndexIVFFlat.  This
    benchmark quantifies the insertion overhead.
    """
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    dim = 512
    # Train with a small set first so the PQ codebooks are ready
    train_vecs = random_normalized_vectors(500, dim, seed=42)
    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, 16, 64, 8)
    idx.train(train_vecs)

    vecs = random_normalized_vectors(32, dim, seed=99)

    def _add():
        idx.add(vecs)

    benchmark(_add)


@pytest.mark.scaling
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 1024])
def test_bench_faiss_ivfpq_add_batch_size(benchmark, batch_size):
    """Benchmark: add vectors to IVFPQ at various batch sizes.

    Creates a fresh (trained, empty) IVFPQ per call to avoid cumulative
    growth across benchmark iterations.
    """
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    dim = 512
    # Pre-train codebooks once — reused across iterations
    train_vecs = random_normalized_vectors(500, dim, seed=42)

    vecs = random_normalized_vectors(batch_size, dim, seed=99)

    def _add():
        quantizer = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIVFPQ(quantizer, dim, 16, 64, 8)
        idx.train(train_vecs)
        idx.add(vecs)

    benchmark(_add)


# ── IVFPQ batch query ──────────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("n_queries", [1, 10, 50])
def test_bench_faiss_ivfpq_multi_query(benchmark, bench_ivfpq_10k, n_queries):
    """Benchmark: IVFPQ batch query with multiple vectors."""
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    idx = bench_ivfpq_10k
    queries = random_normalized_vectors(n_queries, 512, seed=42)

    def _search_batch():
        return idx.search(queries, 50)

    benchmark(_search_batch)


# ── IVFPQ memory footprint ────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_ivfpq_memory_footprint(benchmark):
    """Benchmark: measure IVFPQ memory savings vs Flat/IVF at 10K scale.

    Records theoretical and actual serialised sizes in extra_info
    for comparison:
      - Flat/IVF:  N × d × 4 bytes (raw vectors)
      - IVFPQ:     N × m × nbits/8 bytes (compressed codes)
    """
    import tempfile
    from pathlib import Path

    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    dim = 512
    n = 10_000
    vecs = random_normalized_vectors(n, dim, seed=42)

    # Build Flat index
    flat = faiss.IndexFlatL2(dim)
    flat.add(vecs)

    # Build IVFPQ index
    nlist = 40
    m = 64
    nbits = 8
    quantizer = faiss.IndexFlatL2(dim)
    ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    ivfpq.train(vecs)
    ivfpq.add(vecs)

    # Measure serialised sizes via temp files
    tmp = Path(tempfile.mkdtemp(prefix="twin_bench_mem_"))
    flat_path = str(tmp / "flat.faiss")
    pq_path = str(tmp / "ivfpq.faiss")

    faiss.write_index(flat, flat_path)
    flat_bytes = Path(flat_path).stat().st_size

    faiss.write_index(ivfpq, pq_path)
    pq_bytes = Path(pq_path).stat().st_size

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    # Theoretical sizes
    flat_theory = n * dim * 4  # N × 512 × 4
    pq_theory = n * m * nbits // 8  # N × 64 bytes

    benchmark.extra_info["flat_bytes"] = flat_bytes
    benchmark.extra_info["ivfpq_bytes"] = pq_bytes
    benchmark.extra_info["flat_theory_bytes"] = flat_theory
    benchmark.extra_info["pq_theory_bytes"] = pq_theory
    benchmark.extra_info["compression_ratio"] = round(flat_bytes / max(pq_bytes, 1), 1)

    # Benchmark the search to have a timing reference
    q = random_normalized_vectors(1, dim, seed=99)

    def _search():
        return ivfpq.search(q, 50)

    benchmark(_search)


# ── Cross-index comparison: Flat vs IVF vs IVFPQ vs HNSW at 10K ───────────────


def test_bench_faiss_index_comparison_10k(
    benchmark,
    bench_faiss_10k,
    bench_faiss_ivf_10k,
    bench_ivfpq_10k,
    bench_hnsw_10k,
    bench_query_vec,
):
    """Benchmark: compare all four index types at 10K scale.

    Runs search on each index type with the same query vector and
    records all timings in extra_info for side-by-side comparison.

    This is the single most useful benchmark for choosing an index type.
    """
    q = bench_query_vec

    timings: dict[str, float] = {}

    def _bench_one(idx, name: str, k: int = 50):
        import time

        # Warmup
        for _ in range(3):
            idx.search(q, k)
        # Timed
        t0 = time.perf_counter()
        idx.search(q, k)
        timings[name] = (time.perf_counter() - t0) * 1000  # ms

    _bench_one(bench_faiss_10k, "flat")
    _bench_one(bench_faiss_ivf_10k, "ivf")
    _bench_one(bench_ivfpq_10k, "ivfpq")
    _bench_one(bench_hnsw_10k, "hnsw")

    # Find fastest for relative comparison
    fastest = min(timings.values()) if timings else 1.0
    for name, t in timings.items():
        benchmark.extra_info[f"{name}_ms"] = round(t, 3)
        benchmark.extra_info[f"{name}_rel"] = round(t / fastest, 1)

    # Primary benchmark: IVFPQ search (the feature under test)
    idx = bench_ivfpq_10k

    def _search():
        return idx.search(q, 50)

    benchmark(_search)


# ═══════════════════════════════════════════════════════════════════════════════
# Recall (accuracy) benchmarks — measure how many true neighbours are recovered
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_recall_at_k(
    ground_truth_ids: np.ndarray,
    approx_ids: np.ndarray,
    k: int,
) -> float:
    """Compute recall@K: fraction of true top-K found by approximate search.

    ground_truth_ids: (n_queries, K) from IndexFlatL2
    approx_ids:       (n_queries, K) from approximate index
    Returns:          float in [0, 1]
    """
    total = 0
    for gt, ap in zip(ground_truth_ids, approx_ids):
        total += len(set(gt[:k]) & set(ap[:k]))
    return total / (len(ground_truth_ids) * k)


def _build_ground_truth_and_approx_indexes(
    n: int = 10_000,
    dim: int = 512,
    n_queries: int = 100,
    seed: int = 42,
) -> tuple[
    "np.ndarray",          # database vectors
    "np.ndarray",          # query vectors
    "np.ndarray",          # ground truth IDs (n_queries, K)
    "faiss.IndexFlatL2",   # Flat index (source of truth)
]:
    """Create database, queries, and ground-truth top-50 from IndexFlatL2."""
    from tests.benchmarks.fixtures.synthetic import random_normalized_vectors

    rng = np.random.RandomState(seed)
    db_vecs = random_normalized_vectors(n, dim, seed=rng.randint(0, 2**31))
    query_vecs = random_normalized_vectors(n_queries, dim, seed=rng.randint(0, 2**31))

    # Ground truth via exhaustive Flat search
    flat = faiss.IndexFlatL2(dim)
    flat.add(db_vecs)
    _, gt_ids = flat.search(query_vecs, 50)

    return db_vecs, query_vecs, gt_ids, flat


# ── IVFPQ recall vs nprobe sweep ──────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("nprobe", [1, 2, 4, 8, 16, 32])
def test_bench_faiss_ivfpq_recall_sweep(benchmark, nprobe):
    """Benchmark: IVFPQ recall@50 vs nprobe at 10K scale.

    Higher nprobe = more clusters searched = better recall but slower.
    Records both recall and latency in extra_info for Pareto analysis.
    """
    dim = 512
    nlist = 40
    m = 64
    nbits = 8
    db, queries, gt_ids, _ = _build_ground_truth_and_approx_indexes(10_000, dim)

    # Build and train IVFPQ
    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    idx.nprobe = nprobe
    idx.train(db)
    idx.add(db)

    # Timed search
    import time

    t0 = time.perf_counter()
    _, ap_ids = idx.search(queries, 50)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    recall = _compute_recall_at_k(gt_ids, ap_ids, 50)
    per_query_ms = elapsed_ms / len(queries)

    benchmark.extra_info["recall_at_50"] = round(recall, 4)
    benchmark.extra_info["per_query_us"] = round(per_query_ms * 1000, 1)
    benchmark.extra_info["nprobe"] = nprobe

    def _search():
        return idx.search(queries[:1], 50)

    benchmark(_search)


# ── IVF recall vs nprobe sweep ────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("nprobe", [1, 2, 4, 8, 16, 32])
def test_bench_faiss_ivf_recall_sweep(benchmark, nprobe):
    """Benchmark: IVFFlat recall@50 vs nprobe at 10K scale."""
    dim = 512
    nlist = 40
    db, queries, gt_ids, _ = _build_ground_truth_and_approx_indexes(10_000, dim)

    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    idx.nprobe = nprobe
    idx.train(db)
    idx.add(db)

    import time

    t0 = time.perf_counter()
    _, ap_ids = idx.search(queries, 50)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    recall = _compute_recall_at_k(gt_ids, ap_ids, 50)
    per_query_ms = elapsed_ms / len(queries)

    benchmark.extra_info["recall_at_50"] = round(recall, 4)
    benchmark.extra_info["per_query_us"] = round(per_query_ms * 1000, 1)
    benchmark.extra_info["nprobe"] = nprobe

    def _search():
        return idx.search(queries[:1], 50)

    benchmark(_search)


# ── HNSW recall vs efSearch sweep ─────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("ef_search", [16, 32, 64, 128, 256])
def test_bench_faiss_hnsw_recall_sweep(benchmark, ef_search):
    """Benchmark: HNSW recall@50 vs efSearch at 10K scale."""
    dim = 512
    m = 32
    db, queries, gt_ids, _ = _build_ground_truth_and_approx_indexes(10_000, dim)

    idx = faiss.IndexHNSWFlat(dim, m)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = ef_search
    idx.add(db)

    import time

    t0 = time.perf_counter()
    _, ap_ids = idx.search(queries, 50)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    recall = _compute_recall_at_k(gt_ids, ap_ids, 50)
    per_query_ms = elapsed_ms / len(queries)

    benchmark.extra_info["recall_at_50"] = round(recall, 4)
    benchmark.extra_info["per_query_us"] = round(per_query_ms * 1000, 1)
    benchmark.extra_info["ef_search"] = ef_search

    def _search():
        return idx.search(queries[:1], 50)

    benchmark(_search)


# ── Cross-index recall at default parameters ──────────────────────────────────


@pytest.mark.smoke
def test_bench_faiss_recall_comparison(benchmark):
    """Benchmark: recall@50 for all four index types at default settings (10K).

    Records recall + per-query latency for Flat, IVF, IVFPQ, HNSW in
    extra_info.  Flat is the ground-truth baseline (recall = 1.0).
    """
    dim = 512
    db, queries, gt_ids, flat = _build_ground_truth_and_approx_indexes(10_000)

    import time

    results: dict[str, dict] = {}

    # Flat (ground truth)
    t0 = time.perf_counter()
    _, ap_ids = flat.search(queries, 50)
    flat_ms = (time.perf_counter() - t0) * 1000
    results["flat"] = {
        "recall": 1.0,
        "per_query_us": round(flat_ms / len(queries) * 1000, 1),
    }

    # IVFFlat (nprobe=8)
    nlist = 40
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    ivf.nprobe = 8
    ivf.train(db)
    ivf.add(db)
    t0 = time.perf_counter()
    _, ap_ids = ivf.search(queries, 50)
    ivf_ms = (time.perf_counter() - t0) * 1000
    results["ivf"] = {
        "recall": round(_compute_recall_at_k(gt_ids, ap_ids, 50), 4),
        "per_query_us": round(ivf_ms / len(queries) * 1000, 1),
    }

    # IVFPQ (nprobe=8, M=64, nbits=8)
    quantizer2 = faiss.IndexFlatL2(dim)
    ivfpq = faiss.IndexIVFPQ(quantizer2, dim, nlist, 64, 8)
    ivfpq.nprobe = 8
    ivfpq.train(db)
    ivfpq.add(db)
    t0 = time.perf_counter()
    _, ap_ids = ivfpq.search(queries, 50)
    pq_ms = (time.perf_counter() - t0) * 1000
    results["ivfpq"] = {
        "recall": round(_compute_recall_at_k(gt_ids, ap_ids, 50), 4),
        "per_query_us": round(pq_ms / len(queries) * 1000, 1),
    }

    # HNSW (M=32, efSearch=64)
    hnsw = faiss.IndexHNSWFlat(dim, 32)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    hnsw.add(db)
    t0 = time.perf_counter()
    _, ap_ids = hnsw.search(queries, 50)
    hnsw_ms = (time.perf_counter() - t0) * 1000
    results["hnsw"] = {
        "recall": round(_compute_recall_at_k(gt_ids, ap_ids, 50), 4),
        "per_query_us": round(hnsw_ms / len(queries) * 1000, 1),
    }

    for name, r in results.items():
        benchmark.extra_info[f"{name}_recall"] = r["recall"]
        benchmark.extra_info[f"{name}_us"] = r["per_query_us"]

    # Primary benchmark: IVFPQ
    def _search():
        return ivfpq.search(queries[:1], 50)

    benchmark(_search)


# ── IVFPQ recall vs compression (M / nbits sweep) ─────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("pq_config", [
    (64, 8),   # default: 64 bytes/vector, 256 centroids/sub-space
    (32, 8),   # 32 bytes/vector
    (16, 8),   # 16 bytes/vector (aggressive)
    (64, 6),   # 48 bytes/vector, 64 centroids/sub-space
    (32, 6),   # 24 bytes/vector
])
def test_bench_faiss_ivfpq_compression_recall(benchmark, pq_config):
    """Benchmark: recall@50 vs PQ compression level at 10K.

    Sweeps (M, nbits) combos to show the recall–compression trade-off:
      - Higher M → more sub-quantizers → better recall, more bytes
      - Higher nbits → finer centroids → better recall, more bytes
    """
    m, nbits = pq_config
    dim = 512
    nlist = 40
    db, queries, gt_ids, _ = _build_ground_truth_and_approx_indexes(10_000, dim)

    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    idx.nprobe = 8
    idx.train(db)
    idx.add(db)

    import time

    t0 = time.perf_counter()
    _, ap_ids = idx.search(queries, 50)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    recall = _compute_recall_at_k(gt_ids, ap_ids, 50)
    bytes_per_vec = m * nbits // 8

    benchmark.extra_info["recall_at_50"] = round(recall, 4)
    benchmark.extra_info["per_query_us"] = round(elapsed_ms / len(queries) * 1000, 1)
    benchmark.extra_info["pq_m"] = m
    benchmark.extra_info["pq_nbits"] = nbits
    benchmark.extra_info["bytes_per_vec"] = bytes_per_vec

    def _search():
        return idx.search(queries[:1], 50)

    benchmark(_search)
