"""Benchmarks for the full 4-stage search pipeline.

This is the most important benchmark file — it measures the end-to-end
search flow from CLIP encoding through to tiered results.

The search pipeline:
  1. CLIP encode → 512-dim embedding
  2. Faiss L2 search → top-K candidates
  3. dHash filter (Hamming ≤ threshold)
  4. pHash filter (Hamming ≤ threshold)
  5. SSIM filter (≥ threshold)
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_search_index(n: int, _clip_model) -> None:
    """Populate the indexer singleton with N synthetic images.

    Each image gets a CLIP embedding, dHash, and pHash — so the search
    pipeline exercises all stages.
    """
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.embedding import compute_embedding
    from twin.services.hasher import compute_dhash, compute_phash
    from twin.services.indexer import indexer

    indexer.clear()

    for i in range(n):
        img = random_image((224, 224), seed=42 + i)
        vec = compute_embedding(img)
        dhash = compute_dhash(img)
        phash = compute_phash(img)
        indexer.add_item(
            vec,
            {
                "filename": f"bench_{i:06d}.png",
                "path": f"/tmp/bench/bench_{i:06d}.png",
                "dhash": dhash,
                "phash": phash,
            },
        )


# ── End-to-end search ─────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_search_end_to_end(benchmark, clip_model):
    """Benchmark: full search() pipeline on a small (100 image) index."""
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.indexer import indexer
    from twin.services.search import search

    _build_search_index(100, clip_model)
    query = random_image((224, 224), seed=999)

    benchmark(search, query)

    indexer.clear()


@pytest.mark.scaling
@pytest.mark.parametrize("index_size", [100, 500, 1000])
def test_bench_search_scaling(benchmark, clip_model, index_size):
    """Benchmark: search latency vs index size."""
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.indexer import indexer
    from twin.services.search import search

    _build_search_index(index_size, clip_model)
    query = random_image((224, 224), seed=999)

    benchmark(search, query)

    indexer.clear()


# ── Per-stage timing breakdown ────────────────────────────────────────────────


@pytest.mark.smoke
def test_bench_search_stage_breakdown(benchmark, clip_model):
    """Benchmark: search with manual per-stage timing.

    Uses benchmark.pedantic to record per-stage breakdown in the
    benchmark extras dictionary for later analysis.
    """
    from tests.benchmarks.fixtures.synthetic import near_duplicate_pair, random_image
    from twin.services.embedding import compute_embedding
    from twin.services.hasher import compute_dhash, compute_phash
    from twin.services.indexer import indexer
    from twin.services.search import search

    # Build index with one near-duplicate pair to ensure all stages fire
    indexer.clear()
    base, dup = near_duplicate_pair((224, 224), noise_level=5.0, seed=42)

    # Index 50 random + 1 near-duplicate
    for i in range(50):
        img = random_image((224, 224), seed=100 + i)
        vec = compute_embedding(img)
        dhash = compute_dhash(img)
        phash = compute_phash(img)
        indexer.add_item(
            vec,
            {
                "filename": f"b_{i:04d}.png",
                "path": f"/tmp/b_{i:04d}.png",
                "dhash": dhash,
                "phash": phash,
            },
        )

    # Index the near-duplicate's base
    vec_base = compute_embedding(base)
    dhash_base = compute_dhash(base)
    phash_base = compute_phash(base)
    indexer.add_item(
        vec_base,
        {
            "filename": "base.png",
            "path": "/tmp/base.png",
            "dhash": dhash_base,
            "phash": phash_base,
        },
    )

    # Search with the near-duplicate — should pass all 4 stages
    result = search(dup)
    stages = result.get("stages", {})

    # Record per-stage timing as benchmark extras
    benchmark.extra_info["stages"] = stages
    benchmark.extra_info["total_ms"] = result.get("query_time_ms", 0)
    benchmark.extra_info["count_confirmed"] = sum(
        1 for r in result.get("results", []) if r.get("match_level") == "confirmed"
    )

    # Benchmark the search itself
    def _search():
        return search(dup)

    benchmark(_search)

    indexer.clear()


# ── Threshold sweep ───────────────────────────────────────────────────────────


@pytest.mark.scaling
@pytest.mark.parametrize("dhash_threshold", [6, 10, 14, 18])
def test_bench_search_dhash_threshold(benchmark, clip_model, dhash_threshold):
    """Benchmark: how dHash threshold affects survivor count and latency.

    Lower threshold = stricter filter = fewer survivors = faster SSIM stage.
    """
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.indexer import indexer
    from twin.services.search import search

    _build_search_index(200, clip_model)
    query = random_image((224, 224), seed=777)

    def _search():
        return search(query, dhash_threshold=dhash_threshold)

    result = benchmark(_search)
    benchmark.extra_info["dhash_survivors"] = result.get("stages", {}).get("dhash", {}).get("out", 0)

    indexer.clear()


@pytest.mark.scaling
@pytest.mark.parametrize("phash_threshold", [8, 12, 16, 20])
def test_bench_search_phash_threshold(benchmark, clip_model, phash_threshold):
    """Benchmark: how pHash threshold affects survivor count and latency."""
    from tests.benchmarks.fixtures.synthetic import random_image
    from twin.services.indexer import indexer
    from twin.services.search import search

    _build_search_index(200, clip_model)
    query = random_image((224, 224), seed=777)

    def _search():
        return search(query, phash_threshold=phash_threshold)

    result = benchmark(_search)
    benchmark.extra_info["phash_survivors"] = result.get("stages", {}).get("phash", {}).get("out", 0)

    indexer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline Recall: how Faiss recall impacts confirmed matches
# ═══════════════════════════════════════════════════════════════════════════════


def _build_pipeline_recall_index(
    n_random: int,
    pair_count: int,
    noise_level: float = 5.0,
    seed: int = 42,
    tmp_dir: str = "/tmp/twin_bench_pipeline",
) -> tuple[list, list]:
    """Build an index with N random images + K near-duplicate base images.

    Images are saved to tmp_dir so the SSIM stage can load them from disk.

    Returns (query_images, base_filenames) for pipeline recall measurement.
    """
    import shutil
    from pathlib import Path

    from tests.benchmarks.fixtures.synthetic import near_duplicate_pair, random_image
    from twin.services.embedding import compute_embedding
    from twin.services.hasher import compute_dhash, compute_phash
    from twin.services.indexer import indexer

    indexer.clear()

    base = Path(tmp_dir)
    base.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    queries = []
    base_names = []

    # Index random filler images (save to disk for SSIM)
    for i in range(n_random):
        img = random_image((224, 224), seed=rng.randint(0, 2**31))
        fname = f"rand_{i:05d}.png"
        fpath = base / fname
        img.save(fpath)
        vec = compute_embedding(img)
        dhash = compute_dhash(img)
        phash = compute_phash(img)
        indexer.add_item(
            vec,
            {
                "filename": fname,
                "path": str(fpath),
                "dhash": dhash,
                "phash": phash,
            },
        )

    # Index near-duplicate base images, save queries
    for i in range(pair_count):
        base_img, dup = near_duplicate_pair((224, 224), noise_level=noise_level,
                                            seed=rng.randint(0, 2**31))
        fname = f"base_{i:05d}.png"
        fpath = base / fname
        base_img.save(fpath)
        vec_base = compute_embedding(base_img)
        dhash_base = compute_dhash(base_img)
        phash_base = compute_phash(base_img)
        indexer.add_item(
            vec_base,
            {
                "filename": fname,
                "path": str(fpath),
                "dhash": dhash_base,
                "phash": phash_base,
            },
        )
        queries.append(dup)
        base_names.append(fname)

    return queries, base_names


@pytest.mark.smoke
def test_bench_pipeline_recall_flat(benchmark, clip_model):
    """End-to-end pipeline recall with IndexFlatL2 (exact Faiss, upper bound).

    Measures: of K near-duplicate queries, how many result in a
    'confirmed' match through the full CLIP → Faiss → dHash → pHash → SSIM
    pipeline?  Flat = ground truth for maximum achievable pipeline recall.
    """
    from twin.services.indexer import indexer
    from twin.services.search import search

    n_random = 98   # 98 random + 2 base = 100 total
    n_pairs = 2
    queries, base_names = _build_pipeline_recall_index(n_random, n_pairs)

    confirmed = 0
    for query, expected_name in zip(queries, base_names):
        result = search(query)
        for r in result.get("results", []):
            if r.get("match_level") == "confirmed" and r.get("filename") == expected_name:
                confirmed += 1
                break

    benchmark.extra_info["pipeline_recall"] = round(confirmed / n_pairs, 3) if n_pairs else 0
    benchmark.extra_info["pairs"] = n_pairs
    benchmark.extra_info["faiss_type"] = indexer.index_type_name

    def _search():
        return search(queries[0])

    benchmark(_search)

    indexer.clear()


@pytest.mark.scaling
@pytest.mark.parametrize("index_config", [
    # (type, parameter, pq_cfg)
    ("flat",       None, None),
    ("ivf_flat",   8,    None),
    ("ivf_flat",   16,   None),
    ("ivf_flat",   32,   None),
    ("ivf_pq",     8,    (64, 8)),
    ("ivf_pq",     16,   (64, 8)),
    ("ivf_pq",     32,   (64, 8)),
    ("hnsw",       64,   None),
    ("hnsw",       128,  None),
])
def test_bench_pipeline_recall_sweep(benchmark, clip_model, index_config):
    """Pipeline recall vs Faiss index type and parameters.

    Builds a 500-image index with 5 near-duplicate pairs, runs the full
    search pipeline for each query, and measures what fraction result in
    'confirmed' matches.

    This is the definitive metric: not abstract Faiss recall@50, but actual
    end-to-end confirmed-match rate through all 4 pipeline stages.

    Uses nlist=4 for IVF-based indexes (needs >= 156 training vectors;
    500 satisfies this).  For IVFPQ with small datasets, nbits is
    automatically reduced to ensure training stability.
    """
    import faiss

    from twin.services.embedding import compute_embedding
    from twin.services.hasher import compute_dhash, compute_phash
    from twin.services.indexer import indexer
    from twin.services.search import search

    idx_type, nprobe_or_ef, pq_cfg = index_config
    n_random = 495   # 495 random + 5 base = 500 total — enough for IVF training
    n_pairs = 5
    fixed_nlist = 4  # small nlist → training needs 4*39=156 vectors, 500 is plenty

    # ── Build index manually with desired Faiss type ──
    import shutil
    import tempfile
    from pathlib import Path

    indexer.clear()

    from tests.benchmarks.fixtures.synthetic import near_duplicate_pair, random_image

    tmp_dir = Path(tempfile.mkdtemp(prefix="twin_bench_pipeline_sweep_"))

    rng = np.random.RandomState(42)

    # Collect all images first, save to disk for SSIM
    all_images = []
    all_filenames = []
    all_filepaths = []
    all_labels = []  # "random" or "base_N"

    for i in range(n_random):
        img = random_image((224, 224), seed=rng.randint(0, 2**31))
        fname = f"rand_{i:05d}.png"
        fpath = tmp_dir / fname
        img.save(fpath)
        all_images.append(img)
        all_filenames.append(fname)
        all_filepaths.append(fpath)
        all_labels.append(None)

    queries = []
    base_names = []
    for i in range(n_pairs):
        base_img, dup = near_duplicate_pair((224, 224), noise_level=5.0,
                                            seed=rng.randint(0, 2**31))
        fname = f"base_{i:05d}.png"
        fpath = tmp_dir / fname
        base_img.save(fpath)
        all_images.append(base_img)
        all_filenames.append(fname)
        all_filepaths.append(fpath)
        all_labels.append(fname)
        queries.append(dup)
        base_names.append(fname)

    # Compute CLIP embeddings
    embeddings = np.stack(
        [compute_embedding(img) for img in all_images], axis=0
    ).astype(np.float32)

    # Compute hashes
    dhashes = [compute_dhash(img) for img in all_images]
    phashes = [compute_phash(img) for img in all_images]

    # ── Replace indexer's internal index with the desired type ──
    dim = 512
    n = len(all_images)

    if idx_type == "flat":
        faiss_idx = faiss.IndexFlatL2(dim)
        faiss_idx.add(embeddings)
    elif idx_type == "ivf_flat":
        nlist = fixed_nlist
        quantizer = faiss.IndexFlatL2(dim)
        faiss_idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        faiss_idx.nprobe = nprobe_or_ef
        faiss_idx.train(embeddings)
        faiss_idx.add(embeddings)
    elif idx_type == "ivf_pq":
        nlist = fixed_nlist
        quantizer = faiss.IndexFlatL2(dim)
        m, nbits = pq_cfg
        # For small datasets, auto-reduce nbits for training stability
        # (need n >= 2^nbits for k-means; 500 vectors → nbits ≤ 8 is fine)
        actual_nbits = nbits
        if n < (1 << nbits):
            actual_nbits = max(4, int(np.log2(n)))
            benchmark.extra_info["pq_nbits_adjusted"] = actual_nbits
        faiss_idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, actual_nbits)
        faiss_idx.nprobe = nprobe_or_ef
        faiss_idx.train(embeddings)
        faiss_idx.add(embeddings)
    elif idx_type == "hnsw":
        faiss_idx = faiss.IndexHNSWFlat(dim, 32)
        faiss_idx.hnsw.efConstruction = 200
        faiss_idx.hnsw.efSearch = nprobe_or_ef
        faiss_idx.add(embeddings)
    else:
        raise ValueError(f"Unknown index type: {idx_type}")

    # Replace the indexer's internal index and metadata
    with indexer._lock:
        indexer._index = faiss_idx
        indexer._metadata = [
            {
                "id": i,
                "filename": all_filenames[i],
                "path": str(all_filepaths[i]),
                "dhash": dhashes[i],
                "phash": phashes[i],
            }
            for i in range(n)
        ]

    # ── Run pipeline for each query ──
    confirmed = 0
    faiss_hits = 0
    stage_survivors = {"faiss": 0, "dhash": 0, "phash": 0, "ssim": 0}

    for query, expected_name in zip(queries, base_names):
        result = search(query)
        found_in_faiss = False
        for r in result.get("results", []):
            if r.get("filename") == expected_name:
                found_in_faiss = True
                if r.get("match_level") == "confirmed":
                    confirmed += 1
                break
        if found_in_faiss:
            faiss_hits += 1

        # Track per-stage survivors
        stages = result.get("stages", {})
        for sname in ["dhash", "phash", "ssim"]:
            stage_survivors[sname] += stages.get(sname, {}).get("out", 0)

    pipeline_recall = confirmed / n_pairs if n_pairs else 0
    faiss_recall = faiss_hits / n_pairs if n_pairs else 0

    benchmark.extra_info["pipeline_recall"] = round(pipeline_recall, 3)
    benchmark.extra_info["faiss_recall"] = round(faiss_recall, 3)
    benchmark.extra_info["faiss_type"] = idx_type
    benchmark.extra_info["parameter"] = nprobe_or_ef
    benchmark.extra_info["pairs"] = n_pairs
    benchmark.extra_info["n_total"] = n

    def _search():
        return search(queries[0])

    benchmark(_search)

    indexer.clear()
    shutil.rmtree(tmp_dir, ignore_errors=True)
