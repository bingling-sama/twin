"""Two-stage search with 4-stage funnel verification.

Architecture:
  Stage 1 (Recall):    CLIP embedding → Faiss L2 top-K semantic candidates.
  Stage 2 (Filter 1):  dHash Hamming distance — cheapest, gradient-based.
  Stage 3 (Filter 2):  pHash Hamming distance — DCT-based, complementary.
  Stage 4 (Filter 3):  SSIM structural similarity — most expensive, last resort.

Each filter eliminates non-matches.  Filters only run on survivors.
Results are tiered:
  - "confirmed": passed all 4 filters (Faiss → dHash → pHash → SSIM)
  - "suspected": passed Faiss + at least one hash filter
  - "none":      only passed Faiss (semantically similar, not visually duplicate)
"""

import logging
import time
from pathlib import Path

from PIL import Image

from twin.core.config import settings
from twin.services.embedding import compute_embedding
from twin.services.hasher import (
    compute_dhash,
    compute_phash,
    compute_rotated_dhashes,
    compute_rotated_phashes,
    compute_ssim,
    hamming_distance,
    rotate_image,
)
from twin.services.indexer import indexer
from twin.utils.image import load_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def _passes_hash(query_hash: str, candidate_hash: str, threshold: int) -> tuple[bool, int]:
    """Check a perceptual hash against threshold. Returns (passed, distance)."""
    if not query_hash or not candidate_hash:
        return False, 999
    dist = hamming_distance(query_hash, candidate_hash)
    return dist <= threshold, dist


def _passes_ssim(query_img: Image.Image, cand_path: str, threshold: float) -> tuple[bool, float]:
    if not cand_path or not Path(cand_path).exists():
        return False, 0.0
    cand_img = load_image(cand_path)
    if cand_img is None:
        return False, 0.0
    score = compute_ssim(query_img, cand_img)
    return score >= threshold, score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _empty(t0: float, stages: dict) -> dict:
    """Return an empty-but-well-formed search response dict."""
    elapsed = time.perf_counter() - t0
    for s in stages.values():
        s.setdefault("elapsed_ms", 0)
    return {
        "results": [],
        "count": 0,
        "query_time_ms": round(elapsed * 1000, 3),
        "stages": stages,
    }


def _assign_match_level(stages_passed: int) -> str:
    """Tier based on funnel progression."""
    if stages_passed >= 3:
        return "confirmed"
    elif stages_passed >= 1:
        return "suspected"
    return "none"


def _final_sort(results: list[dict]) -> None:
    """In-place sort: most stages passed first, then dHash ASC, then L2 ASC."""
    results.sort(
        key=lambda r: (
            -r["stages_passed"],
            r["dhash_distance"],
            r["distance"],
        )
    )


def _build_response(results: list[dict], stages: dict, t0: float) -> dict:
    """Clean up internal keys and return API-ready dict."""
    elapsed = time.perf_counter() - t0
    output = [
        {
            "id": c["id"],
            "filename": c["filename"],
            "distance": c["distance"],
            "match_level": _assign_match_level(c["stages_passed"]),
            "stages_passed": c["stages_passed"],
            "dhash_distance": c["dhash_distance"],
            "phash_distance": c["phash_distance"],
            "ssim_score": c["ssim_score"],
            "dhash_hex": c["meta"].get("dhash", ""),
            "phash_hex": c["meta"].get("phash", ""),
            "path": c["meta"].get("path", ""),
        }
        for c in results
    ]
    return {
        "results": output,
        "count": len(output),
        "query_time_ms": round(elapsed * 1000, 3),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------
def search(
    image: Image.Image,
    top_k: int | None = None,
    dhash_threshold: int | None = None,
    phash_threshold: int | None = None,
    ssim_threshold: float | None = None,
    rotation_invariant: bool | None = None,
) -> dict:
    """
    Two-stage image search with funnel-based filtering.

    Stage 1: CLIP/DINOv2 → Faiss top-K semantic candidates.
    Stage 2–4: Sequential filters (dHash → pHash → SSIM), each operating
    only on survivors of the previous stage.
    Supports rotation tolerance in Stage 2 (0°, 90°, 180°, 270°).

    Returns list of result dicts sorted by match_level.
    """
    if top_k is None:
        top_k = settings.top_k
    if dhash_threshold is None:
        dhash_threshold = settings.dhash_threshold
    if phash_threshold is None:
        phash_threshold = settings.phash_threshold
    if ssim_threshold is None:
        ssim_threshold = settings.ssim_threshold
    if rotation_invariant is None:
        rotation_invariant = settings.rotation_invariant

    t0 = time.perf_counter()
    stages: dict[str, dict] = {}  # track per-stage stats + timing

    # ==================================================================
    # Stage 1: Semantic retrieval (Faiss top-K)
    # ==================================================================
    query_vec = compute_embedding(image)
    query_dhash = compute_dhash(image)
    query_phash = compute_phash(image)

    distances, indices = indexer.search(query_vec, k=top_k)
    t1 = time.perf_counter()
    stages["faiss"] = {"in": 0, "out": len(indices), "elapsed_ms": round((t1 - t0) * 1000, 3)}

    if not indices:
        return _empty(t0, stages)

    # Build candidate pool
    candidates: list[dict] = []
    for dist, idx in zip(distances, indices):
        meta = indexer.get_metadata(idx)
        if meta is None:
            continue
        candidates.append(
            {
                "id": idx,
                "filename": meta.get("filename", "unknown"),
                "distance": round(float(dist), 6),
                "meta": meta,
                # metrics filled in as we go
                "dhash_distance": 999,
                "phash_distance": 999,
                "ssim_score": 0.0,
                "stages_passed": 0,
            }
        )

    if not candidates:
        return _empty(t0, stages)

    # ==================================================================
    # Stage 2: dHash filter (cheapest, runs on all Faiss survivors)
    # ==================================================================
    query_dhashes = compute_rotated_dhashes(image) if rotation_invariant else [query_dhash]

    dhash_survivors = []
    for c in candidates:
        cand_dhash = c["meta"].get("dhash", "")
        if not cand_dhash:
            c["dhash_distance"] = 999
            c["_candidate_rotations"] = [0]
            c["_best_rotation"] = 0
            continue

        if rotation_invariant:
            distances_per_angle = [hamming_distance(qh, cand_dhash) for qh in query_dhashes]
            min_dist = min(distances_per_angle)
            c["dhash_distance"] = min_dist
            if min_dist <= dhash_threshold:
                c["_candidate_rotations"] = [
                    i for i, d in enumerate(distances_per_angle) if d <= dhash_threshold
                ]
                c["_best_rotation"] = distances_per_angle.index(min_dist)
                c["stages_passed"] += 1
                dhash_survivors.append(c)
        else:
            dist = hamming_distance(query_dhash, cand_dhash)
            c["dhash_distance"] = dist
            c["_candidate_rotations"] = [0]
            c["_best_rotation"] = 0
            if dist <= dhash_threshold:
                c["stages_passed"] += 1
                dhash_survivors.append(c)
    t2 = time.perf_counter()
    stages["dhash"] = {
        "in": len(candidates),
        "out": len(dhash_survivors),
        "elapsed_ms": round((t2 - t1) * 1000, 3),
    }

    if not dhash_survivors:
        # No dHash survivors — everything goes to "none" tier
        _final_sort(candidates)
        logger.info("Search: %d candidates, 0 passed dHash", len(candidates))
        return _build_response(candidates, stages, t0)

    # ==================================================================
    # Stage 3: pHash filter (runs on dHash survivors)
    # ==================================================================
    query_phashes = compute_rotated_phashes(image) if rotation_invariant else [query_phash]

    phash_survivors = []
    for c in dhash_survivors:
        cand_phash = c["meta"].get("phash", "")
        if not cand_phash:
            c["phash_distance"] = 999
            continue

        if rotation_invariant:
            candidate_rots = c.get("_candidate_rotations", [0])
            phash_evals = [
                (rot_i, hamming_distance(query_phashes[rot_i], cand_phash))
                for rot_i in candidate_rots
            ]
            best_rot, best_p_dist = min(phash_evals, key=lambda x: x[1])
            c["phash_distance"] = best_p_dist
            c["_best_rotation"] = best_rot  # pHash disambiguates the exact rotation angle
            if best_p_dist <= phash_threshold:
                c["stages_passed"] += 1
                phash_survivors.append(c)
        else:
            ok, dist = _passes_hash(query_phash, cand_phash, phash_threshold)
            c["phash_distance"] = dist
            if ok:
                c["stages_passed"] += 1
                phash_survivors.append(c)
    t3 = time.perf_counter()
    stages["phash"] = {
        "in": len(dhash_survivors),
        "out": len(phash_survivors),
        "elapsed_ms": round((t3 - t2) * 1000, 3),
    }

    if not phash_survivors:
        # dHash passed but pHash failed — "suspected" tier
        _final_sort(candidates)
        logger.info("Search: %d → %d dHash → 0 pHash", len(candidates), len(dhash_survivors))
        return _build_response(candidates, stages, t0)

    # ==================================================================
    # Stage 4: SSIM filter (most expensive, runs on pHash survivors, parallel)
    # ==================================================================
    def _ssim_check(c: dict) -> dict:
        cpath = c["meta"].get("path", "")
        rot_idx = c.get("_best_rotation", 0)
        query_candidate_img = (
            rotate_image(image, rot_idx) if (rotation_invariant and rot_idx) else image
        )
        ok, score = _passes_ssim(query_candidate_img, cpath, ssim_threshold)
        c["ssim_score"] = round(score, 4)
        if ok:
            c["stages_passed"] += 1
        return c

    # SSIM on usually 2–5 candidates — simple loop, overhead of pool isn't worth it
    checked = [_ssim_check(c) for c in phash_survivors]

    confirmed = [c for c in checked if c["stages_passed"] >= 3]  # Faiss+dHash+pHash+SSIM
    t4 = time.perf_counter()
    stages["ssim"] = {
        "in": len(phash_survivors),
        "out": len(confirmed),
        "elapsed_ms": round((t4 - t3) * 1000, 3),
    }

    # Merge survivors back into full candidate list for ranking
    finalized = {c["id"]: c for c in candidates}
    for c in checked:
        finalized[c["id"]] = c
    final_list = list(finalized.values())

    _final_sort(final_list)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Search in %.0fms: %d candidates → %d dHash → %d pHash → %d confirmed",
        elapsed * 1000,
        len(candidates),
        len(dhash_survivors),
        len(phash_survivors),
        len(confirmed),
    )

    return _build_response(final_list, stages, t0)


# ---------------------------------------------------------------------------
# Text Search (Multi-modal)
# ---------------------------------------------------------------------------
def search_by_text(
    query: str,
    top_k: int | None = None,
) -> dict:
    """
    Search indexed images using a natural language text query via CLIP.

    Returns candidate images ranked by vector L2 distance / cosine similarity.
    """
    from twin.services.embedding import compute_text_embedding

    if top_k is None:
        top_k = settings.top_k

    t0 = time.perf_counter()
    query_vec = compute_text_embedding(query)

    distances, indices = indexer.search(query_vec, k=top_k)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    if not indices:
        return {
            "results": [],
            "count": 0,
            "query_time_ms": round(elapsed * 1000, 3),
            "query": query,
        }

    results = []
    for dist, idx in zip(distances, indices):
        meta = indexer.get_metadata(idx)
        if meta is None:
            continue
        results.append(
            {
                "id": idx,
                "filename": meta.get("filename", "unknown"),
                "distance": round(float(dist), 6),
                "path": meta.get("path", ""),
                "dhash_hex": meta.get("dhash", ""),
                "phash_hex": meta.get("phash", ""),
            }
        )

    return {
        "results": results,
        "count": len(results),
        "query_time_ms": round(elapsed * 1000, 3),
        "query": query,
    }
