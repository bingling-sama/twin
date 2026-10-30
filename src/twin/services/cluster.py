"""Seedless Similar Texture Discovery via Greedy Leader Elimination (Phase 1).

Discovers batches of mutually similar or duplicate Minecraft textures across
indexed assets without requiring an input query image.

Core Algorithm: Greedy Leader Elimination
1. Maintains an unvisited pool of indexed asset IDs.
2. Selects pivot seed (supports random, sequential, or biased selection).
3. Recalls nearest neighbors via Faiss binary perceptual hash index (IndexBinaryFlat).
4. Verifies candidate similarity against seed using active Verifier (DiffMaskVerifier).
5. If cluster size >= 2 (seed + >= 1 verified neighbors):
   - Constructs a SimilarBatch.
   - Eliminates all cluster members from unvisited pool (prevents transitivity collapse).
6. Repeats until requested max_batches reached or unvisited pool / max_probes exhausted.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image

from twin.core.config import settings
from twin.services.hasher import hamming_distance
from twin.services.indexer import indexer
from twin.services.verifiers import VerifyResult, get_verifier
from twin.utils.image import load_image

logger = logging.getLogger(__name__)


def discover_similar_batches(
    datasource: str = "skin",
    max_batches: int = 5,
    batch_size: int = 5,
    mode: str = "confirmed",  # "confirmed" | "variants"
    max_probes: int = 200,
    seed_strategy: str = "random",  # "random" | "sequential"
    verifier_type: str | None = None,
    images_map: dict[str, Image.Image] | None = None,
) -> dict[str, Any]:
    """Discover batches of similar textures without an input seed image.

    Args:
        datasource: Datasource partition ("skin" or "cape").
        max_batches: Maximum number of batches to return.
        batch_size: Maximum total size of each batch (seed + peers).
        mode: "confirmed" (strict duplicates, diff_ratio <= 0.05) or
              "variants" (recolors/edits, Hamming <= 4, diff_ratio <= 0.20).
        max_probes: Maximum number of seed exploration attempts.
        seed_strategy: "random" or "sequential".
        verifier_type: Override verifier ("diffmask" or "ssim").
        images_map: Optional in-memory mapping from filename to PIL Image
            (useful for testing/benchmarks).

    Returns:
        Dictionary containing discovered batches, count, probes, and performance metrics.
    """
    t0 = time.perf_counter()
    total_items = indexer.get_count(datasource=datasource)

    if total_items < 2:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "batches": [],
            "count": 0,
            "total_probes": 0,
            "query_time_ms": elapsed_ms,
            "datasource": datasource,
            "mode": mode,
        }

    # Threshold calibration based on mode
    if mode == "confirmed":
        max_hamming = 2
        max_diff_ratio = settings.diffmask_diff_ratio_threshold
        target_match_level = "confirmed"
    else:  # "variants"
        max_hamming = 4
        max_diff_ratio = 0.20
        target_match_level = "suspected"

    # Initialize unvisited pool
    unvisited: set[int] = set(range(total_items))
    candidate_seeds: list[int] = list(unvisited)

    if seed_strategy == "random":
        random.shuffle(candidate_seeds)

    active_verifier = get_verifier(verifier_type)
    batches: list[dict[str, Any]] = []
    probes = 0

    for seed_idx in candidate_seeds:
        if seed_idx not in unvisited:
            continue

        probes += 1
        if probes > max_probes or len(batches) >= max_batches:
            break

        seed_meta = indexer.get_metadata(seed_idx, datasource=datasource)
        if not seed_meta:
            unvisited.discard(seed_idx)
            continue

        seed_phash = seed_meta.get("phash", "")
        if not seed_phash:
            unvisited.discard(seed_idx)
            continue

        # Recall top neighbors from Faiss binary pHash index
        recall_k = min(total_items, max(batch_size * 3, 15))
        dists, ids = indexer.search_binary(seed_phash, k=recall_k, datasource=datasource)

        candidate_ids: list[int] = []
        candidate_dists: dict[int, int] = {}
        for d, idx in zip(dists, ids):
            if idx == seed_idx or idx not in unvisited or idx < 0:
                continue
            if d <= max_hamming:
                candidate_ids.append(idx)
                candidate_dists[idx] = int(d)

        if not candidate_ids:
            unvisited.discard(seed_idx)
            continue

        # Load seed image
        seed_path = seed_meta.get("path", "")
        seed_img: Image.Image | None = None
        seed_fname = seed_meta.get("filename", "")
        if images_map and seed_fname in images_map:
            seed_img = images_map[seed_fname]
        elif seed_path and Path(seed_path).exists():
            seed_img = load_image(seed_path)

        matching_members: list[dict[str, Any]] = []
        for cand_idx in candidate_ids:
            cand_meta = indexer.get_metadata(cand_idx, datasource=datasource)
            if not cand_meta:
                continue

            cand_fname = cand_meta.get("filename", "")
            cand_path = cand_meta.get("path", "")
            cand_phash_dist = candidate_dists[cand_idx]
            cand_dhash = cand_meta.get("dhash", "")
            cand_dhash_dist = 999
            if seed_meta.get("dhash") and cand_dhash:
                cand_dhash_dist = hamming_distance(seed_meta["dhash"], cand_dhash)

            cand_img: Image.Image | None = None
            if images_map and cand_fname in images_map:
                cand_img = images_map[cand_fname]
            elif cand_path and Path(cand_path).exists():
                cand_img = load_image(cand_path)

            # Verification execution
            if seed_img is not None and cand_img is not None:
                v_res = active_verifier.verify(seed_img, cand_img)
            elif seed_img is not None and cand_path:
                v_res = active_verifier.verify_path(seed_img, cand_path)
            else:
                # Deterministic fallback when image files are not available on disk
                is_pass = cand_phash_dist <= (2 if mode == "confirmed" else 4)
                score = round(max(0.0, 1.0 - cand_phash_dist / 64.0), 4)
                v_res = VerifyResult(
                    passed=is_pass,
                    score=score,
                    diff_ratio=round(cand_phash_dist / 64.0, 4),
                )

            # Match level calibration
            if mode == "confirmed":
                is_match = v_res.passed or v_res.diff_ratio <= max_diff_ratio
                m_level = "confirmed" if is_match else "none"
            else:
                is_match = v_res.diff_ratio <= max_diff_ratio or cand_phash_dist <= 2
                m_level = "suspected" if is_match else "none"

            if is_match:
                matching_members.append(
                    {
                        "id": cand_idx,
                        "filename": cand_fname,
                        "datasource": datasource,
                        "dhash_distance": cand_dhash_dist,
                        "phash_distance": cand_phash_dist,
                        "verifier_score": round(float(v_res.score), 4),
                        "diff_ratio": round(float(v_res.diff_ratio), 4),
                        "match_level": m_level,
                        "path": cand_path,
                    }
                )
                if len(matching_members) >= batch_size - 1:
                    break

        # If we successfully formed a cluster (seed + >= 1 verified peer)
        if matching_members:
            matching_members.sort(key=lambda m: (-m["verifier_score"], m["phash_distance"]))

            seed_item = {
                "id": seed_idx,
                "filename": seed_fname,
                "datasource": datasource,
                "dhash_distance": 0,
                "phash_distance": 0,
                "verifier_score": 1.0,
                "diff_ratio": 0.0,
                "match_level": target_match_level,
                "path": seed_path,
            }

            max_diff = max(m["diff_ratio"] for m in matching_members)
            batch = {
                "batch_id": len(batches) + 1,
                "size": len(matching_members) + 1,
                "mode": mode,
                "datasource": datasource,
                "seed": seed_item,
                "members": matching_members,
                "max_diff_ratio": round(max_diff, 4),
            }
            batches.append(batch)

            # Dynamic Elimination: remove all cluster members from unvisited pool
            unvisited.discard(seed_idx)
            for m in matching_members:
                unvisited.discard(m["id"])
        else:
            unvisited.discard(seed_idx)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    logger.info(
        "Discovered %d %s batches in %.2fms (probes=%d, remaining_unvisited=%d)",
        len(batches),
        mode,
        elapsed_ms,
        probes,
        len(unvisited),
    )

    return {
        "batches": batches,
        "count": len(batches),
        "total_probes": probes,
        "query_time_ms": elapsed_ms,
        "datasource": datasource,
        "mode": mode,
    }
