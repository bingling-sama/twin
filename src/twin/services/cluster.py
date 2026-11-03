"""Seedless Similar Texture Discovery via Greedy Leader Elimination.

Discovers batches of mutually similar or duplicate Minecraft textures across
indexed assets without requiring an input query image.

Core Algorithm: Greedy Leader Dynamic Elimination
1. Maintains an unvisited pool of indexed asset IDs.
2. Selects pivot seed (supports random or sequential strategy).
3. Recalls nearest neighbors via Faiss binary perceptual hash index (IndexBinaryFlat).
4. Loads images in parallel via load_images() with deduplicated paths and local cache.
5. Verifies candidate similarity against seed using DiffMaskVerifier.
6. If cluster size >= 2 (seed + >= 1 verified neighbors):
   - Constructs a SimilarBatchItem.
   - Eliminates all cluster members from unvisited pool (prevents transitivity collapse).
   - Immediately breaks if max_batches is satisfied (prevents probe inflation).
7. Repeats until requested max_batches reached or unvisited pool / max_probes exhausted.
8. Returns batches with detailed stage-level telemetry metrics (Show your work).
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
from twin.services.verifiers import DiffMaskVerifier
from twin.utils.image import load_images

logger = logging.getLogger(__name__)


def discover_similar_batches(
    datasource: str = "skin",
    max_batches: int = 5,
    batch_size: int = 5,
    mode: str = "confirmed",  # "confirmed" | "variants"
    max_probes: int = 200,
    seed_strategy: str = "random",  # "random" | "sequential"
) -> dict[str, Any]:
    """Discover batches of similar textures without an input seed image.

    Args:
        datasource: Datasource partition ("skin" or "cape").
        max_batches: Maximum number of batches to return.
        batch_size: Maximum total size of each batch (seed + peers).
        mode: "confirmed" (strict duplicates) or "variants" (recolors/edits).
        max_probes: Maximum number of seed exploration attempts.
        seed_strategy: "random" or "sequential".

    Returns:
        Dictionary containing discovered batches, count, probes, performance metrics,
        and stage-level telemetry.
    """
    t0 = time.perf_counter()
    total_items = indexer.get_count(datasource=datasource)

    empty_stages = {
        "recall": {"probes_evaluated": 0, "candidates_recalled": 0, "elapsed_ms": 0.0},
        "io_load": {"images_loaded": 0, "elapsed_ms": 0.0},
        "verify": {
            "verifier": "DiffMaskVerifier",
            "pairs_verified": 0,
            "batches_formed": 0,
            "elapsed_ms": 0.0,
        },
    }

    if total_items < 2:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "batches": [],
            "count": 0,
            "total_probes": 0,
            "query_time_ms": elapsed_ms,
            "datasource": datasource,
            "mode": mode,
            "stages": empty_stages,
        }

    # Threshold calibration based on mode (from centralized settings)
    if mode == "confirmed":
        max_hamming = settings.discover_confirmed_hamming
        max_diff_ratio = settings.discover_confirmed_diff_ratio
        target_match_level = "confirmed"
    else:  # "variants"
        max_hamming = settings.discover_variants_hamming
        max_diff_ratio = settings.discover_variants_diff_ratio
        target_match_level = "suspected"

    # Initialize unvisited pool
    unvisited: set[int] = set(range(total_items))
    candidate_seeds: list[int] = list(unvisited)

    if seed_strategy == "random":
        random.shuffle(candidate_seeds)

    verifier = DiffMaskVerifier()
    batches: list[dict[str, Any]] = []
    probes = 0

    # Telemetry accumulators
    t_recall_total = 0.0
    t_io_total = 0.0
    t_verify_total = 0.0
    total_candidates_recalled = 0
    total_images_loaded = 0
    total_pairs_verified = 0

    # Local session image cache (path -> PIL Image)
    img_cache: dict[str, Image.Image] = {}

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
        t_rec_0 = time.perf_counter()
        recall_k = min(
            total_items,
            max(batch_size * settings.discover_recall_multiplier, settings.discover_min_recall_k),
        )
        dists, ids = indexer.search_binary(seed_phash, k=recall_k, datasource=datasource)
        t_recall_total += time.perf_counter() - t_rec_0

        candidate_ids: list[int] = []
        candidate_dists: dict[int, int] = {}
        for d, idx in zip(dists, ids):
            if idx == seed_idx or idx not in unvisited or idx < 0:
                continue
            if d <= max_hamming:
                candidate_ids.append(idx)
                candidate_dists[idx] = int(d)

        total_candidates_recalled += len(candidate_ids)

        if not candidate_ids:
            unvisited.discard(seed_idx)
            continue

        # Gather paths for parallel image loading
        seed_path = seed_meta.get("path", "")
        cand_metas: dict[int, dict] = {}
        paths_to_load: list[str] = []

        if seed_path and seed_path not in img_cache and Path(seed_path).exists():
            paths_to_load.append(seed_path)

        for cand_idx in candidate_ids:
            c_meta = indexer.get_metadata(cand_idx, datasource=datasource)
            if c_meta:
                cand_metas[cand_idx] = c_meta
                c_path = c_meta.get("path", "")
                if c_path and c_path not in img_cache and Path(c_path).exists():
                    paths_to_load.append(c_path)

        # Batch load missing images in parallel with deduplication
        if paths_to_load:
            paths_to_load = list(dict.fromkeys(paths_to_load))
            t_io_0 = time.perf_counter()
            loaded_imgs, ok_paths, _ = load_images(paths_to_load)
            for p, img in zip(ok_paths, loaded_imgs):
                img_cache[p] = img
            total_images_loaded += len(ok_paths)
            t_io_total += time.perf_counter() - t_io_0

        # Retrieve seed image from cache
        seed_img = img_cache.get(seed_path)
        if seed_img is None:
            logger.warning(
                "Seed texture %s (%s) not found on disk, skipping probe", seed_idx, seed_path
            )
            unvisited.discard(seed_idx)
            continue

        matching_members: list[dict[str, Any]] = []
        for cand_idx in candidate_ids:
            cand_meta = cand_metas.get(cand_idx)
            if not cand_meta:
                continue

            cand_path = cand_meta.get("path", "")
            cand_img = img_cache.get(cand_path)
            if cand_img is None:
                logger.warning("Candidate %s (%s) not found on disk, skipping", cand_idx, cand_path)
                continue

            cand_fname = cand_meta.get("filename", "")
            cand_phash_dist = candidate_dists[cand_idx]
            cand_dhash = cand_meta.get("dhash", "")
            cand_dhash_dist = 999
            if seed_meta.get("dhash") and cand_dhash:
                cand_dhash_dist = hamming_distance(seed_meta["dhash"], cand_dhash)

            # Strict pixel-level verification via DiffMaskVerifier (NO shortcuts)
            t_ver_0 = time.perf_counter()
            v_res = verifier.verify(seed_img, cand_img)
            t_verify_total += time.perf_counter() - t_ver_0
            total_pairs_verified += 1

            # Match criteria strictly respect configured max_diff_ratio (NO bypass via v_res.passed)
            is_match = v_res.diff_ratio <= max_diff_ratio
            if mode == "confirmed":
                m_level = "confirmed" if is_match else "none"
            else:  # variants mode
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
                "filename": seed_meta.get("filename", ""),
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

            # Immediate exit when target max_batches reached to prevent probe inflation
            if len(batches) >= max_batches:
                break
        else:
            unvisited.discard(seed_idx)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    stages = {
        "recall": {
            "probes_evaluated": probes,
            "candidates_recalled": total_candidates_recalled,
            "elapsed_ms": round(t_recall_total * 1000, 3),
        },
        "io_load": {
            "images_loaded": total_images_loaded,
            "elapsed_ms": round(t_io_total * 1000, 3),
        },
        "verify": {
            "verifier": "DiffMaskVerifier",
            "pairs_verified": total_pairs_verified,
            "batches_formed": len(batches),
            "elapsed_ms": round(t_verify_total * 1000, 3),
        },
    }

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
        "stages": stages,
    }
