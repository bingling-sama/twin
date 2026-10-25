"""Deterministic single-pipeline search service.

Architecture (Issue #5):
  Separation by Modality:
    - Image Query (以图搜图): Pure CPU Flat SIMD binary hash retrieval (faiss.IndexBinaryFlat)
      + Unified Verifier (DiffMask by default / SSIM optional). 0% GPU算力.
    - Text Query (以文搜图): Multi-modal CLIP text vector retrieval (faiss.IndexFlatL2).
  Unified Top-K Funnel:
    - No presumptive front-shortcutting (无死板前置短路).
    - No compensatory multi-path union/patching (无代偿性并联去重打分).
    - Retrieval produces Top-K candidates -> Unified Verifier -> Transparent API output.
"""

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from twin.core.config import settings
from twin.services.classifier import TextureType, inspect_texture
from twin.services.converter import single2double
from twin.services.hasher import (
    compute_dhash,
    compute_phash,
    compute_rotated_phashes,
    hamming_distance,
    rotate_image,
)
from twin.services.indexer import indexer
from twin.services.verifiers import get_verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed candidate model
# ---------------------------------------------------------------------------
@dataclass
class SearchCandidate:
    """Internal structured state tracking across retrieval and verification."""

    id: int
    filename: str
    distance: float
    meta: dict
    dhash_distance: int = 999
    phash_distance: int = 999
    verifier_score: float = 0.0
    verifier_type: str = "diffmask"
    datasource: str = "skin"
    ssim_score: float = 0.0
    stages_passed: int = 0
    candidate_rotations: list[int] = field(default_factory=lambda: [0])
    best_rotation: int = 0

    def to_api_dict(self) -> dict:
        v_score = self.verifier_score if self.verifier_score > 0.0 else self.ssim_score
        return {
            "id": self.id,
            "filename": self.filename,
            "distance": self.distance,
            "match_level": _assign_match_level(self.stages_passed),
            "stages_passed": self.stages_passed,
            "dhash_distance": self.dhash_distance,
            "phash_distance": self.phash_distance,
            "verifier_score": v_score,
            "verifier_type": self.verifier_type,
            "datasource": self.datasource,
            "ssim_score": v_score,
            "dhash_hex": self.meta.get("dhash", ""),
            "phash_hex": self.meta.get("phash", ""),
            "path": self.meta.get("path", ""),
        }


def _empty(t0: float, stages: dict, datasource: str = "skin") -> dict:
    """Return an empty-but-well-formed search response dict."""
    elapsed = time.perf_counter() - t0
    for s in stages.values():
        s.setdefault("elapsed_ms", 0)
    return {
        "results": [],
        "count": 0,
        "datasource": datasource,
        "query_time_ms": round(elapsed * 1000, 3),
        "stages": stages,
    }


def _assign_match_level(stages_passed: int) -> str:
    """Tier based on verification result.

    - >= 3: confirmed (passed Verifier, e.g. diffmask <= threshold)
    - == 2: suspected (within perceptual hash threshold)
    - <= 1: none (recalled or unverified)
    """
    if stages_passed >= 3:
        return "confirmed"
    elif stages_passed == 2:
        return "suspected"
    return "none"


def _candidate_to_api_dict(c: SearchCandidate | dict, datasource: str = "skin") -> dict:
    if isinstance(c, SearchCandidate):
        return c.to_api_dict()
    meta = c.get("meta", {}) if isinstance(c.get("meta"), dict) else {}
    v_score = c.get("verifier_score", c.get("ssim_score", 0.0))
    return {
        "id": c["id"],
        "filename": c["filename"],
        "distance": c["distance"],
        "match_level": _assign_match_level(c.get("stages_passed", 0)),
        "stages_passed": c.get("stages_passed", 0),
        "dhash_distance": c.get("dhash_distance", 999),
        "phash_distance": c.get("phash_distance", 999),
        "verifier_score": v_score,
        "verifier_type": c.get("verifier_type", "diffmask"),
        "datasource": c.get("datasource", datasource),
        "ssim_score": v_score,
        "dhash_hex": meta.get("dhash", ""),
        "phash_hex": meta.get("phash", ""),
        "path": meta.get("path", ""),
    }


def _final_sort(results: list[SearchCandidate | dict]) -> None:
    """In-place sort: verified first (stages_passed DESC), then verifier_score DESC,
    then pHash ASC, then dHash ASC, then distance ASC.
    """
    def _key(r):
        if isinstance(r, SearchCandidate):
            v_score = r.verifier_score if r.verifier_score > 0.0 else r.ssim_score
            return (-r.stages_passed, -v_score, r.phash_distance, r.dhash_distance, r.distance)
        v_score = r.get("verifier_score", r.get("ssim_score", 0.0))
        return (
            -r.get("stages_passed", 0),
            -v_score,
            r.get("phash_distance", 999),
            r.get("dhash_distance", 999),
            r.get("distance", 0.0),
        )

    results.sort(key=_key)


def _build_response(
    results: list[SearchCandidate | dict],
    stages: dict,
    t0: float,
    datasource: str = "skin",
) -> dict:
    """Clean up internal keys and return API-ready dict."""
    elapsed = time.perf_counter() - t0
    return {
        "results": [_candidate_to_api_dict(c) for c in results],
        "count": len(results),
        "datasource": datasource,
        "query_time_ms": round(elapsed * 1000, 3),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Main search: Image Query (Single-Pipeline Flat SIMD + Verifier)
# ---------------------------------------------------------------------------
def search(
    image: Image.Image,
    top_k: int | None = None,
    dhash_threshold: int | None = None,
    phash_threshold: int | None = None,
    ssim_threshold: float | None = None,
    rotation_invariant: bool | None = None,
    datasource: str | None = None,
    verifier_type: str | None = None,
) -> dict:
    """
    Deterministic single-pipeline Minecraft texture search.

    Steps:
      1. Classifier & Preprocessing: Detect Skin/Cape, auto-upgrade 64x32 legacy skins.
      2. Extract 0° global pHash (CPU).
      3. Flat SIMD Recall: faiss.IndexBinaryFlat hardware scan in ~3.3ms, zero GPU usage.
      4. Unified Verifier: DiffMask (default) / SSIM verification on all Top-K candidates.
      5. Output Top-K candidates transparently with recall distance and verifier scores.
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
    stages: dict[str, dict] = {}

    # Step 1: Auto-classify texture and determine datasource
    t_cls_0 = time.perf_counter()
    report = inspect_texture(image)
    if datasource is None:
        if report.texture_type == TextureType.UNKNOWN:
            logger.warning(
                "Search query rejected: unrecognized texture format (%dx%d)",
                image.width,
                image.height,
            )
            stages["classifier"] = {
                "texture_type": report.texture_type.value,
                "upgraded": False,
                "datasource": "unknown",
                "elapsed_ms": round((time.perf_counter() - t_cls_0) * 1000, 3),
                "warning": f"Unrecognized Minecraft texture format ({image.width}x{image.height})",
            }
            return _empty(t0, stages, datasource="unknown")

        is_cape = report.texture_type in (TextureType.CAPE, TextureType.ELYTRA)
        target_ds = "cape" if is_cape else "skin"
    else:
        target_ds = datasource

    # Auto-upgrade 64x32 legacy skins to 64x64 for representation parity
    if report.can_auto_upgrade:
        image = single2double(image)
    t_cls_1 = time.perf_counter()

    stages["classifier"] = {
        "texture_type": report.texture_type.value,
        "upgraded": report.can_auto_upgrade,
        "datasource": target_ds,
        "elapsed_ms": round((t_cls_1 - t_cls_0) * 1000, 3),
    }

    # Step 2 & 3: Flat SIMD binary hash retrieval (faiss.IndexBinaryFlat)
    t_simd_0 = time.perf_counter()
    query_phash = compute_phash(image)
    query_dhash = compute_dhash(image)

    # Candidate tracking: cand_id -> (cand_id, min_hamming, best_rotation)
    retrieved_pool: dict[int, tuple[int, int, int]] = {}

    if rotation_invariant:
        query_phashes = compute_rotated_phashes(image)
        for rot_idx, qh in enumerate(query_phashes):
            dists, ids = indexer.search_binary(qh, k=top_k, datasource=target_ds)
            for d, idx in zip(dists, ids):
                if idx < 0:
                    continue
                d_int = int(d)
                if idx not in retrieved_pool or d_int < retrieved_pool[idx][1]:
                    retrieved_pool[idx] = (idx, d_int, rot_idx)
    else:
        dists, ids = indexer.search_binary(query_phash, k=top_k, datasource=target_ds)
        for d, idx in zip(dists, ids):
            if idx < 0:
                continue
            retrieved_pool[idx] = (idx, int(d), 0)

    # Sort candidates by Hamming distance ASC, take top_k
    sorted_recalled = sorted(retrieved_pool.values(), key=lambda x: x[1])[:top_k]

    t_simd_1 = time.perf_counter()
    stages["simd_recall"] = {
        "in": 1,
        "out": len(sorted_recalled),
        "elapsed_ms": round((t_simd_1 - t_simd_0) * 1000, 3),
        "datasource": target_ds,
    }
    # Backward compatibility monitoring keys
    stages["faiss"] = stages["simd_recall"]

    if not sorted_recalled:
        return _empty(t0, stages, datasource=target_ds)

    # Step 4: Unified Verifier (DiffMask / SSIM) on top-K candidates
    active_verifier = get_verifier(verifier_type)
    t_ver_0 = time.perf_counter()
    candidates: list[SearchCandidate] = []

    for idx, p_dist, best_rot in sorted_recalled:
        meta = indexer.get_metadata(idx, datasource=target_ds)
        if meta is None:
            continue
        cpath = meta.get("path", "")

        query_candidate_img = (
            rotate_image(image, best_rot) if (rotation_invariant and best_rot) else image
        )
        cand_dhash = meta.get("dhash", "")
        if cand_dhash:
            if rotation_invariant and best_rot:
                cand_d_dist = hamming_distance(compute_dhash(query_candidate_img), cand_dhash)
            else:
                cand_d_dist = hamming_distance(query_dhash, cand_dhash)
        else:
            cand_d_dist = 999

        v_res = active_verifier.verify_path(query_candidate_img, cpath)

        # Match level & stage passed determination:
        # 3 = confirmed (passed Verifier, e.g. diffmask <= 5% or SSIM >= threshold)
        # 2 = suspected (within pHash Hamming threshold)
        # 1 = none/recalled (candidate in top_k)
        if v_res.passed:
            stages_passed = 3
        elif p_dist <= phash_threshold or cand_d_dist <= dhash_threshold:
            stages_passed = 2
        else:
            stages_passed = 1

        v_score = round(float(v_res.score), 4)
        v_type_name = type(active_verifier).__name__.lower().replace("verifier", "")
        cand = SearchCandidate(
            id=idx,
            filename=meta.get("filename", "unknown"),
            distance=round(float(p_dist), 6),
            meta=meta,
            dhash_distance=cand_d_dist,
            phash_distance=p_dist,
            verifier_score=v_score,
            verifier_type=v_type_name,
            datasource=target_ds,
            ssim_score=v_score,
            stages_passed=stages_passed,
            candidate_rotations=[best_rot],
            best_rotation=best_rot,
        )
        candidates.append(cand)

    t_ver_1 = time.perf_counter()
    stages["verify"] = {
        "verifier": type(active_verifier).__name__,
        "evaluated": len(candidates),
        "elapsed_ms": round((t_ver_1 - t_ver_0) * 1000, 3),
    }
    stages["ssim"] = stages["verify"]

    # Step 5: Final ranking and output
    _final_sort(candidates)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Search in %.2fms: %d candidates recalled via SIMD -> all %d verified (%s)",
        elapsed * 1000,
        len(sorted_recalled),
        len(candidates),
        type(active_verifier).__name__,
    )

    return _build_response(candidates, stages, t0, datasource=target_ds)


# ---------------------------------------------------------------------------
# Text Search: Text Query (CLIP Multi-modal Vector Retrieval)
# ---------------------------------------------------------------------------
def search_by_text(
    query: str,
    top_k: int | None = None,
    datasource: str = "skin",
) -> dict:
    """
    Search indexed textures using natural language text query via CLIP.

    Returns candidate images ranked by vector L2 distance.
    Raises ValueError if active model does not support text embeddings (e.g. DINOv2).
    """
    from twin.models.clip_model import get_model_name
    from twin.services.embedding import (
        compute_text_embedding,
        is_text_supported,
    )

    if not is_text_supported():
        model_display = get_model_name() or settings.model_name
        raise ValueError(
            f"Current model '{model_display}' (DINOv2) is vision-only "
            "and does not support text search. Use CLIP (e.g. ViT-B-32)."
        )

    if top_k is None:
        top_k = settings.top_k

    t0 = time.perf_counter()
    query_vec = compute_text_embedding(query)

    distances, indices = indexer.search(query_vec, k=top_k, datasource=datasource)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    if not indices:
        return {
            "results": [],
            "count": 0,
            "query_time_ms": round(elapsed * 1000, 3),
            "query": query,
            "datasource": datasource,
        }

    results = []
    for dist, idx in zip(distances, indices):
        if idx < 0:
            continue
        meta = indexer.get_metadata(idx, datasource=datasource)
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
                "datasource": datasource,
            }
        )

    return {
        "results": results,
        "count": len(results),
        "query_time_ms": round(elapsed * 1000, 3),
        "query": query,
        "datasource": datasource,
    }
