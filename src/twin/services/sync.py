"""Startup synchronization — ensures all images in the images directory are indexed.

Uses batched CLIP inference + parallel dHash for high throughput.

Exposes a thread-safe progress dictionary via get_sync_status() so the
GET /api/v1/sync/status endpoint can report background sync progress + ETA.
"""

import logging
import threading
import time

from twin.core.config import settings
from twin.services.embedding import compute_embeddings
from twin.services.hasher import compute_dhashes, compute_phashes
from twin.services.index_service import _index_single_from_disk
from twin.services.indexer import indexer
from twin.utils.image import IMAGE_EXTENSIONS, load_images

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared progress state (thread-safe — written by bg sync, read by routes)
# ---------------------------------------------------------------------------
_sync_lock = threading.Lock()
_sync_state: dict = {
    "running": False,
    "started_at": None,       # perf_counter timestamp (None = not started)
    "total_files": 0,
    "indexed_files": 0,
    "skipped_files": 0,
    "failed_files": 0,
    "_new_work_samples": [],  # list of (perf_counter, new_work) for rolling rate
}


def get_sync_status() -> dict:
    """Return a live snapshot of background sync progress.

    Computes elapsed / ETA on the fly so callers always see fresh estimates.
    """
    with _sync_lock:
        s = dict(_sync_state)  # shallow copy — all values are immutable

    if s["running"] and s["started_at"] is not None and s["total_files"] > 0:
        processed = s["indexed_files"] + s["failed_files"]  # indexed_files already includes skipped
        new_work = processed - s["skipped_files"]            # actual indexing, excluding pre-existing
        now = time.perf_counter()
        s["elapsed_ms"] = (now - s["started_at"]) * 1000

        # Recent rate (last 5s rolling window) — for display
        samples = s.get("_new_work_samples", [])
        if samples:
            cutoff = now - 5.0
            recent = [(t, w) for t, w in samples if t >= cutoff]
            if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
                dw = recent[-1][1] - recent[0][1]
                dt = recent[-1][0] - recent[0][0]
                s["rate_img_per_s"] = round(dw / dt, 1) if dt > 0 else 0.0
            elif new_work > 0:
                s["rate_img_per_s"] = round(new_work / max(s["elapsed_ms"] / 1000, 0.001), 1)
            else:
                s["rate_img_per_s"] = 0.0
        elif new_work > 0:
            s["rate_img_per_s"] = round(new_work / max(s["elapsed_ms"] / 1000, 0.001), 1)
        else:
            s["rate_img_per_s"] = 0.0

        # ETA uses cumulative rate (more stable for long-running jobs)
        if new_work > 0:
            cum_rate = new_work / max(s["elapsed_ms"] / 1000, 0.001)
            remaining = s["total_files"] - processed
            s["eta_ms"] = (remaining / cum_rate) * 1000 if cum_rate > 0 else 0
        else:
            s["eta_ms"] = 0

        s["progress_pct"] = round(processed * 100 / s["total_files"], 1)
    else:
        s["elapsed_ms"] = 0.0
        s["eta_ms"] = 0.0
        s["progress_pct"] = 0.0
        s["rate_img_per_s"] = 0.0

    return s


def _reset_sync_state() -> None:
    """Reset the shared sync progress tracker (called at start of a new sync)."""
    with _sync_lock:
        _sync_state.update(
            running=False,
            started_at=None,
            total_files=0,
            indexed_files=0,
            skipped_files=0,
            failed_files=0,
            _new_work_samples=[],
        )


def sync_images_dir() -> dict:
    """
    Scan the images directory and index any files not already in the index.

    Identification is by filename — if a file with the same name is already
    indexed, it is skipped.  Processing uses batched CLIP inference for speed.

    Updates the shared _sync_state progress dictionary at each batch so the
    GET /api/v1/sync/status endpoint can report live progress + ETA.

    Returns a dict with counts: {total, indexed, skipped, failed}.
    """
    t0 = time.perf_counter()
    _reset_sync_state()

    images_path = settings.images_path
    if not images_path.is_dir():
        logger.info("Images directory does not exist: %s — skipping sync", images_path)
        return {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}

    image_files = sorted(
        p for p in images_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.info("Images directory is empty — nothing to sync")
        return {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}

    indexed_set = indexer.get_indexed_filenames()
    to_index = [p for p in image_files if p.name not in indexed_set]
    skipped = len(image_files) - len(to_index)

    total_files = len(image_files)
    total_batches = (len(to_index) + settings.batch_size - 1) // settings.batch_size if to_index else 0

    # Initialize shared progress state
    with _sync_lock:
        _sync_state.update(
            running=True,
            started_at=t0,
            total_files=total_files,
            indexed_files=skipped,   # already-indexed count toward progress
            skipped_files=skipped,
            failed_files=0,
        )

    if not to_index:
        elapsed = time.perf_counter() - t0
        logger.info("All %d images already indexed (%d skipped) in %.1fs", total_files, skipped, elapsed)
        with _sync_lock:
            _sync_state["running"] = False
        return {"total": total_files, "indexed": 0, "skipped": skipped, "failed": 0}

    batch_size = settings.batch_size
    logger.info(
        "Syncing: %d total, %d already indexed, %d to index (batch_size=%d)",
        total_files, skipped, len(to_index), batch_size,
    )

    indexed = 0
    failed = 0
    batch_num = 0

    for i in range(0, len(to_index), batch_size):
        batch_paths = to_index[i : i + batch_size]
        batch_num += 1

        # Step 1: Load images
        imgs, ok_paths, bad = load_images(batch_paths)
        failed += len(bad)

        if not imgs:
            _update_progress(indexed, skipped, failed)
            continue

        try:
            # Step 2: Batch CLIP embedding (one forward pass for the whole batch)
            vectors = compute_embeddings(imgs)

            # Step 3: Parallel dHash + pHash
            dhashes = compute_dhashes(imgs)
            phashes = compute_phashes(imgs)

            # Step 4: Build metadata and batch-add to index
            metas = []
            for path, dh, ph in zip(ok_paths, dhashes, phashes):
                metas.append({
                    "filename": path.name,
                    "path": str(path),
                    "dhash": dh,
                    "phash": ph,
                })

            indexer.add_items(vectors, metas)
            indexed += len(metas)

            pct = (indexed + skipped) * 100 // total_files
            logger.info("Sync progress: %d/%d (%d%%) — batch %d/%d",
                        indexed + skipped, total_files, pct, batch_num, total_batches)

        except Exception:
            logger.warning("Batch failed at offset %d, falling back to individual", i, exc_info=True)
            # Fallback: index one by one for this batch
            for path, img in zip(ok_paths, imgs):
                try:
                    _index_single_from_disk(img, path)
                    indexed += 1
                except Exception:
                    logger.warning("Failed to index: %s", path, exc_info=True)
                    failed += 1

        # Update shared progress after each batch
        _update_progress(indexed, skipped, failed)

        # Free memory
        for img in imgs:
            img.close()

    elapsed = time.perf_counter() - t0
    rate = indexed / elapsed if elapsed > 0 else 0
    logger.info(
        "Sync complete: %d indexed, %d skipped, %d failed in %.1fs (%.0f img/s)",
        indexed, skipped, failed, elapsed, rate,
    )

    with _sync_lock:
        _sync_state["running"] = False

    # Persist immediately — don't wait for the 120s auto-save interval.
    # If the process is killed before auto-save fires, the Faiss index and
    # metadata are lost (vectors.bin survives via append, but .faiss/.json won't).
    if indexed > 0:
        try:
            indexer.save()
            logger.info("Index persisted after sync (%d vectors)", indexer.count)
        except Exception:
            logger.warning("Failed to persist index after sync", exc_info=True)

    return {
        "total": total_files,
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _update_progress(indexed: int, skipped: int, failed: int) -> None:
    """Update the shared progress tracker after a batch completes."""
    with _sync_lock:
        _sync_state["indexed_files"] = indexed + skipped
        _sync_state["skipped_files"] = skipped
        _sync_state["failed_files"] = failed
        # Record sample for rolling rate (keep last 60 samples ≈ 5 min at 1 batch/5s)
        new_work = indexed + failed
        samples = _sync_state["_new_work_samples"]
        samples.append((time.perf_counter(), new_work))
        if len(samples) > 60:
            samples.pop(0)
