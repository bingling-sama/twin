"""Indexing workflows — shared by the HTTP API and startup sync.

Encapsulates the full indexing pipeline (load → embed → hash → store) so
route handlers are thin passthroughs and sync.py can reuse the same logic.
"""

import logging
import threading
import uuid
from pathlib import Path

from PIL import Image

from twin.core.config import settings
from twin.services.embedding import compute_embedding, compute_embeddings
from twin.services.hasher import compute_dhash, compute_dhashes, compute_phash, compute_phashes
from twin.services.indexer import indexer
from twin.utils.image import IMAGE_EXTENSIONS, load_images

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared batch-indexing progress & background tasks (thread-safe)
# ---------------------------------------------------------------------------
_batch_lock = threading.Lock()
_batch_state: dict = {
    "running": False,
    "started_at": None,
    "total": 0,
    "indexed": 0,
    "failed": 0,
    "current_batch": 0,
    "total_batches": 0,
    "_processed_samples": [],  # list of (perf_counter, indexed + failed)
}

_tasks_lock = threading.Lock()
_tasks: dict[str, dict] = {}


def get_task_status(task_id: str) -> dict | None:
    """Return status dictionary for a specific task ID, combined with live progress if active."""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return None
        res = dict(task)

    if res["status"] == "running":
        live = get_batch_status()
        if live["running"]:
            res["total"] = live["total"]
            res["indexed"] = live["indexed"]
            res["failed"] = live["failed"]
            res["progress_pct"] = live["progress_pct"]
            res["time_ms"] = live["elapsed_ms"]
    return res


def index_batch_async(directory: str | Path) -> str:
    """Launch batch indexing in a background daemon thread and return task_id."""
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "started",
            "task_id": task_id,
            "directory": str(dir_path),
            "total": 0,
            "indexed": 0,
            "failed": 0,
            "skipped": 0,
            "progress_pct": 0.0,
            "time_ms": 0.0,
            "error": None,
        }

    def _worker():
        try:
            with _tasks_lock:
                _tasks[task_id]["status"] = "running"
            res = index_batch(dir_path)
            with _tasks_lock:
                _tasks[task_id].update(
                    status="completed",
                    total=res.get("total", 0),
                    indexed=res.get("indexed", 0),
                    failed=res.get("failed", 0),
                    time_ms=res.get("time_ms", 0.0),
                    progress_pct=100.0,
                )
        except Exception as e:
            logger.error("Async batch indexing failed for task %s", task_id, exc_info=True)
            with _tasks_lock:
                _tasks[task_id].update(
                    status="failed",
                    error=str(e),
                )

    thread = threading.Thread(target=_worker, daemon=True, name=f"batch-index-{task_id[:8]}")
    thread.start()
    return task_id


def get_batch_status() -> dict:
    """Return a live snapshot of in-progress batch indexing."""
    import time
    with _batch_lock:
        s = dict(_batch_state)
    if s["running"] and s["started_at"] is not None:
        now = time.perf_counter()
        processed = s["indexed"] + s["failed"]
        s["elapsed_ms"] = (now - s["started_at"]) * 1000

        # Recent rate (last 5s rolling window)
        samples = s.get("_processed_samples", [])
        if samples:
            cutoff = now - 5.0
            recent = [(t, w) for t, w in samples if t >= cutoff]
            if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
                dw = recent[-1][1] - recent[0][1]
                dt = recent[-1][0] - recent[0][0]
                s["rate_img_per_s"] = round(dw / dt, 1) if dt > 0 else 0.0
            elif processed > 0:
                s["rate_img_per_s"] = round(processed / max(s["elapsed_ms"] / 1000, 0.001), 1)
            else:
                s["rate_img_per_s"] = 0.0
        elif processed > 0:
            s["rate_img_per_s"] = round(processed / max(s["elapsed_ms"] / 1000, 0.001), 1)
        else:
            s["rate_img_per_s"] = 0.0

        # ETA uses cumulative rate
        if processed > 0:
            cum_rate = processed / max(s["elapsed_ms"] / 1000, 0.001)
            remaining = s["total"] - processed
            s["eta_ms"] = (remaining / cum_rate) * 1000 if cum_rate > 0 else 0
        else:
            s["eta_ms"] = 0
        s["progress_pct"] = round(processed * 100 / max(s["total"], 1), 1)
    else:
        s["elapsed_ms"] = 0.0
        s["eta_ms"] = 0.0
        s["progress_pct"] = 0.0
        s["rate_img_per_s"] = 0.0
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_single(image: Image.Image, filename: str, content: bytes) -> dict:
    """
    Index a single image uploaded via the API.

    Returns {"status": "indexed"|"already_exists", "id": int, "filename": str}.
    """
    # Dedup check
    if filename in indexer.get_indexed_filenames():
        return {"status": "already_exists", "id": -1, "filename": filename}

    # Persist to disk
    saved_path = settings.images_path / filename
    saved_path.write_bytes(content)

    # Compute and store
    vector = compute_embedding(image)
    meta = {
        "filename": filename,
        "path": str(saved_path),
        "dhash": compute_dhash(image),
        "phash": compute_phash(image),
    }
    idx = indexer.add_item(vector, meta)
    logger.info("Indexed image %d: %s", idx, filename)
    return {"status": "indexed", "id": idx, "filename": filename}


def index_batch(directory: str | Path) -> dict:
    """
    Index all images in a server-side directory.

    Uses batched CLIP inference + parallel hashing for throughput.
    Returns {"status": "completed", "total": int, "indexed": int, "failed": int}.
    """
    import time

    t0 = time.perf_counter()
    directory = Path(directory).expanduser().resolve()

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    image_files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        return {"status": "completed", "total": 0, "indexed": 0, "failed": 0, "time_ms": 0}

    batch_size = settings.batch_size
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    indexed = 0
    failed = 0

    # Initialize shared progress state
    with _batch_lock:
        _batch_state.update(
            running=True,
            started_at=t0,
            total=len(image_files),
            indexed=0,
            failed=0,
            current_batch=0,
            total_batches=total_batches,
            _processed_samples=[],
        )

    try:
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i : i + batch_size]
            batch_num = i // batch_size + 1
            imgs, ok_paths, bad = load_images(batch_paths)
            failed += len(bad)

            if not imgs:
                with _batch_lock:
                    _batch_state.update(current_batch=batch_num, indexed=indexed, failed=failed)
                continue

            try:
                vectors = compute_embeddings(imgs)
                dhashes = compute_dhashes(imgs)
                phashes = compute_phashes(imgs)

                metas = [
                    {"filename": p.name, "path": str(p), "dhash": dh, "phash": ph}
                    for p, dh, ph in zip(ok_paths, dhashes, phashes)
                ]
                indexer.add_items(vectors, metas)
                indexed += len(metas)

            except Exception:
                logger.warning(
                    "Batch failed at offset %d, falling back to individual", i, exc_info=True,
                )
                for path, img in zip(ok_paths, imgs):
                    try:
                        _index_single_from_disk(img, path)
                        indexed += 1
                    except Exception:
                        logger.warning("Failed to index: %s", path, exc_info=True)
                        failed += 1

            # Update shared progress after each batch
            with _batch_lock:
                _batch_state.update(current_batch=batch_num, indexed=indexed, failed=failed)
                samples = _batch_state["_processed_samples"]
                samples.append((time.perf_counter(), indexed + failed))
                if len(samples) > 60:
                    samples.pop(0)

            # Free memory
            for img in imgs:
                img.close()
    finally:
        with _batch_lock:
            _batch_state["running"] = False

    elapsed = time.perf_counter() - t0
    logger.info(
        "Batch index done: %d indexed, %d failed in %.0fms",
        indexed, failed, elapsed * 1000,
    )

    # Persist immediately after batch indexing — don't rely on the 120s
    # auto-save interval or fragile shutdown hooks.
    if indexed > 0:
        try:
            indexer.save()
            logger.info("Index persisted after batch (%d vectors)", indexer.count)
        except Exception:
            logger.warning("Failed to persist index after batch", exc_info=True)

    return {
        "status": "completed",
        "total": len(image_files),
        "indexed": indexed,
        "failed": failed,
        "time_ms": round(elapsed * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _index_single_from_disk(image: Image.Image, path: Path) -> int:
    """Index a single image that's already on disk. Returns assigned ID."""
    vector = compute_embedding(image)
    meta = {
        "filename": path.name,
        "path": str(path),
        "dhash": compute_dhash(image),
        "phash": compute_phash(image),
    }
    return indexer.add_item(vector, meta)
