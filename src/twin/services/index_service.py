"""Indexing workflows — shared by the HTTP API and startup sync.

Encapsulates the full indexing pipeline (load → embed → hash → store) so
route handlers are thin passthroughs and sync.py can reuse the same logic.
"""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from twin.core.config import settings
from twin.services.classifier import TextureType, inspect_texture
from twin.services.converter import single2double
from twin.services.embedding import compute_embedding, compute_embeddings
from twin.services.hasher import (
    compute_ahash,
    compute_ahashes,
    compute_dhash,
    compute_dhashes,
    compute_phash,
    compute_phashes,
)
from twin.services.indexer import indexer
from twin.utils.image import IMAGE_EXTENSIONS, load_images
from twin.utils.metrics import compute_progress_metrics

logger = logging.getLogger(__name__)

# Bounded single-worker queue for async batch indexing (prevents GPU/CPU resource contention)
_batch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-index")

# ---------------------------------------------------------------------------
# Task state & bounded memory management (thread-safe LRU)
# ---------------------------------------------------------------------------
MAX_TASKS = 100
_tasks_lock = threading.Lock()
_tasks: OrderedDict[str, dict] = OrderedDict()


def _prune_tasks_locked() -> None:
    """Evict least recently accessed finished tasks if _tasks exceeds MAX_TASKS."""
    if len(_tasks) <= MAX_TASKS:
        return
    # In OrderedDict, items at the front are the least recently accessed
    removable = [tid for tid, t in _tasks.items() if t.get("status") in ("completed", "failed")]
    for tid in removable:
        if len(_tasks) <= MAX_TASKS:
            break
        _tasks.pop(tid, None)


def get_task_status(task_id: str) -> dict | None:
    """Return status dictionary for a specific task ID with live computed progress."""
    with _tasks_lock:
        if task_id not in _tasks:
            return None
        _tasks.move_to_end(task_id)
        task = _tasks[task_id]
        res = dict(task)
        samples = list(task.get("_processed_samples", []))

    if (
        res["status"] in ("running", "started")
        and res.get("started_at") is not None
        and res.get("total", 0) > 0
    ):
        processed = res.get("indexed", 0) + res.get("failed", 0)
        metrics = compute_progress_metrics(
            processed=processed,
            total=res["total"],
            started_at=res["started_at"],
            samples=samples,
        )
        res["time_ms"] = metrics["elapsed_ms"]
        res["elapsed_ms"] = metrics["elapsed_ms"]
        res["rate_img_per_s"] = metrics["rate_img_per_s"]
        res["eta_ms"] = metrics["eta_ms"]
        res["progress_pct"] = metrics["progress_pct"]
    else:
        res.setdefault("rate_img_per_s", 0.0)
        res.setdefault("eta_ms", 0.0)

    return res


def get_batch_status() -> dict:
    """Return a live snapshot of the most recent or active batch indexing task."""
    with _tasks_lock:
        active_task = None
        for t in reversed(list(_tasks.values())):
            if t.get("status") == "running":
                active_task = t
                break
        if active_task is None and _tasks:
            active_task = list(_tasks.values())[-1]

    if active_task:
        status = get_task_status(active_task["task_id"])
        if status:
            return {
                "running": status.get("status") in ("started", "running"),
                "total": status.get("total", 0),
                "indexed": status.get("indexed", 0),
                "failed": status.get("failed", 0),
                "current_batch": status.get("current_batch", 0),
                "total_batches": status.get("total_batches", 0),
                "progress_pct": status.get("progress_pct", 0.0),
                "elapsed_ms": status.get("elapsed_ms", status.get("time_ms", 0.0)),
                "eta_ms": status.get("eta_ms", 0.0),
                "rate_img_per_s": status.get("rate_img_per_s", 0.0),
            }

    return {
        "running": False,
        "total": 0,
        "indexed": 0,
        "failed": 0,
        "current_batch": 0,
        "total_batches": 0,
        "progress_pct": 0.0,
        "elapsed_ms": 0.0,
        "eta_ms": 0.0,
        "rate_img_per_s": 0.0,
    }


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
            "current_batch": 0,
            "total_batches": 0,
            "progress_pct": 0.0,
            "time_ms": 0.0,
            "started_at": None,
            "_processed_samples": [],
            "error": None,
        }
        _prune_tasks_locked()

    def _worker():
        try:
            with _tasks_lock:
                if task_id in _tasks:
                    _tasks[task_id]["status"] = "running"
            res = index_batch(dir_path, task_id=task_id)
            with _tasks_lock:
                if task_id in _tasks:
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
                if task_id in _tasks:
                    _tasks[task_id].update(
                        status="failed",
                        error=str(e),
                    )

    _batch_executor.submit(_worker)
    return task_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_single(image: Image.Image, filename: str, content: bytes) -> dict:
    """
    Index a single image uploaded via the API.

    Returns {"status": "indexed"|"already_exists", "id": int, "filename": str, "datasource": str}.
    """
    report = inspect_texture(image)
    target_ds = "cape" if report.texture_type in (TextureType.CAPE, TextureType.ELYTRA) else "skin"
    processed_image = single2double(image) if report.can_auto_upgrade else image

    # Dedup check
    if filename in indexer.get_indexed_filenames(datasource=target_ds):
        return {
            "status": "already_exists",
            "id": -1,
            "filename": filename,
            "datasource": target_ds,
        }

    # Persist to disk
    saved_path = settings.images_path / filename
    saved_path.write_bytes(content)

    # Compute and store
    vector = compute_embedding(processed_image)
    meta = {
        "filename": filename,
        "path": str(saved_path),
        "datasource": target_ds,
        "texture_type": report.texture_type.value,
        "upgraded": report.can_auto_upgrade,
        "ahash": compute_ahash(processed_image),
        "dhash": compute_dhash(processed_image),
        "phash": compute_phash(processed_image),
    }
    idx = indexer.add_item(vector, meta, datasource=target_ds)
    logger.info("Indexed image %d (%s): %s", idx, target_ds, filename)
    return {
        "status": "indexed",
        "id": idx,
        "filename": filename,
        "datasource": target_ds,
    }


def index_batch(directory: str | Path, task_id: str | None = None) -> dict:
    """
    Index all images in a server-side directory.

    Uses batched CLIP/DINOv2 inference + parallel hashing for throughput.
    Returns {"status": "completed", "total": int, "indexed": int, "failed": int}.
    """
    t0 = time.perf_counter()
    directory = Path(directory).expanduser().resolve()

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    image_files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        return {"status": "completed", "total": 0, "indexed": 0, "failed": 0, "time_ms": 0}

    batch_size = settings.batch_size
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    indexed = 0
    failed = 0

    if task_id:
        with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id].update(
                    started_at=t0,
                    total=len(image_files),
                    indexed=0,
                    failed=0,
                    current_batch=0,
                    total_batches=total_batches,
                    _processed_samples=[],
                )

    batches = [image_files[i : i + batch_size] for i in range(0, len(image_files), batch_size)]

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-preload") as preloader:
            future_next = preloader.submit(load_images, batches[0]) if batches else None

            for b_idx, batch_paths in enumerate(batches):
                batch_num = b_idx + 1
                try:
                    imgs, ok_paths, bad = future_next.result() if future_next else ([], [], [])
                except Exception as e:
                    logger.error("Failed to preload batch %d: %s", batch_num, e, exc_info=True)
                    imgs, ok_paths, bad = [], [], list(batch_paths)

                # Submit next batch asynchronously while computing current batch
                if b_idx + 1 < len(batches):
                    future_next = preloader.submit(load_images, batches[b_idx + 1])
                else:
                    future_next = None

                failed += len(bad)

                if not imgs:
                    if task_id:
                        with _tasks_lock:
                            if task_id in _tasks:
                                _tasks[task_id].update(
                                    current_batch=batch_num,
                                    indexed=indexed,
                                    failed=failed,
                                )
                    continue

                try:
                    proc_imgs = []
                    ds_list = []
                    reports = []
                    for img in imgs:
                        rep = inspect_texture(img)
                        reports.append(rep)
                        ds = (
                            "cape"
                            if rep.texture_type in (TextureType.CAPE, TextureType.ELYTRA)
                            else "skin"
                        )
                        ds_list.append(ds)
                        proc_imgs.append(single2double(img) if rep.can_auto_upgrade else img)

                    vectors = compute_embeddings(proc_imgs)
                    ahashes = compute_ahashes(proc_imgs)
                    dhashes = compute_dhashes(proc_imgs)
                    phashes = compute_phashes(proc_imgs)

                    groups: dict[str, tuple[list[np.ndarray], list[dict]]] = {}
                    for i, (p, ah, dh, ph, ds, rep) in enumerate(
                        zip(ok_paths, ahashes, dhashes, phashes, ds_list, reports)
                    ):
                        if ds not in groups:
                            groups[ds] = ([], [])
                        groups[ds][0].append(vectors[i])
                        groups[ds][1].append(
                            {
                                "filename": p.name,
                                "path": str(p),
                                "datasource": ds,
                                "texture_type": rep.texture_type.value,
                                "upgraded": rep.can_auto_upgrade,
                                "ahash": ah,
                                "dhash": dh,
                                "phash": ph,
                            }
                        )

                    for ds, (v_list, m_list) in groups.items():
                        indexer.add_items(np.stack(v_list), m_list, datasource=ds)
                        indexed += len(m_list)

                    for p_img, orig_img in zip(proc_imgs, imgs):
                        if p_img is not orig_img:
                            p_img.close()

                except Exception:
                    logger.warning(
                        "Batch failed at offset %d, falling back to individual",
                        b_idx * batch_size,
                        exc_info=True,
                    )
                    for path, img in zip(ok_paths, imgs):
                        try:
                            _index_single_from_disk(img, path)
                            indexed += 1
                        except Exception:
                            logger.warning("Failed to index: %s", path, exc_info=True)
                            failed += 1

                        if task_id:
                            with _tasks_lock:
                                if task_id in _tasks:
                                    _tasks[task_id].update(
                                        current_batch=batch_num,
                                        indexed=indexed,
                                        failed=failed,
                                    )
                                    samples = _tasks[task_id]["_processed_samples"]
                                    samples.append((time.perf_counter(), indexed + failed))
                                    if len(samples) > 60:
                                        samples.pop(0)

                if task_id:
                    with _tasks_lock:
                        if task_id in _tasks:
                            _tasks[task_id].update(
                                current_batch=batch_num,
                                indexed=indexed,
                                failed=failed,
                            )
                            samples = _tasks[task_id]["_processed_samples"]
                            samples.append((time.perf_counter(), indexed + failed))
                            if len(samples) > 60:
                                samples.pop(0)

                # Free memory
                for img in imgs:
                    img.close()
    finally:
        pass

    elapsed = time.perf_counter() - t0
    logger.info(
        "Batch index done: %d indexed, %d failed in %.0fms",
        indexed,
        failed,
        elapsed * 1000,
    )

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
    report = inspect_texture(image)
    target_ds = "cape" if report.texture_type in (TextureType.CAPE, TextureType.ELYTRA) else "skin"
    processed_image = single2double(image) if report.can_auto_upgrade else image

    try:
        vector = compute_embedding(processed_image)
        meta = {
            "filename": path.name,
            "path": str(path),
            "datasource": target_ds,
            "texture_type": report.texture_type.value,
            "upgraded": report.can_auto_upgrade,
            "ahash": compute_ahash(processed_image),
            "dhash": compute_dhash(processed_image),
            "phash": compute_phash(processed_image),
        }
        return indexer.add_item(vector, meta, datasource=target_ds)
    finally:
        if processed_image is not image:
            processed_image.close()
