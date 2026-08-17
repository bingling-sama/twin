"""API route definitions — thin HTTP handlers that delegate to services."""

import asyncio
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from twin.api.schemas import (
    BatchIndexAsyncResponse,
    BatchIndexRequest,
    BatchIndexResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ErrorResponse,
    HealthResponse,
    IndexListResponse,
    IndexStatus,
    SearchResponse,
    SyncStatusResponse,
    TextSearchRequest,
    TextSearchResponse,
)
from twin.core.config import settings
from twin.models.clip_model import get_device, get_gpu_name, get_model_name, is_loaded
from twin.services.index_service import (
    get_batch_status,
    get_task_status,
    index_batch,
    index_batch_async,
    index_single,
)
from twin.services.indexer import indexer
from twin.services.search import search, search_by_text
from twin.services.sync import get_sync_status

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_image(filename: str, content: bytes) -> Image.Image:
    """Validate uploaded content is a readable image, return RGB PIL Image."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    try:
        image = Image.open(BytesIO(content))
        image.load()
        return image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        indexed_count=indexer.count,
        model_loaded=is_loaded(),
        index_type=indexer.index_type_name,
        # CLIP runtime
        device=get_device(),
        model_name=get_model_name(),
        gpu_name=get_gpu_name(),
        # Faiss runtime
        faiss_gpu_enabled=indexer.faiss_gpu_enabled,
        faiss_index_type=settings.faiss_index_type,
        # Config snapshot
        batch_size=settings.batch_size,
        auto_upgrade_enabled=settings.faiss_auto_upgrade,
        auto_save_interval_s=settings.auto_save_interval,
    )


# ---------------------------------------------------------------------------
# Sync status
# ---------------------------------------------------------------------------
@router.get("/sync/status", response_model=SyncStatusResponse)
async def sync_status():
    """Report background sync progress, ETA, and completion status."""
    return get_sync_status()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@router.post(
    "/search",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}},
)
async def search_endpoint(file: UploadFile = File(...)):
    """Upload an image and find similar/duplicate images in the index."""
    content = await file.read()
    image = await asyncio.to_thread(_validate_image, file.filename or "unknown", content)
    result = await asyncio.to_thread(search, image)
    return SearchResponse(**result)


@router.post(
    "/search/text",
    response_model=TextSearchResponse,
    responses={400: {"model": ErrorResponse}},
)
async def search_text_endpoint(body: TextSearchRequest):
    """Search for images using a natural language text query via CLIP."""
    try:
        result = await asyncio.to_thread(search_by_text, body.query, body.k)
        return TextSearchResponse(**result)
    except Exception as e:
        logger.error("Text search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Index (single)
# ---------------------------------------------------------------------------
@router.post(
    "/index",
    response_model=IndexStatus,
    responses={400: {"model": ErrorResponse}},
)
async def index_endpoint(file: UploadFile = File(...)):
    """Upload a single image to the index."""
    content = await file.read()
    image = await asyncio.to_thread(_validate_image, file.filename or "unknown", content)
    result = await asyncio.to_thread(index_single, image, file.filename or "unknown", content)
    return IndexStatus(**result)


# ---------------------------------------------------------------------------
# Index (batch — from directory)
# ---------------------------------------------------------------------------
@router.post(
    "/index/batch",
    response_model=BatchIndexResponse | BatchIndexAsyncResponse,
    responses={400: {"model": ErrorResponse}},
)
async def index_batch_endpoint(
    body: BatchIndexRequest,
    async_mode: bool | None = Query(default=None, description="Override async mode"),
):
    """Index all images in a given directory on disk."""
    use_async = body.async_mode if async_mode is None else async_mode
    if use_async:
        try:
            task_id = index_batch_async(body.directory)
            return BatchIndexAsyncResponse(
                status="started",
                task_id=task_id,
                directory=body.directory,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        result = await asyncio.to_thread(index_batch, body.directory)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BatchIndexResponse(**result)


@router.get("/index/batch/status")
async def batch_status_endpoint():
    """Return live progress of the currently running batch index operation."""
    return get_batch_status()


@router.get(
    "/index/batch/status/{task_id}",
    response_model=BatchIndexAsyncResponse,
    responses={404: {"model": ErrorResponse}},
)
async def batch_task_status_endpoint(task_id: str):
    """Return status and progress for a specific background batch indexing task."""
    status = get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return BatchIndexAsyncResponse(**status)


@router.get("/index/rebuild/status")
async def rebuild_status_endpoint():
    """Return live progress of the currently running index rebuild operation."""
    return indexer.rebuild_status


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------
async def _sse_stream(
    get_state: Callable[[], dict],
    is_done: Callable[[dict], bool],
    interval: float = 0.3,
):
    """Generic SSE generator — yields state changes until done.

    Polls get_state() every *interval* seconds and yields an SSE event
    whenever the JSON-serialized state differs from the last sent value.
    Sends a final ``event: done`` when is_done(state) returns True.
    """
    last_payload = ""
    while True:
        await asyncio.sleep(interval)
        try:
            state = get_state()
        except Exception:
            continue

        payload = json.dumps(state, ensure_ascii=False)
        if payload == last_payload:
            continue  # unchanged — skip

        last_payload = payload
        yield f"data: {payload}\n\n"

        if is_done(state):
            yield "event: done\ndata: {}\n\n"
            return


# ---------------------------------------------------------------------------
# SSE endpoints
# ---------------------------------------------------------------------------
@router.get("/sync/status/stream")
async def sync_status_stream():
    """SSE stream of sync progress. Connects once, pushes updates."""
    return StreamingResponse(
        _sse_stream(
            get_state=get_sync_status,
            is_done=lambda s: not s["running"],
            interval=0.5,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/index/batch/status/stream")
async def batch_status_stream():
    """SSE stream of batch indexing progress."""
    return StreamingResponse(
        _sse_stream(
            get_state=get_batch_status,
            is_done=lambda s: not s["running"],
            interval=0.3,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/index/rebuild/status/stream")
async def rebuild_status_stream():
    """SSE stream of index rebuild progress."""
    return StreamingResponse(
        _sse_stream(
            get_state=lambda: indexer.rebuild_status,
            is_done=lambda s: not s["running"],
            interval=0.3,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# GPU toggle
# ---------------------------------------------------------------------------
@router.post("/index/gpu")
async def set_gpu_endpoint(body: dict = None):
    """Enable or disable GPU acceleration for the Faiss index.

    Body: {"enabled": true} or {"enabled": false}
    HNSW indices are CPU-only — GPU is not available for them.
    """
    body = body or {}
    enabled = body.get("enabled", True)
    return await asyncio.to_thread(indexer.set_gpu_enabled, enabled)


# ---------------------------------------------------------------------------
# Train index (IVF upgrade)
# ---------------------------------------------------------------------------
@router.post("/index/train")
async def train_index_endpoint():
    """Train/upgrade the Faiss index from Flat to IVF for faster search."""
    result = await asyncio.to_thread(indexer.train_index)
    return result


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Return current runtime configuration."""
    return ConfigResponse(
        faiss_index_type=settings.faiss_index_type,
        auto_upgrade_enabled=settings.faiss_auto_upgrade,
        auto_save_interval_s=settings.auto_save_interval,
        batch_size=settings.batch_size,
        top_k=settings.top_k,
        nprobe=settings.faiss_nprobe,
        hnsw_ef_search=settings.faiss_hnsw_ef_search,
        dhash_threshold=settings.dhash_threshold,
        phash_threshold=settings.phash_threshold,
        ssim_threshold=settings.ssim_threshold,
    )


@router.patch("/config", response_model=ConfigResponse)
async def update_config(body: ConfigUpdateRequest):
    """Update runtime configuration. Returns the full config after update.

    Special handling:
    - faiss_index_type change clears and rebuilds the index (destructive).
    - auto_save_interval_s change restarts the background save thread.
    """
    needs_rebuild = False
    restart_autosave = False

    if body.faiss_index_type is not None and body.faiss_index_type != settings.faiss_index_type:
        if body.faiss_index_type not in ("flat", "ivf_flat", "ivf_pq", "hnsw"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid index type: {body.faiss_index_type}. Must be flat, ivf_flat, ivf_pq, or hnsw.",
            )
        settings.faiss_index_type = body.faiss_index_type
        needs_rebuild = True

    if body.auto_upgrade_enabled is not None:
        settings.faiss_auto_upgrade = body.auto_upgrade_enabled

    if body.auto_save_interval_s is not None:
        if body.auto_save_interval_s < 0:
            raise HTTPException(status_code=400, detail="auto_save_interval_s must be >= 0")
        settings.auto_save_interval = body.auto_save_interval_s
        restart_autosave = True

    if body.batch_size is not None:
        if body.batch_size < 1:
            raise HTTPException(status_code=400, detail="batch_size must be >= 1")
        settings.batch_size = body.batch_size

    if body.top_k is not None:
        if body.top_k < 1:
            raise HTTPException(status_code=400, detail="top_k must be >= 1")
        settings.top_k = body.top_k

    if body.nprobe is not None:
        if body.nprobe < 1:
            raise HTTPException(status_code=400, detail="nprobe must be >= 1")
        settings.faiss_nprobe = body.nprobe

    if body.hnsw_ef_search is not None:
        if body.hnsw_ef_search < 1:
            raise HTTPException(status_code=400, detail="hnsw_ef_search must be >= 1")
        settings.faiss_hnsw_ef_search = body.hnsw_ef_search

    if body.dhash_threshold is not None:
        settings.dhash_threshold = body.dhash_threshold

    if body.phash_threshold is not None:
        settings.phash_threshold = body.phash_threshold

    if body.ssim_threshold is not None:
        settings.ssim_threshold = body.ssim_threshold

    # Rebuild index if type changed (preserves data)
    if needs_rebuild:
        logger.info("Index type changed to %s — rebuilding index", settings.faiss_index_type)
        await asyncio.to_thread(indexer.rebuild)

    # Restart auto-save with new interval
    if restart_autosave:
        await asyncio.to_thread(indexer.stop_auto_save)
        await asyncio.to_thread(indexer.start_auto_save)

    return await get_config()


# ---------------------------------------------------------------------------
# Delete index
# ---------------------------------------------------------------------------
@router.delete("/index")
async def clear_index_endpoint():
    """Clear the entire index (in-memory and on-disk)."""
    await asyncio.to_thread(indexer.clear)
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Browse index
# ---------------------------------------------------------------------------
@router.get("/index", response_model=IndexListResponse)
async def list_index_endpoint(
    page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
):
    """Paginated listing of all indexed images."""
    return await asyncio.to_thread(indexer.list_items, page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# Serve images
# ---------------------------------------------------------------------------
@router.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve an image file from the images directory (for preview thumbnails)."""
    file_path = settings.images_path / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
