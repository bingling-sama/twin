"""Twin — Two-stage image similarity search and deduplication system.

Usage:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run uvicorn twin.main:app --reload
"""

import logging
import os
import threading
from contextlib import asynccontextmanager

# Safe fallback — the Makefile sets these before launching.
# On Linux + conda, OMP_NUM_THREADS can be higher (Faiss k-means uses OpenMP).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from twin.api.routes import router
from twin.core.config import settings
from twin.models.clip_model import load as load_model
from twin.services.indexer import indexer
from twin.services.sync import sync_images_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("twin")

def _run_sync_background() -> None:
    """Run sync_images_dir in a background thread so the server is ready immediately.

    Progress is tracked inside sync.py's shared _sync_state; the
    GET /api/v1/sync/status endpoint reads it directly.
    """
    try:
        result = sync_images_dir()
        logger.info(
            "Background sync finished: %d total, %d indexed, %d skipped, %d failed",
            result["total"], result["indexed"],
            result["skipped"], result["failed"],
        )
    except Exception:
        logger.exception("Background sync failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + restore index. Shutdown: persist index."""
    logger.info("=== Twin starting up ===")
    settings.ensure_dirs()
    load_model(
        device=settings.device or None,
        model_name=settings.model_name,
        pretrained=settings.pretrained,
    )
    indexer._init_gpu()  # must precede load() to wrap index in GPU
    indexer.load()
    indexer.start_auto_save()

    if settings.sync_on_startup:
        sync_result = sync_images_dir()
        logger.info(
            "Sync result: %d total, %d indexed, %d skipped, %d failed",
            sync_result["total"], sync_result["indexed"],
            sync_result["skipped"], sync_result["failed"],
        )
    else:
        # Launch background sync so server is responsive immediately
        logger.info("Starting background sync (use TWIN_SYNC_ON_STARTUP=true to block)")
        threading.Thread(
            target=_run_sync_background, name="twin-sync", daemon=True
        ).start()

    logger.info(
        "=== Twin ready (indexed: %d, index: %s) ===",
        indexer.count, indexer.index_type_name,
    )
    yield
    logger.info("=== Twin shutting down ===")
    indexer.stop_auto_save()
    indexer.save()
    logger.info("=== Twin stopped ===")


app = FastAPI(
    title="Twin Image Search",
    description="Two-stage image similarity search with CLIP + Faiss + dHash",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
