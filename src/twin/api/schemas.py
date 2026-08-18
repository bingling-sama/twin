"""Pydantic request/response models."""

from pydantic import BaseModel, Field


# --- Search ---
class SearchResultItem(BaseModel):
    id: int
    filename: str
    distance: float              # L2 distance in CLIP embedding space
    match_level: str             # "confirmed" | "suspected" | "none"
    stages_passed: int           # how many filter stages passed (0–4)
    dhash_distance: int = 999    # Hamming distance of dHash
    phash_distance: int = 999    # Hamming distance of pHash
    ssim_score: float = 0.0      # Structural similarity (1.0 = identical)
    dhash_hex: str = ""          # raw dHash hex string
    phash_hex: str = ""          # raw pHash hex string
    path: str = ""               # file path on server (empty for upload-only images)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    count: int
    query_time_ms: float
    stages: dict = Field(default_factory=dict)


# --- Text Search ---
class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search prompt", min_length=1)
    k: int | None = Field(default=None, description="Number of candidates to retrieve", ge=1, le=1000)


class TextSearchResultItem(BaseModel):
    id: int
    filename: str
    distance: float
    path: str = ""
    dhash_hex: str = ""
    phash_hex: str = ""


class TextSearchResponse(BaseModel):
    query: str
    results: list[TextSearchResultItem]
    count: int
    query_time_ms: float


# --- Batch Index Async ---
class BatchIndexAsyncResponse(BaseModel):
    status: str  # "started" | "running" | "completed" | "failed"
    task_id: str
    directory: str = ""
    total: int = 0
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    progress_pct: float = 0.0
    time_ms: float = 0.0
    error: str | None = None


# --- Index ---
class IndexStatus(BaseModel):
    status: str  # "indexed" | "already_exists"
    id: int
    filename: str


class BatchIndexRequest(BaseModel):
    directory: str = Field(..., description="Path to directory containing images")
    async_mode: bool = Field(default=False, description="Run in background thread without blocking HTTP response")


class BatchIndexResponse(BaseModel):
    status: str  # "completed"
    total: int
    indexed: int
    failed: int
    time_ms: float


# --- Health ---
class HealthResponse(BaseModel):
    status: str  # "ok"
    indexed_count: int
    model_loaded: bool
    index_type: str = "unknown"  # e.g. "IndexFlatL2", "IndexIVFFlat"

    # CLIP runtime
    device: str = ""               # "cuda" | "cpu" | "mps"
    model_name: str = ""           # "ViT-B-32", etc.
    gpu_name: str = ""             # GPU device name if CUDA, e.g. "NVIDIA GeForce RTX 4060"

    # Faiss runtime
    faiss_gpu_enabled: bool = False     # whether current index lives on GPU
    faiss_index_type: str = ""          # configured type from TWIN_FAISS_INDEX_TYPE

    # Configuration snapshot
    batch_size: int = 0
    auto_upgrade_enabled: bool = False
    auto_save_interval_s: int = 0


# --- Browse index ---
class IndexedItem(BaseModel):
    id: int
    filename: str
    dhash: str = ""
    path: str = ""


class IndexListResponse(BaseModel):
    items: list[IndexedItem]
    total: int
    page: int
    page_size: int


# --- Sync status ---
class SyncStatusResponse(BaseModel):
    running: bool
    total_files: int
    indexed_files: int       # indexed + skipped (both count toward progress)
    skipped_files: int
    failed_files: int
    progress_pct: float      # 0.0–100.0
    elapsed_ms: float
    eta_ms: float
    rate_img_per_s: float = 0.0  # current throughput


# --- Runtime config ---
class ConfigUpdateRequest(BaseModel):
    """Partial update — only provided fields are changed. Others stay as-is."""
    faiss_index_type: str | None = None   # "flat" | "ivf_flat" | "ivf_pq" | "hnsw"
    auto_upgrade_enabled: bool | None = None
    auto_save_interval_s: int | None = None  # 0 = disable
    batch_size: int | None = None
    top_k: int | None = None
    nprobe: int | None = None
    hnsw_ef_search: int | None = None
    dhash_threshold: int | None = None
    phash_threshold: int | None = None
    ssim_threshold: float | None = None


class ConfigResponse(BaseModel):
    faiss_index_type: str
    auto_upgrade_enabled: bool
    auto_save_interval_s: int
    batch_size: int
    top_k: int
    nprobe: int
    hnsw_ef_search: int
    dhash_threshold: int
    phash_threshold: int
    ssim_threshold: float


# --- Error ---
class ErrorResponse(BaseModel):
    detail: str
