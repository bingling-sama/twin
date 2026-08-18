"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve relative paths against the project root (4 dirs up: config → core → twin → src → root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


class Settings(BaseSettings):
    """All configurable parameters with sensible defaults for MVP."""

    model_config = SettingsConfigDict(env_prefix="TWIN_", env_file=".env", extra="ignore")

    # --- Model ---
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    embedding_dim: int = 512
    device: str = ""  # 'cuda', 'mps', 'cpu', or '' for auto-detect

    # --- Faiss ---
    index_path: str = "data"
    images_dir: str = "data/images"
    faiss_index_type: str = "ivf_flat"  # "flat" | "ivf_flat" | "ivf_pq" | "hnsw"
    faiss_nlist: int = 0             # IVF centroids; 0 = auto (4 * sqrt(N))
    faiss_nprobe: int = 16           # IVF search-time probes; higher = more recall (was 8)
    faiss_auto_upgrade: bool = True  # auto-convert Flat→IVF when enough vectors (no-op for hnsw)
    faiss_gpu: bool = True           # try GPU Faiss if CUDA is available (ignored for hnsw)

    # --- PQ (Product Quantization, for ivf_pq) ---
    faiss_pq_m: int = 0              # sub-quantizers; 0 = auto (embedding_dim // 8). Must divide dim evenly
    faiss_pq_nbits: int = 8          # bits per PQ code (8 = 256 centroids per sub-space)

    # --- HNSW (graph-based index, no training needed) ---
    faiss_hnsw_m: int = 32                # graph degree (bi-directional links per node, 4-64)
    faiss_hnsw_ef_construction: int = 200  # build-time exploration depth (100-2000)
    faiss_hnsw_ef_search: int = 128        # search-time exploration depth (higher = more recall, was 64)

    # --- Search ---
    top_k: int = 100
    dhash_threshold: int = 10   # max Hamming distance for dHash duplicate
    phash_threshold: int = 12   # max Hamming distance for pHash duplicate
    ssim_threshold: float = 0.90  # min SSIM for structural duplicate
    ssim_size: int = 128  # resize dimension for SSIM comparison (128 = fast, 256 = precise)

    # --- Auto-save ---
    auto_save_interval: int = 120  # seconds between periodic saves; 0 = disable (was 300)

    # --- Performance ---
    batch_size: int = 64  # images per CLIP forward pass during batch indexing (was 32)
    sync_on_startup: bool = False  # auto-sync images dir on startup (blocks until done)

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    def ensure_dirs(self) -> None:
        """Create required directories. Call once at startup."""
        for p in (self.index_dir, self.images_path):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def index_dir(self) -> Path:
        return _resolve(self.index_path)

    @property
    def faiss_path(self) -> Path:
        """Faiss index file for the currently configured type."""
        return self._faiss_path_for(self.faiss_index_type)

    @property
    def metadata_path(self) -> Path:
        """Metadata file for the currently configured type."""
        return self._metadata_path_for(self.faiss_index_type)

    def _faiss_path_for(self, index_type: str) -> Path:
        return self.index_dir / f"index.{index_type}.faiss"

    def _metadata_path_for(self, index_type: str) -> Path:
        return self.index_dir / f"metadata.{index_type}.json"

    @property
    def legacy_faiss_path(self) -> Path:
        """Pre-multi-type fallback (for migration from old format)."""
        return self.index_dir / "index.faiss"

    @property
    def legacy_metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"

    @property
    def images_path(self) -> Path:
        return _resolve(self.images_dir)

    def all_type_faiss_paths(self) -> list[Path]:
        """Return paths for all known index types (for bulk cleanup)."""
        return [self._faiss_path_for(t) for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")]

    def all_type_metadata_paths(self) -> list[Path]:
        """Return paths for all known index types (for bulk cleanup)."""
        return [self._metadata_path_for(t) for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")]


# Module-level singleton — import this everywhere
settings = Settings()
