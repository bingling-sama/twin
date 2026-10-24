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
    model_type: str = "clip"  # "clip" | "dinov2"
    model_name: str = "ViT-B-32"
    dinov2_model_name: str = "vit_small_patch14_dinov2"
    pretrained: str = "openai"
    embedding_dim: int = 512
    device: str = ""  # 'cuda', 'mps', 'cpu', or '' for auto-detect

    # --- Faiss ---
    index_path: str = "data"
    images_dir: str = "data/images"
    faiss_index_type: str = "ivf_flat"  # "flat" | "ivf_flat" | "ivf_pq" | "hnsw"
    faiss_nlist: int = 0  # IVF centroids; 0 = auto (4 * sqrt(N))
    faiss_nprobe: int = 16  # IVF search-time probes; higher = more recall (was 8)
    faiss_auto_upgrade: bool = True  # auto-convert Flat→IVF when enough vectors (no-op for hnsw)
    faiss_gpu: bool = True  # try GPU Faiss if CUDA is available (ignored for hnsw)

    # --- PQ (Product Quantization, for ivf_pq) ---
    faiss_pq_m: int = 0  # sub-quantizers; 0 = auto (embedding_dim // 8). Must divide dim evenly
    faiss_pq_nbits: int = 8  # bits per PQ code (8 = 256 centroids per sub-space)

    # --- HNSW (graph-based index, no training needed) ---
    faiss_hnsw_m: int = 32  # graph degree (bi-directional links per node, 4-64)
    faiss_hnsw_ef_construction: int = 200  # build-time exploration depth (100-2000)
    faiss_hnsw_ef_search: int = 128  # search-time exploration depth (higher = more recall, was 64)

    # --- Search ---
    top_k: int = 100
    rotation_invariant: bool = False  # MC UV orientation is fixed; 0° locked for efficiency
    binary_recall_k: int = 1000  # candidate count from fast binary pHash retrieval
    verification_budget: int = 200  # maximum candidate budget for expensive SSIM verification
    ahash_threshold: int = 10  # max Hamming distance for aHash pre-filter
    dhash_threshold: int = 10  # max Hamming distance for dHash duplicate
    phash_threshold: int = 12  # max Hamming distance for pHash duplicate
    ssim_threshold: float = 0.90  # min SSIM for structural duplicate
    ssim_size: int = 128  # resize dimension for SSIM comparison (128 = fast, 256 = precise)
    ssim_device: str = "auto"  # 'auto' | 'cuda' | 'cpu'

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
    def data_dir(self) -> Path:
        """Alias for index_dir."""
        return self.index_dir

    @property
    def faiss_path(self) -> Path:
        """Faiss index file for the currently configured type (default datasource)."""
        return self._faiss_path_for(self.faiss_index_type, datasource="skin")

    @property
    def metadata_path(self) -> Path:
        """Metadata file for the currently configured type (default datasource)."""
        return self._metadata_path_for(self.faiss_index_type, datasource="skin")

    def _base_dir_for(self, datasource: str | None = None) -> Path:
        base = self.index_dir / datasource if datasource else self.index_dir
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _faiss_path_for(self, index_type: str, datasource: str | None = None) -> Path:
        return self._base_dir_for(datasource) / f"index.{index_type}.faiss"

    def _metadata_path_for(self, index_type: str, datasource: str | None = None) -> Path:
        return self._base_dir_for(datasource) / f"metadata.{index_type}.json"

    def _count_path_for(self, index_type: str, datasource: str | None = None) -> Path:
        return self._base_dir_for(datasource) / f"index.{index_type}.count"

    @property
    def binary_faiss_path(self) -> Path:
        """Binary Faiss index file (default datasource)."""
        return self._binary_faiss_path_for(datasource="skin")

    def _binary_faiss_path_for(self, datasource: str | None = None) -> Path:
        return self._base_dir_for(datasource) / "index.binary.faiss"

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

    def all_type_faiss_paths(self, datasource: str | None = None) -> list[Path]:
        """Return paths for all known index types (for bulk cleanup)."""
        return [
            self._faiss_path_for(t, datasource)
            for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")
        ]

    def all_type_metadata_paths(self, datasource: str | None = None) -> list[Path]:
        """Return paths for all known index types (for bulk cleanup)."""
        return [
            self._metadata_path_for(t, datasource)
            for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")
        ]



# Module-level singleton — import this everywhere
settings = Settings()
