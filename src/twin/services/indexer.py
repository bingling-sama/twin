"""Faiss index wrapper with metadata persistence and thread safety.

Supports:
  - IndexFlatL2 (exhaustive, accurate, default for <100K vectors)
  - IndexIVFFlat (clustered, fast, trainable for >100K vectors)
  - IndexIVFPQ (clustered + product quantization, ~32× memory compression)
  - IndexHNSWFlat (graph-based, no training needed, excellent search QPS)
  - GPU acceleration when faiss-gpu is installed and CUDA is available.
"""

import json
import logging
import math
import threading
from pathlib import Path

import faiss
import numpy as np

from twin.core.config import settings

logger = logging.getLogger(__name__)


class Indexer:
    """Thread-safe Faiss index with metadata persistence and optional GPU.

    Lifecycle:
      1. load()       — restore from disk or create fresh
      2. add_item(s)  — write vectors (marks dirty)
      3. search()     — read nearest neighbours
      4. train_index()— convert Flat→IVF when enough data
      5. save()       — persist to disk
    """

    # Known rebuild phases (used by GET /api/v1/index/rebuild/status)
    REBUILD_PHASES = (
        "saving_current",  # caching old index to disk before switch
        "loading_cached",  # loading pre-built target index from disk
        "extracting",  # extracting vectors from current index
        "building",  # creating new index + adding vectors
        "training",  # training IVF/PQ clusters
        "saving",  # persisting new index to disk
        "done",  # complete
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []
        self._dirty: bool = False
        self._stop_event = threading.Event()
        self._auto_save_thread: threading.Thread | None = None

        # GPU resources — populated lazily
        self._gpu_res: faiss.GpuResources | None = None
        self._gpu_id: int = 0

        # Rebuild progress (polled by GET /api/v1/index/rebuild/status)
        self._rebuild_state: dict = {
            "running": False,
            "phase": "done",
            "n_vectors": 0,
            "started_at": None,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        with self._lock:
            return self._index.ntotal if self._index is not None else 0

    @property
    def faiss_gpu_enabled(self) -> bool:
        """Whether the current index is running on GPU."""
        with self._lock:
            if self._index is None:
                return False
            name = type(self._index).__name__
            return "GpuIndex" in name

    @property
    def index_type_name(self) -> str:
        """Human-readable index type for health/debug endpoints."""
        with self._lock:
            if self._index is None:
                return "none"
            raw = type(self._index).__name__
            # GpuIndex wrappers strip the prefix
            return raw.replace("GpuIndex", "")

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------
    def _init_gpu(self) -> None:
        """Try to acquire GPU resources. No-op on CPU-only systems.

        Checks CUDA availability via torch first (already a dependency), then
        probes for faiss-gpu.  When CUDA hardware is present but faiss-gpu is
        not installed, logs a warning with install instructions.
        """
        if not settings.faiss_gpu:
            logger.info("GPU Faiss disabled via TWIN_FAISS_GPU=0")
            return
        if self._gpu_res is not None:
            return  # already initialised

        # Detect CUDA hardware via torch (already imported as a dependency)
        cuda_hardware = False
        try:
            import torch

            cuda_hardware = torch.cuda.is_available()
        except ImportError:
            pass

        if not cuda_hardware:
            logger.info("No CUDA devices detected, using CPU Faiss")
            self._gpu_res = None
            return

        try:
            n_gpus = faiss.get_num_gpus()
            if n_gpus > 0:
                self._gpu_res = faiss.StandardGpuResources()
                logger.info("GPU Faiss enabled (%d GPU(s) available)", n_gpus)
                return
        except AttributeError:
            logger.warning(
                "CUDA is available but faiss-gpu is not installed. "
                "Install it via conda: conda install faiss-gpu -c pytorch -c nvidia"
            )
        except RuntimeError as e:
            logger.warning("faiss-gpu init failed (%s), falling back to CPU", e)

        self._gpu_res = None

    def _maybe_to_gpu(self, idx: faiss.Index) -> faiss.Index:
        """Wrap *idx* in a GPU index if GPU resources are available."""
        if self._gpu_res is not None and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                return faiss.index_cpu_to_gpu(self._gpu_res, self._gpu_id, idx)
            except Exception as e:
                logger.warning("Cannot move index to GPU (%s), staying on CPU", e)
                self._gpu_res = None  # don't keep trying
        return idx

    def _maybe_to_cpu(self, idx: faiss.Index) -> faiss.Index:
        """Bring *idx* back to CPU if it currently lives on GPU."""
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(idx)
            except Exception:
                pass
        return idx

    # ------------------------------------------------------------------
    # GPU toggle
    # ------------------------------------------------------------------
    def set_gpu_enabled(self, enabled: bool) -> dict:
        """Enable or disable GPU acceleration for the current Faiss index.

        When enabling: wraps the CPU index in a GpuIndex (requires faiss-gpu
        and CUDA).  When disabling: unwraps back to CPU.

        HNSW indices are CPU-only — enabling GPU is a no-op for them.
        Flat/IVF indices benefit most from GPU acceleration.

        Also updates settings.faiss_gpu so future rebuilds honour the choice.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return {"status": "skipped", "reason": "index is empty"}

            current_name = type(self._index).__name__
            currently_gpu = "GpuIndex" in current_name
            is_hnsw = "HNSW" in current_name

            if is_hnsw:
                return {
                    "status": "skipped",
                    "reason": "HNSW is CPU-only (GPU not supported by Faiss)",
                }

            if enabled and not currently_gpu:
                # Try to init GPU resources
                if self._gpu_res is None:
                    self._init_gpu()
                if self._gpu_res is not None:
                    try:
                        self._index = faiss.index_cpu_to_gpu(
                            self._gpu_res, self._gpu_id, self._index
                        )
                        settings.faiss_gpu = True
                        logger.info("GPU Faiss enabled — index now on GPU")
                        return {
                            "status": "enabled",
                            "index_type": type(self._index).__name__,
                        }
                    except Exception as e:
                        return {"status": "error", "reason": str(e)}
                else:
                    return {
                        "status": "unavailable",
                        "reason": (
                            "No GPU resources available (faiss-gpu not installed "
                            "or CUDA unavailable)"
                        ),
                    }

            elif not enabled and currently_gpu:
                try:
                    self._index = faiss.index_gpu_to_cpu(self._index)
                    settings.faiss_gpu = False
                    logger.info("GPU Faiss disabled — index moved to CPU")
                    return {
                        "status": "disabled",
                        "index_type": type(self._index).__name__,
                    }
                except Exception as e:
                    return {"status": "error", "reason": str(e)}

            else:
                state = "GPU" if currently_gpu else "CPU"
                return {"status": "unchanged", "reason": f"already on {state}"}

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------
    def _create_index(self, dim: int | None = None) -> faiss.Index:
        """Create a fresh index according to the configured type.

        When the index is empty (0 vectors) with faiss_index_type='ivf_flat'
        or 'ivf_pq', starts with IndexFlatL2 — IVF/PQ both require training
        data.  HNSW is created directly since it builds the graph incrementally.
        """
        if dim is None:
            from twin.services.embedding import get_embedding_dim

            dim = get_embedding_dim()

        idx_type = settings.faiss_index_type
        n_existing = self._index.ntotal if self._index is not None else 0

        if idx_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, settings.faiss_hnsw_m)
            idx.hnsw.efConstruction = settings.faiss_hnsw_ef_construction
            logger.info(
                "Created new IndexHNSWFlat (dim=%d, M=%d, efConstruction=%d)",
                dim,
                settings.faiss_hnsw_m,
                settings.faiss_hnsw_ef_construction,
            )
            # HNSW is CPU-only — faiss does not support GPU HNSW
            if self._gpu_res is not None:
                logger.info("HNSW index is CPU-only (GPU wrapping skipped)")
            return idx

        if n_existing > 0 and idx_type in ("ivf_flat", "ivf_pq"):
            nlist = self._resolve_nlist(n_existing)
            quantizer = faiss.IndexFlatL2(dim)
            if idx_type == "ivf_pq":
                m = self._resolve_pq_m(dim)
                idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, settings.faiss_pq_nbits)
                logger.info(
                    "Created new IndexIVFPQ (dim=%d, nlist=%d, m=%d, nbits=%d)",
                    dim,
                    nlist,
                    m,
                    settings.faiss_pq_nbits,
                )
            else:
                idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
                logger.info(
                    "Created new IndexIVFFlat (dim=%d, nlist=%d, nprobe=%d)",
                    dim,
                    nlist,
                    settings.faiss_nprobe,
                )
            idx.nprobe = settings.faiss_nprobe
        else:
            idx = faiss.IndexFlatL2(dim)
            logger.info("Created new IndexFlatL2 (dim=%d)", dim)

        # Try GPU wrapping
        idx = self._maybe_to_gpu(idx)
        return idx

    @staticmethod
    def _resolve_nlist(n: int) -> int:
        """Compute nlist for IVF index."""
        if settings.faiss_nlist > 0:
            return max(1, settings.faiss_nlist)
        if n == 0:
            return 100  # sensible default for an empty index that might be trained later
        # Rule of thumb: 4 * sqrt(N), clamped to [1, 65536]
        return max(1, min(65536, int(4 * math.sqrt(n))))

    @staticmethod
    def _resolve_pq_m(dim: int) -> int:
        """Compute m (number of sub-quantizers) for Product Quantization.

        The embedding dimension must be evenly divisible by m.
        Default: dim // 8, yielding 8-dimensional sub-vectors.
        """
        if settings.faiss_pq_m > 0:
            return settings.faiss_pq_m
        # Default: 8-dimensional sub-vectors
        return max(1, dim // 8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_path_for(type_key: str) -> Path:
        """Path to the freshness counter for a cached index type."""
        return settings.index_dir / f"index.{type_key}.count"

    # ------------------------------------------------------------------
    # Internal save (caller MUST hold self._lock)
    # ------------------------------------------------------------------
    def _save_unlocked(self) -> None:
        """Persist index and metadata to disk atomically. Caller must hold _lock.

        Uses temp-file + atomic-rename to prevent corruption if the process is
        killed mid-write — the previous valid file survives until the new one
        is fully written.
        """
        import os as _os

        if self._index is None or self._index.ntotal == 0:
            return

        faiss_path = settings.faiss_path
        meta_path = settings.metadata_path
        count_path = self._count_path_for(settings.faiss_index_type)
        n = self._index.ntotal

        # Safety check: never overwrite a valid index with fewer vectors.
        # If the existing file has MORE vectors than we're about to save,
        # something is wrong — refuse and keep the larger one.
        if faiss_path.exists():
            try:
                existing = faiss.read_index(str(faiss_path))
                if existing.ntotal > n:
                    logger.error(
                        "REFUSING to shrink index from %d → %d vectors. "
                        "Delete %s manually if intentional.",
                        existing.ntotal,
                        n,
                        faiss_path,
                    )
                    return
            except Exception:
                pass  # existing file is corrupt — safe to overwrite

        # Write to temp files first, then atomically rename.
        # If we crash during write, the temp files are orphaned but the
        # previous valid files survive.
        faiss_tmp = str(faiss_path) + ".tmp"
        meta_tmp = str(meta_path) + ".tmp"
        count_tmp = str(count_path) + ".tmp"

        try:
            cpu_idx = self._maybe_to_cpu(self._index)
            faiss.write_index(cpu_idx, faiss_tmp)
            meta_tmp_path = Path(meta_tmp)
            meta_tmp_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2))
            count_tmp_path = Path(count_tmp)
            count_tmp_path.write_text(str(n))

            # Atomic rename — this is the commit point
            _os.replace(faiss_tmp, str(faiss_path))
            _os.replace(meta_tmp, str(meta_path))
            _os.replace(count_tmp, str(count_path))

            self._dirty = False
            logger.info("Saved index (%d vectors) to %s", n, faiss_path)
        finally:
            # Clean up any leftover temp files
            for tmp in (faiss_tmp, meta_tmp, count_tmp):
                if _os.path.exists(tmp):
                    _os.unlink(tmp)

    # ------------------------------------------------------------------
    # Load / Save / Clear
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Restore index and metadata from disk, or create a fresh index.

        Search order:
          1. Configured type (e.g. data/index.ivf_flat.faiss)
          2. Any other type that has saved files (pick the one with most vectors)
          3. Legacy paths (data/index.faiss) for migration from old format
          4. Fresh empty index
        """
        with self._lock:
            # Build candidate list: configured type first, then all others sorted
            # by vector count (largest first), then legacy.
            candidates: list[tuple[Path, Path]] = []

            # 1. Configured type
            candidates.append((settings.faiss_path, settings.metadata_path))

            # 2. Any other type with saved files, sorted by vector count desc
            others = []
            for t in ("flat", "ivf_flat", "ivf_pq", "hnsw"):
                if t == settings.faiss_index_type:
                    continue
                fp = settings._faiss_path_for(t)
                mp = settings._metadata_path_for(t)
                cp = self._count_path_for(t)
                if fp.exists() and mp.exists():
                    n = 0
                    if cp.exists():
                        try:
                            n = int(cp.read_text().strip())
                        except Exception:
                            pass
                    others.append((n, fp, mp))
            others.sort(key=lambda x: x[0], reverse=True)  # most vectors first
            for _, fp, mp in others:
                candidates.append((fp, mp))

            # 3. Legacy
            candidates.append((settings.legacy_faiss_path, settings.legacy_metadata_path))

            # Diagnostic: log what we're looking for
            logger.info(
                "Searching for existing index (configured type: %s) ...", settings.faiss_index_type
            )

            # Try each candidate
            for faiss_p, meta_p in candidates:
                faiss_exists = faiss_p.exists()
                meta_exists = meta_p.exists()
                logger.info(
                    "  Checking %s (faiss=%s size=%d, meta=%s)",
                    faiss_p.name,
                    faiss_exists,
                    faiss_p.stat().st_size if faiss_exists else 0,
                    meta_exists,
                )

                if faiss_exists and meta_exists:
                    try:
                        cpu_idx = faiss.read_index(str(faiss_p))
                        self._metadata = json.loads(meta_p.read_text())

                        # Check vector count matches metadata
                        n_faiss = cpu_idx.ntotal
                        n_meta = len(self._metadata)
                        if n_faiss != n_meta:
                            logger.warning(
                                "  Vector count mismatch: faiss=%d, metadata=%d — using metadata",
                                n_faiss,
                                n_meta,
                            )

                        # Restore runtime params not persisted by Faiss
                        if hasattr(cpu_idx, "nprobe"):
                            cpu_idx.nprobe = settings.faiss_nprobe
                        if hasattr(cpu_idx, "hnsw"):
                            cpu_idx.hnsw.efSearch = settings.faiss_hnsw_ef_search

                        self._index = self._maybe_to_gpu(cpu_idx)
                        logger.info(
                            "  ✓ Loaded %s with %d vectors from %s",
                            type(self._index).__name__,
                            self._index.ntotal,
                            faiss_p,
                        )
                        # Migrate legacy files to type-specific on next save
                        if faiss_p == settings.legacy_faiss_path:
                            self._dirty = True
                        return
                    except Exception as e:
                        logger.warning("  ✗ Corrupt — %s: %s", faiss_p.name, e)

            logger.warning(
                "  No valid index found — starting fresh (checked %d candidates)",
                len(candidates),
            )
            self._index = self._create_index()
            self._metadata = []

    def save(self) -> None:
        """Persist index and metadata to disk (thread-safe, acquires lock)."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                logger.debug("Skipping save — empty index")
                return
            self._save_unlocked()

    def clear(self) -> None:
        """Reset the index and metadata in memory and on disk."""
        with self._lock:
            self._index = None
            self._index = self._create_index()
            self._metadata = []
            self._dirty = False
            count_paths = [self._count_path_for(t) for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")]
            for p in (
                settings.all_type_faiss_paths()
                + settings.all_type_metadata_paths()
                + count_paths
                + [settings.legacy_faiss_path, settings.legacy_metadata_path]
            ):
                if p.exists():
                    p.unlink()
        logger.info("Index cleared")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_item(self, vector: np.ndarray, meta: dict) -> int:
        """Add a single vector with metadata. Returns the assigned ID."""
        with self._lock:
            idx = self._index.ntotal if self._index is not None else 0
            if self._index is not None:
                self._index.add(vector.reshape(1, -1).astype(np.float32))
            meta["id"] = idx
            self._metadata.append(meta)
            logger.debug("Added item %d: %s", idx, meta.get("filename", "?"))
            self._dirty = True
        return idx

    def add_items(self, vectors: np.ndarray, metas: list[dict]) -> list[int]:
        """Batch add vectors. Returns list of assigned IDs.

        If auto_upgrade is enabled and the index has enough vectors for IVF
        training, automatically converts Flat→IVF after adding.
        """
        with self._lock:
            if self._index is None:
                return []
            start_id = self._index.ntotal
            self._index.add(vectors.astype(np.float32))
            ids = []
            for i, meta in enumerate(metas):
                assigned = start_id + i
                meta["id"] = assigned
                self._metadata.append(meta)
                ids.append(assigned)
            self._dirty = True

        # Auto-upgrade outside the lock to avoid holding it during training
        if settings.faiss_auto_upgrade and settings.faiss_index_type in ("ivf_flat", "ivf_pq"):
            n = self.count
            nlist = self._resolve_nlist(n)
            if n >= nlist * 39:  # faiss recommends >= 39*nlist for training
                current_name = type(self._index).__name__
                target_pq = settings.faiss_index_type == "ivf_pq"

                # Tier 1: Flat → IVF / IVFPQ
                if "Flat" in current_name and "IVF" not in current_name:
                    logger.info(
                        "Auto-triggering %s upgrade (%d vectors, nlist=%d)",
                        settings.faiss_index_type,
                        n,
                        nlist,
                    )
                    self.train_index()

                # Tier 2: IVFFlat → IVFPQ
                elif target_pq and "IVFFlat" in current_name:
                    logger.info(
                        "Auto-triggering IVFFlat → IVFPQ upgrade (%d vectors, nlist=%d)",
                        n,
                        nlist,
                    )
                    self.train_index()

        return ids

    def get_metadata(self, idx: int) -> dict | None:
        """Retrieve metadata for a given index ID."""
        with self._lock:
            if 0 <= idx < len(self._metadata):
                return self._metadata[idx]
            return None

    def get_indexed_filenames(self) -> set[str]:
        """Return the set of filenames currently in the index."""
        with self._lock:
            return {m.get("filename", "") for m in self._metadata}

    def list_items(self, page: int = 1, page_size: int = 50) -> dict:
        """Paginated listing of indexed items."""
        with self._lock:
            total = len(self._metadata)
            start = (page - 1) * page_size
            end = start + page_size
            items = [
                {
                    "id": m.get("id", i),
                    "filename": m.get("filename", "unknown"),
                    "dhash": m.get("dhash", ""),
                    "path": m.get("path", ""),
                }
                for i, m in enumerate(self._metadata[start:end])
            ]
            return {"items": items, "total": total, "page": page, "page_size": page_size}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int | None = None) -> tuple[list[float], list[int]]:
        """Search for k nearest neighbors. Returns (distances, ids)."""
        if k is None:
            k = settings.top_k

        with self._lock:
            if self._index is None:
                return [], []
            total = self._index.ntotal
            if total == 0:
                return [], []
            k = min(k, total)

            # Restore runtime params that may drift (not persisted by Faiss)
            if hasattr(self._index, "nprobe"):
                self._index.nprobe = settings.faiss_nprobe
            if hasattr(self._index, "hnsw"):
                self._index.hnsw.efSearch = settings.faiss_hnsw_ef_search

            distances, indices = self._index.search(query.reshape(1, -1).astype(np.float32), k)
            return distances[0].tolist(), indices[0].tolist()

    # ------------------------------------------------------------------
    # IVF training
    # ------------------------------------------------------------------
    def train_index(self) -> dict:
        """Train / upgrade the current index toward the configured target type.

        Upgrade paths (all require faiss_auto_upgrade=True for automatic
        triggering; manual POST /index/train always works):

            IndexFlatL2  ──→  IndexIVFFlat   (target: ivf_flat)
            IndexFlatL2  ──→  IndexIVFPQ     (target: ivf_pq)
            IndexIVFFlat ──→  IndexIVFPQ     (target: ivf_pq, NEW)

        Returns a status dict suitable for the /api/v1/index/train response.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return {"status": "skipped", "reason": "index is empty"}

            n = self._index.ntotal
            current_name = type(self._index).__name__

            # HNSW — no training
            if "HNSW" in current_name:
                return {"status": "skipped", "reason": "HNSW index does not require training"}

            # Check if we're already at the configured target
            target_pq = settings.faiss_index_type == "ivf_pq"
            target_ivf = settings.faiss_index_type in ("ivf_flat", "ivf_pq")

            if not target_ivf:
                return {
                    "status": "skipped",
                    "reason": (
                        f"faiss_index_type={settings.faiss_index_type}, "
                        "not ivf_flat or ivf_pq"
                    ),
                }

            # Already at target?
            if not target_pq and "IVFFlat" in current_name:
                return {"status": "skipped", "reason": "already an IndexIVFFlat"}
            if target_pq and "IVFPQ" in current_name:
                return {"status": "skipped", "reason": "already an IndexIVFPQ"}

            # Determine the transition
            from_ivf_flat = "IVFFlat" in current_name

            nlist = self._resolve_nlist(n)
            if n < nlist * 39:
                return {
                    "status": "skipped",
                    "reason": f"need >= {nlist * 39} vectors to train (have {n})",
                    "nlist": nlist,
                    "n_vectors": n,
                    "min_required": nlist * 39,
                }

            if from_ivf_flat:
                logger.info("Upgrading IVFFlat → IVFPQ (nlist=%d, n_vectors=%d) ...", nlist, n)
            else:
                logger.info(
                    "Training %s index (nlist=%d, n_vectors=%d) ...",
                    settings.faiss_index_type,
                    nlist,
                    n,
                )

            # Extract all vectors from current index
            cpu_idx = self._maybe_to_cpu(self._index)
            vectors = cpu_idx.reconstruct_n(0, n)

            # Build and train new index
            from twin.services.embedding import get_embedding_dim

            dim = cpu_idx.d if cpu_idx is not None else get_embedding_dim()
            quantizer = faiss.IndexFlatL2(dim)

            if target_pq:
                m = self._resolve_pq_m(dim)
                new_idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, settings.faiss_pq_nbits)
                new_idx.nprobe = settings.faiss_nprobe
                new_idx.train(vectors)
                new_idx.add(vectors)
                idx_label = "IndexIVFPQ"
                logger.info(
                    "IVFPQ training complete (nlist=%d, m=%d, nbits=%d, nprobe=%d, %d vectors)",
                    nlist,
                    m,
                    settings.faiss_pq_nbits,
                    settings.faiss_nprobe,
                    n,
                )
            else:
                new_idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
                new_idx.nprobe = settings.faiss_nprobe
                new_idx.train(vectors)
                new_idx.add(vectors)
                idx_label = "IndexIVFFlat"
                logger.info(
                    "IVF training complete (nlist=%d, nprobe=%d, %d vectors)",
                    nlist,
                    settings.faiss_nprobe,
                    n,
                )

            # Swap
            self._index = self._maybe_to_gpu(new_idx)
            self._dirty = True

            status = "upgraded" if from_ivf_flat else "trained"
            return {
                "status": status,
                "index_type": idx_label,
                "nlist": nlist,
                "nprobe": settings.faiss_nprobe,
                "n_vectors": n,
            }

    def upgrade_to_ivf(self) -> dict:
        """Public alias — same as train_index()."""
        return self.train_index()

    # ------------------------------------------------------------------
    # Index rebuild (type switch without data loss)
    # ------------------------------------------------------------------
    @property
    def rebuild_status(self) -> dict:
        """Return a snapshot of the current/last rebuild operation."""
        import time

        s = dict(self._rebuild_state)
        if s["running"] and s["started_at"] is not None:
            s["elapsed_ms"] = (time.perf_counter() - s["started_at"]) * 1000
        return s

    def _runtime_type_key(self) -> str:
        """Map the current runtime index class to a cache key.

        IndexFlatL2 / GpuIndexFlat → "flat"
        IndexIVFFlat / GpuIndexIVFFlat → "ivf_flat"
        IndexIVFPQ / GpuIndexIVFPQ → "ivf_pq"
        IndexHNSWFlat → "hnsw"
        """
        if self._index is None:
            return "none"
        name = type(self._index).__name__.replace("GpuIndex", "")
        if "IVFPQ" in name:
            return "ivf_pq"
        if "IVFFlat" in name:
            return "ivf_flat"
        if "HNSW" in name:
            return "hnsw"
        if "Flat" in name:
            return "flat"
        return "unknown"

    def _save_for_type(self, type_key: str) -> None:
        """Persist current index under a specific type key.

        Unlike _save_unlocked(), this writes to the path for *type_key*
        regardless of settings.faiss_index_type.  Also writes a .count file
        so rebuild() can check freshness before loading a cached index.
        Caller must hold _lock.
        """
        if self._index is None or self._index.ntotal == 0:
            return
        cpu_idx = self._maybe_to_cpu(self._index)
        faiss.write_index(cpu_idx, str(settings._faiss_path_for(type_key)))
        settings._metadata_path_for(type_key).write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2)
        )
        self._count_path_for(type_key).write_text(str(self._index.ntotal))
        logger.debug("Saved index as type=%s (%d vectors)", type_key, self._index.ntotal)

    def _set_rebuild_phase(self, phase: str, n_vectors: int = 0) -> None:
        """Update shared rebuild progress (caller may hold _lock)."""
        import time

        self._rebuild_state.update(
            running=(phase != "done"),
            phase=phase,
            n_vectors=n_vectors,
            started_at=time.perf_counter()
            if phase != "done" and self._rebuild_state.get("started_at") is None
            else self._rebuild_state.get("started_at"),
        )

    def rebuild(self) -> dict:
        """Switch to the current faiss_index_type, preserving all vectors.

        Fast path: if the target type has a valid saved index on disk, load it
        directly (sub-second for typical sizes).

        Slow path: extract all vectors from the current index, build a fresh
        index of the target type, train if IVF, add vectors, and save to disk.

        Updates self._rebuild_state at each phase so the
        GET /api/v1/index/rebuild/status endpoint can report live progress.

        Returns a status dict with the new index type and vector count.
        """

        try:
            with self._lock:
                old_type_key = self._runtime_type_key()
                n = self._index.ntotal if self._index else 0

                # ── Phase: saving_current ──
                if self._index is not None and n > 0 and old_type_key != "unknown":
                    self._set_rebuild_phase("saving_current", n)
                    self._save_for_type(old_type_key)
                    logger.info("Cached %s index (%d vectors) before switch", old_type_key, n)

                target_type = settings.faiss_index_type
                target_faiss = settings._faiss_path_for(target_type)
                target_meta = settings._metadata_path_for(target_type)

                # ── Fast path: loading_cached (fresh or incrementally updatable) ──
                count_path = self._count_path_for(target_type)
                cached_count = 0
                if target_faiss.exists() and target_meta.exists() and count_path.exists():
                    try:
                        cached_count = int(count_path.read_text().strip())
                    except Exception:
                        cached_count = 0

                if cached_count > 0:
                    self._set_rebuild_phase("loading_cached", n)
                    try:
                        cpu_idx = faiss.read_index(str(target_faiss))

                        if hasattr(cpu_idx, "nprobe"):
                            cpu_idx.nprobe = settings.faiss_nprobe
                        if hasattr(cpu_idx, "hnsw"):
                            cpu_idx.hnsw.efSearch = settings.faiss_hnsw_ef_search

                        # ── Incremental: add missing vectors ──
                        if cached_count < n:
                            new_count = n - cached_count
                            logger.info(
                                "Incrementally adding %d new vectors to cached %s index",
                                new_count,
                                target_type,
                            )
                            # Extract only new vectors from active index (IDs cached_count .. n-1)
                            active_cpu = self._maybe_to_cpu(self._index)
                            new_vecs = active_cpu.reconstruct_n(cached_count, new_count)
                            cpu_idx.add(new_vecs.astype(np.float32))

                        self._index = self._maybe_to_gpu(cpu_idx)
                        self._metadata = list(self._metadata)  # use current full metadata
                        self._dirty = True
                        new_name = type(self._index).__name__.replace("GpuIndex", "")

                        if cached_count < n:
                            # Save updated cache immediately (don't wait for auto-save)
                            self._save_for_type(target_type)
                            logger.info(
                                "Updated cached %s: %d → %d vectors", new_name, cached_count, n
                            )
                        else:
                            logger.info("Switched to cached %s (%d vectors) — instant", new_name, n)

                        self._set_rebuild_phase("done", self._index.ntotal)
                        return {
                            "status": "switched",
                            "index_type": new_name,
                            "n_vectors": self._index.ntotal,
                            "cached": True,
                            "incremental": cached_count < n,
                            "added": max(0, n - cached_count),
                        }
                    except Exception as e:
                        logger.warning(
                            "Cached %s index failed (%s), doing full rebuild", target_type, e
                        )

                # ── Empty index ──
                if n == 0:
                    self._index = None
                    self._index = self._create_index()
                    self._metadata = []
                    self._dirty = False
                    new_name = type(self._index).__name__.replace("GpuIndex", "")
                    self._set_rebuild_phase("done", 0)
                    logger.info("Rebuilt empty index as %s", new_name)
                    return {
                        "status": "rebuilt",
                        "index_type": new_name,
                        "n_vectors": 0,
                        "cached": False,
                    }

                logger.info("Building %s index from %d vectors ...", target_type, n)

                # ── Phase: extracting vectors ──
                self._set_rebuild_phase("extracting", n)
                cpu_idx = self._maybe_to_cpu(self._index)
                vectors = cpu_idx.reconstruct_n(0, n)
                old_metadata = list(self._metadata)

                # ── Phase: building ──
                self._set_rebuild_phase("building", n)
                self._index = None
                self._index = self._create_index()
                self._index.add(vectors.astype(np.float32))
                self._metadata = old_metadata
                self._dirty = True

                new_name = type(self._index).__name__.replace("GpuIndex", "")
                logger.info("Built %s with %d vectors", new_name, n)

            # ── Phase: training (outside lock) ──
            if settings.faiss_auto_upgrade and settings.faiss_index_type in ("ivf_flat", "ivf_pq"):
                nlist = self._resolve_nlist(n)
                if n >= nlist * 39:
                    cn = type(self._index).__name__
                    if "Flat" in cn and "IVF" not in cn:
                        self._set_rebuild_phase("training", n)
                        logger.info("Auto-training after rebuild (%d vectors, nlist=%d)", n, nlist)
                        self.train_index()
                        new_name = type(self._index).__name__.replace("GpuIndex", "")
                        self._dirty = True

            # ── Phase: saving ──
            self._set_rebuild_phase("saving", n)
            with self._lock:
                self._save_for_type(settings.faiss_index_type)

            self._set_rebuild_phase("done", n)
            return {
                "status": "rebuilt",
                "index_type": new_name,
                "n_vectors": n,
                "cached": False,
            }
        except Exception:
            self._set_rebuild_phase("done", 0)
            raise

    # ------------------------------------------------------------------
    # Auto-save background thread
    # ------------------------------------------------------------------
    def _auto_save_loop(self) -> None:
        """Periodically persist the index if dirty. Runs in a background thread."""
        interval = settings.auto_save_interval
        if interval <= 0:
            return

        logger.info("Auto-save started (interval=%ds)", interval)
        while not self._stop_event.wait(interval):
            if self._dirty:
                self.save()
                logger.debug("Auto-save completed")

    def start_auto_save(self) -> None:
        """Launch the periodic auto-save background thread."""
        if settings.auto_save_interval <= 0:
            logger.info("Auto-save disabled (TWIN_AUTO_SAVE_INTERVAL=0)")
            return
        if self._auto_save_thread is not None and self._auto_save_thread.is_alive():
            logger.warning("Auto-save thread already running")
            return
        self._stop_event.clear()
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop, name="twin-auto-save", daemon=True
        )
        self._auto_save_thread.start()

    def stop_auto_save(self) -> None:
        """Signal the background thread to stop and join."""
        if self._auto_save_thread is None:
            return
        self._stop_event.set()
        self._auto_save_thread.join(timeout=5)
        if self._auto_save_thread.is_alive():
            logger.warning("Auto-save thread did not stop within 5s")
        self._auto_save_thread = None


# Module-level singleton
indexer = Indexer()
