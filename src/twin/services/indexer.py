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


def phash_to_binary_bytes(phash_hex: str) -> np.ndarray:
    """Convert 16-character hex pHash string to 8-byte uint8 numpy vector."""
    if not phash_hex or len(phash_hex) != 16:
        return np.zeros(8, dtype=np.uint8)
    try:
        raw = bytes.fromhex(phash_hex)
        return np.frombuffer(raw, dtype=np.uint8)
    except Exception:
        return np.zeros(8, dtype=np.uint8)


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

    DEFAULT_DATASOURCE = "skin"
    KNOWN_DATASOURCES = ("skin", "cape")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._indices: dict[str, faiss.Index | None] = {ds: None for ds in self.KNOWN_DATASOURCES}
        self._binary_indices: dict[str, faiss.IndexBinaryFlat | None] = {
            ds: None for ds in self.KNOWN_DATASOURCES
        }
        self._metadatas: dict[str, list[dict]] = {ds: [] for ds in self.KNOWN_DATASOURCES}
        self._dirty_flags: dict[str, bool] = {ds: False for ds in self.KNOWN_DATASOURCES}
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
    # Properties & Datasource Accessors (Thread-safe via _lock)
    # ------------------------------------------------------------------
    @property
    def _index(self) -> faiss.Index | None:
        with self._lock:
            return self._indices.get(self.DEFAULT_DATASOURCE)

    @_index.setter
    def _index(self, val: faiss.Index | None) -> None:
        with self._lock:
            self._indices[self.DEFAULT_DATASOURCE] = val

    @property
    def _binary_index(self) -> faiss.IndexBinaryFlat | None:
        with self._lock:
            return self._binary_indices.get(self.DEFAULT_DATASOURCE)

    @_binary_index.setter
    def _binary_index(self, val: faiss.IndexBinaryFlat | None) -> None:
        with self._lock:
            self._binary_indices[self.DEFAULT_DATASOURCE] = val

    @property
    def _metadata(self) -> list[dict]:
        with self._lock:
            if self.DEFAULT_DATASOURCE not in self._metadatas:
                self._metadatas[self.DEFAULT_DATASOURCE] = []
            return self._metadatas[self.DEFAULT_DATASOURCE]

    @_metadata.setter
    def _metadata(self, val: list[dict]) -> None:
        with self._lock:
            self._metadatas[self.DEFAULT_DATASOURCE] = val

    @property
    def _dirty(self) -> bool:
        with self._lock:
            return any(self._dirty_flags.values())

    @_dirty.setter
    def _dirty(self, val: bool) -> None:
        with self._lock:
            for ds in self.KNOWN_DATASOURCES:
                self._dirty_flags[ds] = val

    @property
    def count(self) -> int:
        """Total vectors across all registered datasources."""
        with self._lock:
            total = 0
            for idx in self._indices.values():
                if idx is not None:
                    total += idx.ntotal
            return total

    def get_count(self, datasource: str = DEFAULT_DATASOURCE) -> int:
        """Vector count for a specific datasource."""
        with self._lock:
            idx = self._indices.get(datasource)
            return idx.ntotal if idx is not None else 0

    @property
    def faiss_gpu_enabled(self) -> bool:
        """Whether the default index is running on GPU."""
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
        """Enable or disable GPU acceleration across all Faiss indices.

        Applies to all datasources (skin and cape) holding vectors.
        HNSW indices are CPU-only — GPU is skipped for them.
        """
        with self._lock:
            total_vectors = sum(
                idx.ntotal for idx in self._indices.values() if idx is not None
            )
            if total_vectors == 0:
                return {"status": "skipped", "reason": "index is empty"}

            if enabled and self._gpu_res is None:
                self._init_gpu()
            if enabled and self._gpu_res is None:
                return {
                    "status": "unavailable",
                    "reason": (
                        "No GPU resources available (faiss-gpu not installed "
                        "or CUDA unavailable)"
                    ),
                }

            results: dict[str, dict] = {}
            for ds in self.KNOWN_DATASOURCES:
                idx_obj = self._indices.get(ds)
                if idx_obj is None or idx_obj.ntotal == 0:
                    continue
                current_name = type(idx_obj).__name__
                currently_gpu = "GpuIndex" in current_name
                is_hnsw = "HNSW" in current_name

                if is_hnsw:
                    results[ds] = {
                        "status": "skipped",
                        "reason": "HNSW is CPU-only (GPU not supported by Faiss)",
                    }
                    continue

                if enabled and not currently_gpu:
                    try:
                        self._indices[ds] = faiss.index_cpu_to_gpu(
                            self._gpu_res, self._gpu_id, idx_obj
                        )
                        results[ds] = {
                            "status": "enabled",
                            "index_type": type(self._indices[ds]).__name__,
                        }
                    except Exception as e:
                        results[ds] = {"status": "error", "reason": str(e)}
                elif not enabled and currently_gpu:
                    try:
                        self._indices[ds] = faiss.index_gpu_to_cpu(idx_obj)
                        results[ds] = {
                            "status": "disabled",
                            "index_type": type(self._indices[ds]).__name__,
                        }
                    except Exception as e:
                        results[ds] = {"status": "error", "reason": str(e)}
                else:
                    state = "GPU" if currently_gpu else "CPU"
                    results[ds] = {"status": "unchanged", "reason": f"already on {state}"}

            settings.faiss_gpu = enabled
            return {
                "status": "enabled" if enabled else "disabled",
                "datasources": results,
                "index_type": type(self._index).__name__ if self._index is not None else "none",
            }

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------
    def _create_index(
        self, dim: int | None = None, datasource: str = DEFAULT_DATASOURCE
    ) -> faiss.Index:
        """Create a fresh index according to the configured type.

        When the index is empty (0 vectors) with faiss_index_type='ivf_flat'
        or 'ivf_pq', starts with IndexFlatL2 — IVF/PQ both require training
        data.  HNSW is created directly since it builds the graph incrementally.
        """
        if dim is None:
            from twin.services.embedding import get_embedding_dim

            dim = get_embedding_dim()

        idx_type = settings.faiss_index_type
        existing_idx = self._indices.get(datasource)
        n_existing = existing_idx.ntotal if existing_idx is not None else 0

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

    def _create_binary_index(self) -> faiss.IndexBinaryFlat:
        """Create a 64-bit IndexBinaryFlat for fast pHash Hamming search."""
        return faiss.IndexBinaryFlat(64)

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
    def _count_path_for(type_key: str, datasource: str | None = DEFAULT_DATASOURCE) -> Path:
        """Path to the freshness counter for a cached index type."""
        return settings._count_path_for(type_key, datasource=datasource)

    # ------------------------------------------------------------------
    # Internal save (caller MUST hold self._lock)
    # ------------------------------------------------------------------
    def _save_unlocked(self) -> None:
        """Persist index and metadata to disk atomically for all dirty datasources.
        Caller must hold _lock.

        Uses temp-file + atomic-rename to prevent corruption if the process is
        killed mid-write — the previous valid file survives until the new one
        is fully written.
        """
        import os as _os

        for ds in self.KNOWN_DATASOURCES:
            idx_obj = self._indices.get(ds)
            if idx_obj is None or idx_obj.ntotal == 0 or not self._dirty_flags.get(ds, False):
                continue

            n = idx_obj.ntotal
            faiss_path = settings._faiss_path_for(settings.faiss_index_type, datasource=ds)
            meta_path = settings._metadata_path_for(settings.faiss_index_type, datasource=ds)
            count_path = self._count_path_for(settings.faiss_index_type, datasource=ds)
            bin_path = settings._binary_faiss_path_for(datasource=ds)
            faiss_path.parent.mkdir(parents=True, exist_ok=True)

            # Safety check: never overwrite a valid index with fewer vectors.
            if faiss_path.exists():
                try:
                    existing = faiss.read_index(str(faiss_path))
                    if existing.ntotal > n:
                        logger.error(
                            "REFUSING to shrink index for %s from %d → %d vectors. "
                            "Delete %s manually if intentional.",
                            ds,
                            existing.ntotal,
                            n,
                            faiss_path,
                        )
                        continue
                except Exception:
                    pass  # existing file is corrupt — safe to overwrite

            faiss_tmp = str(faiss_path) + ".tmp"
            meta_tmp = str(meta_path) + ".tmp"
            count_tmp = str(count_path) + ".tmp"
            bin_tmp = str(bin_path) + ".tmp"

            try:
                cpu_idx = self._maybe_to_cpu(idx_obj)
                faiss.write_index(cpu_idx, faiss_tmp)
                meta_tmp_path = Path(meta_tmp)
                meta_tmp_path.write_text(
                    json.dumps(self._metadatas[ds], ensure_ascii=False, indent=2)
                )
                count_tmp_path = Path(count_tmp)
                count_tmp_path.write_text(str(n))

                # Binary pHash index persistence
                bin_idx = self._binary_indices.get(ds)
                if bin_idx is not None and bin_idx.ntotal > 0:
                    faiss.write_index_binary(bin_idx, bin_tmp)
                    _os.replace(bin_tmp, str(bin_path))

                # Atomic rename — this is the commit point
                _os.replace(faiss_tmp, str(faiss_path))
                _os.replace(meta_tmp, str(meta_path))
                _os.replace(count_tmp, str(count_path))

                self._dirty_flags[ds] = False
                logger.info("Saved %s index (%d vectors) to %s", ds, n, faiss_path)
            finally:
                for tmp in (faiss_tmp, meta_tmp, count_tmp, bin_tmp):
                    if _os.path.exists(tmp):
                        _os.unlink(tmp)

    # ------------------------------------------------------------------
    # Load / Save / Clear
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Restore index and metadata from disk, or create fresh indices for all known datasources.

        For each datasource:
          1. Configured type in datasource directory (e.g. data/skin/index.flat.faiss)
          2. Any other type that has saved files for that datasource
          3. For default datasource ('skin'), fall back to root-level legacy files
             (data/index.faiss)
          4. Fresh empty index
        """
        with self._lock:
            for ds in self.KNOWN_DATASOURCES:
                candidates: list[tuple[Path, Path]] = []

                # 1. Configured type for this datasource
                configured_faiss = settings._faiss_path_for(
                    settings.faiss_index_type, datasource=ds
                )
                configured_meta = settings._metadata_path_for(
                    settings.faiss_index_type, datasource=ds
                )
                candidates.append((configured_faiss, configured_meta))

                # 2. Any other type with saved files for this datasource
                others = []
                for t in ("flat", "ivf_flat", "ivf_pq", "hnsw"):
                    if t == settings.faiss_index_type:
                        continue
                    fp = settings._faiss_path_for(t, datasource=ds)
                    mp = settings._metadata_path_for(t, datasource=ds)
                    cp = self._count_path_for(t, datasource=ds)
                    if fp.exists() and mp.exists():
                        n = 0
                        if cp.exists():
                            try:
                                n = int(cp.read_text().strip())
                            except Exception:
                                pass
                        others.append((n, fp, mp))
                others.sort(key=lambda x: x[0], reverse=True)
                for _, fp, mp in others:
                    candidates.append((fp, mp))

                # 3. For default datasource ('skin'), check root-level legacy files
                #    for backwards compatibility
                if ds == self.DEFAULT_DATASOURCE:
                    root_others = []
                    for t in ("flat", "ivf_flat", "ivf_pq", "hnsw"):
                        fp = settings._faiss_path_for(t, datasource=None)
                        mp = settings._metadata_path_for(t, datasource=None)
                        cp = self._count_path_for(t, datasource=None)
                        if fp.exists() and mp.exists():
                            n = 0
                            if cp.exists():
                                try:
                                    n = int(cp.read_text().strip())
                                except Exception:
                                    pass
                            root_others.append((n, fp, mp))
                    root_others.sort(key=lambda x: x[0], reverse=True)
                    for _, fp, mp in root_others:
                        candidates.append((fp, mp))
                    candidates.append((settings.legacy_faiss_path, settings.legacy_metadata_path))

                logger.info(
                    "Searching for existing index for datasource '%s' (configured: %s) ...",
                    ds,
                    settings.faiss_index_type,
                )

                loaded = False
                for faiss_p, meta_p in candidates:
                    if faiss_p.exists() and meta_p.exists():
                        try:
                            cpu_idx = faiss.read_index(str(faiss_p))
                            meta_list = json.loads(meta_p.read_text())

                            if hasattr(cpu_idx, "nprobe"):
                                cpu_idx.nprobe = settings.faiss_nprobe
                            if hasattr(cpu_idx, "hnsw"):
                                cpu_idx.hnsw.efSearch = settings.faiss_hnsw_ef_search

                            self._indices[ds] = self._maybe_to_gpu(cpu_idx)
                            self._metadatas[ds] = meta_list
                            logger.info(
                                "  ✓ Loaded %s for datasource '%s' with %d vectors from %s",
                                type(cpu_idx).__name__,
                                ds,
                                cpu_idx.ntotal,
                                faiss_p,
                            )
                            # If loaded from legacy or root path, mark dirty so save() migrates it
                            if faiss_p.parent == settings.index_dir:
                                self._dirty_flags[ds] = True
                            loaded = True
                            break
                        except Exception as e:
                            logger.warning(
                                "  ✗ Corrupt %s for datasource '%s': %s", faiss_p.name, ds, e
                            )

                if not loaded:
                    logger.info("  No valid index found for datasource '%s' — starting fresh", ds)
                    self._indices[ds] = self._create_index(datasource=ds)
                    self._metadatas[ds] = []
                    self._dirty_flags[ds] = False

                # Load or rebuild binary pHash index
                bin_path = settings._binary_faiss_path_for(datasource=ds)
                bin_loaded = False
                if bin_path.exists():
                    try:
                        self._binary_indices[ds] = faiss.read_index_binary(str(bin_path))
                        bin_loaded = True
                        logger.info(
                            "  ✓ Loaded binary index for '%s' (%d items)",
                            ds,
                            self._binary_indices[ds].ntotal,
                        )
                    except Exception as e:
                        logger.warning("  ✗ Corrupt binary index for '%s': %s", ds, e)
                        self._binary_indices[ds] = None

                meta_len = len(self._metadatas.get(ds, []))
                bin_count = (
                    self._binary_indices[ds].ntotal
                    if self._binary_indices.get(ds) is not None
                    else 0
                )
                if not bin_loaded or bin_count != meta_len:
                    bin_idx = self._create_binary_index()
                    if meta_len > 0:
                        bin_vecs = np.stack(
                            [
                                phash_to_binary_bytes(m.get("phash", ""))
                                for m in self._metadatas[ds]
                            ]
                        )
                        bin_idx.add(bin_vecs)
                        self._dirty_flags[ds] = True
                    self._binary_indices[ds] = bin_idx
                    logger.info(
                        "  ✓ Initialized binary index for '%s' (%d items from metadata)",
                        ds,
                        bin_idx.ntotal,
                    )

    def save(self) -> None:
        """Persist index and metadata to disk (thread-safe, acquires lock)."""
        with self._lock:
            if not self._dirty:
                logger.debug("Skipping save — no dirty datasources")
                return
            self._save_unlocked()

    def clear(self, datasource: str | None = None) -> None:
        """Reset the index and metadata in memory and on disk.

        If datasource is None, resets all known datasources and cleans up legacy files.
        If datasource is specified, resets only the given datasource.
        """
        with self._lock:
            targets = [datasource] if datasource is not None else list(self.KNOWN_DATASOURCES)
            for ds in targets:
                self._indices[ds] = None
                self._indices[ds] = self._create_index(datasource=ds)
                self._binary_indices[ds] = self._create_binary_index()
                self._metadatas[ds] = []
                self._dirty_flags[ds] = False

                count_paths = [
                    self._count_path_for(t, datasource=ds)
                    for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")
                ]
                bin_p = settings._binary_faiss_path_for(datasource=ds)
                for p in (
                    settings.all_type_faiss_paths(datasource=ds)
                    + settings.all_type_metadata_paths(datasource=ds)
                    + count_paths
                    + [bin_p]
                ):
                    if p.exists():
                        p.unlink()

            if datasource is None or datasource == self.DEFAULT_DATASOURCE:
                legacy_count_paths = [
                    self._count_path_for(t, datasource=None)
                    for t in ("flat", "ivf_flat", "ivf_pq", "hnsw")
                ]
                legacy_bin_p = settings._binary_faiss_path_for(datasource=None)
                for p in (
                    settings.all_type_faiss_paths(datasource=None)
                    + settings.all_type_metadata_paths(datasource=None)
                    + legacy_count_paths
                    + [
                        settings.legacy_faiss_path,
                        settings.legacy_metadata_path,
                        legacy_bin_p,
                    ]
                ):
                    if p.exists():
                        p.unlink()

        logger.info("Index cleared (datasource=%s)", datasource or "all")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_item(self, vector: np.ndarray, meta: dict, datasource: str = DEFAULT_DATASOURCE) -> int:
        """Add a single vector with metadata. Returns the assigned ID."""
        with self._lock:
            idx_obj = self._indices.get(datasource)
            if idx_obj is None:
                idx_obj = self._create_index(datasource=datasource)
                self._indices[datasource] = idx_obj
                self._metadatas[datasource] = []

            idx = idx_obj.ntotal
            idx_obj.add(vector.reshape(1, -1).astype(np.float32))

            # Add to binary pHash index
            bin_idx = self._binary_indices.get(datasource)
            if bin_idx is None:
                bin_idx = self._create_binary_index()
                self._binary_indices[datasource] = bin_idx
            bin_vec = phash_to_binary_bytes(meta.get("phash", ""))
            bin_idx.add(bin_vec.reshape(1, 8))

            meta["id"] = idx
            meta["datasource"] = datasource
            self._metadatas[datasource].append(meta)
            logger.debug("Added item %d (%s): %s", idx, datasource, meta.get("filename", "?"))
            self._dirty_flags[datasource] = True
        return idx

    def add_items(
        self,
        vectors: np.ndarray,
        metas: list[dict],
        datasource: str = DEFAULT_DATASOURCE,
    ) -> list[int]:
        """Batch add vectors. Returns list of assigned IDs.

        If auto_upgrade is enabled and the index has enough vectors for IVF
        training, automatically converts Flat→IVF after adding.
        """
        with self._lock:
            idx_obj = self._indices.get(datasource)
            if idx_obj is None:
                idx_obj = self._create_index(datasource=datasource)
                self._indices[datasource] = idx_obj
                self._metadatas[datasource] = []

            start_id = idx_obj.ntotal
            idx_obj.add(vectors.astype(np.float32))

            # Add to binary pHash index
            bin_idx = self._binary_indices.get(datasource)
            if bin_idx is None:
                bin_idx = self._create_binary_index()
                self._binary_indices[datasource] = bin_idx
            bin_vecs = np.stack([phash_to_binary_bytes(m.get("phash", "")) for m in metas])
            bin_idx.add(bin_vecs)

            ids = []
            for i, meta in enumerate(metas):
                assigned = start_id + i
                meta["id"] = assigned
                meta["datasource"] = datasource
                self._metadatas[datasource].append(meta)
                ids.append(assigned)
            self._dirty_flags[datasource] = True

            # Auto-upgrade under RLock protection to prevent concurrency race
            if settings.faiss_auto_upgrade and settings.faiss_index_type in ("ivf_flat", "ivf_pq"):
                n = idx_obj.ntotal
                nlist = self._resolve_nlist(n)
                if n >= nlist * 39:  # faiss recommends >= 39*nlist for training
                    current_name = type(idx_obj).__name__
                    target_pq = settings.faiss_index_type == "ivf_pq"

                    # Tier 1: Flat → IVF / IVFPQ
                    if "Flat" in current_name and "IVF" not in current_name:
                        logger.info(
                            "Auto-triggering %s upgrade for %s (%d vectors, nlist=%d)",
                            settings.faiss_index_type,
                            datasource,
                            n,
                            nlist,
                        )
                        self.train_index(datasource=datasource)

                    # Tier 2: IVFFlat → IVFPQ
                    elif target_pq and "IVFFlat" in current_name:
                        logger.info(
                            "Auto-triggering IVFFlat → IVFPQ upgrade for %s (%d vectors, nlist=%d)",
                            datasource,
                            n,
                            nlist,
                        )
                        self.train_index(datasource=datasource)

        return ids

    def get_metadata(self, idx: int, datasource: str = DEFAULT_DATASOURCE) -> dict | None:
        """Retrieve metadata for a given index ID."""
        with self._lock:
            meta_list = self._metadatas.get(datasource, [])
            if 0 <= idx < len(meta_list):
                return meta_list[idx]
            return None

    def get_indexed_filenames(self, datasource: str | None = None) -> set[str]:
        """Return the set of filenames currently in the index."""
        with self._lock:
            if datasource is not None:
                return {m.get("filename", "") for m in self._metadatas.get(datasource, [])}
            filenames: set[str] = set()
            for meta_list in self._metadatas.values():
                filenames.update(m.get("filename", "") for m in meta_list)
            return filenames

    def list_items(
        self,
        page: int = 1,
        page_size: int = 50,
        datasource: str = DEFAULT_DATASOURCE,
    ) -> dict:
        """Paginated listing of indexed items."""
        with self._lock:
            meta_list = self._metadatas.get(datasource, [])
            total = len(meta_list)
            start = (page - 1) * page_size
            end = start + page_size
            items = [
                {
                    "id": m.get("id", i),
                    "filename": m.get("filename", "unknown"),
                    "dhash": m.get("dhash", ""),
                    "path": m.get("path", ""),
                    "datasource": m.get("datasource", datasource),
                }
                for i, m in enumerate(meta_list[start:end])
            ]
            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "datasource": datasource,
            }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: np.ndarray,
        k: int | None = None,
        datasource: str = DEFAULT_DATASOURCE,
    ) -> tuple[list[float], list[int]]:
        """Search for k nearest neighbors. Returns (distances, ids)."""
        if k is None:
            k = settings.top_k

        with self._lock:
            idx_obj = self._indices.get(datasource)
            if idx_obj is None:
                return [], []
            total = idx_obj.ntotal
            if total == 0:
                return [], []
            k = min(k, total)

            # Restore runtime params that may drift (not persisted by Faiss)
            if hasattr(idx_obj, "nprobe"):
                idx_obj.nprobe = settings.faiss_nprobe
            if hasattr(idx_obj, "hnsw"):
                idx_obj.hnsw.efSearch = settings.faiss_hnsw_ef_search

            distances, indices = idx_obj.search(query.reshape(1, -1).astype(np.float32), k)
            return distances[0].tolist(), indices[0].tolist()

    def search_binary(
        self,
        query_phash_hex: str,
        k: int | None = None,
        datasource: str = DEFAULT_DATASOURCE,
    ) -> tuple[list[int], list[int]]:
        """Search for k nearest neighbors using 64-bit binary pHash index.

        Returns (hamming_distances, ids).
        """
        if k is None:
            k = settings.binary_recall_k

        with self._lock:
            bin_idx = self._binary_indices.get(datasource)
            if bin_idx is None or bin_idx.ntotal == 0:
                return [], []
            k = min(k, bin_idx.ntotal)
            q_vec = phash_to_binary_bytes(query_phash_hex).reshape(1, 8)
            distances, indices = bin_idx.search(q_vec, k)
            return distances[0].tolist(), indices[0].tolist()

    # ------------------------------------------------------------------
    # IVF training
    # ------------------------------------------------------------------
    def train_index(self, datasource: str = DEFAULT_DATASOURCE) -> dict:
        """Train / upgrade the index for the given datasource toward the configured target type.

        Upgrade paths (all require faiss_auto_upgrade=True for automatic
        triggering; manual POST /index/train always works):

            IndexFlatL2  ──→  IndexIVFFlat   (target: ivf_flat)
            IndexFlatL2  ──→  IndexIVFPQ     (target: ivf_pq)
            IndexIVFFlat ──→  IndexIVFPQ     (target: ivf_pq, NEW)

        Returns a status dict suitable for the /api/v1/index/train response.
        """
        with self._lock:
            idx_obj = self._indices.get(datasource)
            if idx_obj is None or idx_obj.ntotal == 0:
                return {
                    "status": "skipped",
                    "reason": f"index for '{datasource}' is empty",
                    "datasource": datasource,
                }

            n = idx_obj.ntotal
            current_name = type(idx_obj).__name__

            # HNSW — no training
            if "HNSW" in current_name:
                return {
                    "status": "skipped",
                    "reason": "HNSW index does not require training",
                    "datasource": datasource,
                }

            # Check if we're already at the configured target
            target_pq = settings.faiss_index_type == "ivf_pq"
            target_ivf = settings.faiss_index_type in ("ivf_flat", "ivf_pq")

            if not target_ivf:
                return {
                    "status": "skipped",
                    "reason": (
                        f"faiss_index_type={settings.faiss_index_type}, not ivf_flat or ivf_pq"
                    ),
                    "datasource": datasource,
                }

            # Already at target?
            if not target_pq and "IVFFlat" in current_name:
                return {
                    "status": "skipped",
                    "reason": "already an IndexIVFFlat",
                    "datasource": datasource,
                }
            if target_pq and "IVFPQ" in current_name:
                return {
                    "status": "skipped",
                    "reason": "already an IndexIVFPQ",
                    "datasource": datasource,
                }

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
                    "datasource": datasource,
                }

            if from_ivf_flat:
                logger.info(
                    "Upgrading IVFFlat → IVFPQ for %s (nlist=%d, n_vectors=%d) ...",
                    datasource,
                    nlist,
                    n,
                )
            else:
                logger.info(
                    "Training %s index for %s (nlist=%d, n_vectors=%d) ...",
                    settings.faiss_index_type,
                    datasource,
                    nlist,
                    n,
                )

            # Extract all vectors from current index
            cpu_idx = self._maybe_to_cpu(idx_obj)
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
                    "IVFPQ training complete for %s (nlist=%d, m=%d, nprobe=%d, %d vectors)",
                    datasource,
                    nlist,
                    m,
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
                    "IVF training complete for %s (nlist=%d, nprobe=%d, %d vectors)",
                    datasource,
                    nlist,
                    settings.faiss_nprobe,
                    n,
                )

            # Swap specific datasource
            self._indices[datasource] = self._maybe_to_gpu(new_idx)
            self._dirty_flags[datasource] = True

            status = "upgraded" if from_ivf_flat else "trained"
            return {
                "status": status,
                "index_type": idx_label,
                "nlist": nlist,
                "nprobe": settings.faiss_nprobe,
                "n_vectors": n,
                "datasource": datasource,
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

    def _runtime_type_key(self, datasource: str = DEFAULT_DATASOURCE) -> str:
        """Map the current runtime index class to a cache key for datasource.

        IndexFlatL2 / GpuIndexFlat → "flat"
        IndexIVFFlat / GpuIndexIVFFlat → "ivf_flat"
        IndexIVFPQ / GpuIndexIVFPQ → "ivf_pq"
        IndexHNSWFlat → "hnsw"
        """
        idx = self._indices.get(datasource)
        if idx is None:
            return "none"
        name = type(idx).__name__.replace("GpuIndex", "")
        if "IVFPQ" in name:
            return "ivf_pq"
        if "IVFFlat" in name:
            return "ivf_flat"
        if "HNSW" in name:
            return "hnsw"
        if "Flat" in name:
            return "flat"
        return "unknown"

    def _save_for_type(self, type_key: str, datasource: str = DEFAULT_DATASOURCE) -> None:
        """Persist current index under a specific type key for datasource.

        Unlike _save_unlocked(), this writes to the path for *type_key*
        regardless of settings.faiss_index_type.  Also writes a .count file
        so rebuild() can check freshness before loading a cached index.
        Caller must hold _lock.
        """
        idx = self._indices.get(datasource)
        meta = self._metadatas.get(datasource, [])
        if idx is None or idx.ntotal == 0:
            return
        cpu_idx = self._maybe_to_cpu(idx)
        fpath = settings._faiss_path_for(type_key, datasource=datasource)
        mpath = settings._metadata_path_for(type_key, datasource=datasource)
        cpath = self._count_path_for(type_key, datasource=datasource)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(cpu_idx, str(fpath))
        mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        cpath.write_text(str(idx.ntotal))
        logger.debug(
            "Saved index as type=%s (%d vectors) for ds=%s", type_key, idx.ntotal, datasource
        )

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

    def _rebuild_datasource(self, datasource: str) -> dict:
        """Rebuild index for a specific datasource. Caller must hold _lock."""
        old_type_key = self._runtime_type_key(datasource=datasource)
        idx = self._indices.get(datasource)
        meta = self._metadatas.get(datasource, [])
        n = idx.ntotal if idx else 0

        # ── Phase: saving_current ──
        if idx is not None and n > 0 and old_type_key != "unknown":
            self._set_rebuild_phase(f"saving_current_{datasource}", n)
            self._save_for_type(old_type_key, datasource=datasource)
            logger.info(
                "Cached %s index (%d vectors) for ds=%s before switch",
                old_type_key,
                n,
                datasource,
            )

        target_type = settings.faiss_index_type
        target_faiss = settings._faiss_path_for(target_type, datasource=datasource)
        target_meta = settings._metadata_path_for(target_type, datasource=datasource)

        # ── Fast path: loading_cached (fresh or incrementally updatable) ──
        count_path = self._count_path_for(target_type, datasource=datasource)
        cached_count = 0
        if target_faiss.exists() and target_meta.exists() and count_path.exists():
            try:
                cached_count = int(count_path.read_text().strip())
            except Exception:
                cached_count = 0

        if cached_count > 0:
            self._set_rebuild_phase(f"loading_cached_{datasource}", n)
            try:
                cpu_idx = faiss.read_index(str(target_faiss))

                if hasattr(cpu_idx, "nprobe"):
                    cpu_idx.nprobe = settings.faiss_nprobe
                if hasattr(cpu_idx, "hnsw"):
                    cpu_idx.hnsw.efSearch = settings.faiss_hnsw_ef_search

                # ── Incremental: add missing vectors ──
                if cached_count < n and idx is not None:
                    new_count = n - cached_count
                    logger.info(
                        "Incrementally adding %d new vectors to cached %s index for ds=%s",
                        new_count,
                        target_type,
                        datasource,
                    )
                    active_cpu = self._maybe_to_cpu(idx)
                    new_vecs = active_cpu.reconstruct_n(cached_count, new_count)
                    cpu_idx.add(new_vecs.astype(np.float32))

                new_idx = self._maybe_to_gpu(cpu_idx)
                self._indices[datasource] = new_idx
                self._dirty_flags[datasource] = True
                new_name = type(new_idx).__name__.replace("GpuIndex", "")

                if cached_count < n:
                    # Save updated cache immediately (don't wait for auto-save)
                    self._save_for_type(target_type, datasource=datasource)
                    logger.info(
                        "Updated cached %s: %d → %d vectors for ds=%s",
                        new_name,
                        cached_count,
                        n,
                        datasource,
                    )
                else:
                    logger.info(
                        "Switched to cached %s (%d vectors) for ds=%s — instant",
                        new_name,
                        n,
                        datasource,
                    )

                return {
                    "datasource": datasource,
                    "status": "switched",
                    "index_type": new_name,
                    "n_vectors": new_idx.ntotal,
                    "cached": True,
                    "incremental": cached_count < n,
                    "added": max(0, n - cached_count),
                }
            except Exception as e:
                logger.warning(
                    "Cached %s index failed for ds=%s (%s), doing full rebuild",
                    target_type,
                    datasource,
                    e,
                )

        # ── Empty index ──
        if n == 0:
            new_idx = self._create_index()
            self._indices[datasource] = new_idx
            self._metadatas[datasource] = []
            self._dirty_flags[datasource] = False
            new_name = type(new_idx).__name__.replace("GpuIndex", "")
            logger.info("Rebuilt empty index as %s for ds=%s", new_name, datasource)
            return {
                "datasource": datasource,
                "status": "rebuilt",
                "index_type": new_name,
                "n_vectors": 0,
                "cached": False,
            }

        logger.info("Building %s index from %d vectors for ds=%s ...", target_type, n, datasource)

        # ── Phase: extracting vectors ──
        self._set_rebuild_phase(f"extracting_{datasource}", n)
        cpu_idx = self._maybe_to_cpu(idx)
        vectors = cpu_idx.reconstruct_n(0, n)
        old_metadata = list(meta)

        # ── Phase: building ──
        self._set_rebuild_phase(f"building_{datasource}", n)
        self._indices[datasource] = None
        new_idx = self._create_index(datasource=datasource)
        new_idx.add(vectors.astype(np.float32))
        self._indices[datasource] = new_idx
        self._metadatas[datasource] = old_metadata
        self._dirty_flags[datasource] = True

        new_name = type(new_idx).__name__.replace("GpuIndex", "")
        logger.info("Built %s with %d vectors for ds=%s", new_name, n, datasource)

        # ── Phase: training ──
        target_ivf = settings.faiss_index_type in ("ivf_flat", "ivf_pq")
        if settings.faiss_auto_upgrade and target_ivf:
            nlist = self._resolve_nlist(n)
            if n >= nlist * 39:
                cn = type(new_idx).__name__
                if "Flat" in cn and "IVF" not in cn:
                    self._set_rebuild_phase(f"training_{datasource}", n)
                    logger.info(
                        "Auto-training after rebuild (%d vectors, nlist=%d) for ds=%s",
                        n,
                        nlist,
                        datasource,
                    )
                    self.train_index(datasource=datasource)
                    new_idx = self._indices.get(datasource)
                    new_name = type(new_idx).__name__.replace("GpuIndex", "")
                    self._dirty_flags[datasource] = True

        # ── Phase: saving ──
        self._set_rebuild_phase(f"saving_{datasource}", n)
        self._save_for_type(settings.faiss_index_type, datasource=datasource)

        return {
            "datasource": datasource,
            "status": "rebuilt",
            "index_type": new_name,
            "n_vectors": n,
            "cached": False,
        }

    def rebuild(self, datasource: str | None = None) -> dict:
        """Switch to the current faiss_index_type, preserving all vectors across datasources.

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
                if datasource is not None:
                    res = self._rebuild_datasource(datasource)
                    self._set_rebuild_phase("done", res.get("n_vectors", 0))
                    return res

                # Rebuild all mounted/known datasources
                results = {}
                total_vectors = 0
                target_datasources = (
                    list(self._indices.keys()) if self._indices else list(self.KNOWN_DATASOURCES)
                )
                for ds in target_datasources:
                    res = self._rebuild_datasource(ds)
                    results[ds] = res
                    total_vectors += res.get("n_vectors", 0)

                self._set_rebuild_phase("done", total_vectors)
                primary_res = results.get(
                    self.DEFAULT_DATASOURCE, next(iter(results.values()), {})
                )
                return {
                    **primary_res,
                    "datasources": results,
                    "total_vectors": total_vectors,
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
