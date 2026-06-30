# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies (Python + frontend) — CPU-only (portable)
# Uses faiss-cpu from PyPI via the [cpu] optional dep group.
make install-cpu

# Install dependencies with GPU Faiss (conda, CUDA hosts)
# Requires conda env 'twin-gpu' with faiss-gpu pre-installed.
# Does NOT install faiss-cpu — expects faiss-gpu from conda.
make install

# Start both backend + frontend
make dev

# Run all tests
make test

# Run a single test file
OMP_NUM_THREADS=1 uv run pytest tests/test_hasher.py -v

# Lint
make lint

# Format
make format
```

## Project Flow

This section walks through every major flow in the system end-to-end.

### 1. Startup (`main.py` lifespan)

```
make dev  →  uvicorn twin.main:app

1. main.py: os.environ.setdefault("OMP_NUM_THREADS", "1")
             os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
             ↑ Must run BEFORE any import of torch or faiss

2. lifespan startup:
   ├─ settings.ensure_dirs()
   │   └─ Creates data/ and data/images/ if missing
   │
   ├─ clip_model.load(device, model_name, pretrained)
   │   ├─ _get_device(): torch.cuda.is_available()? → "cuda"
   │   │                torch.backends.mps.is_available()? → "mps"
   │   │                else → "cpu"
   │   ├─ open_clip.create_model_and_transforms(model_name, pretrained)
   │   ├─ model.to(device), model.eval()
   │   └─ Idempotent: if _model is already set, return immediately
   │
   ├─ indexer.load()
   │   ├─ Check data/index.faiss + data/metadata.json exist?
   │   │   ├─ Yes → faiss.read_index() + json.loads()
   │   │   └─ No (or corrupt) → _create_index() (IndexFlatL2, IndexHNSWFlat, or IndexIVFFlat per config)
   │   └─ Acquires _lock for the entire operation
   │
   ├─ sync_images_dir()
   │   ├─ Scan settings.images_path for image files
   │   ├─ Query indexer.get_indexed_filenames() → set of already-indexed names
   │   ├─ Compute diff: files not yet indexed
   │   ├─ For each batch (size = TWIN_BATCH_SIZE, default 64):
   │   │   ├─ utils.image.load_images(paths) → PIL list
   │   │   ├─ embedding.compute_embeddings(imgs) → batch CLIP → (32, 512)
   │   │   ├─ hasher.compute_dhashes(imgs) → ThreadPoolExecutor → hex strings
   │   │   ├─ hasher.compute_phashes(imgs) → ThreadPoolExecutor → hex strings
   │   │   └─ indexer.add_items(vectors, metas) → batch Faiss insert
   │   └─ Fallback: if batch fails, _index_single_from_disk() one-by-one
   │   └─ Updates _sync_state at each batch → GET /api/v1/sync/status reads live progress + ETA
   │
   ├─ indexer.start_auto_save()
   │   └─ Spawns daemon thread: every TWIN_AUTO_SAVE_INTERVAL sec (default 120)
   │       check _dirty flag → if True, indexer.save()
   │
   └─ Log: "=== Twin ready (indexed: N) ==="

3. lifespan shutdown (on SIGTERM / Ctrl+C):
   ├─ indexer.stop_auto_save() → signal + join(5s)
   ├─ indexer.save() → final persist (acquires lock, writes .faiss + .json)
   └─ Log: "=== Twin stopped ==="
```

### 2. Search Flow (`POST /api/v1/search`)

```
Client uploads query.jpg
  │
  ├─ routes.search_endpoint()
  │   ├─ file.file.read() → bytes
  │   ├─ _validate_image(filename, bytes)
  │   │   ├─ Check extension ∈ {.jpg, .jpeg, .png, .webp, .bmp}
  │   │   ├─ Image.open(BytesIO(bytes))
  │   │   ├─ img.load()  ← decodes pixels, raises on corruption
  │   │   └─ img.convert("RGB")
  │   └─ search(image) → dict → SearchResponse(**dict)
  │
  └─ services/search.py: search(image)
      │
      ├─ [Stage 0: Prepare]
      │   ├─ query_vec = compute_embedding(image)
      │   │   └─ clip_model.encode_image() → torch → L2-normalize → (512,) float32
      │   ├─ query_dhash = compute_dhash(image) → imagehash.dhash() → 16-char hex
      │   └─ query_phash = compute_phash(image) → imagehash.phash() → 16-char hex
      │
      ├─ [Stage 1: Semantic Recall]
      │   ├─ distances, ids = indexer.search(query_vec, k=50)
      │   │   └─ faiss.IndexFlatL2 / IndexIVFFlat / IndexHNSWFlat.search() → L2 nearest neighbors
      │   ├─ Build candidate pool: for each (dist, id) → load metadata
      │   └─ t1 = now; stages["faiss"] = {in:0, out:N, elapsed_ms: ...}
      │
      ├─ [Stage 2: dHash Filter]
      │   ├─ For each candidate:
      │   │   └─ _passes_hash(query_dhash, cand_dhash, threshold=10)
      │   │       └─ int(h1,16) ^ int(h2,16) → .bit_count() → dist ≤ 10?
      │   ├─ dhash_survivors = [c for c in candidates if passed]
      │   └─ If 0 survivors → return _build_response (all "none" tier)
      │
      ├─ [Stage 3: pHash Filter]
      │   ├─ For each surviving candidate:
      │   │   └─ _passes_hash(query_phash, cand_phash, threshold=12)
      │   ├─ phash_survivors = [c for c in dhash_survivors if passed]
      │   └─ If 0 survivors → return _build_response (dHash survivors "suspected")
      │
      ├─ [Stage 4: SSIM Filter]
      │   ├─ For each surviving candidate (parallel via simple loop):
      │   │   ├─ Load candidate image from disk (if path exists)
      │   │   └─ hasher.compute_ssim(query_img, cand_img)
      │   │       ├─ Resize both to 256×256 grayscale
      │   │       └─ skimage.metrics.structural_similarity(a, b)
      │   ├─ confirmed = [c for c in phash_survivors if SSIM ≥ 0.90]
      │   └─ stages["ssim"] = {in, out, elapsed_ms}
      │
      └─ [Finalize]
          ├─ _assign_match_level(stages_passed)
          │   ├─ ≥3 → "confirmed"
          │   ├─ ≥1 → "suspected"
          │   └─ 0   → "none"
          ├─ _final_sort: stages_passed DESC, dHash ASC, L2 ASC
          └─ _build_response → {results, count, query_time_ms, stages}
```

### 3. Index Flow (Single Upload)

```
POST /api/v1/index  (file=@photo.jpg)

routes.index_endpoint()
  ├─ file.file.read() → bytes
  ├─ _validate_image() → RGB PIL Image
  └─ index_service.index_single(image, filename, content)
      ├─ Dedup: filename already in indexer.get_indexed_filenames()?
      │   └─ Yes → return {"status": "already_exists", ...}
      ├─ Persist: settings.images_path / filename → .write_bytes(content)
      ├─ embedding.compute_embedding(image) → (512,) float32
      ├─ hasher.compute_dhash(image) → hex string
      ├─ hasher.compute_phash(image) → hex string
      ├─ indexer.add_item(vector, {filename, path, dhash, phash})
      │   ├─ Acquire _lock
      │   ├─ _index.add(vector.reshape(1, -1))
      │   ├─ _metadata.append(meta)
      │   ├─ _dirty = True
      │   └─ Release _lock
      └─ Return IndexStatus
```

### 4. Index Flow (Batch — from directory)

```
POST /api/v1/index/batch  {"directory": "/home/user/images"}

routes.index_batch_endpoint()
  └─ index_service.index_batch(directory)
      ├─ Validate: directory.is_dir()
      ├─ Scan: iter_image_files() → sorted list of image Paths
      ├─ For each chunk of TWIN_BATCH_SIZE images:
      │   ├─ utils.image.load_images(paths) → (valid_imgs, ok_paths, failed)
      │   ├─ embedding.compute_embeddings(imgs)
      │   │   └─ clip_model.encode_images() → single GPU forward pass
      │   ├─ hasher.compute_dhashes(imgs) → ThreadPoolExecutor
      │   ├─ hasher.compute_phashes(imgs) → ThreadPoolExecutor
      │   ├─ indexer.add_items(vectors, metas) → batch insert
      │   └─ Fallback: if batch fails → _index_single_from_disk() per image
      └─ Return BatchIndexResponse {total, indexed, failed, time_ms}
```

### 5. Indexer Auto-Save Flow

```
start_auto_save()
  └─ Spawn daemon thread: twin-auto-save

_auto_save_loop():
  while not _stop_event.wait(TWIN_AUTO_SAVE_INTERVAL):
      if self._dirty:
          self.save()           ← acquires _lock
          └─ _save_unlocked()  ← writes .faiss + .json
              └─ _dirty = False

Write operations (add_item, add_items) set _dirty = True but do NOT save.
The auto-save thread is the sole trigger for periodic persistence.
Final save on shutdown is explicit: stop_auto_save() then save().

Clear() resets _dirty = False and deletes .faiss + .json files.
```

### 6. Frontend Flow

```
Browser loads http://localhost:5173
  │
  ├─ Vite dev server proxies /api/v1 → localhost:8000
  │
  ├─ App.tsx mounts → default tab: "search"
  │   ├─ Search tab: drag-drop zone + results grid + zoomable preview overlay
  │   ├─ Browse tab: paginated thumbnail grid (50/page) + click preview
  │   └─ Manage tab: inline health status + index controls
  │
  ├─ Search flow (frontend):
  │   1. User drops/clicks file → preview shows immediately (onChange)
  │   2. Click "Search" → api/searchImage(file) → POST /api/v1/search
  │   3. Response renders: summary (count + ms) + stage pipeline + result cards
  │   4. Click card → fullscreen overlay with ZoomImage (wheel zoom, drag pan)
  │   5. Overlay shows: dimensions, path, dHash hex, pHash hex, all metrics
  │
  └─ Browse flow:
      1. On mount: api/listIndex(1, 50) → GET /api/v1/index?page=1&page_size=50
      2. Images loaded lazily (loading="lazy") from /api/v1/images/{filename}
      3. Pagination: ← Prev / Page N / Next →
      4. Click thumbnail → overlay with large preview
```

### 7. Module Call Graph

```
main.py (lifespan)
  ├── config.py::settings.ensure_dirs()
  ├── clip_model.py::load(device, model_name, pretrained)
  ├── indexer.py::indexer.load()
  ├── sync.py::sync_images_dir()
  │     ├── indexer.py::get_indexed_filenames()
  │     ├── utils/image.py::load_images()
  │     ├── embedding.py::compute_embeddings()
  │     │     └── clip_model.py::encode_images()
  │     ├── hasher.py::compute_dhashes() / compute_phashes()
  │     └── indexer.py::add_items()
  └── indexer.py::indexer.start_auto_save()

routes.py (HTTP handlers)
  ├── [Search]  → services/search.py::search()
  │     ├── embedding.py::compute_embedding()
  │     │     └── clip_model.py::encode_image()
  │     ├── hasher.py::compute_dhash() / compute_phash()
  │     ├── indexer.py::search()
  │     ├── utils/image.py::load_image()   (for SSIM candidates)
  │     └── hasher.py::compute_ssim()
  ├── [Index]   → services/index_service.py::index_single()
  │     ├── indexer.py::get_indexed_filenames()
  │     ├── embedding.py::compute_embedding()
  │     ├── hasher.py::compute_dhash() / compute_phash()
  │     └── indexer.py::add_item()
  ├── [Batch]   → services/index_service.py::index_batch()
  │     ├── utils/image.py::iter_image_files() / load_images()
  │     ├── embedding.py::compute_embeddings()
  │     ├── hasher.py::compute_dhashes() / compute_phashes()
  │     └── indexer.py::add_items()
  ├── [List]    → indexer.py::indexer.list_items()
  ├── [Health]  → indexer.py::count + clip_model.py::is_loaded()
  ├── [Clear]   → indexer.py::indexer.clear()
  ├── [Images]  → FileResponse(settings.images_path / filename)
  └── [Sync Status] → services/sync.py::get_sync_status()

sync.py (startup)
  ├── utils/image.py::iter_image_files() / load_images()
  ├── embedding.py::compute_embeddings()
  ├── hasher.py::compute_dhashes() / compute_phashes()
  ├── indexer.py::add_items()
  ├── index_service.py::_index_single_from_disk()  (fallback reuse)
  └── get_sync_status() → thread-safe progress snapshot (elapsed + ETA)
```

## Architecture

### Architecture Layers

```
HTTP Layer     api/routes.py  ──→  8 thin endpoints, delegates to services
               api/schemas.py ──→  Pydantic v2 validation
                    │
Orchestration   search.py      ──→  2-stage + 4-filter funnel
               index_service.py ──→  index_single(), index_batch() workflows
               sync.py        ──→  startup auto-sync from images dir
                    │
Computation     embedding.py   ──→  CLIP → 512d float32 vectors (single + batch)
               hasher.py      ──→  dHash, pHash, SSIM, Hamming distance
               indexer.py     ──→  Faiss IndexFlatL2 / IndexIVFFlat / IndexHNSWFlat + metadata + auto-save
                    │
Domain          clip_model.py  ──→  CLIP singleton, CUDA/MPS/CPU auto-detect
               config.py      ──→  TWIN_ env vars, path resolution
               utils/image.py ──→  shared image I/O utilities
```

### Component Map

| Layer | Module | Role |
|-------|--------|------|
| **HTTP** | `api/routes.py` | 8 endpoints — thin handlers, 0 business logic. Delegates to `search()`, `index_service`, `indexer`, `get_sync_status()` |
| HTTP | `api/schemas.py` | Pydantic v2 models: SearchResultItem, SearchResponse, IndexStatus, BatchIndexResponse, etc. |
| **Service** | `services/search.py` | Orchestrator: CLIP → Faiss → dHash → pHash → SSIM funnel. Returns tiered results + per-stage timing |
| Service | `services/index_service.py` | Encapsulates index workflows: `index_single()`, `index_batch()`, `_index_single_from_disk()`. Shared by routes and sync |
| Service | `services/sync.py` | Startup sync: scans images dir, batch-indexes files not yet in index. Tracks progress in thread-safe `_sync_state` (exposed via `get_sync_status()`). Reuses `_index_single_from_disk()` for fallback |
| Service | `services/embedding.py` | `compute_embedding()` + `compute_embeddings()` — thin wrapper around clip_model |
| Service | `services/hasher.py` | dHash, pHash (imagehash), SSIM (skimage), Hamming distance. Batch+parallel variants |
| Service | `services/indexer.py` | Faiss IndexFlatL2 / IndexIVFFlat / IndexHNSWFlat + JSON metadata, thread-safe (threading.RLock), auto-save daemon thread with dirty flag |
| **Domain** | `models/clip_model.py` | Singleton. `load()` idempotent, `encode_image()` + `encode_images()`. Auto-detects CUDA > MPS > CPU |
| Domain | `core/config.py` | `BaseSettings` with `TWIN_` prefix. Paths resolved against project root. `ensure_dirs()` called at startup |
| Domain | `utils/image.py` | Shared `load_image()`, `load_images()`, `iter_image_files()`, `IMAGE_EXTENSIONS` |

### Data Flow (Search)

```
POST /search (UploadFile)
  → routes.py: _validate_image() → RGB PIL Image
  → services/search.py: search(image)
      ├─ Stage 1: CLIP encode → Faiss L2 top-K
      ├─ Stage 2: dHash filter (Hamming ≤10) → survivors
      ├─ Stage 3: pHash filter (Hamming ≤12, on survivors only)
      ├─ Stage 4: SSIM filter (≥0.90, on survivors only, parallel)
      └─ Tier: confirmed (3/3) | suspected (1-2/3) | none (0/3)
  → SearchResponse JSON (results + per-stage counts + per-stage timing)
```

### Data Flow (Index)

```
POST /api/v1/index (single)
  → routes.py → index_service.index_single()
      ├─ dedup check (by filename)
      ├─ save → settings.images_path / filename
      ├─ compute_embedding() + dHash + pHash
      └─ indexer.add_item()

POST /api/v1/index/batch (directory)
  → routes.py → index_service.index_batch()
      ├─ scan directory for image files
      ├─ for each batch (size = settings.batch_size):
      │   ├─ load_images() → PIL list
      │   ├─ compute_embeddings() → batch CLIP
      │   ├─ compute_dhashes() + compute_phashes() → parallel
      │   └─ indexer.add_items() → batch Faiss insert
      └─ fallback: _index_single_from_disk() on batch failure
```

### Computation Layer Internals

```
┌─────────────────────────────────────────────────────────────┐
│  embedding.py                                               │
│                                                             │
│  compute_embedding(image) → (512,) float32                   │
│    └─ clip_model.encode_image() → torch.Tensor → .cpu().numpy()
│                                                             │
│  compute_embeddings([images]) → (N, 512) float32             │
│    └─ clip_model.encode_images() → batch forward → .cpu().numpy()
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  hasher.py                                                  │
│                                                             │
│  ┌─ dHash ───────────────────────────────────────┐         │
│  │ compute_dhash(img) → "a1b2..." (16-char hex)   │         │
│  │   └─ imagehash.dhash() → 64-bit binary → hex    │         │
│  │                                                  │         │
│  │ compute_dhashes(imgs, workers=8) → ["a1b2",...] │         │
│  │   └─ ThreadPoolExecutor.map(compute_dhash)       │         │
│  └─────────────────────────────────────────────────┘         │
│                                                             │
│  ┌─ pHash ───────────────────────────────────────┐         │
│  │ compute_phash(img) → "c3d4..." (16-char hex)   │         │
│  │   └─ imagehash.phash() → DCT freq → 64-bit hex  │         │
│  │                                                  │         │
│  │ compute_phashes(imgs, workers=8) → ["c3d4",...] │         │
│  │   └─ ThreadPoolExecutor.map(compute_phash)       │         │
│  └─────────────────────────────────────────────────┘         │
│                                                             │
│  ┌─ SSIM ────────────────────────────────────────┐         │
│  │ compute_ssim(img1, img2) → float [-1, 1]       │         │
│  │   ├─ resize both → 256×256 grayscale            │         │
│  │   └─ skimage.metrics.structural_similarity()    │         │
│  └─────────────────────────────────────────────────┘         │
│                                                             │
│  ┌─ Distance ─────────────────────────────────────┐         │
│  │ hamming_distance(h1, h2) → int                  │         │
│  │   └─ int(h1,16) ^ int(h2,16) → .bit_count()     │         │
│  │                                                  │         │
│  │ is_duplicate(h1, h2, threshold=10) → bool        │         │
│  └─────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  indexer.py — class Indexer (module-level singleton)        │
│                                                             │
│  State:                                                      │
│    _lock        = threading.RLock()    # all ops serialized  │
│    _index       = faiss.IndexFlatL2 / IndexIVFFlat / IndexHNSWFlat  (512-dim) │
│    _metadata    = list[dict]           # [{filename,path,    │
│    _dirty       = bool                 #   dhash,phash,id}]  │
│    _auto_save_thread                   # daemon background   │
│                                                             │
│  Lifecycle:                                                  │
│    load()  → disk → _index + _metadata (or fresh)            │
│    save()  → _save_unlocked() → disk (public, acquires lock) │
│    clear() → reset _index + _metadata, delete disk files    │
│                                                             │
│  Write (always inside _lock):                                │
│    add_item(v, meta) → int id                               │
│    add_items(V, metas) → list[int]                           │
│      └─ mark _dirty = True (triggers next auto-save)        │
│                                                             │
│  Read (always inside _lock):                                 │
│    count       → int                                         │
│    search(q,k) → (distances[], ids[])                        │
│    get_metadata(id) → dict | None                            │
│    get_indexed_filenames() → set[str]                        │
│    list_items(page, page_size) → {items, total}              │
│                                                             │
│  Auto-save (daemon thread):                                  │
│    start_auto_save() → _auto_save_loop()                     │
│      └─ every N sec: if _dirty → save() → _dirty = False    │
│    stop_auto_save() → signal + join(5s timeout)              │
│                                                             │
│  Key invariant: _lock acquired before any _index or          │
│  _metadata mutation. _save_unlocked() must be called with    │
│  _lock already held (avoids deadlock on non-reentrant lock). │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns

- **Singleton model**: `clip_model.py` uses module-level `_model` global. `load()` idempotent — called once in FastAPI `lifespan` startup.
- **Module-level indexer**: `services/indexer.py` exports `indexer = Indexer()` singleton. All routes and search share this instance.
- **Thread safety**: `Indexer._lock` protects all Faiss operations. Route handlers are sync `def` (FastAPI runs them in threadpool).
- **Persistence**: Index auto-restored on startup via `indexer.load()`. `_dirty` flag triggers periodic save via daemon thread (`TWIN_AUTO_SAVE_INTERVAL`). Final save on shutdown via `lifespan`.
- **Funnel filtering**: Each stage only runs on survivors of the previous stage. Early termination if a stage produces zero survivors. dHash/pHash stages use integer XOR (microseconds), SSIM only runs on 2-5 candidates.
- **CORS**: Enabled for all origins in dev (see `main.py`).
- **Frontend proxy**: Vite dev server proxies `/api/v1` → `localhost:8000`. Production build should use nginx or similar.

### Config (TWIN_ environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `TWIN_MODEL_NAME` | `ViT-B-32` | CLIP variant |
| `TWIN_PRETRAINED` | `openai` | Weights tag |
| `TWIN_DEVICE` | (auto) | `cuda` / `mps` / `cpu` |
| `TWIN_TOP_K` | `100` | Faiss candidates |
| `TWIN_DHASH_THRESHOLD` | `10` | dHash Hamming threshold |
| `TWIN_PHASH_THRESHOLD` | `12` | pHash Hamming threshold |
| `TWIN_SSIM_THRESHOLD` | `0.90` | SSIM threshold |
| `TWIN_FAISS_INDEX_TYPE` | `ivf_flat` | `flat` / `ivf_flat` / `hnsw` |
| `TWIN_FAISS_NLIST` | `0` | IVF centroids (0=auto) |
| `TWIN_FAISS_NPROBE` | `16` | IVF search probes |
| `TWIN_FAISS_AUTO_UPGRADE` | `true` | Auto Flat→IVF upgrade |
| `TWIN_FAISS_GPU` | `true` | Try GPU Faiss (ignored for HNSW) |
| `TWIN_FAISS_HNSW_M` | `32` | HNSW graph degree (4–64) |
| `TWIN_FAISS_HNSW_EF_CONSTRUCTION` | `200` | HNSW build-time depth |
| `TWIN_FAISS_HNSW_EF_SEARCH` | `128` | HNSW search-time depth |
| `TWIN_BATCH_SIZE` | `64` | Images per CLIP batch |
| `TWIN_AUTO_SAVE_INTERVAL` | `120` | Auto-save seconds |
| `TWIN_SYNC_ON_STARTUP` | `false` | Block startup until sync done |
| `TWIN_INDEX_PATH` | `data` | Index storage dir |
| `TWIN_IMAGES_DIR` | `data/images` | Source images dir |

### Critical Gotcha: OpenMP on macOS

Torch and faiss-cpu each link `libomp`. Loading both in one process causes `OMP: Error #15`. Workaround: set `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` before launching. `main.py` and `conftest.py` call `os.environ.setdefault()` for these, but they must be set **before** the Python interpreter imports torch/faiss — hence still passed as shell env vars.

## Performance

### Indexing Pipeline — Threading Model

```
                    ┌──────────────────────────────────────────┐
                    │       索引流水线 (per batch, 32 images)    │
                    │                                          │
image files ───────►│ load_images()    串行 PIL I/O             │
                    │       │                                  │
                    │       ▼                                  │
                    │ CLIP encode      单次 GPU forward         │
                    │ (batch=32)       torch.set_num_threads(1)│
                    │       │                                  │
                    │       ├──────────┬──────────┐            │
                    │       ▼          ▼          ▼            │
                    │    dHash       pHash      ...             │
                    │  ThreadPool   ThreadPool                 │
                    │  workers=8    workers=8                  │
                    │       │          │                       │
                    │       └──────────┘                       │
                    │              │                            │
                    │              ▼                            │
                    │       Faiss add                          │
                    │       RLock 串行                         │
                    │       (0.19ms / 32 vectors)              │
                    └──────────────────────────────────────────┘
```

| Stage | Concurrency | Mechanism | Bottleneck? |
|-------|------------|-----------|-------------|
| Image I/O (`load_images`) | Serial | `for` loop over paths | 🟡 ~32ms/batch, 30% of total |
| CLIP encode (`encode_images`) | Single GPU forward | `torch.set_num_threads(1)` | 🔴 ~70ms/batch, 67% — **dominant** |
| dHash (`compute_dhashes`) | Parallel | `ThreadPoolExecutor(max_workers=8)` | 🟢 ~0.9ms/batch |
| pHash (`compute_phashes`) | Parallel | `ThreadPoolExecutor(max_workers=8)` | 🟢 ~1.8ms/batch |
| Faiss add (`add_items`) | Serial | `threading.RLock` | 🟢 ~0.2ms/batch |
| SSIM (search funnel) | Serial | `for` loop, 2–5 candidates | 🟢 not on indexing path |
| Auto-save | Background | Independent daemon thread | 🟢 async, non-blocking |

### Per-Batch Time Budget (batch_size=32, ViT-B-32, GPU)

```
CLIP batch encode     70.48 ms  ████████████████████████████  67%
Image I/O             ~32  ms  ████████████                  30%
dHash (8 workers)     ~0.9 ms  ▏                             <1%
pHash (8 workers)     ~1.8 ms  ▏                             <1%
Faiss add             ~0.2 ms  ▏                             <1%
─────────────────────────────────────────
Total                ~105 ms

Throughput: 32 / 0.105 ≈ 305 img/s
Single-image latency (index_single): 24.65 ms (CLIP + dHash + pHash + Faiss, no disk I/O: 4.64 ms)
```

### Optimization Priorities

All measured against the current ~305 img/s baseline (batch_size=32, ViT-B-32, GPU).

| Priority | Change | Expected Gain | Rationale |
|----------|--------|---------------|-----------|
| **P0** | Parallelize image I/O with `ThreadPoolExecutor` | 305 → ~400 img/s | 30% of batch time is serial PIL decode; 8 workers would cut it to ~5ms |
| **P1** | Increase `TWIN_BATCH_SIZE` to 128 | 305 → ~360 img/s | Better GPU utilization; CLIP forward is sub-linear in batch size (64 is now default) |
| **P2** | Pipeline: preload batch N+1 while encoding batch N | 305 → ~380 img/s | Overlaps I/O wait with GPU compute; requires double-buffering |
| **P3** | GPU Faiss for indexing | negligible | `add_items` is already <0.2ms per 32 vectors; GPU transfer overhead would dominate. GPU Faiss matters for **search** on large indices, not indexing throughput |

### Search Latency Budget (per query)

```
CLIP encode (single)   3.56 ms  ██████████████████████  90%
Faiss L2 top-50        0.03 ms  ▏                       <1%
dHash filter (50)      0.01 ms  ▏                       <1%
pHash filter (~5)      0.01 ms  ▏                       <1%
SSIM verify (~3)       2.62 ms  ████████████████        7%  (per pair)
─────────────────────────────────────────
End-to-end             ~4.0 ms
```

dHash/pHash run on all 50 Faiss candidates but are O(1) integer XOR — sub-microsecond per comparison. SSIM is the only expensive filter and only runs on 2–5 pHash survivors. The funnel design ensures SSIM cost is bounded regardless of index size.

### GPU Faiss Impact on Search

| Index size | CPU Flat | GPU Flat | Speedup |
|-----------|----------|----------|---------|
| 1K | 0.03 ms | ~0.02 ms | 1.5× (overhead dominates) |
| 10K | 0.15 ms | ~0.05 ms | 3× |
| 100K | 1.5 ms | ~0.2 ms | 7.5× |
| 1M | 15 ms | ~1.5 ms | 10× |

GPU Faiss only matters for search on large indices (>10K vectors). For indexing, Faiss add is <0.2ms per batch — GPU acceleration would make no measurable difference.

## Design Context

- **Register**: Product (internal dev tool, not a marketing surface)
- **Brand Personality**: Precision · Efficient · Transparent
- **Design System**: MongoDB-inspired (see `DESIGN.md` for tokens). Euclid Circular A, segmented tab nav, pill buttons (`--radius-full`), card-base with `--hairline` borders
- **Anti-patterns to avoid**: No SaaS marketing hero banners, no gradient backgrounds, no decorative animations. Performance > decoration.
- **Principles**: (1) Show your work — every pipeline stage visible with counts + timing. (2) Be fast, not fancy. (3) Raw scores, not happy signals. (4) One thing per screen.
- **Accessibility**: Target WCAG 2.1 AA. Contrast ≥4.5:1. Reduced motion respected.
- **Live mode**: Configured at `.impeccable/live/config.json`. Run `/impeccable live` for in-browser variant iteration.
- **Frontend**: `src/example/web/` — Vite + React. Proxy `/api/v1` → `localhost:8000`. Color tokens in `index.css` (MongoDB-based palette: `--brand-green`, `--brand-teal-deep`, `--ink`, etc.)
