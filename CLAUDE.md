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
Client uploads query.png
  │
  ├─ routes.search_endpoint()
  │   ├─ file.file.read() → bytes
  │   ├─ _validate_image(filename, bytes)
  │   │   ├─ Check extension ∈ {.jpg, .jpeg, .png, .webp, .bmp}
  │   │   ├─ Image.open(BytesIO(bytes))
  │   │   ├─ img.load()  ← decodes pixels, raises on corruption
  │   │   └─ img.convert("RGBA" if has_transparency else "RGB")
  │   └─ search(image) → dict → SearchResponse(**dict)
  │
  └─ services/search.py: search(image)
      │
      ├─ [Step 1: Classification & Preprocessing (CPU)]
      │   ├─ classifier.classify_texture(image) → TextureKind (SkinDouble / SkinSingle / Cape / Unknown)
      │   ├─ Target datasource routed: "skin" (SkinDouble / SkinSingle / Unknown) vs "cape" (Cape)
      │   └─ If SkinSingle (64x32): converter.single2double(image) → 64x64 standard UV layout
      │
      ├─ [Step 2: 0° Global pHash Extraction (CPU, 0% GPU)]
      │   └─ query_phash = compute_phash(image) → 16-char hex (64-bit DCT binary)
      │
      ├─ [Step 3: Flat SIMD Binary Recall (faiss.IndexBinaryFlat)]
      │   ├─ indexer.search_binary(query_phash, k=top_k, datasource=target_ds)
      │   │   └─ Hardware SIMD AVX2/AVX-512 POPCNT scan (~3.3ms across entire collection)
      │   └─ Candidates: Top-K items sorted by Hamming distance
      │
      ├─ [Step 4: Unified Top-K Verifier (Pixel Confirmation)]
      │   ├─ verifiers.get_verifier(settings.verifier_type) → DiffMaskVerifier (default) or SSIMVerifier
      │   └─ For each candidate in Top-K:
      │       ├─ Load candidate PIL Image from disk
      │       └─ verifier.verify(query_image, cand_image)
      │           ├─ DiffMask: Alpha dynamic union mask, point-by-point tolerance (|RGB Δ| ≤ 5) (~0.12ms)
      │           └─ Score: match_ratio (1.0 - diff_ratio) or SSIM similarity
      │
      └─ [Step 5: Finalize & Format]
          ├─ Assign match_level based on verification threshold
          └─ _build_response → {results (Top-K + Hamming distance + verifier_score), count, query_time_ms, stages}
```

### 2.1 Text Search Flow (`POST /api/v1/search/text`)

```
User enters query text (e.g. "alex green dress")
  │
  ├─ routes.search_text_endpoint() → services/search.py: search_by_text(query)
  │
  ├─ [Step 1: CLIP Text Encoder]
  │   └─ clip_model.encode_text(query) → (512,) normalized vector
  │
  ├─ [Step 2: Vector Semantic Recall (faiss.IndexFlatL2)]
  │   └─ indexer.search(query_vec, k=top_k, datasource="skin") → Top-K semantic candidates
  │
  └─ [Step 3: Format & Return]
      └─ Return semantic candidates with L2 distance
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
HTTP Layer     api/routes.py  ──→  Thin REST & SSE endpoints, delegates to services
               api/schemas.py ──→  Pydantic v2 validation
                    │
Orchestration   search.py      ──→  Dual-engine recall (SIMD Hash / ViT Text) + Unified Top-K Verifier
               index_service.py ──→  index_single(), async index_batch(), classification & conversion
               sync.py        ──→  startup auto-sync with texture classification & dual-datasource index
                    │
Domain/Texture  classifier.py  ──→  Minecraft UV layout heuristic classifier (SkinDouble, SkinSingle, Cape)
               converter.py   ──→  Minecraft 1.7 single (64x32) to 1.8 double (64x64) UV layout converter
               verifiers.py   ──→  Unified Top-K verification (DiffMaskVerifier, SSIMVerifier)
                    │
Computation     embedding.py   ──→  CLIP / DINOv2 float32 vectors (single + batch)
               hasher.py      ──→  aHash, dHash, pHash (imagehash), SSIM, Hamming
               indexer.py     ──→  faiss.IndexBinaryFlat (Hash) + faiss.IndexFlatL2 (Vector) per datasource ("skin", "cape")
                    │
Domain          clip_model.py  ──→  CLIP singleton, CUDA/MPS/CPU auto-detect
               dinov2_model.py ──→  DINOv2 vision transformer singleton
               config.py      ──→  TWIN_ env vars, multi-datasource paths
               utils/image.py ──→  shared parallel image I/O utilities
               utils/metrics.py──→  rolling throughput rate & ETA metrics
```

### Component Map

| Layer | Module | Role |
|-------|--------|------|
| **HTTP** | `api/routes.py` | REST & SSE endpoints — thin handlers, 0 business logic. Delegates to `search()`, `search_by_text()`, `index_service`, `indexer`, `get_sync_status()` |
| HTTP | `api/schemas.py` | Pydantic v2 models: SearchResultItem, SearchResponse, TextSearchResponse, BatchIndexAsyncResponse, HealthResponse, etc. |
| **Service** | `services/search.py` | Orchestrator: SIMD pHash binary recall (image) / CLIP text recall (text) → Unified Top-K Verifier (DiffMask/SSIM). Returns verified candidates |
| Service | `services/classifier.py` | Texture classifier: classifies SkinDouble, SkinSingle, Cape, Elytra, Unknown via UV alpha inspection |
| Service | `services/converter.py` | UV converter: elevates legacy 64x32 single-layer skins to standard 64x64 double-layer layout |
| Service | `services/verifiers.py` | Verification strategies: DiffMask (Alpha-masked pixel tolerance) and SSIM verifier |
| Service | `services/index_service.py` | Encapsulates index workflows: classify + convert + index single/batch across dual datasources |
| Service | `services/sync.py` | Startup sync: scans images dir, classifies textures and populates skin and cape indexers |
| Service | `services/embedding.py` | Multi-model embedding abstraction for CLIP and DINOv2 |
| Service | `services/hasher.py` | aHash, dHash, pHash (imagehash), SSIM (PyTorch CUDA / skimage), Hamming distance |
| Service | `services/indexer.py` | Multi-datasource index manager (`skin`, `cape`), each holding `IndexBinaryFlat` (pHash) and `IndexFlatL2/IVF` (vector) |
| **Domain** | `models/clip_model.py` | Singleton. `load()` idempotent, `encode_image()`, `encode_images()`, `encode_text()`. Auto-detects CUDA > MPS > CPU |
| Domain | `models/dinov2_model.py` | Singleton. DINOv2 vision transformer via timm / PyTorch |
| Domain | `core/config.py` | `BaseSettings` with `TWIN_` prefix. Paths resolved per datasource (`data/{datasource}/`). `ensure_dirs()` called at startup |
| Domain | `utils/image.py` | Shared `load_image()`, parallel `load_images()`, `IMAGE_EXTENSIONS` |
| Domain | `utils/metrics.py` | Shared `compute_progress_metrics()`: rolling window throughput rate, progress percentage, ETA |

### Data Flow (Search)

```
POST /api/v1/search (Image query)
  → routes.py: _validate_image() → PIL Image
  → services/search.py: search(image)
      ├─ Step 1: classify_texture(image) → route to "skin" or "cape"
      ├─ Step 2: if single-layer skin → convert to 64x64 double-layer
      ├─ Step 3: compute_phash(image) (0° lock, 0% GPU)
      ├─ Step 4: IndexBinaryFlat SIMD scan (Hamming distance Top-K, ~3.3ms)
      ├─ Step 5: Unified Top-K Verifier (DiffMaskVerifier, ~0.12ms/pair)
      └─ Match level: confirmed / suspected / none based on verification score
  → SearchResponse JSON (results with distance & verification score)

POST /api/v1/search/text (Text query)
  → routes.py: search_text_endpoint()
  → services/search.py: search_by_text(query)
      ├─ Step 1: CLIP encode_text(query) → 512-dim vector
      ├─ Step 2: IndexFlatL2 search on skin datasource → Top-K semantic candidates
      └─ Step 3: Top-K Verifier validation
  → TextSearchResponse JSON
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
│  State (Multi-datasource: "skin", "cape"):                  │
│    _lock          = threading.RLock()   # all ops serialized│
│    _indices       = dict[str, faiss.Index] (vector 512-dim) │
│    _binary_indices= dict[str, faiss.IndexBinaryFlat] (64-bit)│
│    _metadatas     = dict[str, list[dict]]                   │
│    _dirty_flags   = dict[str, bool]     # per datasource    │
│    _auto_save_thread                    # daemon background │
│                                                             │
│  Lifecycle:                                                  │
│    load()  → disk → loads all datasources (skin & cape)     │
│    save()  → _save_unlocked() → disk per datasource         │
│    clear() → reset all indices, delete disk files           │
│                                                             │
│  Write (always inside _lock):                                │
│    add_item(v, meta, hash_val, datasource)                  │
│    add_items(V, metas, hashes, datasource)                  │
│      └─ mark _dirty_flags[ds] = True (triggers auto-save)   │
│                                                             │
│  Read (always inside _lock):                                 │
│    count(datasource?)       → int                           │
│    search(q, k, ds)         → (distances[], ids[])          │
│    search_binary(q_hash, k, ds) → (distances[], ids[])     │
│    get_metadata(id, ds)     → dict | None                   │
│    get_indexed_filenames(ds)→ set[str]                      │
│    list_items(page, page_size, ds) → {items, total}         │
│                                                             │
│  Auto-save (daemon thread):                                  │
│    start_auto_save() → _auto_save_loop()                     │
│      └─ every N sec: if _dirty → save()                     │
│    stop_auto_save() → signal + join(5s timeout)              │
│                                                             │
│  Key invariant: _lock acquired before any _indices or        │
│  _metadatas mutation or read.                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns

- **Singleton model**: `clip_model.py` uses module-level `_model` global. `load()` idempotent — called once in FastAPI `lifespan` startup.
- **Module-level indexer**: `services/indexer.py` exports `indexer = Indexer()` singleton managing isolated `skin` and `cape` datasources.
- **Thread safety**: `Indexer._lock` protects all Faiss operations across all datasources. Route handlers are sync `def` (FastAPI runs them in threadpool).
- **Persistence**: Index auto-restored on startup via `indexer.load()`. `_dirty` flag triggers periodic save via daemon thread (`TWIN_AUTO_SAVE_INTERVAL`). Final save on shutdown via `lifespan`.
- **Modality-Decoupled Recall**: Image-to-image queries bypass GPU completely, performing SIMD hardware POPCNT scan over `faiss.IndexBinaryFlat` (~3.3ms) routed by texture type (`skin` vs `cape`). Text-to-image queries leverage CLIP text encoder and `faiss.IndexFlatL2`.
- **Unified Top-K Verifier**: All candidate recalls converge into a single linear pipeline — Top-K candidates are directly verified by `DiffMaskVerifier` (default, ~0.12ms) or `SSIMVerifier`, outputting verifiable pixel metrics.
- **CORS**: Enabled for all origins in dev (see `main.py`).
- **Frontend proxy**: Vite dev server proxies `/api/v1` → `localhost:8000`. Production build should use nginx or similar.

### Config (TWIN_ environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `TWIN_MODEL_TYPE` | `clip` | Embedding model architecture (`clip` or `dinov2`) |
| `TWIN_MODEL_NAME` | `ViT-B-32` | CLIP variant |
| `TWIN_PRETRAINED` | `openai` | CLIP weights tag |
| `TWIN_DINOV2_MODEL_NAME` | `vit_small_patch14_dinov2` | DINOv2 model architecture name |
| `TWIN_DEVICE` | (auto) | `cuda` / `mps` / `cpu` |
| `TWIN_TOP_K` | `100` | Candidates retrieved for verification |
| `TWIN_VERIFIER_TYPE` | `diffmask` | Verifier implementation: `diffmask` or `ssim` |
| `TWIN_DIFFMASK_TOLERANCE` | `5` | DiffMask RGB delta threshold |
| `TWIN_DIFFMASK_ALPHA_THRESHOLD` | `10` | DiffMask alpha cutoff |
| `TWIN_DIFFMASK_DIFF_RATIO_THRESHOLD` | `0.05` | DiffMask max mismatch ratio |
| `TWIN_DHASH_THRESHOLD` | `10` | dHash Hamming threshold |
| `TWIN_PHASH_THRESHOLD` | `12` | pHash Hamming threshold |
| `TWIN_SSIM_THRESHOLD` | `0.90` | SSIM threshold |
| `TWIN_SSIM_DEVICE` | (auto) | SSIM compute device (`cuda` / `mps` / `cpu`) |
| `TWIN_SSIM_SIZE` | `256` | SSIM comparison resolution |
| `TWIN_ROTATION_INVARIANT` | `false` | 90°/180°/270° orthogonal orientation tolerance |
| `TWIN_FAISS_INDEX_TYPE` | `ivf_flat` | `flat` / `ivf_flat` / `ivf_pq` / `hnsw` |
| `TWIN_FAISS_NLIST` | `0` | IVF centroids (0=auto) |
| `TWIN_FAISS_NPROBE` | `16` | IVF search probes |
| `TWIN_FAISS_AUTO_UPGRADE` | `true` | Auto Flat→IVF upgrade |
| `TWIN_FAISS_GPU` | `true` | Try GPU Faiss (ignored for HNSW) |
| `TWIN_FAISS_HNSW_M` | `32` | HNSW graph degree (4–64) |
| `TWIN_FAISS_HNSW_EF_CONSTRUCTION` | `200` | HNSW build-time depth |
| `TWIN_FAISS_HNSW_EF_SEARCH` | `128` | HNSW search-time depth |
| `TWIN_BATCH_SIZE` | `64` | Images per forward pass / batch |
| `TWIN_AUTO_SAVE_INTERVAL` | `120` | Auto-save seconds |
| `TWIN_SYNC_ON_STARTUP` | `false` | Block startup until sync done |
| `TWIN_INDEX_PATH` | `data` | Index storage dir |
| `TWIN_IMAGES_DIR` | `data/images` | Source images dir |

### Critical Gotcha: OpenMP on macOS & Windows

Torch and faiss-cpu each link `libomp`. Loading both in one process causes `OMP: Error #15`. Workaround: set `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` before launching. `main.py`, `conftest.py`, and `dev.ps1` configure these automatically.

## Performance

### Indexing Pipeline — Double-Buffering Preload Model

```
                    ┌────────────────────────────────────────────────────────┐
                    │          双缓冲流水线 (Double-Buffering Pipeline)        │
                    │                                                        │
                    │  Batch N (GPU Compute)      Batch N+1 (I/O Preload)    │
                    │  ┌──────────────────────┐   ┌───────────────────────┐  │
                    │  │ CLIP/DINOv2 encode   │   │ ThreadPoolExecutor    │  │
                    │  │ aHash + dHash + pHash│   │ load_images(batch_N+1)│  │
                    │  │ Faiss RLock add_items│   │ decode PIL in bg      │  │
                    │  └──────────────────────┘   └───────────────────────┘  │
                    │              ▲                          │              │
                    │              └──────────────────────────┘              │
                    │             (Overlapped I/O hides behind compute)      │
                    └────────────────────────────────────────────────────────┘
```

| Stage | Concurrency | Mechanism | Bottleneck? |
|-------|------------|-----------|-------------|
| Image I/O (`load_images`) | Overlapped | Background `ThreadPoolExecutor(max_workers=1)` double-buffering preloader | 🟢 Overlapped behind GPU compute |
| Forward pass (`compute_embeddings`) | Single GPU forward | CLIP / DINOv2 (`torch.set_num_threads(1)`) | 🔴 ~70ms/batch (dominant) |
| Hashing (`compute_*hashes`) | Parallel | `ThreadPoolExecutor(max_workers=8)` for aHash/dHash/pHash | 🟢 ~2.5ms/batch |
| Faiss add (`add_items`) | Serial | `threading.RLock` | 🟢 ~0.2ms/batch |
| SSIM (search funnel) | Serial / GPU | skimage CPU or PyTorch CUDA conv2d | 🟢 not on indexing path |
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

- **P0: Parallelize image I/O** (Implemented ✅) — `load_images` multithreaded decode.
- **P1: Batch size tuning** (Implemented ✅) — `TWIN_BATCH_SIZE=64` default.
- **P2: Pipelined double-buffering** (Implemented ✅) — background thread preloads batch N+1 while GPU computes batch N, lifting ingestion throughput by +19.6%.
- **P3: GPU Faiss for indexing** — Evaluated: negligible gain for insert (<0.2ms per batch). Retained on GPU for large index search speedup.

### Search Latency Budget (Image-to-Image Query, Pure CPU)

```
Texture classify & UV convert  ~0.4 ms  ███████                 8%
0° DCT pHash compute           ~1.2 ms  ██████████████████      24%
Flat SIMD binary scan (660k)   ~3.3 ms  █████████████████████████ 66%
DiffMask verify (Top-50 cands) ~0.1 ms  █                       2%
─────────────────────────────────────────
End-to-end                     ~5.0 ms  (0% GPU usage)
```

Image-to-image queries bypass GPU inference entirely, achieving ~5ms end-to-end response times with 100% exact Hamming recall and hardware AVX2/AVX-512 SIMD execution. Verification cost is minimal (~0.12ms per pair with DiffMask).

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
