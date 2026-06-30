# Twin — Image Similarity Search & Deduplication

Two-stage image retrieval system with **4-stage funnel verification**:

```
Stage 1 (Recall)                    Stage 2 (Verification)
CLIP ViT-B/32 → Faiss L2 top-K  →  dHash → pHash → SSIM
       ↓                                    ↓
  semantic candidates              funnel filtering → confirmed/suspected
```

## Quick Start

```bash
# Prerequisites: conda environment with faiss-gpu (for GPU acceleration)
conda create -n twin-gpu python=3.12 faiss-gpu -c pytorch -c nvidia -y

# 1. Install backend + frontend deps（GPU path — faiss-gpu from conda）
make install

# Or CPU-only（faiss-cpu from PyPI）：
make install-cpu

# 2. Start backend + frontend
make dev

# 3. Open the web UI
open http://localhost:5173
```

**Dependency split**: `faiss-cpu` is NOT a core dependency — it lives in the `[cpu]` optional group.
- **GPU hosts**: `make install` / `make install-gpu` → `pip install -e ".[dev]"`（no `faiss-cpu`; expects `faiss-gpu` pre-installed via conda）
- **CPU hosts**: `make install-cpu` → `uv sync --extra cpu`（explicitly pulls `faiss-cpu` from PyPI）

Runtime GPU detection is automatic: if `faiss-gpu` is available（`faiss.get_num_gpus()` + `StandardGpuResources()` + `index_cpu_to_gpu()`）, the indexer wraps every new index in a GPU index. Otherwise it falls back to CPU with a warning.

## API Endpoints

| Method   | Path                        | Description                                               |
| -------- | --------------------------- | --------------------------------------------------------- |
| `GET`    | `/api/v1/health`            | Indexed count + model status                              |
| `POST`   | `/api/v1/search`            | Upload image → find similar/duplicates                    |
| `POST`   | `/api/v1/index`             | Upload single image → add to index                        |
| `POST`   | `/api/v1/index/batch`       | `{"directory": "/path"}` → index all images               |
| `GET`    | `/api/v1/index`             | Paginated list of indexed images (`?page=1&page_size=50`) |
| `GET`    | `/api/v1/images/{filename}` | Serve image file for preview thumbnails                   |
| `DELETE` | `/api/v1/index`             | Clear entire index                                        |
| `GET`    | `/api/v1/sync/status`       | Background sync progress, ETA, and completion status      |

## Examples

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Sync status (progress + ETA during background sync)
curl http://localhost:8000/api/v1/sync/status

# Index an image
curl -X POST -F "file=@photo.jpg" http://localhost:8000/api/v1/index

# Index all images in a directory
curl -X POST -H "Content-Type: application/json" \
  -d '{"directory":"/home/user/pictures"}' \
  http://localhost:8000/api/v1/index/batch

# Search for similar images
curl -s -X POST -F "file=@query.jpg" http://localhost:8000/api/v1/search | python3 -m json.tool

# Browse indexed images (paginated)
curl "http://localhost:8000/api/v1/index?page=1&page_size=20"
```

## Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP Layer (api/)                       │
│  routes.py — thin handlers, 7 endpoints, 0 business logic   │
│  schemas.py — Pydantic v2 request/response validation       │
└───────────────┬───────────────────────────────┬─────────────┘
                │                               │
    ┌───────────▼───────────┐     ┌────────────▼────────────┐
    │   Service Layer        │     │   Service Layer          │
    │   (orchestration)      │     │   (computation)          │
    │                        │     │                          │
    │  search.py             │     │  embedding.py            │
    │  ├─ Stage 1: Faiss     │     │  ├─ CLIP → 512d vector  │
    │  ├─ Stage 2: dHash     │     │  └─ batch encode        │
    │  ├─ Stage 3: pHash     │     │                          │
    │  └─ Stage 4: SSIM      │     │  hasher.py               │
    │                        │     │  ├─ dHash / pHash        │
    │  index_service.py      │     │  ├─ SSIM (skimage)       │
    │  ├─ index_single()     │     │  └─ Hamming distance     │
    │  └─ index_batch()      │     │                          │
    │                        │     │  indexer.py              │
    │  sync.py               │     │  ├─ Faiss IndexFlatL2    │
    │  └─ startup auto-sync  │     │  └─ metadata + persist   │
    └───────────┬────────────┘     └────────────┬─────────────┘
                │                               │
    ┌───────────▼───────────────────────────────▼─────────────┐
    │                    Domain Layer                          │
    │                                                         │
    │  models/clip_model.py    core/config.py                 │
    │  ├─ load() idempotent    ├─ TWIN_ env vars              │
    │  ├─ encode_image()       ├─ path resolution              │
    │  └─ encode_images()      └─ settings singleton          │
    │                                                         │
    │  utils/image.py                                         │
    │  ├─ load_image() / load_images()                        │
    │  └─ IMAGE_EXTENSIONS                                    │
    └─────────────────────────────────────────────────────────┘
```

### Search Pipeline

```
                    ┌──────────┐
                    │  Query   │
                    │  Image   │
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │  Stage 1: Recall     │
              │                      │
              │  CLIP ViT-B/32       │
              │  → 512d L2-norm vec  │
              │  → Faiss.search(k=50)│
              │                      │
              │  Output: 50 candidates│
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Stage 2: dHash       │
              │                       │
              │  64-bit gradient hash │
              │  Hamming ≤ 10?        │
              │                       │
              │  Survivors: ~10       │
              └──────────┬────────────┘
                         │
              ┌──────────▼───────────┐
              │  Stage 3: pHash       │
              │                       │
              │  64-bit DCT hash      │
              │  Hamming ≤ 12?        │
              │                       │
              │  Survivors: ~5        │
              └──────────┬────────────┘
                         │
              ┌──────────▼───────────┐
              │  Stage 4: SSIM        │
              │                       │
              │  Luminance + Contrast │
              │  + Structure ≥ 0.90?  │
              │                       │
              │  Confirmed: ~3        │
              └──────────┬────────────┘
                         │
              ┌──────────▼───────────┐
              │  Tier Classification  │
              │                       │
              │  🟢 confirmed  (4/4)  │
              │  🟡 suspected  (2-3)  │
              │  ⚪ none       (1)    │
              └───────────────────────┘
```

### Data Flow (Search)

```
POST /api/v1/search
  │
  ├─ routes.search_endpoint()
  │   ├─ _validate_image() → RGB PIL Image
  │   └─ services.search.search(image)
  │
  └─ search()
      ├─ compute_embedding(image)    → (512,) float32
      ├─ compute_dhash(image)        → 16-char hex
      ├─ compute_phash(image)        → 16-char hex
      ├─ indexer.search(vec, k=50)   → distances[], ids[]
      │
      ├─ For each candidate:
      │   ├─ metadata lookup → dhash, phash, path
      │   ├─ if path exists → load_image() → fresh dHash + pHash + SSIM
      │   └─ score: dHash_ok? pHash_ok? SSIM_ok?
      │
      └─ Sort: stages_passed DESC, dHash ASC, L2 ASC
         → SearchResponse JSON (results + stages timing + counts)
```

### Data Flow (Index)

```
POST /api/v1/index                    POST /api/v1/index/batch
  │                                      │
  ├─ _validate_image()                   ├─ validate directory
  ├─ index_service.index_single()        └─ index_service.index_batch()
  │   ├─ dedup check                         ├─ iter_image_files()
  │   ├─ save → data/images/{name}           ├─ for batch in chunks:
  │   ├─ compute_embedding()                 │   ├─ load_images()
  │   ├─ compute_dhash() + phash()           │   ├─ compute_embeddings()
  │   └─ indexer.add_item()                  │   ├─ compute_dhashes() + phashes()
  │                                          │   └─ indexer.add_items()
  └─ IndexStatus JSON                        │
                                             └─ BatchIndexResponse JSON
```

### Project Structure

```
src/twin/
├── main.py                    # FastAPI app, CORS, lifespan (model→index→sync→auto-save)
├── core/
│   └── config.py              # TWIN_ env vars, path resolution, ensure_dirs()
├── models/
│   └── clip_model.py          # CLIP singleton, encode_image(), encode_images()
├── services/
│   ├── embedding.py           # compute_embedding(), compute_embeddings()
│   ├── hasher.py              # dHash, pHash, SSIM, Hamming distance
│   ├── indexer.py             # Faiss Flat / IVF / HNSW + metadata + auto-save thread
│   ├── index_service.py       # index_single(), index_batch() workflows
│   ├── search.py              # 2-stage + 4-filter funnel orchestrator
│   └── sync.py                # Startup: scan images dir, batch-index missing
├── api/
│   ├── routes.py              # 7 endpoints — thin HTTP handlers
│   └── schemas.py             # Pydantic v2 request/response models
└── utils/
    └── image.py               # load_image(), load_images(), IMAGE_EXTENSIONS

tests/
├── conftest.py                # OMP env vars
├── test_api.py                # Integration tests (7 endpoints)
├── test_embedding.py          # CLIP embedding tests
├── test_hasher.py             # dHash/pHash/SSIM tests
└── test_indexer.py            # Faiss indexer unit tests

src/example/web/               # React frontend (Vite)
├── src/
│   ├── App.jsx                # Tab layout (Search | Browse | Manage)
│   ├── api/twin.js            # Axios API client
│   ├── pages/
│   │   ├── Search.jsx         # Drag-drop upload + zoomable results
│   │   ├── Browse.jsx         # Paginated thumbnail gallery
│   │   └── Manage.jsx         # Health status + index controls
│   └── index.css              # MongoDB-inspired design tokens
└── vite.config.js             # Proxy /api/v1 → localhost:8000
```

## Configuration

Set via environment variables with `TWIN_` prefix (or `.env` file):

| Variable                  | Default       | Description                                                         |
| ------------------------- | ------------- | ------------------------------------------------------------------- |
| `TWIN_MODEL_NAME`         | `ViT-B-32`    | CLIP model variant                                                  |
| `TWIN_PRETRAINED`         | `openai`      | Pretrained weights tag                                              |
| `TWIN_DEVICE`             | (auto)        | `cuda`, `mps`, `cpu`, or empty for auto-detect                      |
| `TWIN_TOP_K`              | `50`          | Faiss candidates to retrieve                                        |
| `TWIN_DHASH_THRESHOLD`    | `10`          | Max dHash Hamming distance for match                                |
| `TWIN_PHASH_THRESHOLD`    | `12`          | Max pHash Hamming distance for match                                |
| `TWIN_SSIM_THRESHOLD`     | `0.90`        | Min SSIM for structural match                                       |
| `TWIN_BATCH_SIZE`         | `32`          | Images per CLIP forward pass                                        |
| `TWIN_AUTO_SAVE_INTERVAL` | `300`         | Seconds between auto-saves (0=disable)                              |
| `TWIN_FAISS_INDEX_TYPE`              | `ivf_flat`    | Faiss index type: `flat`, `ivf_flat`, or `hnsw` (IndexHNSWFlat) |
| `TWIN_FAISS_NLIST`                   | `0`           | IVF cluster count; 0 = auto (`4 * sqrt(N)`)                   |
| `TWIN_FAISS_NPROBE`                  | `8`           | IVF search-time probe count; higher = more recall              |
| `TWIN_FAISS_AUTO_UPGRADE`            | `true`        | Auto-convert Flat→IVF when enough vectors accumulated         |
| `TWIN_FAISS_GPU`                     | `true`        | Try GPU Faiss if CUDA available (requires `faiss-gpu`; ignored for HNSW) |
| `TWIN_FAISS_HNSW_M`                  | `32`          | HNSW graph degree (4–64)                                       |
| `TWIN_FAISS_HNSW_EF_CONSTRUCTION`    | `200`         | HNSW build-time exploration depth (100–2000)                   |
| `TWIN_FAISS_HNSW_EF_SEARCH`          | `64`          | HNSW search-time depth; higher = more recall                   |
| `TWIN_SYNC_ON_STARTUP`               | `false`       | Block startup until sync completes; `false` = background thread |
| `TWIN_INDEX_PATH`                    | `data`        | Faiss index storage directory                                  |
| `TWIN_IMAGES_DIR`                    | `data/images` | Source images directory (auto-synced on startup)               |

## Frontend

The React dev server is at `src/example/web/`. It proxies `/api/v1` to the backend.
Three pages:

- **Health** — server status + index count
- **Search** — upload image, view results with funnel metrics and per-stage timing
- **Browse** — paginated thumbnail gallery of all indexed images
- **Index** — upload images, batch index, clear index

## Roadmap

Items mapped to recommendations from [图像检索方案技术研究](./图像检索方案技术研究.md).

### Already Done ✅

| #   | Task                                                                                     | Report Reference    |
| --- | ---------------------------------------------------------------------------------------- | ------------------- |
| 1   | **Async I/O** — `async def` routes + `asyncio.to_thread` for CPU-bound ops               | § 异步模型与阻塞 I/O 的处理   |
| 2   | **IndexIVFFlat** — K-Means clustering with auto-upgrade (Flat→IVF) + `nprobe` tuning     | § Faiss 索引机制与性能优化   |
| 3   | **4-stage funnel** — CLIP → Faiss L2 → dHash → pHash → SSIM, each on survivors only      | § 双阶段检索架构           |
| 4   | **Pydantic v2** — request/response validation (SearchResponse, BatchIndexResponse, etc.) | § Pydantic v2 与性能边界 |
| 5   | **UploadFile streaming** — chunked uploads via FastAPI `UploadFile`                      | § 大文件上传与存储管道优化      |
| 6   | **Project structure** — api/ services/ models/ core/ utils/ layered architecture         | § 生产级项目结构与服务分层      |
| 7   | **Singleton model loading** — CLIP loaded once at startup, shared across requests        | § 模型管理              |
| 8   | **GPU Faiss code path** — `_init_gpu()` / `_maybe_to_gpu()` wrapper in place             | § 硬件原生的异构计算         |
| 9   | **Background sync** — startup scan images dir, batch-index new files in daemon thread    | § 背景任务              |
| 10  | **Transparent staging** — per-stage timing + survivor counts returned in search response | § 第二阶段：局部细节比对       |
| 11  | **GPU Faiss (runtime)** — switch `faiss-cpu` → `faiss-gpu` on CUDA hosts                 | § 硬件原生的异构计算               |
| 12  | **HNSW index** — `IndexHNSWFlat` as index option alongside Flat + IVF, configurable M/efConstruction/efSearch | § Faiss 索引机制, Table: IndexHNSW |
| 13  | **Sync status endpoint** — `GET /api/v1/sync/status` to report background sync progress + ETA | § 背景任务                         | Sync runs in background with no visibility; frontend needs progress feedback      |

### High Priority 🔴

| #   | Task                                                                                          | Report Reference               | Notes                                                                             |
| --- | --------------------------------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------- |
| 1   | **DINOv2 embedding** — optional DINOv2 backend (`TWIN_MODEL_TYPE=dinov2`)                     | § 嵌入模型的选择, Table: DINOv2        | ViT-L/14 1024-dim; superior for fine-grained retrieval, natural species, medical |
| 2   | **IndexIVFPQ** — Product Quantization for memory compression at >10M scale             | § Faiss 索引机制, Table: IndexIVFPQ | Not needed at current 664K scale; code infrastructure should be ready            |
| 3   | **aHash** — average-hash as a supplemental cheap pre-filter                            | § 感知哈希的稳健性评估 §1                 | Fastest to compute, least robust; could go before dHash in funnel                |
| 4   | **Background task queue** — proper job queue (ARQ / Celery) for batch-index operations | § 背景任务                          | Batch index currently synchronous in request handler; large dirs cause timeouts  |
| 5   | **CUDA hash / SSIM** — GPU-accelerate perceptual hash and SSIM computation             | § 硬件原生的异构计算                     | Currently CPU-only via ThreadPoolExecutor and skimage                            |

### Low Priority / Future 🟢

| #   | Task                                                                                                 | Report Reference                     | Notes                                                                         |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------- |
| 1   | **Rotation handling** — multi-angle hash or SIFT alignment for rotation >15°                         | § 感知哈希的稳健性评估                         | dHash/pHash break beyond ~15° rotation; pre-align candidates before hashing   |
| 2   | **Adaptive thresholds** — ML-driven threshold tuning per image domain (anime, photo, document)       | § 自适应阈值学习                            | Replaces hardcoded `TWIN_DHASH_THRESHOLD` etc.                                |
| 3   | **Multi-modal search** — text-to-image / text+image hybrid via CLIP dual encoder                     | § 多模态融合检索                            | CLIP already encodes text; needs text embedding endpoint + frontend UI        |
| 4   | **Preprocessing pipeline** — configurable interpolation (`INTER_AREA`, `INTER_CUBIC`), CIELab option | § 图像预处理对检索一致性的影响                     | PIL currently delegates to defaults; OpenCV `INTER_AREA` yields better hashes |
| 5   | **Distributed sharding** — manual shard routing or migrate to Milvus / Qdrant / pgvector             | § 扩展性：Faiss 与 purpose-built 向量数据库的博弈 | Beyond single-machine scope; relevant at 10M+ images                          |
| 6   | **Batch progress streaming** — SSE/WebSocket for real-time progress during large indexing jobs       | § 大文件上传与存储管道优化                       | Frontend gets no progress until batch completes                               |

## Run Tests

```bash
# All tests (GPU env)
make test

# CPU-only tests
make test-cpu

# Single file
OMP_NUM_THREADS=1 uv run pytest tests/test_hasher.py -v
```

## Makefile Targets

```bash
make dev            # Start backend + frontend
make dev-backend    # Backend only
make dev-frontend   # Frontend only
make install        # GPU Faiss deps via conda
make install-cpu    # CPU-only deps via uv + faiss-cpu
make test           # Run tests (GPU env)
make test-cpu       # Run tests (CPU env)
make lint           # Ruff check
make format         # Ruff format
make clean          # Remove build artifacts
```
