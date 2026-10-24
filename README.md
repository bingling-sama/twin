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

| Method   | Path                                           | Description                                                 |
| -------- | ---------------------------------------------- | ----------------------------------------------------------- |
| `GET`    | `/api/v1/health`                               | Indexed count + active model / GPU runtime status           |
| `POST`   | `/api/v1/search`                               | Upload image → find similar/duplicates                      |
| `POST`   | `/api/v1/search/text`                          | `{"query": "text prompt", "k": 50}` → text-to-image search   |
| `POST`   | `/api/v1/index`                                | Upload single image → add to index                          |
| `POST`   | `/api/v1/index/batch`                          | `{"directory": "/path", "async_mode": true}` → index images |
| `GET`    | `/api/v1/index/batch/status`                   | Snapshot of latest batch indexing progress                  |
| `GET`    | `/api/v1/index/batch/status/{task_id}`         | Snapshot for specific async batch indexing task             |
| `GET`    | `/api/v1/index/batch/status/{task_id}/stream`  | SSE real-time stream of async batch indexing progress       |
| `GET`    | `/api/v1/index/batch/status/stream`            | SSE stream of global batch indexing progress                |
| `GET`    | `/api/v1/index/rebuild/status`                 | Status of in-progress index rebuild                         |
| `GET`    | `/api/v1/index/rebuild/status/stream`          | SSE stream of index rebuild progress                        |
| `GET`    | `/api/v1/index`                                | Paginated list of indexed images (`?page=1&page_size=50`)   |
| `GET`    | `/api/v1/images/{filename}`                    | Serve image file for preview thumbnails                     |
| `DELETE` | `/api/v1/index`                                | Clear entire index                                          |
| `POST`   | `/api/v1/index/gpu`                            | Toggle Faiss GPU acceleration (`{"enabled": true/false}`)   |
| `POST`   | `/api/v1/index/train`                          | Force train/upgrade Faiss index to IVF                      |
| `GET`    | `/api/v1/sync/status`                          | Background sync progress, ETA, and completion status        |
| `GET`    | `/api/v1/sync/status/stream`                   | SSE stream of background sync progress                      |
| `GET`    | `/api/v1/config`                               | View dynamic runtime configuration snapshot                 |
| `PATCH`  | `/api/v1/config`                               | Update runtime configuration (index type, batch size, etc.) |

## Examples

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Text-to-image multi-modal search
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "a photo of a cat", "k": 10}' \
  http://localhost:8000/api/v1/search/text

# Sync status (progress + ETA during background sync)
curl http://localhost:8000/api/v1/sync/status

# Index an image
curl -X POST -F "file=@photo.jpg" http://localhost:8000/api/v1/index

# Index all images in a directory (async mode)
curl -X POST -H "Content-Type: application/json" \
  -d '{"directory":"/home/user/pictures", "async_mode": true}' \
  http://localhost:8000/api/v1/index/batch

# Check async batch task progress
curl http://localhost:8000/api/v1/index/batch/status/<task_id>

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
│  routes.py — thin handlers, thin endpoints, 0 business logic│
│  schemas.py — Pydantic v2 request/response validation       │
└───────────────┬───────────────────────────────┬─────────────┘
                │                               │
    ┌───────────▼───────────┐     ┌────────────▼────────────┐
    │   Service Layer        │     │   Service Layer          │
    │   (orchestration)      │     │   (computation)          │
    │                        │     │                          │
    │  search.py             │     │  embedding.py            │
    │  ├─ Stage 1: Faiss     │     │  ├─ CLIP / DINOv2 vector │
    │  ├─ Stage 2: dHash     │     │  └─ batch encode        │
    │  ├─ Stage 3: pHash     │     │                          │
    │  └─ Stage 4: SSIM      │     │  hasher.py               │
    │                        │     │  ├─ aHash / dHash / pHash│
    │  index_service.py      │     │  ├─ SSIM (CUDA/skimage)  │
    │  ├─ index_single()     │     │  └─ Hamming distance     │
    │  ├─ index_batch()      │     │                          │
    │  └─ preloading pipeline│     │  indexer.py              │
    │                        │     │  ├─ Faiss Flat/IVF/HNSW  │
    │  sync.py               │     │  └─ metadata + persist   │
    │  ├─ startup auto-sync  │     └────────────┬─────────────┘
    │  └─ preloading pipeline│                  │
    └───────────┬────────────┘                  │
                │                               │
    ┌───────────▼───────────────────────────────▼─────────────┐
    │                    Domain Layer                          │
    │                                                         │
    │  models/clip_model.py    models/dinov2_model.py         │
    │  ├─ load() idempotent    ├─ load() idempotent           │
    │  ├─ encode_image(s)      ├─ encode_image(s)             │
    │  └─ encode_text()        └─ PyTorch vision transformer  │
    │                                                         │
    │  core/config.py          utils/metrics.py               │
    │  ├─ TWIN_ env vars       └─ rolling rate / ETA metrics  │
    │  └─ settings singleton   utils/image.py                 │
    │                          └─ load_images() parallel I/O  │
    └─────────────────────────────────────────────────────────┘
```

### Search Pipeline (Issue #5 RFC)

```
                                  Query 输入
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼ 图像 Query (以图搜图)                         ▼ 文本 Query (以文搜图)
     [Step 1: 材质分类与预处理 (CPU)]                 [CLIP Text Encoder (GPU/CPU)]
   - 区分 SkinDouble / SkinSingle / Cape                      │
   - 单层皮肤 64x32 升维至标准 64x64 UV                       │
              │                                               │
              ▼                                               │
   [Step 2: 提取全局 pHash (0°)]                              │
              │                                               │
              ▼                                               ▼
   【引擎 A: Flat SIMD 哈希召回】                   【引擎 B: ViT 向量语义召回】
   (faiss.IndexBinaryFlat)                         (faiss.IndexFlatL2)
   - 路由至 Skin 库或 Cape 库                      - 跨模态文本语义匹配
   - 3.3ms 全库精确扫描                            - 提取语义 Top-K
   - 提取汉明距离最近的 Top-K 候选                            │
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                      【统一 Top-K 候选进入 Verifier】
                      Config.Verifier.verify(Top-K)
                      - 默认: DiffMask (Alpha 动态掩码像素比对，单次 0.12ms)
                      - 可选: SSIM (配置切换兼容保留)
                                      │
                                      ▼
                      【最终输出 Top-K 及其 Verifier 评分】
```

### Empirical Verification Matrix (收益对照)

| 评估维度 | 重构前通用漏斗 | 本 PR 落地架构 (Issue #5) |
| :--- | :--- | :--- |
| **以图搜图单次耗时** | 30 ~ 50 ms (强行走 GPU ViT) | **~5 ms** (CPU 预处理 + 3.3ms SIMD 扫描) |
| **以图搜图 GPU 占用率** | 100% 消耗 GPU 推理与显存 | **0% (完全脱离 GPU，纯 CPU 搞定)** |
| **以图搜图漏检率** | 严重 (微调图易被 ViT 挤出 Top-K) | **0% 绝对召回** (全库精确扫描，无数学盲区) |
| **跨版本皮肤格式查重** | 汉明距离 36 (永久漏检) | **距离 0 ~ 1** (升维预处理后精确对齐) |
| **皮肤/披风品类误检** | 相互混杂污染 | **0% 误检** (分类器物理隔离至独立库) |
| **系统控制流复杂度** | 级联过滤、前置短路、并联补丁 | **单管道线性流转** (召回 Top-K -> Verifier -> 输出) |

### Funnel Architecture Limitations & Analysis (Why We Refactored)

Twin 当前采用 **CLIP/DINOv2 + Faiss 粗召回 → dHash → pHash → SSIM** 的四级漏斗过滤架构，在实际应用和高负载场景下存在以下结构性缺陷与改进空间：

1. **召回阶段一票否决与算力倒置 (Top-K Recall Bottleneck)**
   - **漏检不可逆**：严格串联漏斗机制下，如果目标图因轻微噪点、局部裁切或风格差异未排进 Faiss 的 `top_k`，后续 dHash/pHash/SSIM 永远无法接触到该候选。
   - **算力倒置**：感知哈希（微秒级）计算开销远低于 Transformer 向量提取（毫秒级）。目前架构优先运行重模型粗筛，未利用轻量哈希构建极速倒排索引或预过滤通道。

2. **强依赖串行的“雪崩式”误杀 (Fail-Fast Cascading False Negatives)**
   - **dHash 先行截断**：`Stage 2 (dHash)` 先行，仅留存者进入 `Stage 3 (pHash)`。dHash 基于像素单向梯度，对明暗渐变、对比度/Gamma 调节敏感；而基于频域 DCT 的 pHash 抗调色和模糊能力更强。一旦 dHash 误判剔除，更鲁棒的 pHash 和 SSIM 均失去校验机会。
   - **缺乏多维特征融合打分**：目前为布尔硬阈值漏斗，未引入加权评分机制（例如 dHash 略微超标但 pHash 极低且 SSIM 接近 1.0 的高置信度样本会被直接丢弃）。

3. **Stage 4 (SSIM) 存储与 I/O 放大瓶颈 (Disk I/O & Decoding Overhead)**
   - **磁盘全量回表**：`_ssim_check` 需实时通过 `load_image(cand_path)` 从磁盘读取原图并解码。并发搜索或命中多个候选时，高频磁盘随机 I/O 和 JPEG/PNG 解码成为主要耗时瓶颈（远慢于内存比对）。
   - **形变与拉伸失真**：比对时统一强制缩放至 `(ssim_size, ssim_size)`（默认 128x128/256x256）进行灰度比对，未保持原图长宽比，可能引入拉伸形变失真。

4. **匹配层级判定粗糙 (Coarse Tiering Semantics)**
   - 当前以通过阶段计数判定层级（`≥3` confirmed, `≥1` suspected, `0` none），Faiss 召回但未通过 dHash 的候选直接标为 `none`，无法区分“语义高度相似但非重复”与“完全无关”；`suspected` 仅代表通过单哈希，缺乏置信度量化。

### Data Flow (Search)

```
POST /api/v1/search (Image query)
  │
  ├─ routes.search_endpoint()
  │   ├─ _validate_image() → PIL Image
  │   └─ services.search.search(image)
  │
  └─ search()
      ├─ classify_texture(image) → route to "skin" or "cape"
      ├─ single2double(image) if 64x32 single-layer skin
      ├─ compute_phash(image) (0° lock, 0% GPU)
      ├─ indexer.search_binary(hash, k=top_k) → Flat SIMD scan (~3.3ms)
      ├─ Verifier (DiffMask default / SSIM optional) on Top-K
      └─ Return Top-K with Hamming distance & verification score
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
.
├── dev.ps1                    # Native Windows concurrent dev script (FastAPI + Vite)
├── src/twin/
│   ├── main.py                # FastAPI app, CORS, lifespan (model→index→sync→auto-save)
│   ├── core/
│   │   └── config.py          # TWIN_ env vars, path resolution, ensure_dirs()
│   ├── models/
│   │   ├── clip_model.py      # CLIP singleton, encode_image(s), encode_text()
│   │   └── dinov2_model.py    # DINOv2 vision transformer singleton
│   ├── services/
│   │   ├── embedding.py       # Multi-model embedding abstraction (CLIP / DINOv2)
│   │   ├── hasher.py          # aHash, dHash, pHash, SSIM (PyTorch CUDA / skimage)
│   │   ├── indexer.py         # Faiss Flat / IVF / HNSW + metadata + auto-save thread
│   │   ├── index_service.py   # index_single, async index_batch, double-buffering preloader
│   │   ├── search.py          # 2-stage + 4-filter funnel orchestrator + text search
│   │   └── sync.py            # Startup auto-sync with double-buffering preloader
│   ├── api/
│   │   ├── routes.py          # REST & SSE streaming endpoints
│   │   └── schemas.py         # Pydantic v2 request/response models
│   └── utils/
│       ├── image.py           # Parallel load_images(), formats, validation
│       └── metrics.py         # Progress metrics, rolling rate, ETA calculation
│
├── tests/
│   ├── conftest.py            # Fixtures & env setup
│   ├── benchmarks/            # Comprehensive benchmark suite (CLIP, DINOv2, Faiss, SSIM, I/O)
│   ├── test_api.py            # Integration tests for core endpoints
│   ├── test_concurrency_tasks.py # Async task LRU, isolation, and preload resilience tests
│   ├── test_dinov2.py         # DINOv2 model tests
│   ├── test_rotation_search.py# Rotation invariance tests
│   └── ...                    # Service & router unit tests
│
└── src/example/web/           # React frontend (Vite + TypeScript)
    ├── src/
    │   ├── App.tsx            # Tab layout (Search | Browse | Manage)
    │   ├── api/twin.ts        # Typed Axios API client
    │   ├── pages/
    │   │   ├── Search.tsx     # Image & text multi-modal search + zoom preview
    │   │   ├── Browse.tsx     # Paginated thumbnail gallery
    │   │   └── Manage.tsx     # Live status, SSE task progress, and runtime config
    │   └── index.css          # Design tokens (Euclid Circular A, pill buttons)
    └── vite.config.ts         # Proxy /api/v1 → localhost:8000
```

## Configuration

Set via environment variables with `TWIN_` prefix (or `.env` file):

| Variable                  | Default       | Description                                                         |
| ------------------------- | ------------- | ------------------------------------------------------------------- |
| `TWIN_MODEL_TYPE`         | `clip`        | Embedding model architecture (`clip` or `dinov2`)                  |
| `TWIN_MODEL_NAME`         | `ViT-B-32`    | CLIP model variant                                                  |
| `TWIN_PRETRAINED`         | `openai`      | Pretrained weights tag for CLIP                                     |
| `TWIN_DINOV2_MODEL_NAME`  | `vit_small_patch14_dinov2` | DINOv2 model architecture name                        |
| `TWIN_DEVICE`             | (auto)        | `cuda`, `mps`, `cpu`, or empty for auto-detect                      |
| `TWIN_TOP_K`              | `100`         | Faiss candidates to retrieve                                        |
| `TWIN_DHASH_THRESHOLD`    | `10`          | Max dHash Hamming distance for match                                |
| `TWIN_PHASH_THRESHOLD`    | `12`          | Max pHash Hamming distance for match                                |
| `TWIN_SSIM_THRESHOLD`     | `0.90`        | Min SSIM for structural match                                       |
| `TWIN_SSIM_DEVICE`        | (auto)        | SSIM compute device (`cuda`, `mps`, `cpu`, or empty for auto)       |
| `TWIN_SSIM_SIZE`          | `256`         | Resolution for SSIM grayscale comparison (N×N)                      |
| `TWIN_ROTATION_INVARIANT` | `false`       | Tolerates 90°/180°/270° orthogonal image rotations                 |
| `TWIN_BATCH_SIZE`         | `64`          | Images per forward pass / I/O batch                                 |
| `TWIN_AUTO_SAVE_INTERVAL` | `120`         | Seconds between auto-saves (0=disable)                              |
| `TWIN_FAISS_INDEX_TYPE`              | `ivf_flat`    | Faiss index type: `flat`, `ivf_flat`, `ivf_pq`, or `hnsw`       |
| `TWIN_FAISS_NLIST`                   | `0`           | IVF cluster count; 0 = auto (`4 * sqrt(N)`)                   |
| `TWIN_FAISS_NPROBE`                  | `16`          | IVF search-time probe count; higher = more recall              |
| `TWIN_FAISS_AUTO_UPGRADE`            | `true`        | Auto-convert Flat→IVF when enough vectors accumulated         |
| `TWIN_FAISS_GPU`                     | `true`        | Try GPU Faiss if CUDA available (requires `faiss-gpu`; ignored for HNSW) |
| `TWIN_FAISS_HNSW_M`                  | `32`          | HNSW graph degree (4–64)                                       |
| `TWIN_FAISS_HNSW_EF_CONSTRUCTION`    | `200`         | HNSW build-time exploration depth (100–2000)                   |
| `TWIN_FAISS_HNSW_EF_SEARCH`          | `128`         | HNSW search-time depth; higher = more recall                   |
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
| 13  | **Sync status endpoint** — `GET /api/v1/sync/status` to report background sync progress + ETA | § 背景任务                         |
| 14  | **IndexIVFPQ** — Product Quantization with auto-upgrade & sub-space quantizers           | § Faiss 索引机制, Table: IndexIVFPQ |
| 15  | **Parallel Image I/O** — `load_images()` thread pool for 30% faster batch ingestion     | § 生产级项目结构与性能瓶颈      |
| 16  | **aHash & Rotation Invariance** — average-hash & 4-orthogonal orientation tolerance       | § 感知哈希的稳健性评估 §1        |
| 17  | **Multi-modal Text Search** — `POST /api/v1/search/text` via CLIP text transformer      | § 多模态融合检索            |
| 18  | **Async Batch Indexing** — Background task manager with `GET /index/batch/status/{task_id}` | § 背景任务                  |
| 19  | **DINOv2 embedding** — `TWIN_MODEL_TYPE=dinov2` via timm/PyTorch Vision Transformers       | § 嵌入模型的选择, Table: DINOv2 |
| 20  | **CUDA SSIM** — GPU-accelerate PyTorch 2D Convolution SSIM computation with auto-fallback | § 硬件原生的异构计算         |
| 21  | **Pipelined Double-Buffering** — Overlap disk I/O and GPU inference in indexing and startup sync (+19.6% throughput) | § 生产级项目结构与性能瓶颈 |
| 22  | **Task SSE Progress Streaming** — Real-time event streams (`/sync/status/stream`, `/index/batch/status/{task_id}/stream`) with live ETA | § 大文件上传与存储管道优化 |

### High Priority 🔴

All high priority engineering items have been implemented and verified. ✅

### Low Priority / Future 🟢

| #   | Task                                                                                                 | Report Reference                     | Notes                                                                         |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------- |
| 1   | **Adaptive thresholds** — ML-driven threshold tuning per image domain (anime, photo, document)       | § 自适应阈值学习                            | Replaces hardcoded `TWIN_DHASH_THRESHOLD` etc.                                |
| 2   | **Preprocessing pipeline** — configurable interpolation (`INTER_AREA`, `INTER_CUBIC`), CIELab option | § 图像预处理对检索一致性的影响                     | PIL currently delegates to defaults; OpenCV `INTER_AREA` yields better hashes |
| 3   | **Distributed sharding** — manual shard routing or migrate to Milvus / Qdrant / pgvector             | § 扩展性：Faiss 与 purpose-built 向量数据库的博弈 | Beyond single-machine scope; relevant at 10M+ images                          |

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
