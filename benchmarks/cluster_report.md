# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report

> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**
> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) and pixel-level `DiffMaskVerifier`.

## 🏆 Executive Summary & Strategy Matrix

| Scenario | Mode | Scale (Items / GT) | Latency | Throughput | Formed Batches | Purity | 100% Pure | GT Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Duplicate Swarms** | `confirmed` | 105 / 3 | **146.9ms** | 380 prob/s | 15 batches (64 skins) | **100.0%** | 100.0% | **100.0%** |
| **Massive 50-Skin Duplicate Swarm** | `confirmed` | 70 / 1 | **79.2ms** | 378 prob/s | 10 batches (50 skins) | **100.0%** | 100.0% | **100.0%** |
| **Derivative Variant Clusters** | `variants` | 70 / 5 | **119.7ms** | 392 prob/s | 17 batches (40 skins) | **56.4%** | 17.6% | **80.0%** |
| **Sparse Real Haystack (Confirmed)** | `confirmed` | 245 / 10 | **63.7ms** | 3,127 prob/s | 13 batches (32 skins) | **100.0%** | 100.0% | **100.0%** |
| **Sparse Real Haystack (Variants)** | `variants` | 245 / 5 | **423.9ms** | 471 prob/s | 16 batches (45 skins) | **96.9%** | 93.8% | **100.0%** |
| **Sequential Seed Strategy** | `confirmed` | 245 / 10 | **62.4ms** | 3,192 prob/s | 13 batches (31 skins) | **100.0%** | 100.0% | **100.0%** |

## 🔬 Telemetry & Stage Latency Decomposition

| Scenario | Binary Recall (ms / %) | Parallel I/O (ms / %) | DiffMask Verify (ms / %) | Verified Pairs | Verify Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Exact Duplicate Swarms` | 8.1ms (6.3%) | 45.2ms (34.8%) | 76.4ms (58.9%) | 422 | 2,866 pairs/s |
| `Massive 50-Skin Duplicate Swarm` | 2.3ms (3.3%) | 29.0ms (42.1%) | 37.6ms (54.5%) | 207 | 2,607 pairs/s |
| `Derivative Variant Clusters` | 3.8ms (3.6%) | 33.6ms (31.3%) | 70.0ms (65.2%) | 381 | 3,177 pairs/s |
| `Sparse Real Haystack (Confirmed)` | 7.8ms (14.6%) | 34.3ms (64.2%) | 11.4ms (21.2%) | 48 | 750 pairs/s |
| `Sparse Real Haystack (Variants)` | 16.8ms (4.4%) | 120.5ms (31.5%) | 245.2ms (64.1%) | 1004 | 2,366 pairs/s |
| `Sequential Seed Strategy` | 9.1ms (17.4%) | 31.7ms (60.4%) | 11.7ms (22.3%) | 50 | 798 pairs/s |

---

## 📝 Detailed Scenario Analysis

### Scenario: Exact Duplicate Swarms (`mode=confirmed`)
- **Dataset Scale**: 105 items (3 GT clusters)
- **Execution Time**: **146.92 ms** | **380.4 Probes/sec**
- **Batches Formed**: 15 (covering 64 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (3 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0000 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 8.14 ms (6.3%) | 490 recalled
  - **Parallel I/O**: 45.21 ms (34.8%) | 105 images decoded
  - **DiffMask Verify**: 76.42 ms (58.9%) | 422 pairs (2,866.5 pairs/sec)

### Scenario: Massive 50-Skin Duplicate Swarm (`mode=confirmed`)
- **Dataset Scale**: 70 items (1 GT clusters)
- **Execution Time**: **79.15 ms** | **377.8 Probes/sec**
- **Batches Formed**: 10 (covering 50 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (1 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0000 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 2.29 ms (3.3%) | 274 recalled
  - **Parallel I/O**: 29.04 ms (42.1%) | 70 images decoded
  - **DiffMask Verify**: 37.59 ms (54.5%) | 207 pairs (2,606.6 pairs/sec)

### Scenario: Derivative Variant Clusters (`mode=variants`)
- **Dataset Scale**: 70 items (5 GT clusters)
- **Execution Time**: **119.66 ms** | **391.9 Probes/sec**
- **Batches Formed**: 17 (covering 40 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **56.4%** (100% pure: 17.6%)
  - **Cluster Recall**: **80.0%** (5 ground-truth groups)
  - **Max Observed DiffRatio**: 0.2995 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 3.84 ms (3.6%) | 381 recalled
  - **Parallel I/O**: 33.56 ms (31.3%) | 70 images decoded
  - **DiffMask Verify**: 69.99 ms (65.2%) | 381 pairs (3,176.8 pairs/sec)

### Scenario: Sparse Real Haystack (Confirmed) (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Execution Time**: **63.72 ms** | **3,126.8 Probes/sec**
- **Batches Formed**: 13 (covering 32 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (10 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0473 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 7.80 ms (14.6%) | 48 recalled
  - **Parallel I/O**: 34.31 ms (64.2%) | 54 images decoded
  - **DiffMask Verify**: 11.36 ms (21.2%) | 48 pairs (750.4 pairs/sec)

### Scenario: Sparse Real Haystack (Variants) (`mode=variants`)
- **Dataset Scale**: 245 items (5 GT clusters)
- **Execution Time**: **423.85 ms** | **471.4 Probes/sec**
- **Batches Formed**: 16 (covering 45 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **96.9%** (100% pure: 93.8%)
  - **Cluster Recall**: **100.0%** (5 ground-truth groups)
  - **Max Observed DiffRatio**: 0.1602 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 16.84 ms (4.4%) | 1004 recalled
  - **Parallel I/O**: 120.50 ms (31.5%) | 206 images decoded
  - **DiffMask Verify**: 245.15 ms (64.1%) | 1004 pairs (2,366.3 pairs/sec)

### Scenario: Sequential Seed Strategy (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Execution Time**: **62.38 ms** | **3,192.1 Probes/sec**
- **Batches Formed**: 13 (covering 31 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (10 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0473 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 9.09 ms (17.4%) | 50 recalled
  - **Parallel I/O**: 31.65 ms (60.4%) | 54 images decoded
  - **DiffMask Verify**: 11.66 ms (22.3%) | 50 pairs (798.0 pairs/sec)
