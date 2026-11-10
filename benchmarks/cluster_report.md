# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report

> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**
> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) and pixel-level `DiffMaskVerifier`.

## 🏆 Executive Summary & Strategy Matrix

| Scenario | Mode | Scale (Items / GT) | Latency | Throughput | Formed Batches | Purity | 100% Pure | GT Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Duplicate Swarms** | `confirmed` | 105 / 3 | **164.6ms** | 364 prob/s | 13 batches (58 skins) | **100.0%** | 100.0% | **100.0%** |
| **Massive 50-Skin Duplicate Swarm** | `confirmed` | 70 / 1 | **87.8ms** | 340 prob/s | 10 batches (50 skins) | **100.0%** | 100.0% | **100.0%** |
| **Derivative Variant Clusters** | `variants` | 70 / 5 | **118.7ms** | 353 prob/s | 20 batches (48 skins) | **66.7%** | 30.0% | **100.0%** |
| **Sparse Real Haystack (Confirmed)** | `confirmed` | 245 / 10 | **68.1ms** | 2,923 prob/s | 9 batches (23 skins) | **100.0%** | 100.0% | **90.0%** |
| **Sparse Real Haystack (Variants)** | `variants` | 245 / 5 | **421.4ms** | 474 prob/s | 16 batches (40 skins) | **96.9%** | 93.8% | **300.0%** |
| **Sequential Seed Strategy** | `confirmed` | 245 / 10 | **57.7ms** | 3,450 prob/s | 10 batches (25 skins) | **100.0%** | 100.0% | **100.0%** |

## 🔬 Telemetry & Stage Latency Decomposition

| Scenario | Binary Recall (ms / %) | Parallel I/O (ms / %) | DiffMask Verify (ms / %) | Verified Pairs | Verify Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Exact Duplicate Swarms` | 8.8ms (6.0%) | 47.0ms (32.3%) | 89.6ms (61.7%) | 447 | 2,710 pairs/s |
| `Massive 50-Skin Duplicate Swarm` | 2.5ms (3.3%) | 33.9ms (44.0%) | 40.5ms (52.6%) | 207 | 2,349 pairs/s |
| `Derivative Variant Clusters` | 5.9ms (5.6%) | 34.7ms (33.2%) | 64.0ms (61.2%) | 322 | 2,705 pairs/s |
| `Sparse Real Haystack (Confirmed)` | 12.3ms (21.3%) | 34.4ms (59.4%) | 11.2ms (19.3%) | 41 | 599 pairs/s |
| `Sparse Real Haystack (Variants)` | 12.6ms (3.3%) | 121.6ms (32.0%) | 245.2ms (64.6%) | 1038 | 2,461 pairs/s |
| `Sequential Seed Strategy` | 7.7ms (15.9%) | 30.6ms (63.1%) | 10.2ms (21.0%) | 42 | 725 pairs/s |

---

## 📝 Detailed Scenario Analysis

### Scenario: Exact Duplicate Swarms (`mode=confirmed`)
- **Dataset Scale**: 105 items (3 GT clusters)
- **Execution Time**: **164.60 ms** | **363.8 Probes/sec**
- **Batches Formed**: 13 (covering 58 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (3 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0000 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 8.78 ms (6.0%) | 504 recalled
  - **Parallel I/O**: 46.97 ms (32.3%) | 105 images decoded
  - **DiffMask Verify**: 89.63 ms (61.7%) | 447 pairs (2,710.2 pairs/sec)

### Scenario: Massive 50-Skin Duplicate Swarm (`mode=confirmed`)
- **Dataset Scale**: 70 items (1 GT clusters)
- **Execution Time**: **87.81 ms** | **340.5 Probes/sec**
- **Batches Formed**: 10 (covering 50 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (1 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0000 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 2.54 ms (3.3%) | 274 recalled
  - **Parallel I/O**: 33.86 ms (44.0%) | 70 images decoded
  - **DiffMask Verify**: 40.46 ms (52.6%) | 207 pairs (2,349.3 pairs/sec)

### Scenario: Derivative Variant Clusters (`mode=variants`)
- **Dataset Scale**: 70 items (5 GT clusters)
- **Execution Time**: **118.73 ms** | **352.8 Probes/sec**
- **Batches Formed**: 20 (covering 48 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **66.7%** (100% pure: 30.0%)
  - **Cluster Recall**: **100.0%** (5 ground-truth groups)
  - **Max Observed DiffRatio**: 0.1667 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 5.90 ms (5.6%) | 322 recalled
  - **Parallel I/O**: 34.68 ms (33.2%) | 70 images decoded
  - **DiffMask Verify**: 63.97 ms (61.2%) | 322 pairs (2,705.2 pairs/sec)

### Scenario: Sparse Real Haystack (Confirmed) (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Execution Time**: **68.11 ms** | **2,923.3 Probes/sec**
- **Batches Formed**: 9 (covering 23 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **90.0%** (10 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0000 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 12.33 ms (21.3%) | 41 recalled
  - **Parallel I/O**: 34.43 ms (59.4%) | 46 images decoded
  - **DiffMask Verify**: 11.21 ms (19.3%) | 41 pairs (599.3 pairs/sec)

### Scenario: Sparse Real Haystack (Variants) (`mode=variants`)
- **Dataset Scale**: 245 items (5 GT clusters)
- **Execution Time**: **421.39 ms** | **474.1 Probes/sec**
- **Batches Formed**: 16 (covering 40 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **96.9%** (100% pure: 93.8%)
  - **Cluster Recall**: **300.0%** (5 ground-truth groups)
  - **Max Observed DiffRatio**: 0.1683 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 12.65 ms (3.3%) | 1038 recalled
  - **Parallel I/O**: 121.59 ms (32.0%) | 209 images decoded
  - **DiffMask Verify**: 245.24 ms (64.6%) | 1038 pairs (2,460.7 pairs/sec)

### Scenario: Sequential Seed Strategy (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Execution Time**: **57.73 ms** | **3,450.1 Probes/sec**
- **Batches Formed**: 10 (covering 25 skins)
- **Clustering Quality**:
  - **Cluster Purity**: **100.0%** (100% pure: 100.0%)
  - **Cluster Recall**: **100.0%** (10 ground-truth groups)
  - **Max Observed DiffRatio**: 0.0000 (strictly within bounds)
- **Stages Telemetry & Latency Breakdown**:
  - **Binary Recall**: 7.71 ms (15.9%) | 42 recalled
  - **Parallel I/O**: 30.64 ms (63.1%) | 48 images decoded
  - **DiffMask Verify**: 10.19 ms (21.0%) | 42 pairs (724.5 pairs/sec)
