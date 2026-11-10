# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report

> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**
> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) and pixel-level `DiffMaskVerifier`.

## 🖥️ Execution Environment Metadata

| Parameter | Value |
| :--- | :--- |
| **Commit SHA** | `49f7de9` |
| **Python** | `3.12.11` |
| **OS Platform** | `Windows 11 (AMD64)` |
| **CPU** | `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD` |
| **Faiss** | `1.15.0` |
| **Pillow** | `12.3.0` |
| **PRNG Seed** | `42 (Deterministic)` |

## 🏆 Executive Summary & Strategy Matrix

> **Note on Latencies & Rates**: `Latency` reflects Wall-clock Median (P95) over multiple statistical rounds. Throughput (`prob/s`) uses `end_to_end_ms` as the strict rate denominator.

| Scenario | Mode | Scale (Items/GT) | Wall Latency (Med/P95) | Throughput | Formed Batches | Purity | Coverage | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Duplicate Swarms** | `confirmed` | 105 / 3 | **224.6ms** (244.3ms) | 276 prob/s | 13 batches (56 skins) | **100.0%** | 69.4% | **100.0%** |
| **Massive 50-Skin Duplicate Swarm** | `confirmed` | 70 / 1 | **120.3ms** (121.8ms) | 249 prob/s | 10 batches (50 skins) | **100.0%** | 81.6% | **100.0%** |
| **Derivative Variant Clusters** | `variants` | 70 / 5 | **195.0ms** (220.7ms) | 231 prob/s | 17 batches (42 skins) | **60.0%** | 32.0% | **80.0%** |
| **Sparse Real Haystack (Confirmed)** | `confirmed` | 245 / 10 | **133.9ms** (727.9ms) | 1,493 prob/s | 10 batches (25 skins) | **100.0%** | 100.0% | **100.0%** |
| **Sparse Real Haystack (Variants)** | `variants` | 245 / 5 | **599.4ms** (1322.8ms) | 334 prob/s | 16 batches (47 skins) | **96.9%** | 100.0% | **100.0%** |
| **Sequential Seed Strategy** | `confirmed` | 245 / 10 | **109.4ms** (111.1ms) | 1,828 prob/s | 10 batches (25 skins) | **100.0%** | 100.0% | **100.0%** |
| **Threshold Boundary Gating** | `confirmed` | 4 / 1 | **9.1ms** (9.8ms) | 329 prob/s | 1 batches (2 skins) | **100.0%** | 100.0% | **100.0%** |
| **Anti-Transitivity Elimination** | `variants` | 3 / 1 | **4.5ms** (5.4ms) | 442 prob/s | 1 batches (2 skins) | **100.0%** | 100.0% | **100.0%** |

## 🔬 Telemetry & Fine-Grained Latency Decomposition

> **Decomposition**: `service_query_ms` = `instrumented_stage_ms` (`recall` + `io` + `verify`) + `unattributed_ms` (dynamic elimination loop overhead & metadata retrieval).

| Scenario | Service / Wall | Binary Recall | Parallel I/O | DiffMask Verify | Loop Overhead | Verified Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exact Duplicate Swarms` | 224.2ms / 224.6ms | 20.3ms (10.2%) | 64.7ms (32.6%) | 113.5ms (57.2%) | 25.7ms | 446 (1,986 pairs/s) |
| `Massive 50-Skin Duplicate Swarm` | 120.0ms / 120.3ms | 9.7ms (9.2%) | 41.0ms (38.9%) | 54.6ms (51.8%) | 14.7ms | 207 (1,720 pairs/s) |
| `Derivative Variant Clusters` | 194.5ms / 195.0ms | 28.1ms (16.1%) | 58.7ms (33.6%) | 88.1ms (50.4%) | 19.6ms | 326 (1,672 pairs/s) |
| `Sparse Real Haystack (Confirmed)` | 133.6ms / 133.9ms | 31.1ms (27.0%) | 60.1ms (52.1%) | 24.2ms (20.9%) | 18.2ms | 68 (508 pairs/s) |
| `Sparse Real Haystack (Variants)` | 598.9ms / 599.4ms | 50.8ms (9.3%) | 215.9ms (39.4%) | 280.7ms (51.3%) | 51.6ms | 949 (1,583 pairs/s) |
| `Sequential Seed Strategy` | 109.1ms / 109.4ms | 24.5ms (26.0%) | 49.5ms (52.6%) | 20.1ms (21.3%) | 15.1ms | 68 (621 pairs/s) |
| `Threshold Boundary Gating` | 8.9ms / 9.1ms | 2.3ms (28.7%) | 4.8ms (59.4%) | 1.0ms (11.9%) | 0.8ms | 4 (438 pairs/s) |
| `Anti-Transitivity Elimination` | 4.3ms / 4.5ms | 0.4ms (10.8%) | 2.7ms (71.2%) | 0.7ms (18.1%) | 0.6ms | 2 (442 pairs/s) |

## ⚖️ Algorithmic Trade-offs & Variants Mode Honest Disclosure

> [!IMPORTANT]
> **Variants Mode Trade-off Disclosure**:
> In `mode='variants'`, search constraints are intentionally relaxed to Hamming $\le 12$ and DiffMask $\le 0.35$ > in order to discover derivative community recolors, minor clothing modifications, and edit series.
> When exposed to hundreds of arbitrary community background distractors, cluster purity naturally settles at **~56% ~ 68%**.
> This is a deliberate, mathematically sound engineering trade-off: **trading single-pass precision to maximize recall on loose artistic variants**.
> Clustering suggestions generated under `variants` mode are intended for human review workflows and must never trigger unsupervised destructive deduplication.

---

## 📝 Detailed Scenario Analysis

### Scenario: Exact Duplicate Swarms (`mode=confirmed`)
- **Dataset Scale**: 105 items (3 GT clusters)
- **Wall-Clock Latency**: **224.55 ms** (service internal: 224.19 ms) | **276.1 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 13 (covering 56 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0)
  - **Target Group Recall**: **100.0%** (3 ground-truth groups)
  - **Member Coverage**: **69.4%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 20.28 ms (10.2%) | 491 candidates recalled
  - **Parallel Image I/O**: 64.69 ms (32.6%) | 105 images decoded in thread pool
  - **DiffMask Verify**: 113.52 ms (57.2%) | 446 pairs verified (1,986.2 pairs/sec)
  - **Unattributed Loop Overhead**: 25.70 ms (loop branching, metadata lookups, elimination)

### Scenario: Massive 50-Skin Duplicate Swarm (`mode=confirmed`)
- **Dataset Scale**: 70 items (1 GT clusters)
- **Wall-Clock Latency**: **120.33 ms** (service internal: 120.00 ms) | **249.3 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 10 (covering 50 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0)
  - **Target Group Recall**: **100.0%** (1 ground-truth groups)
  - **Member Coverage**: **81.6%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 9.72 ms (9.2%) | 274 candidates recalled
  - **Parallel Image I/O**: 41.01 ms (38.9%) | 70 images decoded in thread pool
  - **DiffMask Verify**: 54.57 ms (51.8%) | 207 pairs verified (1,720.2 pairs/sec)
  - **Unattributed Loop Overhead**: 14.70 ms (loop branching, metadata lookups, elimination)

### Scenario: Derivative Variant Clusters (`mode=variants`)
- **Dataset Scale**: 70 items (5 GT clusters)
- **Wall-Clock Latency**: **194.97 ms** (service internal: 194.50 ms) | **230.8 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 17 (covering 42 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **60.0%** (100% pure batches: 23.5%, impure batches: 13)
  - **Target Group Recall**: **80.0%** (5 ground-truth groups)
  - **Member Coverage**: **32.0%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.3477 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 28.08 ms (16.1%) | 331 candidates recalled
  - **Parallel Image I/O**: 58.73 ms (33.6%) | 70 images decoded in thread pool
  - **DiffMask Verify**: 88.05 ms (50.4%) | 326 pairs verified (1,672.1 pairs/sec)
  - **Unattributed Loop Overhead**: 19.64 ms (loop branching, metadata lookups, elimination)

### Scenario: Sparse Real Haystack (Confirmed) (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Wall-Clock Latency**: **133.94 ms** (service internal: 133.60 ms) | **1,493.2 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 10 (covering 25 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0)
  - **Target Group Recall**: **100.0%** (10 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 31.15 ms (27.0%) | 68 candidates recalled
  - **Parallel Image I/O**: 60.09 ms (52.1%) | 60 images decoded in thread pool
  - **DiffMask Verify**: 24.16 ms (20.9%) | 68 pairs verified (507.7 pairs/sec)
  - **Unattributed Loop Overhead**: 18.19 ms (loop branching, metadata lookups, elimination)

### Scenario: Sparse Real Haystack (Variants) (`mode=variants`)
- **Dataset Scale**: 245 items (5 GT clusters)
- **Wall-Clock Latency**: **599.42 ms** (service internal: 598.93 ms) | **333.7 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 16 (covering 47 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **96.9%** (100% pure batches: 93.8%, impure batches: 1)
  - **Target Group Recall**: **100.0%** (5 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.3481 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 50.80 ms (9.3%) | 949 candidates recalled
  - **Parallel Image I/O**: 215.87 ms (39.4%) | 208 images decoded in thread pool
  - **DiffMask Verify**: 280.71 ms (51.3%) | 949 pairs verified (1,583.2 pairs/sec)
  - **Unattributed Loop Overhead**: 51.55 ms (loop branching, metadata lookups, elimination)

### Scenario: Sequential Seed Strategy (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Wall-Clock Latency**: **109.42 ms** (service internal: 109.13 ms) | **1,827.9 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 10 (covering 25 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0)
  - **Target Group Recall**: **100.0%** (10 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 24.48 ms (26.0%) | 68 candidates recalled
  - **Parallel Image I/O**: 49.49 ms (52.6%) | 60 images decoded in thread pool
  - **DiffMask Verify**: 20.06 ms (21.3%) | 68 pairs verified (621.5 pairs/sec)
  - **Unattributed Loop Overhead**: 15.10 ms (loop branching, metadata lookups, elimination)

### Scenario: Threshold Boundary Gating (`mode=confirmed`)
- **Dataset Scale**: 4 items (1 GT clusters)
- **Wall-Clock Latency**: **9.12 ms** (service internal: 8.94 ms) | **328.8 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 1 (covering 2 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0)
  - **Target Group Recall**: **100.0%** (1 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.0449 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 2.33 ms (28.7%) | 4 candidates recalled
  - **Parallel Image I/O**: 4.82 ms (59.4%) | 4 images decoded in thread pool
  - **DiffMask Verify**: 0.96 ms (11.9%) | 4 pairs verified (438.4 pairs/sec)
  - **Unattributed Loop Overhead**: 0.82 ms (loop branching, metadata lookups, elimination)

### Scenario: Anti-Transitivity Elimination (`mode=variants`)
- **Dataset Scale**: 3 items (1 GT clusters)
- **Wall-Clock Latency**: **4.53 ms** (service internal: 4.32 ms) | **441.7 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 1 (covering 2 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0)
  - **Target Group Recall**: **100.0%** (1 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of non-seed GT cluster members recovered)
  - **Max Observed DiffRatio**: 0.1797 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 0.40 ms (10.8%) | 2 candidates recalled
  - **Parallel Image I/O**: 2.67 ms (71.2%) | 3 images decoded in thread pool
  - **DiffMask Verify**: 0.68 ms (18.1%) | 2 pairs verified (441.7 pairs/sec)
  - **Unattributed Loop Overhead**: 0.57 ms (loop branching, metadata lookups, elimination)
