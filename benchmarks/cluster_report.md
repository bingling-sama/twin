# 📊 Minecraft Texture Seedless Discovery & Clustering Benchmark Report

> Benchmark of `discover_similar_batches()` using **Greedy Leader Dynamic Elimination**
> over native binary Faiss (`IndexBinaryFlat` SIMD pHash) and pixel-level `DiffMaskVerifier`.

## 🖥️ Execution Environment Metadata

| Parameter | Value |
| :--- | :--- |
| **Generated At** | `2026-11-11T12:03:16+08:00` |
| **Commit SHA** | `c393ffe (clean)` |
| **Source Digest (SHA-256)** | `ab0b74e9718c83ae` |
| **Python** | `3.12.11` |
| **OS Platform** | `Windows 11 (AMD64)` |
| **CPU** | `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 logical cores)` |
| **GPU** | `NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB)` |
| **Torch / Faiss / Pillow** | `2.6.0+cu124 / 1.15.0 / 12.3.0` |
| **Statistical Rounds** | `5` |
| **Haystack Config** | `samples=200, use_zip=True` |
| **images.zip Fingerprint** | `5,412,351,199 bytes | 664,633 PNGs | sha256=fb75e5017085aa5b0f951a2f26b6a3a4b5149f8d63cdb347c92676e67ca5d6fe` |
| **Confirmed Thresholds** | `hamming <= 4, diff_ratio <= 0.05` |
| **Variants Thresholds** | `hamming <= 12, diff_ratio <= 0.35` |
| **Recall Window** | `min_k=15, multiplier=3` |

## 🏆 Executive Summary & Strategy Matrix

> **Note on Latencies & Rates**: `Latency` reports `Median (P95) [Min - Max]` wall-clock ms over `Statistical Rounds`. Throughput (`prob/s`) uses `end_to_end_ms` as the strict denominator. `Coverage` counts all recovered GT cluster members (seeds + peers).

| Scenario | Mode | Scale (Items/GT) | Wall Latency Med/P95 [Min-Max] | Throughput | Formed Batches | Purity (Wrong) | Coverage | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Duplicate Swarms** | `confirmed` | 105 / 3 | **64.5ms** (66.5ms) [61.6-66.8] | 822 prob/s | 13 batches (65 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Massive 50-Skin Duplicate Swarm** | `confirmed` | 70 / 1 | **50.8ms** (62.5ms) [47.9-65.2] | 590 prob/s | 10 batches (50 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Derivative Variant Clusters** | `variants` | 70 / 5 | **52.2ms** (56.9ms) [46.1-57.8] | 958 prob/s | 5 batches (25 skins) | **100.0%** (0) | 83.3% | **100.0%** |
| **Sparse Real Haystack (Confirmed)** | `confirmed` | 245 / 10 | **54.9ms** (55.2ms) [51.4-55.3] | 3,642 prob/s | 10 batches (25 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Sparse Real Haystack (Variants)** | `variants` | 245 / 5 | **314.8ms** (337.1ms) [303.7-338.2] | 635 prob/s | 16 batches (47 skins) | **93.8%** (1) | 100.0% | **100.0%** |
| **Sequential Seed Strategy** | `confirmed` | 245 / 10 | **55.8ms** (61.5ms) [52.4-61.8] | 3,583 prob/s | 10 batches (25 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Threshold Boundary Gating (Confirmed)** | `confirmed` | 8 / 2 | **7.5ms** (9.3ms) [6.9-9.6] | 668 prob/s | 2 batches (5 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Threshold Boundary Gating (Variants)** | `variants` | 6 / 2 | **5.3ms** (8.1ms) [4.6-8.8] | 748 prob/s | 2 batches (4 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Anti-Transitivity Elimination** | `variants` | 3 / 1 | **3.4ms** (3.9ms) [3.0-4.0] | 593 prob/s | 1 batches (2 skins) | **100.0%** (0) | 100.0% | **100.0%** |
| **Adversarial Torture Suite** | `confirmed` | 13 / 3 | **14.0ms** (16.2ms) [12.7-16.4] | 716 prob/s | 3 batches (6 skins) | **100.0%** (0) | 100.0% | **100.0%** |

## 🔧 Scenario Reproducibility Matrix

| Scenario | Dataset Seed | PRNG Seed | Seed Strategy | batch_size | max_batches | max_probes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exact Duplicate Swarms` | `101` | `42` | `random` | 5 | 50 | 200 |
| `Massive 50-Skin Duplicate Swarm` | `102` | `42` | `sequential` | 5 | 50 | 200 |
| `Derivative Variant Clusters` | `201` | `42` | `random` | 5 | 30 | 200 |
| `Sparse Real Haystack (Confirmed)` | `301` | `42` | `random` | 5 | 30 | 200 |
| `Sparse Real Haystack (Variants)` | `301` | `42` | `random` | 5 | 30 | 200 |
| `Sequential Seed Strategy` | `301` | `42` | `sequential` | 5 | 30 | 200 |
| `Threshold Boundary Gating (Confirmed)` | `401` | `42` | `sequential` | 5 | 5 | 200 |
| `Threshold Boundary Gating (Variants)` | `401` | `42` | `sequential` | 5 | 5 | 200 |
| `Anti-Transitivity Elimination` | `501` | `42` | `sequential` | 5 | 5 | 200 |
| `Adversarial Torture Suite` | `600` | `42` | `random` | 5 | 10 | 200 |

## 🔬 Telemetry & Fine-Grained Latency Decomposition

> **Decomposition**: `end_to_end_ms` = `instrumented_stage_ms` (`recall` + `io` + `verify`) + `unattributed_ms` (loop branching, metadata lookups, elimination, harness overhead); `service_query_ms` records internal `discover_similar_batches()` elapsed time.

| Scenario | Service / Wall | Binary Recall | Parallel I/O | DiffMask Verify | Unattributed (e2e) | Verified Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exact Duplicate Swarms` | 64.2ms / 64.5ms | 3.3ms (6.0%) | 37.8ms (70.0%) | 12.9ms (23.9%) | 10.5ms | 52 (806 pairs/s) |
| `Massive 50-Skin Duplicate Swarm` | 50.5ms / 50.8ms | 1.3ms (3.0%) | 30.0ms (72.0%) | 10.4ms (25.0%) | 9.2ms | 40 (787 pairs/s) |
| `Derivative Variant Clusters` | 51.8ms / 52.2ms | 1.6ms (3.7%) | 32.4ms (75.4%) | 9.0ms (20.8%) | 9.2ms | 31 (594 pairs/s) |
| `Sparse Real Haystack (Confirmed)` | 54.6ms / 54.9ms | 2.9ms (6.3%) | 33.1ms (70.8%) | 10.7ms (22.9%) | 8.2ms | 68 (1,238 pairs/s) |
| `Sparse Real Haystack (Variants)` | 314.2ms / 314.8ms | 7.0ms (2.5%) | 118.2ms (42.1%) | 155.3ms (55.4%) | 34.3ms | 949 (3,014 pairs/s) |
| `Sequential Seed Strategy` | 55.6ms / 55.8ms | 3.0ms (6.3%) | 33.3ms (70.3%) | 11.1ms (23.5%) | 8.4ms | 68 (1,218 pairs/s) |
| `Threshold Boundary Gating (Confirmed)` | 7.3ms / 7.5ms | 0.2ms (2.5%) | 5.0ms (79.1%) | 1.2ms (18.4%) | 1.1ms | 6 (802 pairs/s) |
| `Threshold Boundary Gating (Variants)` | 5.2ms / 5.3ms | 0.1ms (2.9%) | 3.8ms (82.6%) | 0.7ms (14.5%) | 0.8ms | 3 (561 pairs/s) |
| `Anti-Transitivity Elimination` | 3.2ms / 3.4ms | 0.1ms (2.4%) | 2.4ms (82.9%) | 0.4ms (14.6%) | 0.5ms | 2 (593 pairs/s) |
| `Adversarial Torture Suite` | 13.8ms / 14.0ms | 0.5ms (4.2%) | 8.7ms (75.4%) | 2.4ms (20.4%) | 2.4ms | 7 (501 pairs/s) |

## ⚖️ Algorithmic Trade-offs & Variants Mode Honest Disclosure

> [!IMPORTANT]
> **Variants Mode Trade-off Disclosure**:
> In `mode='variants'`, search constraints are relaxed to Hamming $\le 12$ and DiffMask $\le 0.35$ to discover derivative community recolors, minor clothing modifications, and edit series.
> When exposed to arbitrary community skins (`Sparse Real Haystack (Variants)`), unlabeled background lookalikes may join discovered batches (see measured `Purity` and `Wrong` counts in the matrix above).
> This reflects a deliberate recall-vs-precision engineering trade-off: clustering suggestions under `variants` mode are intended for human review workflows rather than unsupervised deletion.

---

## 📝 Detailed Scenario Analysis

### Scenario: Exact Duplicate Swarms (`mode=confirmed`)
- **Dataset Scale**: 105 items (3 GT clusters)
- **Wall-Clock Latency**: **64.51 ms** (service internal: 64.17 ms) | **821.5 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 13 (covering 65 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (3 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 3.27 ms (6.0%) | 115 candidates recalled
  - **Parallel Image I/O**: 37.80 ms (70.0%) | 65 images decoded in thread pool
  - **DiffMask Verify**: 12.93 ms (23.9%) | 52 pairs verified (806.0 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 10.52 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Massive 50-Skin Duplicate Swarm (`mode=confirmed`)
- **Dataset Scale**: 70 items (1 GT clusters)
- **Wall-Clock Latency**: **50.82 ms** (service internal: 50.47 ms) | **590.3 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 10 (covering 50 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (1 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 1.25 ms (3.0%) | 103 candidates recalled
  - **Parallel Image I/O**: 29.97 ms (72.0%) | 50 images decoded in thread pool
  - **DiffMask Verify**: 10.41 ms (25.0%) | 40 pairs verified (787.1 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 9.18 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Derivative Variant Clusters (`mode=variants`)
- **Dataset Scale**: 70 items (5 GT clusters)
- **Wall-Clock Latency**: **52.17 ms** (service internal: 51.82 ms) | **958.5 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 5 (covering 25 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (5 ground-truth groups)
  - **Member Coverage**: **83.3%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.3477 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 1.60 ms (3.7%) | 35 candidates recalled
  - **Parallel Image I/O**: 32.39 ms (75.4%) | 41 images decoded in thread pool
  - **DiffMask Verify**: 8.95 ms (20.8%) | 31 pairs verified (594.2 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 9.23 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Sparse Real Haystack (Confirmed) (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Wall-Clock Latency**: **54.91 ms** (service internal: 54.65 ms) | **3,642.0 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 10 (covering 25 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (10 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 2.92 ms (6.3%) | 68 candidates recalled
  - **Parallel Image I/O**: 33.07 ms (70.8%) | 60 images decoded in thread pool
  - **DiffMask Verify**: 10.69 ms (22.9%) | 68 pairs verified (1,238.3 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 8.24 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Sparse Real Haystack (Variants) (`mode=variants`)
- **Dataset Scale**: 245 items (5 GT clusters)
- **Wall-Clock Latency**: **314.83 ms** (service internal: 314.16 ms) | **635.3 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 16 (covering 47 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **93.8%** (100% pure batches: 93.8%, impure batches: 1, wrong peers: 1)
  - **Target Group Recall**: **100.0%** (5 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.3481 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 7.00 ms (2.5%) | 949 candidates recalled
  - **Parallel Image I/O**: 118.23 ms (42.1%) | 208 images decoded in thread pool
  - **DiffMask Verify**: 155.32 ms (55.4%) | 949 pairs verified (3,014.3 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 34.28 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Sequential Seed Strategy (`mode=confirmed`)
- **Dataset Scale**: 245 items (10 GT clusters)
- **Wall-Clock Latency**: **55.83 ms** (service internal: 55.56 ms) | **3,582.5 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 10 (covering 25 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (10 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 2.97 ms (6.3%) | 68 candidates recalled
  - **Parallel Image I/O**: 33.33 ms (70.3%) | 60 images decoded in thread pool
  - **DiffMask Verify**: 11.13 ms (23.5%) | 68 pairs verified (1,218.1 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 8.40 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Threshold Boundary Gating (Confirmed) (`mode=confirmed`)
- **Dataset Scale**: 8 items (2 GT clusters)
- **Wall-Clock Latency**: **7.48 ms** (service internal: 7.32 ms) | **668.3 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 2 (covering 5 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (2 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.0500 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 0.16 ms (2.5%) | 6 candidates recalled
  - **Parallel Image I/O**: 5.04 ms (79.1%) | 7 images decoded in thread pool
  - **DiffMask Verify**: 1.17 ms (18.4%) | 6 pairs verified (802.0 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 1.12 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Threshold Boundary Gating (Variants) (`mode=variants`)
- **Dataset Scale**: 6 items (2 GT clusters)
- **Wall-Clock Latency**: **5.35 ms** (service internal: 5.21 ms) | **748.0 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 2 (covering 4 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (2 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.3500 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 0.13 ms (2.9%) | 3 candidates recalled
  - **Parallel Image I/O**: 3.75 ms (82.6%) | 5 images decoded in thread pool
  - **DiffMask Verify**: 0.66 ms (14.5%) | 3 pairs verified (561.0 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 0.81 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Anti-Transitivity Elimination (`mode=variants`)
- **Dataset Scale**: 3 items (1 GT clusters)
- **Wall-Clock Latency**: **3.37 ms** (service internal: 3.22 ms) | **593.2 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 1 (covering 2 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (1 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.1800 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 0.07 ms (2.4%) | 2 candidates recalled
  - **Parallel Image I/O**: 2.37 ms (82.9%) | 3 images decoded in thread pool
  - **DiffMask Verify**: 0.42 ms (14.6%) | 2 pairs verified (593.2 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 0.51 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)

### Scenario: Adversarial Torture Suite (`mode=confirmed`)
- **Dataset Scale**: 13 items (3 GT clusters)
- **Wall-Clock Latency**: **13.97 ms** (service internal: 13.80 ms) | **715.7 Probes/sec** (denominator: end_to_end_ms)
- **Batches Formed**: 3 (covering 6 skins)
- **Clustering Quality & Precision**:
  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, impure batches: 0, wrong peers: 0)
  - **Target Group Recall**: **100.0%** (3 ground-truth groups)
  - **Member Coverage**: **100.0%** (fraction of target GT cluster members—including seeds and peers—recovered)
  - **Max Observed DiffRatio**: 0.0096 (strictly within configured limits)
- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:
  - **Binary Recall**: 0.48 ms (4.2%) | 8 candidates recalled
  - **Parallel Image I/O**: 8.74 ms (75.4%) | 12 images decoded in thread pool
  - **DiffMask Verify**: 2.37 ms (20.4%) | 7 pairs verified (501.0 pairs/sec)
  - **Unattributed Overhead (e2e - stages)**: 2.38 ms (loop branching, metadata lookups, elimination, harness/FFI overhead)
