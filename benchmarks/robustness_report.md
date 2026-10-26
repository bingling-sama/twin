## 🏆 Strategy Comparison Summary

| Strategy | Recall@1 | Recall@5 | MRR | Suspected | Confirmed | P50 Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **current_diffmask** | **92.92%** | 93.33% | 0.9330 | 92.08% | 0.00% | 2.01ms | 2.02ms |
| **ssim_verifier** | **92.92%** | 93.33% | 0.9330 | 92.08% | 0.00% | 1.81ms | 1.80ms |
| **pure_phash** | **92.92%** | 93.33% | 0.9330 | 92.08% | 0.00% | 1.80ms | 1.80ms |
| **pure_clip** | **80.42%** | 92.50% | 0.8578 | 0.00% | 0.00% | 9.86ms | 10.24ms |

### Strategy: current_diffmask
- **Total Queries**: 240
- **Recall@1**: 92.92%
- **Recall@5**: 93.33%
- **MRR**: 0.9330
- **Detected (Confirmed + Suspected)**: 92.08%
- **Confirmed Rate**: 0.00%
- **Suspected Rate**: 92.08%
- **Latency (P50 / P95 / Mean)**: 2.01ms / 2.67ms / 2.02ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 2.08ms |
| `composite` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 2.00ms |
| `cool_tint` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 2.00ms |
| `grayscale` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 2.08ms |
| `hue_shift_15pct` | 30 | 90.0% | 93.3% | 93.3% | 0.0% | 2.03ms |
| `jpeg_q75` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.98ms |
| `pixel_noise` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.87ms |
| `warm_tint` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.99ms |

### Strategy: ssim_verifier
- **Total Queries**: 240
- **Recall@1**: 92.92%
- **Recall@5**: 93.33%
- **MRR**: 0.9330
- **Detected (Confirmed + Suspected)**: 92.08%
- **Confirmed Rate**: 0.00%
- **Suspected Rate**: 92.08%
- **Latency (P50 / P95 / Mean)**: 1.81ms / 2.27ms / 1.80ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.80ms |
| `composite` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.75ms |
| `cool_tint` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.81ms |
| `grayscale` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.73ms |
| `hue_shift_15pct` | 30 | 90.0% | 93.3% | 93.3% | 0.0% | 1.77ms |
| `jpeg_q75` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.90ms |
| `pixel_noise` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.80ms |
| `warm_tint` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.87ms |

### Strategy: pure_phash
- **Total Queries**: 240
- **Recall@1**: 92.92%
- **Recall@5**: 93.33%
- **MRR**: 0.9330
- **Detected (Confirmed + Suspected)**: 92.08%
- **Confirmed Rate**: 0.00%
- **Suspected Rate**: 92.08%
- **Latency (P50 / P95 / Mean)**: 1.80ms / 2.30ms / 1.80ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.75ms |
| `composite` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.79ms |
| `cool_tint` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.70ms |
| `grayscale` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.82ms |
| `hue_shift_15pct` | 30 | 90.0% | 93.3% | 93.3% | 0.0% | 1.89ms |
| `jpeg_q75` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.76ms |
| `pixel_noise` | 30 | 93.3% | 93.3% | 90.0% | 0.0% | 1.86ms |
| `warm_tint` | 30 | 93.3% | 93.3% | 93.3% | 0.0% | 1.78ms |

### Strategy: pure_clip
- **Total Queries**: 240
- **Recall@1**: 80.42%
- **Recall@5**: 92.50%
- **MRR**: 0.8578
- **Detected (Confirmed + Suspected)**: 0.00%
- **Confirmed Rate**: 0.00%
- **Suspected Rate**: 0.00%
- **Latency (P50 / P95 / Mean)**: 9.86ms / 13.77ms / 10.24ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 30 | 100.0% | 100.0% | 0.0% | 0.0% | 9.81ms |
| `composite` | 30 | 60.0% | 90.0% | 0.0% | 0.0% | 10.04ms |
| `cool_tint` | 30 | 93.3% | 100.0% | 0.0% | 0.0% | 9.86ms |
| `grayscale` | 30 | 50.0% | 56.7% | 0.0% | 0.0% | 9.91ms |
| `hue_shift_15pct` | 30 | 73.3% | 96.7% | 0.0% | 0.0% | 9.72ms |
| `jpeg_q75` | 30 | 70.0% | 96.7% | 0.0% | 0.0% | 10.02ms |
| `pixel_noise` | 30 | 100.0% | 100.0% | 0.0% | 0.0% | 9.74ms |
| `warm_tint` | 30 | 96.7% | 100.0% | 0.0% | 0.0% | 9.69ms |