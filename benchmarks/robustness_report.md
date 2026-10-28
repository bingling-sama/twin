## 🏆 Strategy Comparison Summary

| Strategy | Recall@1 | Recall@5 | MRR | Suspected | Confirmed | P50 Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **current_diffmask** | **81.67%** | 91.94% | 0.8599 | 82.22% | 11.39% | 14.16ms | 13.92ms |
| **ssim_verifier** | **95.00%** | 95.00% | 0.9500 | 16.11% | 78.06% | 56.93ms | 55.36ms |
| **pure_phash** | **74.17%** | 86.11% | 0.7926 | 82.22% | 11.39% | 11.87ms | 11.83ms |
| **pure_clip** | **82.50%** | 93.61% | 0.8738 | 0.00% | 0.00% | 12.02ms | 12.61ms |

### Strategy: current_diffmask
- **Total Queries**: 360
- **Recall@1**: 81.67%
- **Recall@5**: 91.94%
- **MRR**: 0.8599
- **Detected (Confirmed + Suspected)**: 93.61%
- **Confirmed Rate**: 11.39%
- **Suspected Rate**: 82.22%
- **Latency (P50 / P95 / Mean)**: 14.16ms / 18.34ms / 13.92ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 40 | 77.5% | 92.5% | 95.0% | 0.0% | 14.10ms |
| `composite` | 40 | 57.5% | 82.5% | 92.5% | 0.0% | 14.13ms |
| `cool_tint` | 40 | 67.5% | 95.0% | 95.0% | 0.0% | 14.31ms |
| `grayscale` | 40 | 90.0% | 92.5% | 90.0% | 5.0% | 14.48ms |
| `hue_shift_15pct` | 40 | 95.0% | 97.5% | 90.0% | 5.0% | 14.03ms |
| `identity` | 40 | 92.5% | 92.5% | 0.0% | 92.5% | 14.13ms |
| `jpeg_q75` | 40 | 95.0% | 95.0% | 92.5% | 0.0% | 14.29ms |
| `pixel_noise` | 40 | 92.5% | 92.5% | 90.0% | 0.0% | 14.18ms |
| `warm_tint` | 40 | 67.5% | 87.5% | 95.0% | 0.0% | 13.92ms |

### Strategy: ssim_verifier
- **Total Queries**: 360
- **Recall@1**: 95.00%
- **Recall@5**: 95.00%
- **MRR**: 0.9500
- **Detected (Confirmed + Suspected)**: 94.17%
- **Confirmed Rate**: 78.06%
- **Suspected Rate**: 16.11%
- **Latency (P50 / P95 / Mean)**: 56.93ms / 76.06ms / 55.36ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 40 | 97.5% | 97.5% | 2.5% | 95.0% | 57.88ms |
| `composite` | 40 | 95.0% | 95.0% | 42.5% | 50.0% | 58.86ms |
| `cool_tint` | 40 | 95.0% | 95.0% | 32.5% | 62.5% | 55.05ms |
| `grayscale` | 40 | 95.0% | 95.0% | 2.5% | 92.5% | 56.13ms |
| `hue_shift_15pct` | 40 | 97.5% | 97.5% | 12.5% | 85.0% | 54.66ms |
| `identity` | 40 | 92.5% | 92.5% | 0.0% | 92.5% | 59.47ms |
| `jpeg_q75` | 40 | 95.0% | 95.0% | 2.5% | 90.0% | 57.49ms |
| `pixel_noise` | 40 | 92.5% | 92.5% | 7.5% | 82.5% | 58.15ms |
| `warm_tint` | 40 | 95.0% | 95.0% | 42.5% | 52.5% | 56.69ms |

### Strategy: pure_phash
- **Total Queries**: 360
- **Recall@1**: 74.17%
- **Recall@5**: 86.11%
- **MRR**: 0.7926
- **Detected (Confirmed + Suspected)**: 93.61%
- **Confirmed Rate**: 11.39%
- **Suspected Rate**: 82.22%
- **Latency (P50 / P95 / Mean)**: 11.87ms / 15.43ms / 11.83ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 40 | 70.0% | 80.0% | 95.0% | 0.0% | 11.69ms |
| `composite` | 40 | 42.5% | 60.0% | 92.5% | 0.0% | 12.17ms |
| `cool_tint` | 40 | 47.5% | 87.5% | 95.0% | 0.0% | 11.94ms |
| `grayscale` | 40 | 90.0% | 92.5% | 90.0% | 5.0% | 11.72ms |
| `hue_shift_15pct` | 40 | 95.0% | 97.5% | 90.0% | 5.0% | 11.67ms |
| `identity` | 40 | 92.5% | 92.5% | 0.0% | 92.5% | 11.91ms |
| `jpeg_q75` | 40 | 95.0% | 95.0% | 92.5% | 0.0% | 11.94ms |
| `pixel_noise` | 40 | 92.5% | 92.5% | 90.0% | 0.0% | 11.77ms |
| `warm_tint` | 40 | 42.5% | 77.5% | 95.0% | 0.0% | 12.07ms |

### Strategy: pure_clip
- **Total Queries**: 360
- **Recall@1**: 82.50%
- **Recall@5**: 93.61%
- **MRR**: 0.8738
- **Detected (Confirmed + Suspected)**: 0.00%
- **Confirmed Rate**: 0.00%
- **Suspected Rate**: 0.00%
- **Latency (P50 / P95 / Mean)**: 12.02ms / 17.99ms / 12.61ms

| Perturbation | Count | Recall@1 | Recall@5 | Suspected | Confirmed | P50 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bright_contrast` | 40 | 100.0% | 100.0% | 0.0% | 0.0% | 11.72ms |
| `composite` | 40 | 62.5% | 92.5% | 0.0% | 0.0% | 12.61ms |
| `cool_tint` | 40 | 95.0% | 100.0% | 0.0% | 0.0% | 12.23ms |
| `grayscale` | 40 | 47.5% | 57.5% | 0.0% | 0.0% | 11.98ms |
| `hue_shift_15pct` | 40 | 72.5% | 95.0% | 0.0% | 0.0% | 12.22ms |
| `identity` | 40 | 100.0% | 100.0% | 0.0% | 0.0% | 12.22ms |
| `jpeg_q75` | 40 | 70.0% | 97.5% | 0.0% | 0.0% | 11.31ms |
| `pixel_noise` | 40 | 100.0% | 100.0% | 0.0% | 0.0% | 11.58ms |
| `warm_tint` | 40 | 95.0% | 100.0% | 0.0% | 0.0% | 12.30ms |