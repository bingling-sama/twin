"""Robustness perturbation and benchmark evaluation package."""

from benchmarks.robustness.dataset import (
    AnchorSample,
    PerturbedQuery,
    build_evaluation_queries,
    sample_anchor_dataset,
)
from benchmarks.robustness.evaluator import (
    BenchmarkReport,
    QueryResultMetric,
    evaluate_strategy,
    populate_benchmark_index,
)
from benchmarks.robustness.perturbations import (
    ALL_PERTURBATIONS,
    apply_brightness_contrast,
    apply_color_tint,
    apply_composite,
    apply_grayscale,
    apply_hue_shift,
    apply_jpeg_compression,
    apply_pixel_noise,
)

__all__ = [
    "ALL_PERTURBATIONS",
    "apply_grayscale",
    "apply_color_tint",
    "apply_hue_shift",
    "apply_brightness_contrast",
    "apply_jpeg_compression",
    "apply_pixel_noise",
    "apply_composite",
    "AnchorSample",
    "PerturbedQuery",
    "sample_anchor_dataset",
    "build_evaluation_queries",
    "QueryResultMetric",
    "BenchmarkReport",
    "populate_benchmark_index",
    "evaluate_strategy",
]
