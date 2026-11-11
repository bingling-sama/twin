"""Provenance metadata helpers for cluster discovery benchmarks."""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLUSTER_SOURCE_PATHS: tuple[Path, ...] = (
    _PROJECT_ROOT / "src" / "twin" / "core" / "config.py",
    _PROJECT_ROOT / "src" / "twin" / "core" / "types.py",
    _PROJECT_ROOT / "src" / "twin" / "services" / "cluster.py",
    _PROJECT_ROOT / "src" / "twin" / "services" / "hasher.py",
    _PROJECT_ROOT / "src" / "twin" / "services" / "verifiers.py",
    _PROJECT_ROOT / "benchmarks" / "cluster" / "dataset.py",
    _PROJECT_ROOT / "benchmarks" / "cluster" / "evaluator.py",
    _PROJECT_ROOT / "benchmarks" / "cluster" / "meta.py",
    _PROJECT_ROOT / "benchmarks" / "cluster_bench.py",
    _PROJECT_ROOT / "benchmarks" / "bench_report.py",
)

CLUSTER_DATASET_PATHS: tuple[Path, ...] = (
    _PROJECT_ROOT / "benchmarks" / "cluster" / "dataset.py",
    _PROJECT_ROOT / "tests" / "benchmarks" / "test_cluster_performance.py",
)


def compute_paths_sha256(paths: tuple[Path, ...] | list[Path]) -> str:
    """Compute full 64-char SHA-256 over normalized file paths and LF-normalized content."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        resolved = path.resolve()
        rel = resolved.relative_to(_PROJECT_ROOT).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        if resolved.is_file():
            content = resolved.read_bytes().replace(b"\r\n", b"\n")
            digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def compute_cluster_source_digest() -> str:
    """Return full 64-char SHA-256 covering discovery core and benchmark harness files."""
    return compute_paths_sha256(CLUSTER_SOURCE_PATHS)


def get_cluster_dataset_fingerprint() -> str:
    """Return deterministic 64-char SHA-256 fingerprint for cluster benchmark datasets."""
    return compute_paths_sha256(CLUSTER_DATASET_PATHS)


def collect_cluster_metadata() -> dict[str, str]:
    """Return provenance metadata dict stored in machine_info and markdown reports."""
    return {
        "source_digest": compute_cluster_source_digest(),
        "dataset_fingerprint": get_cluster_dataset_fingerprint(),
    }

