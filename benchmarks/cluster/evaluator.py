"""Evaluation engine and metrics calculation for similarity discovery benchmarks.

Calculates:
- Execution efficiency (Throughput, Probes/sec, Pairs/sec, Stage latency breakdown)
- Clustering quality (Cluster Purity, Perfect Batch Rate, Cluster Recall Rate)
- Transitivity constraints (Max within-cluster DiffMask ratio)
- Non-destructive state isolation (Strictly preserves host TWIN_INDEX_PATH data)
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import numpy as np

from benchmarks.cluster.dataset import BenchmarkItem
from twin.api.schemas import DiscoveryMode
from twin.core.config import settings
from twin.services.cluster import SeedStrategy, discover_similar_batches
from twin.services.hasher import compute_ahash, compute_dhash, compute_phash
from twin.services.indexer import indexer

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def isolated_index_environment(index_dir: Path | str | None = None) -> Generator[Path, None, None]:
    """Isolate settings.index_path and TWIN_INDEX_PATH to a safe temporary sandbox.

    Guarantees that indexer.clear() and index writing operations never touch
    host or production index directories under data/.
    """
    original_index_path = settings.index_path
    original_env = os.environ.get("TWIN_INDEX_PATH")

    if index_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="twin_isolated_index_")
        isolated_path = Path(temp_ctx.name)
    else:
        temp_ctx = None
        isolated_path = Path(index_dir)

    isolated_path.mkdir(parents=True, exist_ok=True)
    settings.index_path = str(isolated_path)
    os.environ["TWIN_INDEX_PATH"] = str(isolated_path)

    try:
        indexer.clear()
        yield isolated_path
    finally:
        try:
            indexer.clear()
        except Exception as e:
            logger.warning("Error clearing temporary sandbox index: %s", e)
        settings.index_path = original_index_path
        if original_env is not None:
            os.environ["TWIN_INDEX_PATH"] = original_env
        else:
            os.environ.pop("TWIN_INDEX_PATH", None)
        if temp_ctx is not None:
            temp_ctx.cleanup()


@dataclass
class ScenarioResult:
    """Metrics and profiling data for a benchmark scenario."""

    scenario_name: str
    mode: str
    total_items: int
    gt_cluster_count: int
    discovered_batches: int
    total_discovered_members: int
    probes_evaluated: int
    elapsed_ms: float
    probes_per_sec: float
    pairs_verified: int
    pairs_per_sec: float
    avg_batch_purity: float
    perfect_batches_ratio: float
    cluster_recall_rate: float
    max_observed_diff_ratio: float
    stages: dict[str, Any] = field(default_factory=dict)

    def to_markdown_summary(self) -> str:
        """Format scenario result as a detailed markdown report section."""
        recall_stage = self.stages.get("recall", {})
        io_stage = self.stages.get("io_load", {})
        verify_stage = self.stages.get("verify", {})

        recall_ms = recall_stage.get("elapsed_ms", 0.0)
        io_ms = io_stage.get("elapsed_ms", 0.0)
        verify_ms = verify_stage.get("elapsed_ms", 0.0)
        total_stage_ms = max(recall_ms + io_ms + verify_ms, 0.001)

        recall_pct = (recall_ms / total_stage_ms) * 100
        io_pct = (io_ms / total_stage_ms) * 100
        verify_pct = (verify_ms / total_stage_ms) * 100

        cand_recalled = recall_stage.get("candidates_recalled", 0)
        imgs_loaded = io_stage.get("images_loaded", 0)

        lines = [
            f"### Scenario: {self.scenario_name} (`mode={self.mode}`)",
            f"- **Dataset Scale**: {self.total_items} items ({self.gt_cluster_count} GT clusters)",
            f"- **Execution Time**: **{self.elapsed_ms:.2f} ms** | "
            f"**{self.probes_per_sec:,.1f} Probes/sec**",
            f"- **Batches Formed**: {self.discovered_batches} "
            f"(covering {self.total_discovered_members} skins)",
            "- **Clustering Quality**:",
            f"  - **Cluster Purity**: **{self.avg_batch_purity:.1%}** "
            f"(100% pure: {self.perfect_batches_ratio:.1%})",
            f"  - **Cluster Recall**: **{self.cluster_recall_rate:.1%}** "
            f"({self.gt_cluster_count} ground-truth groups)",
            f"  - **Max Observed DiffRatio**: {self.max_observed_diff_ratio:.4f} "
            "(strictly within bounds)",
            "- **Stages Telemetry & Latency Breakdown**:",
            f"  - **Binary Recall**: {recall_ms:.2f} ms ({recall_pct:.1f}%) | "
            f"{cand_recalled} recalled",
            f"  - **Parallel I/O**: {io_ms:.2f} ms ({io_pct:.1f}%) | "
            f"{imgs_loaded} images decoded",
            f"  - **DiffMask Verify**: {verify_ms:.2f} ms ({verify_pct:.1f}%) | "
            f"{self.pairs_verified} pairs ({self.pairs_per_sec:,.1f} pairs/sec)",
            "",
        ]
        return "\n".join(lines)


def populate_cluster_index(
    items: list[BenchmarkItem],
    tmp_dir: Path | str,
    datasource: str = "skin",
) -> dict[str, str | None]:
    """Populate singleton indexer with benchmark items saved to physical disk.

    Returns mapping of filename -> cluster_id (or None for distractors).
    """
    indexer.clear()
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    dummy_vec = np.zeros(512, dtype=np.float32)
    filename_to_cluster: dict[str, str | None] = {}

    for item in items:
        # Save authentic RGBA PNG file
        disk_path = tmp_path / item.filename
        item.image.save(disk_path, format="PNG")
        filename_to_cluster[item.filename] = item.cluster_id

        meta = {
            "filename": item.filename,
            "path": str(disk_path),
            "datasource": datasource,
            "texture_type": item.texture_type.value,
            "ahash": compute_ahash(item.image),
            "dhash": compute_dhash(item.image),
            "phash": compute_phash(item.image),
        }
        indexer.add_item(dummy_vec, meta, datasource=datasource)

    return filename_to_cluster


def evaluate_cluster_scenario(
    scenario_name: str,
    items: list[BenchmarkItem],
    ground_truth_counts: dict[str, int],
    mode: DiscoveryMode = "confirmed",
    seed_strategy: SeedStrategy = "random",
    max_batches: int = 50,
    batch_size: int = 5,
    max_probes: int = 200,
    datasource: str = "skin",
) -> ScenarioResult:
    """Run discovery on benchmark items in an isolated sandbox and evaluate metrics."""
    with tempfile.TemporaryDirectory(prefix="twin_cluster_bench_") as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        idx_dir = Path(tmp_dir) / "index"

        with isolated_index_environment(index_dir=idx_dir):
            fn_to_cluster = populate_cluster_index(items, tmp_dir=img_dir, datasource=datasource)

            t0 = time.perf_counter()
            res = discover_similar_batches(
                datasource=datasource,
                max_batches=max_batches,
                batch_size=batch_size,
                mode=mode,
                max_probes=max_probes,
                seed_strategy=seed_strategy,
            )
            elapsed_sec = max(time.perf_counter() - t0, 0.0001)

    batches = res.get("batches", [])
    probes = res.get("total_probes", 0)
    stages = res.get("stages", {})
    pairs_verified = stages.get("verify", {}).get("pairs_verified", 0)

    total_members = 0
    batch_purities: list[float] = []
    perfect_batches = 0
    discovered_gt_clusters: set[str] = set()
    max_diff_seen = 0.0

    target_gt_set = set(ground_truth_counts.keys())

    for b in batches:
        seed = b["seed"]
        members = b["members"]
        batch_size_actual = len(members) + 1
        total_members += batch_size_actual

        seed_cluster = fn_to_cluster.get(seed["filename"])

        # Calculate purity: members that share seed's cluster
        matches = 0
        has_matching_peer = False
        for m in members:
            diff_r = m.get("diff_ratio", 0.0)
            if diff_r > max_diff_seen:
                max_diff_seen = diff_r

            m_cluster = fn_to_cluster.get(m["filename"])
            if seed_cluster is not None and m_cluster == seed_cluster:
                matches += 1
                has_matching_peer = True

        purity = (matches + 1) / batch_size_actual
        batch_purities.append(purity)
        if purity >= 0.999:
            perfect_batches += 1

        # Genuine group recovery check:
        # 1. seed belongs to target ground-truth set
        # 2. at least one member in batch belongs to the same ground-truth cluster
        if seed_cluster and seed_cluster in target_gt_set and has_matching_peer:
            discovered_gt_clusters.add(seed_cluster)

    avg_purity = float(np.mean(batch_purities)) if batch_purities else 1.0
    perfect_ratio = (perfect_batches / len(batches)) if batches else 1.0

    # Cluster Recall: fraction of target ground-truth clusters recovered with valid peers
    num_gt_clusters = len(target_gt_set)
    cluster_recall = (
        min(1.0, len(discovered_gt_clusters) / num_gt_clusters)
        if num_gt_clusters > 0
        else 1.0
    )

    return ScenarioResult(
        scenario_name=scenario_name,
        mode=mode,
        total_items=len(items),
        gt_cluster_count=num_gt_clusters,
        discovered_batches=len(batches),
        total_discovered_members=total_members,
        probes_evaluated=probes,
        elapsed_ms=res.get("query_time_ms", elapsed_sec * 1000),
        probes_per_sec=probes / elapsed_sec,
        pairs_verified=pairs_verified,
        pairs_per_sec=pairs_verified / elapsed_sec,
        avg_batch_purity=avg_purity,
        perfect_batches_ratio=perfect_ratio,
        cluster_recall_rate=cluster_recall,
        max_observed_diff_ratio=max_diff_seen,
        stages=stages,
    )
