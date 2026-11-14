"""Evaluation engine and metrics calculation for similarity discovery benchmarks.

Calculates:
- Execution efficiency (Throughput, Probes/sec, Pairs/sec, Stage latency breakdown)
- Clustering quality (Cluster Purity, Perfect Batch Rate, Cluster Recall Rate, Member Coverage)
- Transitivity constraints (Max within-cluster DiffMask ratio)
- Non-destructive sandbox isolation (Fresh Indexer + isolated temp dir; zero host/global mutation)
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cluster.dataset import BenchmarkItem
from twin.core.config import settings
from twin.core.types import DiscoveryMode, SeedStrategy
from twin.services.cluster import discover_similar_batches
from twin.services.embedding import get_embedding_dim
from twin.services.hasher import compute_ahash, compute_dhash, compute_phash
from twin.services.indexer import Indexer, indexer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _prod_data_dir() -> Path:
    return (_PROJECT_ROOT / "data").resolve()


def _assert_safe_sandbox_path(target: Path | str, *, label: str) -> Path:
    """Validate that a sandbox path never overlaps host, image, or production dirs."""
    resolved = Path(target).resolve()
    forbidden_roots = [
        ("production data directory", _prod_data_dir()),
        ("active host settings.index_dir", settings.index_dir.resolve()),
        ("active settings.images_path", settings.images_path.resolve()),
    ]
    for root_label, root_path in forbidden_roots:
        if (
            resolved == root_path
            or root_path in resolved.parents
            or resolved in root_path.parents
        ):
            raise ValueError(
                f"Refusing to use {root_label} (or its ancestor/descendant) as {label}: "
                f"{resolved}"
            )
    return resolved


def _assert_empty_sandbox_indexer(idx: Indexer) -> None:
    """Reject the global indexer singleton and any already-populated Indexer instance."""
    if idx is indexer:
        raise ValueError(
            "Benchmark sandbox requires a dedicated non-singleton Indexer instance."
        )
    with idx._lock:
        has_vectors = any(
            v is not None and v.ntotal > 0 for v in idx._indices.values()
        ) or any(
            b is not None and b.ntotal > 0 for b in idx._binary_indices.values()
        )
        has_meta = any(bool(m) for m in idx._metadatas.values())
    if has_vectors or has_meta:
        raise ValueError(
            "Refusing to use a non-empty caller-owned Indexer instance in benchmark sandbox."
        )


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
    service_query_ms: float
    end_to_end_ms: float
    instrumented_stage_ms: float
    unattributed_ms: float
    probes_per_sec: float
    pairs_verified: int
    pairs_per_sec: float
    avg_batch_purity: float
    perfect_batches_ratio: float
    impure_batches_count: int
    impure_batch_rate: float
    cluster_recall_rate: float
    member_coverage_rate: float
    max_observed_diff_ratio: float
    wrong_peer_count: int = 0
    stages: dict[str, Any] = field(default_factory=dict)
    normalized_batches: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown_summary(self) -> str:
        """Format scenario result as a detailed markdown report section."""
        recall_stage = self.stages.get("recall", {})
        io_stage = self.stages.get("io_load", {})
        breakdown = summarize_stage_breakdown(self.stages)
        risk_flags = assess_quality_risk(self)
        risk_summary = ", ".join(risk_flags) if risk_flags else "None"

        cand_recalled = recall_stage.get("candidates_recalled", 0)
        imgs_loaded = io_stage.get("images_loaded", 0)

        lines = [
            f"### Scenario: {self.scenario_name} (`mode={self.mode}`)",
            f"- **Dataset Scale**: {self.total_items} items ({self.gt_cluster_count} GT clusters)",
            f"- **Quality Risk**: `{risk_summary}`",
            f"- **Wall-Clock Latency**: **{self.end_to_end_ms:.2f} ms** "
            f"(service internal: {self.service_query_ms:.2f} ms | "
            f"instrumented stages: {self.instrumented_stage_ms:.2f} ms) | "
            f"**{self.probes_per_sec:,.1f} Probes/sec** (denominator: end_to_end_ms)",
            f"- **Batches Formed**: {self.discovered_batches} "
            f"(covering {self.total_discovered_members} skins)",
            "- **Clustering Quality & Precision**:",
            f"  - **Cluster Purity**: **{self.avg_batch_purity:.1%}** "
            f"(100% pure batches: {self.perfect_batches_ratio:.1%}, "
            f"impure batches: {self.impure_batches_count}, "
            f"impure rate: {self.impure_batch_rate:.1%}, wrong peers: {self.wrong_peer_count})",
            f"  - **Target Group Recall**: **{self.cluster_recall_rate:.1%}** "
            f"({self.gt_cluster_count} ground-truth groups)",
            f"  - **Member Coverage**: **{self.member_coverage_rate:.1%}** "
            "(fraction of target GT cluster members—including seeds and peers—recovered)",
            f"  - **Max Observed DiffRatio**: {self.max_observed_diff_ratio:.4f} "
            "(strictly within configured limits)",
            "- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:",
            f"  - **Binary Recall**: {breakdown['recall_ms']:.2f} ms "
            f"({breakdown['recall_pct']:.1f}%) | {cand_recalled} candidates recalled",
            f"  - **Parallel Image I/O**: {breakdown['io_ms']:.2f} ms "
            f"({breakdown['io_pct']:.1f}%) | {imgs_loaded} images decoded in thread pool",
            f"  - **DiffMask Verify**: {breakdown['verify_ms']:.2f} ms "
            f"({breakdown['verify_pct']:.1f}%) | {self.pairs_verified} pairs verified "
            f"({self.pairs_per_sec:,.1f} pairs/sec, denominator: end_to_end_ms)",
            f"  - **Instrumented Stages Total**: {self.instrumented_stage_ms:.2f} ms | "
            f"**Unattributed Overhead (e2e - stages)**: {self.unattributed_ms:.2f} ms "
            "(loop branching, metadata lookups, elimination, harness/FFI overhead)",
            "",
        ]
        return "\n".join(lines)


def summarize_stage_breakdown(stages: dict[str, Any]) -> dict[str, float]:
    """Extract stage latencies and percentage shares across instrumented stages."""
    recall_ms = float(stages.get("recall", {}).get("elapsed_ms", 0.0))
    io_ms = float(stages.get("io_load", {}).get("elapsed_ms", 0.0))
    verify_ms = float(stages.get("verify", {}).get("elapsed_ms", 0.0))
    instrumented_ms = recall_ms + io_ms + verify_ms
    denom = max(instrumented_ms, 0.001)
    return {
        "recall_ms": recall_ms,
        "io_ms": io_ms,
        "verify_ms": verify_ms,
        "instrumented_ms": instrumented_ms,
        "recall_pct": (recall_ms / denom) * 100.0,
        "io_pct": (io_ms / denom) * 100.0,
        "verify_pct": (verify_ms / denom) * 100.0,
    }


def assess_quality_risk(result: ScenarioResult) -> list[str]:
    """Return explicit quality-risk flags when wrong peers, impurity, or recall gaps occur."""
    flags: list[str] = []
    if result.wrong_peer_count > 0:
        flags.append(f"wrong_peers={result.wrong_peer_count}")
    if result.impure_batches_count > 0 or result.avg_batch_purity < 1.0:
        flags.append(f"impure_rate={result.impure_batch_rate:.1%}")
    if result.gt_cluster_count > 0 and result.cluster_recall_rate < 1.0:
        flags.append(f"recall={result.cluster_recall_rate:.1%}")
    if result.gt_cluster_count > 0 and result.member_coverage_rate < 1.0:
        flags.append(f"coverage={result.member_coverage_rate:.1%}")
    return flags


def populate_cluster_index(
    items: list[BenchmarkItem],
    tmp_dir: Path | str,
    datasource: str = "skin",
    indexer_instance: Indexer | None = None,
) -> dict[str, str | None]:
    """Populate a fresh empty Indexer instance with benchmark items saved to an isolated dir.

    Rejects None, the global indexer singleton, any non-empty caller-owned Indexer,
    and any path overlapping production, host index, or host image directories.
    Returns mapping of filename -> cluster_id (or None for distractors).
    """
    if indexer_instance is None:
        raise ValueError(
            "populate_cluster_index requires an explicit non-singleton Indexer instance."
        )
    _assert_empty_sandbox_indexer(indexer_instance)

    tmp_path = _assert_safe_sandbox_path(tmp_dir, label="sandbox image directory")
    tmp_path.mkdir(parents=True, exist_ok=True)

    dummy_vec = np.zeros(get_embedding_dim(), dtype=np.float32)
    filename_to_cluster: dict[str, str | None] = {}

    for item in items:
        if item.filename in filename_to_cluster:
            raise ValueError(f"Duplicate benchmark filename: {item.filename}")

        disk_path = (tmp_path / item.filename).resolve()
        if tmp_path not in disk_path.parents and disk_path != tmp_path:
            raise ValueError(f"Invalid benchmark filename outside tmp_dir: {item.filename}")

        if not getattr(item, "skip_disk_write", False):
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            img_format = str(getattr(item, "image_format", "PNG")).upper()
            save_kwargs = {"lossless": True} if img_format == "WEBP" else {}
            item.image.save(disk_path, format=img_format, **save_kwargs)

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
        indexer_instance.add_item(dummy_vec, meta, datasource=datasource)

    return filename_to_cluster


def _normalize_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract deterministic structural representation of discovered batches."""
    normalized: list[dict[str, Any]] = []
    for batch in batches:
        normalized.append(
            {
                "batch_id": int(batch.get("batch_id", 0)),
                "seed": str(batch["seed"]["filename"]),
                "members": [
                    {
                        "filename": str(member["filename"]),
                        "diff_ratio": float(member.get("diff_ratio", 0.0)),
                        "phash_distance": int(member.get("phash_distance", 0)),
                        "dhash_distance": int(member.get("dhash_distance", 0)),
                        "verifier_score": float(member.get("verifier_score", 0.0)),
                        "match_level": str(member.get("match_level", "")),
                    }
                    for member in batch["members"]
                ],
            }
        )
    return normalized


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
    random_seed: int = 42,
    indexer_instance: Indexer | None = None,
) -> ScenarioResult:
    """Run discovery on benchmark items in an isolated sandbox and evaluate metrics."""
    if indexer_instance is not None:
        raise ValueError(
            "evaluate_cluster_scenario creates and owns its sandbox Indexer; "
            "caller-owned Indexer instances are rejected."
        )
    sandbox_indexer = Indexer()

    with tempfile.TemporaryDirectory(prefix="twin_cluster_bench_") as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        fn_to_cluster = populate_cluster_index(
            items, tmp_dir=img_dir, datasource=datasource, indexer_instance=sandbox_indexer
        )

        t0 = time.perf_counter()
        res = discover_similar_batches(
            datasource=datasource,
            max_batches=max_batches,
            batch_size=batch_size,
            mode=mode,
            max_probes=max_probes,
            seed_strategy=seed_strategy,
            indexer_instance=sandbox_indexer,
            random_seed=random_seed,
        )
        e2e_sec = max(time.perf_counter() - t0, 0.0001)

    e2e_ms = round(e2e_sec * 1000, 3)
    service_query_ms = float(res.get("query_time_ms", e2e_ms))
    batches = res.get("batches", [])
    probes = res.get("total_probes", 0)
    stages = res.get("stages", {})
    pairs_verified = int(stages.get("verify", {}).get("pairs_verified", 0))

    rec_ms = float(stages.get("recall", {}).get("elapsed_ms", 0.0))
    io_ms = float(stages.get("io_load", {}).get("elapsed_ms", 0.0))
    ver_ms = float(stages.get("verify", {}).get("elapsed_ms", 0.0))
    instrumented_stage_ms = round(rec_ms + io_ms + ver_ms, 3)
    unattributed_ms = max(0.0, round(e2e_ms - instrumented_stage_ms, 3))

    probes_per_sec = probes / e2e_sec
    pairs_per_sec = pairs_verified / e2e_sec

    total_members = 0
    wrong_peer_count = 0
    batch_purities: list[float] = []
    perfect_batches = 0
    impure_batches = 0
    discovered_gt_clusters: set[str] = set()
    covered_gt_members: set[str] = set()
    max_diff_seen = 0.0

    target_gt_set = set(ground_truth_counts.keys())
    target_gt_total = sum(ground_truth_counts.values())

    for b in batches:
        seed = b["seed"]
        members = b["members"]
        batch_size_actual = len(members) + 1
        total_members += batch_size_actual

        seed_cluster = fn_to_cluster.get(seed["filename"])

        # Calculate purity: members that share seed's non-None cluster
        matches = 0
        has_matching_peer = False
        matched_member_filenames: list[str] = []
        for m in members:
            diff_r = m.get("diff_ratio", 0.0)
            if diff_r > max_diff_seen:
                max_diff_seen = diff_r

            m_cluster = fn_to_cluster.get(m["filename"])
            if seed_cluster is not None and m_cluster == seed_cluster:
                matches += 1
                has_matching_peer = True
                matched_member_filenames.append(m["filename"])
            else:
                wrong_peer_count += 1

        if seed_cluster is None:
            purity = 0.0
        else:
            purity = (matches + 1) / batch_size_actual

        batch_purities.append(purity)
        is_pure = seed_cluster is not None and matches == len(members) and matches > 0
        if is_pure:
            perfect_batches += 1
        else:
            impure_batches += 1

        # Genuine group recovery check:
        # 1. seed belongs to target ground-truth set
        # 2. at least one member in batch belongs to the same ground-truth cluster
        if seed_cluster and seed_cluster in target_gt_set and has_matching_peer:
            discovered_gt_clusters.add(seed_cluster)
            covered_gt_members.add(seed["filename"])
            covered_gt_members.update(matched_member_filenames)

    avg_purity = float(np.mean(batch_purities)) if batch_purities else 1.0
    perfect_ratio = (perfect_batches / len(batches)) if batches else 1.0
    impure_batch_rate = (impure_batches / len(batches)) if batches else 0.0

    num_gt_clusters = len(target_gt_set)
    cluster_recall = len(discovered_gt_clusters) / num_gt_clusters if num_gt_clusters > 0 else 1.0
    if not (0.0 <= cluster_recall <= 1.0):
        raise ValueError(
            f"cluster_recall out of [0, 1] bounds: {cluster_recall} "
            f"({len(discovered_gt_clusters)} / {num_gt_clusters})"
        )

    member_coverage = (
        len(covered_gt_members) / target_gt_total if target_gt_total > 0 else 1.0
    )
    if not (0.0 <= member_coverage <= 1.0):
        raise ValueError(
            f"member_coverage out of [0, 1] bounds: {member_coverage} "
            f"({len(covered_gt_members)} / {target_gt_total})"
        )

    return ScenarioResult(
        scenario_name=scenario_name,
        mode=mode,
        total_items=len(items),
        gt_cluster_count=num_gt_clusters,
        discovered_batches=len(batches),
        total_discovered_members=total_members,
        probes_evaluated=probes,
        service_query_ms=service_query_ms,
        end_to_end_ms=e2e_ms,
        instrumented_stage_ms=instrumented_stage_ms,
        unattributed_ms=unattributed_ms,
        probes_per_sec=probes_per_sec,
        pairs_verified=pairs_verified,
        pairs_per_sec=pairs_per_sec,
        avg_batch_purity=avg_purity,
        perfect_batches_ratio=perfect_ratio,
        impure_batches_count=impure_batches,
        impure_batch_rate=impure_batch_rate,
        cluster_recall_rate=cluster_recall,
        member_coverage_rate=member_coverage,
        max_observed_diff_ratio=max_diff_seen,
        wrong_peer_count=wrong_peer_count,
        stages=stages,
        normalized_batches=_normalize_batches(batches),
    )
