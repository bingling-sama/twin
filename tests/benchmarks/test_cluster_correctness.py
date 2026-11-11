"""Correctness, metric, boundary, and contract tests for seedless discovery (Gates 2 & 3).

Runs without pytest-benchmark fixture or @pytest.mark.smoke so performance timing runs
and correctness verification remain cleanly separated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from benchmarks.cluster.dataset import (
    BenchmarkItem,
    build_adversarial_dataset,
    build_boundary_dataset,
    build_duplicate_swarms_dataset,
    build_transitivity_chain_dataset,
    create_base_skin,
    create_variant_skin,
    verify_variant_contracts,
)
from benchmarks.cluster.evaluator import (
    ScenarioResult,
    assess_quality_risk,
    evaluate_cluster_scenario,
    populate_cluster_index,
    summarize_stage_breakdown,
)
from twin.services.cluster import discover_similar_batches
from twin.services.indexer import Indexer


def _build_cluster_dataset(
    num_duplicates: int = 30,
    num_variants: int = 15,
    num_noise: int = 25,
) -> list[BenchmarkItem]:
    """Generate in-memory benchmark items for correctness tests."""
    items: list[BenchmarkItem] = []

    base_dup = create_base_skin(seed=42)
    for i in range(num_duplicates):
        items.append(
            BenchmarkItem(
                filename=f"dup_{i:04d}.png",
                image=base_dup.copy(),
                cluster_id="cluster_dup_1",
            )
        )

    base_var = create_base_skin(seed=100)
    var_imgs = [create_variant_skin(base_var, variant_idx=i + 1) for i in range(num_variants)]
    if var_imgs:
        verify_variant_contracts(base_var, var_imgs)
    for i, var_img in enumerate(var_imgs):
        items.append(
            BenchmarkItem(
                filename=f"var_{i:04d}.png",
                image=var_img,
                cluster_id="cluster_var_1",
            )
        )

    for i in range(num_noise):
        noise_img = create_base_skin(seed=1000 + i * 17)
        items.append(
            BenchmarkItem(
                filename=f"noise_{i:04d}.png",
                image=noise_img,
                cluster_id=None,
            )
        )

    return items


def _run_sandbox_discovery(
    tmp_path: Path,
    items: list[BenchmarkItem],
    **discover_kwargs: Any,
) -> dict[str, Any]:
    """Populate a fresh isolated Indexer and execute discover_similar_batches."""
    sandbox = Indexer()
    populate_cluster_index(
        items, tmp_dir=tmp_path / "images", datasource="skin", indexer_instance=sandbox
    )
    return discover_similar_batches(
        datasource="skin",
        indexer_instance=sandbox,
        **discover_kwargs,
    )


def test_cluster_confirmed_duplicates_correctness(tmp_path: Path) -> None:
    """Verify confirmed mode recovers pure duplicate batches with zero pixel diff."""
    items = _build_cluster_dataset(num_duplicates=25, num_variants=0, num_noise=20)
    res = _run_sandbox_discovery(
        tmp_path,
        items,
        max_batches=10,
        batch_size=5,
        mode="confirmed",
        max_probes=100,
        seed_strategy="random",
        random_seed=42,
    )

    batches = res["batches"]
    assert len(batches) >= 4
    for batch in batches:
        assert batch["mode"] == "confirmed"
        assert batch["max_diff_ratio"] <= 0.05
        for member in batch["members"]:
            assert member["diff_ratio"] == 0.0


def test_cluster_adaptive_recall_scaling_correctness(tmp_path: Path) -> None:
    """Verify adaptive recall expansion across a 50-item identical duplicate swarm."""
    items = _build_cluster_dataset(num_duplicates=50, num_variants=0, num_noise=10)
    res = _run_sandbox_discovery(
        tmp_path,
        items,
        max_batches=20,
        batch_size=5,
        mode="confirmed",
        max_probes=200,
        seed_strategy="sequential",
        random_seed=42,
    )

    batches = res["batches"]
    assert len(batches) == 10
    assert sum(b["size"] for b in batches) == 50

    stages = res["stages"]
    assert stages["recall"]["probes_evaluated"] >= 10
    assert stages["verify"]["batches_formed"] == 10


def test_cluster_variants_mode_correctness(tmp_path: Path) -> None:
    """Verify variants mode discovers recolored derivative batches within diff bounds."""
    items = _build_cluster_dataset(num_duplicates=10, num_variants=20, num_noise=20)
    res = _run_sandbox_discovery(
        tmp_path,
        items,
        max_batches=10,
        batch_size=5,
        mode="variants",
        max_probes=100,
        seed_strategy="random",
        random_seed=42,
    )

    batches = res["batches"]
    assert len(batches) >= 2
    for batch in batches:
        assert batch["mode"] == "variants"
        assert batch["max_diff_ratio"] <= 0.35


def test_cluster_deterministic_reproducibility() -> None:
    """Verify identical batch structure and metrics across runs with the same PRNG seed."""
    items = _build_cluster_dataset(num_duplicates=15, num_variants=10, num_noise=15)
    gt_counts = {"cluster_dup_1": 15, "cluster_var_1": 10}

    res1 = evaluate_cluster_scenario(
        scenario_name="Deterministic Run 1",
        items=items,
        ground_truth_counts=gt_counts,
        mode="confirmed",
        random_seed=42,
        seed_strategy="random",
    )
    res2 = evaluate_cluster_scenario(
        scenario_name="Deterministic Run 2",
        items=items,
        ground_truth_counts=gt_counts,
        mode="confirmed",
        random_seed=42,
        seed_strategy="random",
    )

    assert res1.discovered_batches == res2.discovered_batches
    assert res1.avg_batch_purity == res2.avg_batch_purity
    assert res1.impure_batches_count == res2.impure_batches_count
    assert res1.impure_batch_rate == res2.impure_batch_rate
    assert res1.wrong_peer_count == res2.wrong_peer_count
    assert res1.cluster_recall_rate == res2.cluster_recall_rate
    assert res1.member_coverage_rate == res2.member_coverage_rate
    assert res1.max_observed_diff_ratio == res2.max_observed_diff_ratio
    assert res1.normalized_batches == res2.normalized_batches


@pytest.mark.parametrize("mode,max_diff", [("confirmed", 0.05), ("variants", 0.35)])
def test_cluster_boundary_rejection(mode: str, max_diff: float) -> None:
    """Verify strict inclusion on exact boundary and rejection beyond threshold in both modes."""
    items, gt_counts = build_boundary_dataset(seed=401, mode=mode)  # type: ignore[arg-type]

    res = evaluate_cluster_scenario(
        scenario_name=f"Boundary Rejection ({mode})",
        items=items,
        ground_truth_counts=gt_counts,
        mode=mode,  # type: ignore[arg-type]
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
    )

    assert res.discovered_batches == 2
    assert res.avg_batch_purity == 1.0
    assert res.impure_batch_rate == 0.0
    assert res.cluster_recall_rate == 1.0
    assert res.member_coverage_rate == 1.0
    assert res.wrong_peer_count == 0
    assert res.max_observed_diff_ratio <= max_diff
    assert assess_quality_risk(res) == []

    clustered_names = {b["seed"] for b in res.normalized_batches} | {
        m["filename"] for b in res.normalized_batches for m in b["members"]
    }
    for trap in (
        "conf_diff_distractor_beyond.png",
        "conf_ham_distractor_beyond.png",
        "var_diff_distractor_beyond.png",
        "var_ham_distractor_beyond.png",
    ):
        assert trap not in clustered_names


@pytest.mark.parametrize("batch_size", [5, 10, 50])
def test_cluster_member_coverage_invariant_across_batch_sizes(batch_size: int) -> None:
    """Member coverage must equal 100% across batch_size=5/10/50 when all items are recovered."""
    items, gt_counts = build_duplicate_swarms_dataset(
        swarm_sizes=(50,), distractor_count=10, seed=701
    )
    res = evaluate_cluster_scenario(
        scenario_name=f"Coverage Invariance bs={batch_size}",
        items=items,
        ground_truth_counts=gt_counts,
        mode="confirmed",
        batch_size=batch_size,
        max_batches=50,
        max_probes=200,
        seed_strategy="sequential",
    )
    assert res.cluster_recall_rate == 1.0
    assert res.member_coverage_rate == 1.0
    assert res.avg_batch_purity == 1.0
    assert res.impure_batch_rate == 0.0
    assert res.wrong_peer_count == 0


def test_cluster_noise_seed_purity_and_wrong_peer_accounting() -> None:
    """Noise-seed batches must report purity=0.0, impure_batch_rate=1.0, and wrong_peer_count."""
    base = create_base_skin(seed=801, valid_pixel_count=3000)
    arr = np.array(base).copy()
    coords = np.argwhere(arr[:, :, 3] > 10)
    for y, x in coords[:60]:
        arr[y, x, 2] = (int(arr[y, x, 2]) + 30) % 256
    noise_peer = Image.fromarray(arr, mode="RGBA")
    items = [
        BenchmarkItem(filename="unlabeled_a.png", image=base, cluster_id=None),
        BenchmarkItem(filename="unlabeled_b.png", image=noise_peer, cluster_id=None),
    ]
    res = evaluate_cluster_scenario(
        scenario_name="Noise Seed Purity Test",
        items=items,
        ground_truth_counts={},
        mode="confirmed",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
    )
    assert res.discovered_batches == 1
    assert res.avg_batch_purity == 0.0
    assert res.perfect_batches_ratio == 0.0
    assert res.impure_batches_count == 1
    assert res.impure_batch_rate == 1.0
    assert res.wrong_peer_count == 1


def test_evaluator_impure_batch_rate_and_exact_purity_metrics() -> None:
    """Mixed pure + impure batches must compute exact impure_batch_rate and risk flags."""
    pure_img = create_base_skin(seed=910, valid_pixel_count=3000)

    mixed_base = create_base_skin(seed=911, valid_pixel_count=3000)
    arr = np.array(mixed_base).copy()
    coords = np.argwhere(arr[:, :, 3] > 10)
    for y, x in coords[:40]:
        arr[y, x, 2] = (int(arr[y, x, 2]) + 25) % 256
    mixed_noise = Image.fromarray(arr, mode="RGBA")

    items = [
        # Batch 1 (sequential idx 0..1): strictly pure GT cluster
        BenchmarkItem(filename="pure_0.png", image=pure_img, cluster_id="gt_pure"),
        BenchmarkItem(filename="pure_1.png", image=pure_img.copy(), cluster_id="gt_pure"),
        # Batch 2 (sequential idx 2..4): 1 seed + 1 GT peer + 1 unlabeled wrong peer -> 2/3 pure
        BenchmarkItem(filename="mix_seed.png", image=mixed_base, cluster_id="gt_mixed"),
        BenchmarkItem(filename="mix_peer.png", image=mixed_base.copy(), cluster_id="gt_mixed"),
        BenchmarkItem(filename="mix_noise.png", image=mixed_noise, cluster_id=None),
    ]
    res = evaluate_cluster_scenario(
        scenario_name="Mixed Purity Metric Test",
        items=items,
        ground_truth_counts={"gt_pure": 2, "gt_mixed": 2},
        mode="confirmed",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
    )

    assert res.discovered_batches == 2
    assert res.wrong_peer_count == 1
    assert res.impure_batches_count == 1
    assert res.perfect_batches_ratio == pytest.approx(0.5)
    assert res.impure_batch_rate == pytest.approx(0.5)
    assert res.avg_batch_purity == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)

    flags = assess_quality_risk(res)
    assert any(flag.startswith("wrong_peers=1") for flag in flags)
    assert any(flag.startswith("impure_rate=50.0%") for flag in flags)

    md = res.to_markdown_summary()
    assert "impure rate: 50.0%" in md
    assert "instrumented stages:" in md
    assert "denominator: end_to_end_ms" in md
    assert "**Quality Risk**:" in md


def test_summarize_stage_breakdown_and_quality_risk_helpers() -> None:
    """Stage breakdown and quality risk helpers must calculate consistent shares and flags."""
    breakdown = summarize_stage_breakdown(
        {
            "recall": {"elapsed_ms": 10.0},
            "io_load": {"elapsed_ms": 20.0},
            "verify": {"elapsed_ms": 70.0},
        }
    )
    assert breakdown["instrumented_ms"] == pytest.approx(100.0)
    assert breakdown["recall_pct"] == pytest.approx(10.0)
    assert breakdown["io_pct"] == pytest.approx(20.0)
    assert breakdown["verify_pct"] == pytest.approx(70.0)

    degraded = ScenarioResult(
        scenario_name="Degraded Sample",
        mode="variants",
        total_items=100,
        gt_cluster_count=5,
        discovered_batches=4,
        total_discovered_members=12,
        probes_evaluated=10,
        service_query_ms=10.0,
        end_to_end_ms=12.0,
        instrumented_stage_ms=9.0,
        unattributed_ms=3.0,
        probes_per_sec=800.0,
        pairs_verified=20,
        pairs_per_sec=1600.0,
        avg_batch_purity=0.95,
        perfect_batches_ratio=0.75,
        impure_batches_count=1,
        impure_batch_rate=0.25,
        cluster_recall_rate=0.80,
        member_coverage_rate=0.60,
        max_observed_diff_ratio=0.31,
        wrong_peer_count=2,
    )
    flags = assess_quality_risk(degraded)
    assert flags == [
        "wrong_peers=2",
        "impure_rate=25.0%",
        "recall=80.0%",
        "coverage=60.0%",
    ]


@pytest.mark.parametrize("mode", ["confirmed", "variants"])
def test_cluster_adversarial_torture_suite(mode: str) -> None:
    """Verify Spec Section 5 adversarial cases: HD, RGB WebP, 64x32 upgrade, traps."""
    items, gt_counts = build_adversarial_dataset(seed=600)
    res = evaluate_cluster_scenario(
        scenario_name=f"Adversarial Suite ({mode})",
        items=items,
        ground_truth_counts=gt_counts,
        mode=mode,  # type: ignore[arg-type]
        batch_size=5,
        max_batches=10,
        random_seed=42,
    )
    assert res.discovered_batches == 3
    assert res.cluster_recall_rate == 1.0
    assert res.member_coverage_rate == 1.0
    assert res.avg_batch_purity == 1.0
    assert res.impure_batch_rate == 0.0
    assert res.wrong_peer_count == 0

    clustered_names = {b["seed"] for b in res.normalized_batches} | {
        m["filename"] for b in res.normalized_batches for m in b["members"]
    }
    for trap in (
        "adv_aspect_mismatch.png",
        "adv_false_phash_a.png",
        "adv_false_phash_b.png",
        "adv_alpha_base.png",
        "adv_alpha_corrupted.png",
        "adv_missing_anchor.png",
        "adv_missing_ghost.png",
    ):
        assert trap not in clustered_names


def test_cluster_transitivity_chain_elimination() -> None:
    """Verify Greedy Leader Dynamic Elimination prevents non-transitive cluster collapse."""
    items, gt_counts = build_transitivity_chain_dataset(seed=501)

    res = evaluate_cluster_scenario(
        scenario_name="Transitivity Chain Test",
        items=items,
        ground_truth_counts=gt_counts,
        mode="variants",
        batch_size=5,
        max_batches=5,
        seed_strategy="sequential",
    )

    assert res.discovered_batches == 1
    assert res.avg_batch_purity == 1.0
    assert res.impure_batch_rate == 0.0
    assert res.cluster_recall_rate == 1.0
    assert res.member_coverage_rate == 1.0
    assert res.max_observed_diff_ratio <= 0.35


def test_evaluator_recall_bounded_in_zero_to_one() -> None:
    """Regression test: verify Recall numerator is strictly filtered to target GT set."""
    items = _build_cluster_dataset(num_duplicates=10, num_variants=10, num_noise=10)

    res = evaluate_cluster_scenario(
        scenario_name="Mixed Dataset Bounded Recall Test",
        items=items,
        ground_truth_counts={"cluster_var_1": 10},
        mode="variants",
        max_batches=10,
        batch_size=5,
        random_seed=42,
    )
    assert 0.0 <= res.cluster_recall_rate <= 1.0
    assert 0.0 <= res.member_coverage_rate <= 1.0
    assert res.cluster_recall_rate == 1.0


def test_evaluator_group_recovery_requires_peer() -> None:
    """Regression test: verify a cluster is NOT recovered if batch members are only distractors."""
    phantom_base = create_base_skin(seed=999)
    items: list[BenchmarkItem] = [
        BenchmarkItem(
            filename="phantom_seed.png",
            image=phantom_base,
            cluster_id="phantom_cluster",
        )
    ]
    for i in range(20):
        items.append(
            BenchmarkItem(
                filename=f"distractor_{i:03d}.png",
                image=create_base_skin(seed=10000 + i * 37),
                cluster_id=None,
            )
        )

    res = evaluate_cluster_scenario(
        scenario_name="No-Peer Phantom Test",
        items=items,
        ground_truth_counts={"phantom_cluster": 1},
        mode="confirmed",
        max_batches=5,
        batch_size=5,
        random_seed=42,
    )
    assert res.cluster_recall_rate == 0.0

