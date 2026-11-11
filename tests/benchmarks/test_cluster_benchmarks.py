"""Benchmarks and correctness tests for seedless texture discovery via Greedy Leader Elimination.

Four-tiered verification:
  1. Performance Benchmarks: Integrates with pytest-benchmark (--benchmark-only)
  2. Memory & Disk Sentinel Isolation: Verifies global indexer singleton & host disk are untouched
  3. Deterministic Reproducibility: Verifies identical batches with fixed PRNG seed
  4. Mathematical Contracts & Boundaries: Verifies strict threshold rejection & anti-transitivity
"""

from __future__ import annotations

import hashlib
import os
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
    _prod_data_dir,
    evaluate_cluster_scenario,
    isolated_index_environment,
    populate_cluster_index,
)
from twin.core.config import settings
from twin.services.cluster import discover_similar_batches
from twin.services.embedding import get_embedding_dim
from twin.services.indexer import Indexer, indexer

try:
    import pytest_benchmark  # noqa: F401
except ImportError:

    @pytest.fixture
    def benchmark():
        """Fallback runner when pytest-benchmark is not installed."""

        def _runner(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _runner


def _build_cluster_dataset(
    num_duplicates: int = 30,
    num_variants: int = 15,
    num_noise: int = 25,
) -> list[BenchmarkItem]:
    """Generate in-memory benchmark items for test population."""
    items: list[BenchmarkItem] = []

    # 1. Duplicates
    base_dup = create_base_skin(seed=42)
    for i in range(num_duplicates):
        items.append(
            BenchmarkItem(
                filename=f"dup_{i:04d}.png",
                image=base_dup.copy(),
                cluster_id="cluster_dup_1",
            )
        )

    # 2. Variants (with physical contract verification)
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

    # 3. Noise
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


def _run_benchmarked_discovery(
    benchmark: Any,
    tmp_path: Path,
    items: list[BenchmarkItem],
    **discover_kwargs: Any,
) -> dict[str, Any]:
    """Populate a fresh sandbox Indexer inside an isolated dir and run benchmark."""
    img_dir = tmp_path / "images"
    idx_dir = tmp_path / "index"
    sandbox = Indexer()
    with isolated_index_environment(index_dir=idx_dir):
        populate_cluster_index(
            items, tmp_dir=img_dir, datasource="skin", indexer_instance=sandbox
        )
        return benchmark(
            discover_similar_batches,
            datasource="skin",
            indexer_instance=sandbox,
            **discover_kwargs,
        )


@pytest.mark.smoke
def test_bench_cluster_confirmed_duplicates(benchmark, tmp_path: Path) -> None:
    """Benchmark confirmed mode: pure duplicates discovery with zero diff."""
    items = _build_cluster_dataset(num_duplicates=25, num_variants=0, num_noise=20)
    res = _run_benchmarked_discovery(
        benchmark,
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
    for b in batches:
        assert b["mode"] == "confirmed"
        assert b["max_diff_ratio"] <= 0.05
        for m in b["members"]:
            assert m["diff_ratio"] == 0.0


@pytest.mark.smoke
def test_bench_cluster_adaptive_recall_scaling(benchmark, tmp_path: Path) -> None:
    """Benchmark adaptive recall expansion on a large swarm of 50 identical duplicates."""
    items = _build_cluster_dataset(num_duplicates=50, num_variants=0, num_noise=10)
    res = _run_benchmarked_discovery(
        benchmark,
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
    total_items_in_batches = sum(b["size"] for b in batches)
    assert total_items_in_batches == 50

    stages = res["stages"]
    assert stages["recall"]["probes_evaluated"] >= 10
    assert stages["verify"]["batches_formed"] == 10


@pytest.mark.smoke
def test_bench_cluster_variants_mode(benchmark, tmp_path: Path) -> None:
    """Benchmark variants mode on recolored derivative textures."""
    items = _build_cluster_dataset(num_duplicates=10, num_variants=20, num_noise=20)
    res = _run_benchmarked_discovery(
        benchmark,
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
    for b in batches:
        assert b["mode"] == "variants"
        assert b["max_diff_ratio"] <= 0.35


def _snapshot_files(root: Path) -> dict[Path, tuple[int, int, str | None]]:
    """Capture fast state of index artifacts while skipping huge image-corpus trees."""
    if not root.is_dir():
        return {}
    state: dict[Path, tuple[int, int, str | None]] = {}

    def _record_file(p: Path) -> None:
        st = p.stat()
        digest = (
            hashlib.sha256(p.read_bytes()).hexdigest()
            if st.st_size <= 16 * 1024 * 1024
            else None
        )
        state[p.relative_to(root)] = (st.st_size, st.st_mtime_ns, digest)

    for entry in root.iterdir():
        if entry.is_file():
            _record_file(entry)
        elif entry.is_dir():
            if entry.name == "images":
                st = entry.stat()
                state[entry.relative_to(root)] = (0, st.st_mtime_ns, "dir:images")
                continue
            for p in entry.rglob("*"):
                if p.is_file():
                    _record_file(p)
    return state


def test_cluster_memory_and_disk_isolation_sentinel(tmp_path: Path) -> None:
    """Verify benchmark execution preserves active/host disk files and memory singleton."""
    orig_path = settings.index_path
    orig_env = os.environ.get("TWIN_INDEX_PATH")
    orig_dir = settings.index_dir.resolve()
    orig_files = _snapshot_files(orig_dir)

    host_dir = tmp_path / "host_data"
    host_dir.mkdir(parents=True)
    sentinel_file = host_dir / "sentinel.txt"
    sentinel_file.write_text("CRITICAL_HOST_INDEX_DATA", encoding="utf-8")

    sentinel_faiss = host_dir / "skin" / "index.flat.faiss"
    sentinel_faiss.parent.mkdir(parents=True)
    sentinel_faiss.write_text("FAISS_BINARY_CONTENT", encoding="utf-8")

    sentinel_meta = host_dir / "skin" / "metadata.flat.json"
    sentinel_meta.write_text('[{"filename": "host_meta_0.png"}]', encoding="utf-8")

    empty_stage = tmp_path / "empty_stage"
    empty_stage.mkdir(parents=True)

    with indexer._lock:
        saved_indices = dict(indexer._indices)
        saved_binary = dict(indexer._binary_indices)
        saved_meta = {k: [dict(m) for m in v] for k, v in indexer._metadatas.items()}
        saved_dirty = dict(indexer._dirty_flags)
        indexer._indices = {}
        indexer._binary_indices = {}
        indexer._metadatas = {}
        indexer._dirty_flags = {ds: False for ds in indexer.KNOWN_DATASOURCES}

    try:
        # Populate isolated in-memory singleton without touching disk
        settings.index_path = str(empty_stage)
        dummy_vec = np.zeros(get_embedding_dim(), dtype=np.float32)
        for i in range(10):
            indexer.add_item(
                dummy_vec,
                {"filename": f"host_singleton_{i}.png", "path": f"/host/{i}.png"},
                datasource="skin",
            )
        assert indexer.get_count("skin") == 10

        # Point active settings/env to host_dir containing sentinel disk artifacts
        settings.index_path = str(host_dir)
        os.environ["TWIN_INDEX_PATH"] = str(host_dir)

        items = _build_cluster_dataset(num_duplicates=10, num_variants=0, num_noise=5)
        res = evaluate_cluster_scenario(
            scenario_name="Sentinel Host Protection Test",
            items=items,
            ground_truth_counts={"cluster_dup_1": 10},
            mode="confirmed",
            max_batches=5,
            batch_size=5,
        )
        assert res.discovered_batches == 2

        # Assert host sentinel files (including metadata.flat.json) are byte-identical
        assert sentinel_file.read_text(encoding="utf-8") == "CRITICAL_HOST_INDEX_DATA"
        assert sentinel_faiss.read_text(encoding="utf-8") == "FAISS_BINARY_CONTENT"
        assert sentinel_meta.read_text(encoding="utf-8") == '[{"filename": "host_meta_0.png"}]'

        # Assert pre-existing disk files in orig_dir were NEVER deleted or modified
        assert _snapshot_files(orig_dir) == orig_files

        # Assert global in-memory singleton remained unpolluted
        assert indexer.get_count("skin") == 10
        first_meta = indexer.get_metadata(0, datasource="skin")
        assert first_meta is not None and first_meta["filename"] == "host_singleton_0.png"
        assert settings.index_path == str(host_dir)
        assert os.environ.get("TWIN_INDEX_PATH") == str(host_dir)
    finally:
        settings.index_path = orig_path
        if orig_env is not None:
            os.environ["TWIN_INDEX_PATH"] = orig_env
        else:
            os.environ.pop("TWIN_INDEX_PATH", None)
        with indexer._lock:
            indexer._indices = saved_indices
            indexer._binary_indices = saved_binary
            indexer._metadatas = saved_meta
            indexer._dirty_flags = saved_dirty


def test_isolated_index_environment_rejects_prod_and_active_host_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative isolation tests: disallow data/ and currently active settings.index_dir."""
    prod_dir = _prod_data_dir()
    with pytest.raises(ValueError, match="production data directory"):
        with isolated_index_environment(index_dir=prod_dir):
            pass

    with pytest.raises(ValueError, match="production data directory"):
        with isolated_index_environment(index_dir=prod_dir / "skin"):
            pass

    active_dir = tmp_path / "active_host"
    active_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, "index_path", str(active_dir))
    with pytest.raises(ValueError, match="active host settings.index_dir"):
        with isolated_index_environment(index_dir=active_dir):
            pass


def test_populate_cluster_index_rejects_singleton_or_duplicates(tmp_path: Path) -> None:
    """Negative guards: reject global singleton, None, and duplicate filenames."""
    items = _build_cluster_dataset(num_duplicates=2, num_variants=0, num_noise=0)
    with isolated_index_environment(index_dir=tmp_path / "idx"):
        with pytest.raises(ValueError, match="non-singleton Indexer"):
            populate_cluster_index(items, tmp_dir=tmp_path / "img1", indexer_instance=None)
        with pytest.raises(ValueError, match="non-singleton Indexer"):
            populate_cluster_index(items, tmp_dir=tmp_path / "img2", indexer_instance=indexer)

        dup_items = [items[0], items[0]]
        with pytest.raises(ValueError, match="Duplicate benchmark filename"):
            populate_cluster_index(dup_items, tmp_dir=tmp_path / "img3", indexer_instance=Indexer())


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
    assert res.cluster_recall_rate == 1.0
    assert res.member_coverage_rate == 1.0
    assert res.wrong_peer_count == 0
    assert res.max_observed_diff_ratio <= max_diff

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
        swarm_sizes=[50], distractor_count=10, seed=701
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
    assert res.wrong_peer_count == 0


def test_cluster_noise_seed_purity_and_wrong_peer_accounting() -> None:
    """Noise-seed batches must report purity=0.0 and count wrong peers accurately."""
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
    assert res.impure_batches_count == 1
    assert res.wrong_peer_count == 1


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

    # A ~ B (~0.18), B ~ C (~0.30), A !~ C (~0.48)
    # Under variants mode (threshold = 0.35):
    # Cluster AB should form, B eliminated, C must NOT be in cluster AB
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
