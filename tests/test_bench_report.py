"""Unit tests for benchmark regression gate and median/p95 reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.bench_report import _group_name, _stats_p95, evaluate_regression, main
from benchmarks.cluster.meta import (
    compute_cluster_source_digest,
    get_cluster_dataset_fingerprint,
)
from benchmarks.cluster_bench import (
    _EXPECTED_SCENARIO_NAMES,
    _SCENARIO_REPRO_SPECS,
    _format_runtime_metadata,
    _format_settings_metadata,
    _format_zip_fingerprint,
    verify_cluster_report,
)
from benchmarks.cluster_bench import main as cluster_bench_main

_DEFAULT_SOURCE_DIGEST = "a" * 64
_DEFAULT_DATASET_FP = "b" * 64


def _make_payload(
    *,
    system: str = "Windows",
    cpu: str = "TestCPU",
    python_version: str = "3.12.11",
    gpu_name: str = "NVIDIA GeForce RTX 3060 Laptop GPU",
    torch_version: str = "2.6.0+cu124",
    faiss_version: str = "1.15.0",
    faiss_gpu: str = "False",
    pil_version: str = "12.3.0",
    source_digest: str = _DEFAULT_SOURCE_DIGEST,
    dataset_fingerprint: str = _DEFAULT_DATASET_FP,
    samples: list[float] | None = None,
    name: str = "test_bench_cluster_confirmed_duplicates",
) -> dict:
    data = samples or [0.010] * 10
    ordered = sorted(data)
    mid = len(ordered) // 2
    median = (
        ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    )
    return {
        "machine_info": {
            "system": system,
            "cpu": {"brand_raw": cpu},
            "python_version": python_version,
            "gpu": {"name": gpu_name},
            "packages": {
                "torch": torch_version,
                "faiss": faiss_version,
                "faiss_gpu": faiss_gpu,
                "PIL": pil_version,
            },
            "cluster_metadata": {
                "source_digest": source_digest,
                "dataset_fingerprint": dataset_fingerprint,
            },
        },
        "benchmarks": [
            {
                "name": name,
                "stats": {
                    "min": min(data),
                    "max": max(data),
                    "mean": sum(data) / len(data),
                    "median": median,
                    "rounds": len(data),
                    "iterations": 1,
                    "data": data,
                },
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_cluster_group_classification() -> None:
    assert _group_name("test_bench_cluster_confirmed_duplicates") == "cluster"
    assert _group_name("test_bench_cluster_variants_mode") == "cluster"


def test_regression_gate_passes_same_env_within_threshold(tmp_path: Path) -> None:
    bl_path = _write_json(tmp_path / "base.json", _make_payload(samples=[0.010] * 10))
    cur_path = _write_json(tmp_path / "cur.json", _make_payload(samples=[0.0105] * 10))
    rc = main([cur_path, "--compare", bl_path, "--fail-threshold", "10"])
    assert rc == 0


def test_regression_gate_fails_on_median_regression(tmp_path: Path) -> None:
    bl_path = _write_json(tmp_path / "base.json", _make_payload(samples=[0.010] * 10))
    cur_path = _write_json(tmp_path / "cur.json", _make_payload(samples=[0.012] * 10))
    rc = main([cur_path, "--compare", bl_path, "--fail-threshold", "10"])
    assert rc == 1


def test_regression_gate_fails_on_tail_p95_regression_only(tmp_path: Path) -> None:
    # Median stays 0.010 (0% delta), while highest sample drives P95 above +10%
    base_samples = [0.010] * 20
    tail_spike = [0.010] * 18 + [0.025, 0.030]
    assert _stats_p95({"data": tail_spike}) > _stats_p95({"data": base_samples}) * 1.10

    bl_path = _write_json(tmp_path / "base.json", _make_payload(samples=base_samples))
    cur_path = _write_json(tmp_path / "cur.json", _make_payload(samples=tail_spike))
    rc = main([cur_path, "--compare", bl_path, "--fail-threshold", "10"])
    assert rc == 1


def test_regression_gate_rejects_cross_env_unless_overridden(tmp_path: Path) -> None:
    bl_path = _write_json(
        tmp_path / "base.json",
        _make_payload(
            system="Linux",
            cpu="AMD Ryzen 7 8845H",
            python_version="3.13.13",
            gpu_name="none",
            torch_version="2.7.0",
            faiss_version="1.15.0",
            faiss_gpu="False",
            pil_version="12.2.0",
            samples=[0.010] * 10,
        ),
    )
    cur_path = _write_json(
        tmp_path / "cur.json",
        _make_payload(
            system="Windows",
            cpu="Intel",
            python_version="3.12.11",
            gpu_name="NVIDIA RTX 3060",
            torch_version="2.6.0+cu124",
            faiss_version="1.14.0",
            faiss_gpu="True",
            pil_version="12.3.0",
            samples=[0.010] * 10,
        ),
    )
    assert main([cur_path, "--compare", bl_path, "--fail-threshold", "10"]) == 1
    assert (
        main([cur_path, "--compare", bl_path, "--fail-threshold", "10", "--allow-cross-env"])
        == 0
    )


def test_regression_gate_enforces_provenance_even_when_cross_env_allowed() -> None:
    baseline = _make_payload()

    diff_digest = _make_payload(system="Linux", source_digest="c" * 64)
    errs = evaluate_regression(diff_digest, baseline, fail_threshold=10, allow_cross_env=True)
    assert any("source_digest mismatch" in e for e in errs)

    diff_dataset = _make_payload(system="Linux", dataset_fingerprint="d" * 64)
    errs = evaluate_regression(diff_dataset, baseline, fail_threshold=10, allow_cross_env=True)
    assert any("dataset_fingerprint mismatch" in e for e in errs)

    truncated_digest = _make_payload(source_digest="a" * 16)
    errs = evaluate_regression(truncated_digest, baseline, fail_threshold=10, allow_cross_env=True)
    assert any("non-64-char SHA-256" in e for e in errs)


def test_regression_gate_fails_on_missing_required_env_or_provenance_fields() -> None:
    baseline = _make_payload()

    no_pkg = _make_payload()
    no_pkg["machine_info"]["packages"].pop("torch")
    no_pkg["machine_info"]["packages"].pop("faiss_gpu")
    errs = evaluate_regression(no_pkg, baseline, fail_threshold=10, allow_cross_env=True)
    assert any("Missing required environment metadata" in e for e in errs)
    assert any("Torch" in e and "Faiss GPU" in e for e in errs)

    no_gpu = _make_payload()
    no_gpu["machine_info"].pop("gpu")
    errs = evaluate_regression(no_gpu, baseline, fail_threshold=10)
    assert any("GPU" in e for e in errs)

    no_prov = _make_payload()
    no_prov["machine_info"].pop("cluster_metadata")
    errs = evaluate_regression(no_prov, baseline, fail_threshold=10, allow_cross_env=True)
    assert any("missing dataset_fingerprint" in e and "missing source_digest" in e for e in errs)


def test_regression_gate_fails_on_missing_baseline_entry(tmp_path: Path) -> None:
    bl_path = _write_json(
        tmp_path / "base.json",
        _make_payload(name="test_bench_cluster_confirmed_duplicates"),
    )
    cur_path = _write_json(
        tmp_path / "cur.json",
        _make_payload(name="test_bench_cluster_variants_mode"),
    )
    assert main([cur_path, "--compare", bl_path, "--fail-threshold", "10"]) == 1


def _build_valid_cluster_report(
    *,
    commit_line: str,
    source_digest: str | None = None,
    dataset_fp: str | None = None,
    meta_overrides: dict[str, str] | None = None,
    omit_meta_key: str | None = None,
    omit_scenario: str | None = None,
    summary_overrides: dict[str, dict[str, str]] | None = None,
    repro_overrides: dict[str, dict[str, object]] | None = None,
) -> str:
    src_dig = compute_cluster_source_digest() if source_digest is None else source_digest
    ds_fp = get_cluster_dataset_fingerprint() if dataset_fp is None else dataset_fp
    runtime_meta = _format_runtime_metadata()
    settings_meta = _format_settings_metadata()
    meta = {
        "Generated At": "2026-11-11T19:00:00+08:00",
        "Commit SHA": commit_line,
        "Source Digest (SHA-256)": src_dig,
        "Dataset Fingerprint (SHA-256)": ds_fp,
        **runtime_meta,
        "Statistical Rounds": "5",
        "Haystack Config": "samples=200, use_zip=False",
        "images.zip Fingerprint": _format_zip_fingerprint(use_zip=False),
        **settings_meta,
    }
    if meta_overrides:
        meta.update(meta_overrides)
    if omit_meta_key:
        meta.pop(omit_meta_key, None)

    lines = [
        "# Benchmark Report",
        "",
        "## 🖥️ Execution Environment Metadata",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
    ]
    for k, v in meta.items():
        lines.append(f"| **{k}** | `{v}` |")

    lines.extend(
        [
            "",
            "## 🏆 Executive Summary & Strategy Matrix",
            "",
            "| Scenario | Mode | Scale (Items/GT) | Wall Latency Med/P95 [Min-Max] | "
            "Throughput | Formed Batches | Purity / Impure Rate (Wrong) | Coverage | "
            "Recall | Quality Risk |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name in _EXPECTED_SCENARIO_NAMES:
        if name == omit_scenario:
            continue
        spec = _SCENARIO_REPRO_SPECS[name]
        ov = (summary_overrides or {}).get(name, {})
        mode = ov.get("mode", spec["mode"])
        lat = ov.get("latency", "**10.0ms** (11.0ms) [9.0-11.0]")
        pur = ov.get("purity", "**100.0%** / 0.0% (0)")
        cov = ov.get("coverage", "100.0%")
        rec = ov.get("recall", "**100.0%**")
        risk = ov.get("risk", "Clean")
        lines.append(
            f"| **{name}** | `{mode}` | 100 / 2 | {lat} | 500 prob/s | "
            f"2 batches (4 skins) | {pur} | {cov} | {rec} | `{risk}` |"
        )

    lines.extend(
        [
            "",
            "## 🔧 Scenario Reproducibility Matrix",
            "",
            "| Scenario | Dataset Seed | PRNG Seed | Seed Strategy | "
            "batch_size | max_batches | max_probes |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name in _EXPECTED_SCENARIO_NAMES:
        if name == omit_scenario:
            continue
        spec = dict(_SCENARIO_REPRO_SPECS[name])
        if repro_overrides and name in repro_overrides:
            spec.update(repro_overrides[name])
        lines.append(
            f"| `{name}` | `{spec['dataset_seed']}` | `{spec['random_seed']}` | "
            f"`{spec['seed_strategy']}` | {spec['batch_size']} | "
            f"{spec['max_batches']} | {spec['max_probes']} |"
        )

    lines.extend(
        [
            "",
            "## 🔬 Telemetry & Fine-Grained Latency Decomposition",
            "",
            "> **Decomposition**: `end_to_end_ms` = `instrumented_stage_ms` "
            "(`recall` + `io` + `verify`) + `unattributed_ms` "
            "(loop branching, metadata lookups, elimination, harness overhead); "
            "`service_query_ms` records internal `discover_similar_batches()` elapsed time; "
            "`pairs/s` uses `end_to_end_ms` as denominator.",
            "",
            "| Scenario | Service / Stages / Wall | Binary Recall | Parallel I/O | "
            "DiffMask Verify | Unattributed (e2e) | Verified Pairs (e2e) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name in _EXPECTED_SCENARIO_NAMES:
        if name == omit_scenario:
            continue
        lines.append(
            f"| `{name}` | 8.0ms / 7.0ms / 10.0ms | 2.0ms (28.6%) | 2.0ms (28.6%) | "
            f"3.0ms (42.9%) | 3.0ms | 10 (1,000 pairs/s, denominator: end_to_end_ms) |"
        )

    lines.extend(
        [
            "",
            "## 📝 Detailed Scenario Analysis",
            "",
        ]
    )
    for name in _EXPECTED_SCENARIO_NAMES:
        if name != omit_scenario:
            spec = _SCENARIO_REPRO_SPECS[name]
            lines.extend(
                [
                    f"### Scenario: {name} (`mode={spec['mode']}`)",
                    "- **Dataset Scale**: 100 items (2 GT clusters)",
                    "- **Quality Risk**: `Clean`",
                    "- **Wall-Clock Latency**: **10.00 ms** (service internal: 8.00 ms | "
                    "instrumented stages: 7.00 ms) | "
                    "**500.0 Probes/sec** (denominator: end_to_end_ms)",
                    "- **Batches Formed**: 2 (covering 4 skins)",
                    "- **Clustering Quality & Precision**:",
                    "  - **Cluster Purity**: **100.0%** (100% pure batches: 100.0%, "
                    "impure batches: 0, impure rate: 0.0%, wrong peers: 0)",
                    "  - **Target Group Recall**: **100.0%** (2 ground-truth groups)",
                    "  - **Member Coverage**: **100.0%** "
                    "(fraction of target GT cluster members—including seeds and peers—recovered)",
                    "  - **Max Observed DiffRatio**: 0.0000 (strictly within configured limits)",
                    "- **Stages Telemetry & Latency Breakdown (Share of Instrumented Stages)**:",
                    "  - **Binary Recall**: 2.00 ms (28.6%) | 10 candidates recalled",
                    "  - **Parallel Image I/O**: 2.00 ms (28.6%) | "
                    "10 images decoded in thread pool",
                    "  - **DiffMask Verify**: 3.00 ms (42.9%) | 10 pairs verified "
                    "(1,000.0 pairs/sec, denominator: end_to_end_ms)",
                    "  - **Instrumented Stages Total**: 7.00 ms | "
                    "**Unattributed Overhead (e2e - stages)**: 3.00 ms "
                    "(loop branching, metadata lookups, elimination, harness/FFI overhead)",
                    "",
                ]
            )
    return "\n".join(lines)


def test_verify_cluster_report_passes_clean_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_head = "0123456789abcdef" * 2 + "01234567"
    monkeypatch.setattr(
        "benchmarks.cluster_bench._git_head_and_clean",
        lambda _root=None: (fake_head, True),
    )
    report_path = tmp_path / "cluster_report.md"
    report_path.write_text(
        _build_valid_cluster_report(commit_line=f"{fake_head} (clean)"),
        encoding="utf-8",
    )
    assert verify_cluster_report(report_path, expected_rounds=5, expected_samples=200) == []
    assert (
        cluster_bench_main(
            [
                "--verify-report",
                str(report_path),
                "--expected-rounds",
                "5",
                "--expected-samples",
                "200",
            ]
        )
        == 0
    )


def test_verify_cluster_report_rejects_stale_dirty_tampered_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_head = "0123456789abcdef" * 2 + "01234567"
    monkeypatch.setattr(
        "benchmarks.cluster_bench._git_head_and_clean",
        lambda _root=None: (fake_head, True),
    )

    # Stale commit SHA
    stale_report = tmp_path / "stale.md"
    stale_report.write_text(
        _build_valid_cluster_report(commit_line="deadbeef" * 5 + " (clean)"),
        encoding="utf-8",
    )
    assert any("Commit SHA mismatch" in err for err in verify_cluster_report(stale_report))
    assert cluster_bench_main(["--verify-report", str(stale_report)]) == 1

    # Dirty tag in report even if commit matches
    dirty_report = tmp_path / "dirty.md"
    dirty_report.write_text(
        _build_valid_cluster_report(commit_line=f"{fake_head} (dirty)"),
        encoding="utf-8",
    )
    assert any("Commit SHA mismatch or dirty" in err for err in verify_cluster_report(dirty_report))

    # Live working tree dirty
    monkeypatch.setattr(
        "benchmarks.cluster_bench._git_head_and_clean",
        lambda _root=None: (fake_head, False),
    )
    clean_text_report = tmp_path / "clean_text.md"
    clean_text_report.write_text(
        _build_valid_cluster_report(commit_line=f"{fake_head} (clean)"),
        encoding="utf-8",
    )
    assert any("Working tree is dirty" in err for err in verify_cluster_report(clean_text_report))

    # Restore clean tree, check tampered digests & missing metadata/scenario
    monkeypatch.setattr(
        "benchmarks.cluster_bench._git_head_and_clean",
        lambda _root=None: (fake_head, True),
    )
    bad_digest = tmp_path / "bad_digest.md"
    bad_digest.write_text(
        _build_valid_cluster_report(
            commit_line=f"{fake_head} (clean)",
            source_digest="abcd" * 4,  # truncated 16-char digest must fail
            dataset_fp="f" * 64,
        ),
        encoding="utf-8",
    )
    errs = verify_cluster_report(bad_digest)
    assert any("64-char SHA-256" in e for e in errs)
    assert any("Dataset Fingerprint mismatch" in e for e in errs)

    missing_field = tmp_path / "missing_field.md"
    missing_field.write_text(
        _build_valid_cluster_report(
            commit_line=f"{fake_head} (clean)",
            omit_meta_key="Recall Window",
            omit_scenario="Adversarial Torture Suite",
        ),
        encoding="utf-8",
    )
    errs = verify_cluster_report(missing_field)
    assert any("Missing required report metadata field: Recall Window" in e for e in errs)
    assert any("Adversarial Torture Suite" in e for e in errs)


def test_verify_cluster_report_enforces_settings_runtime_zip_and_repro_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_head = "0123456789abcdef" * 2 + "01234567"
    monkeypatch.setattr(
        "benchmarks.cluster_bench._git_head_and_clean",
        lambda _root=None: (fake_head, True),
    )
    fake_zip_fp = {
        "exists": True,
        "path": "data/images.zip",
        "size_bytes": 123456,
        "png_count": 200,
        "sha256": "c" * 64,
    }

    # Exact images.zip SHA-256 parity passes when matching
    zip_ok = tmp_path / "zip_ok.md"
    zip_ok.write_text(
        _build_valid_cluster_report(
            commit_line=f"{fake_head} (clean)",
            meta_overrides={
                "Haystack Config": "samples=200, use_zip=True",
                "images.zip Fingerprint": _format_zip_fingerprint(use_zip=True, zip_fp=fake_zip_fp),
            },
        ),
        encoding="utf-8",
    )
    assert (
        verify_cluster_report(
            zip_ok,
            expected_rounds=5,
            expected_samples=200,
            expected_use_zip=True,
            zip_fp_override=fake_zip_fp,
        )
        == []
    )

    # Tampered thresholds, runtime, rounds/samples, zip SHA-256, repro seeds/budgets, and risk badge
    tampered = tmp_path / "tampered.md"
    tampered.write_text(
        _build_valid_cluster_report(
            commit_line=f"{fake_head} (clean)",
            meta_overrides={
                "Generated At": "not-an-iso-timestamp",
                "Python": "3.9.0",
                "Torch / Faiss / Pillow": "1.0.0 / 1.0.0 / 9.0.0",
                "Confirmed Thresholds": "hamming <= 99, diff_ratio <= 0.99",
                "Recall Window": "min_k=10, multiplier=3",
                "Statistical Rounds": "3",
                "Haystack Config": "samples=100, use_zip=True",
                "images.zip Fingerprint": _format_zip_fingerprint(
                    use_zip=True,
                    zip_fp={**fake_zip_fp, "sha256": "d" * 64},
                ),
            },
            summary_overrides={
                "Exact Duplicate Swarms": {
                    "mode": "variants",
                    "latency": "10.0ms",
                    "purity": "**80.0%** / 10.0% (2)",
                    "risk": "Clean",
                },
                "Massive 50-Skin Duplicate Swarm": {
                    "risk": "RISK: fake_warning",
                },
            },
            repro_overrides={
                "Derivative Variant Clusters": {
                    "dataset_seed": 999,
                    "max_probes": 10,
                }
            },
        ),
        encoding="utf-8",
    )
    errs = verify_cluster_report(
        tampered,
        expected_rounds=5,
        expected_samples=200,
        expected_use_zip=False,
        zip_fp_override=fake_zip_fp,
        verify_runtime=True,
    )
    assert any("Invalid ISO-8601" in e for e in errs)
    assert any("Report runtime Python mismatch" in e for e in errs)
    assert any("Report runtime Torch / Faiss / Pillow mismatch" in e for e in errs)
    assert any("Report Confirmed Thresholds mismatch" in e for e in errs)
    assert any("Report Recall Window mismatch" in e for e in errs)
    assert any("Statistical Rounds mismatch" in e for e in errs)
    assert any("Haystack samples mismatch" in e for e in errs)
    assert any("Haystack use_zip mismatch" in e for e in errs)
    assert any("images.zip Fingerprint mismatch" in e for e in errs)
    assert any("Exact Duplicate Swarms mode mismatch" in e for e in errs)
    assert any("Exact Duplicate Swarms missing Med/P95" in e for e in errs)
    assert any("Exact Duplicate Swarms has degraded metrics" in e for e in errs)
    assert any("Massive 50-Skin Duplicate Swarm is fully clean" in e for e in errs)
    assert any("Derivative Variant Clusters reproducibility params mismatch" in e for e in errs)

    # verify_runtime=False waives OS/Python/package checks but keeps provenance & budgets
    cross_errs = verify_cluster_report(
        tampered,
        expected_rounds=5,
        expected_samples=200,
        zip_fp_override=fake_zip_fp,
        verify_runtime=False,
    )
    assert not any("Report runtime" in e for e in cross_errs)
    assert any("Statistical Rounds mismatch" in e for e in cross_errs)
    assert any(
        "Derivative Variant Clusters reproducibility params mismatch" in e for e in cross_errs
    )

    # CLI flags reject mismatched rounds/samples
    assert (
        cluster_bench_main(
            [
                "--verify-report",
                str(tampered),
                "--expected-rounds",
                "5",
                "--expected-samples",
                "200",
            ]
        )
        == 1
    )


def test_verify_cluster_report_rejects_missing_scenario_telemetry_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_head = "0123456789abcdef" * 2 + "01234567"
    monkeypatch.setattr(
        "benchmarks.cluster_bench._git_head_and_clean",
        lambda _root=None: (fake_head, True),
    )
    base_text = _build_valid_cluster_report(commit_line=f"{fake_head} (clean)")

    # 1. Reviewer's reproduction sample:
    # Delete all telemetry rows & metrics, keeping only scenario titles & global keywords
    gutted_lines = [
        line
        for line in base_text.splitlines()
        if not (
            line.startswith("- **")
            or line.startswith("  - **")
            or (line.startswith("| `") and "denominator: end_to_end_ms" in line)
        )
    ]
    gutted_lines.append("denominator: end_to_end_ms and instrumented stages:")
    gutted_report = tmp_path / "gutted.md"
    gutted_report.write_text("\n".join(gutted_lines), encoding="utf-8")
    gutted_errs = verify_cluster_report(gutted_report)
    assert len(gutted_errs) > 0
    assert any("missing Wall-Clock Latency" in e for e in gutted_errs)
    assert any("Missing Telemetry Decomposition row" in e for e in gutted_errs)
    assert cluster_bench_main(["--verify-report", str(gutted_report)]) == 1

    # 2. Omit a single scenario's telemetry row in Telemetry table
    omit_telemetry_report = tmp_path / "omit_telemetry.md"
    target_scen = "Exact Duplicate Swarms"
    omit_tel_lines = [
        line
        for line in base_text.splitlines()
        if not (line.startswith(f"| `{target_scen}`") and "denominator: end_to_end_ms" in line)
    ]
    omit_telemetry_report.write_text("\n".join(omit_tel_lines), encoding="utf-8")
    tel_errs = verify_cluster_report(omit_telemetry_report)
    assert any(
        f"Missing Telemetry Decomposition row for scenario: {target_scen}" in e for e in tel_errs
    )

    # 3. Omit service internal in Detailed Scenario section
    omit_service_report = tmp_path / "omit_service.md"
    omit_svc_text = base_text.replace("service internal: 8.00 ms | ", "")
    omit_service_report.write_text(omit_svc_text, encoding="utf-8")
    svc_errs = verify_cluster_report(omit_service_report)
    assert any("missing service internal ms value" in e for e in svc_errs)

    # 4. Omit impure batches count in Detailed Scenario section
    omit_impure_report = tmp_path / "omit_impure.md"
    omit_impure_text = base_text.replace("impure batches: 0, ", "")
    omit_impure_report.write_text(omit_impure_text, encoding="utf-8")
    imp_errs = verify_cluster_report(omit_impure_report)
    assert any("missing impure batches count" in e for e in imp_errs)

    # 5. Omit stage breakdown telemetry in Detailed Scenario section
    omit_stage_report = tmp_path / "omit_stage.md"
    omit_stage_text = base_text.replace(
        "  - **Binary Recall**: 2.00 ms (28.6%) | 10 candidates recalled", ""
    )
    omit_stage_report.write_text(omit_stage_text, encoding="utf-8")
    stg_errs = verify_cluster_report(omit_stage_report)
    assert any("missing Binary Recall stage telemetry" in e for e in stg_errs)

