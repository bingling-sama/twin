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
    omit_meta_key: str | None = None,
    omit_scenario: str | None = None,
) -> str:
    src_dig = compute_cluster_source_digest() if source_digest is None else source_digest
    ds_fp = get_cluster_dataset_fingerprint() if dataset_fp is None else dataset_fp
    meta = {
        "Generated At": "2026-11-11T19:00:00+08:00",
        "Commit SHA": commit_line,
        "Source Digest (SHA-256)": src_dig,
        "Dataset Fingerprint (SHA-256)": ds_fp,
        "Python": "3.12.11",
        "OS Platform": "Windows 11 (AMD64)",
        "CPU": "AMD64 (16 logical cores)",
        "GPU": "NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB)",
        "Torch / Faiss / Pillow": "2.6.0+cu124 / 1.15.0 / 12.3.0",
        "Statistical Rounds": "5",
        "Haystack Config": "samples=200, use_zip=True",
        "images.zip Fingerprint": "disabled (--no-zip)",
        "Confirmed Thresholds": "hamming <= 4, diff_ratio <= 0.05",
        "Variants Thresholds": "hamming <= 12, diff_ratio <= 0.35",
        "Recall Window": "min_k=10, multiplier=3",
    }
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
            "| Scenario | Quality Risk |",
            "| :--- | :--- |",
        ]
    )
    for name in _EXPECTED_SCENARIO_NAMES:
        if name != omit_scenario:
            lines.append(f"| **{name}** | `Clean` |")

    lines.extend(
        [
            "",
            "## 🔧 Scenario Reproducibility Matrix",
            "",
            "## 🔬 Telemetry & Fine-Grained Latency Decomposition",
            "",
            "Pairs verified (1,000.0 pairs/s, denominator: end_to_end_ms)",
            "",
            "## 📝 Detailed Scenario Analysis",
            "",
        ]
    )
    for name in _EXPECTED_SCENARIO_NAMES:
        if name != omit_scenario:
            lines.extend(
                [
                    f"### Scenario: {name} (`mode=confirmed`)",
                    "- **Instrumented Stages Total**: 5.00 ms (instrumented stages: 5.00 ms)",
                    "- **Pairs**: 100.0 pairs/sec, denominator: end_to_end_ms",
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
    assert verify_cluster_report(report_path) == []
    assert cluster_bench_main(["--verify-report", str(report_path)]) == 0


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

