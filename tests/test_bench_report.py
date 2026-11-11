"""Unit tests for benchmark regression gate and median/p95 reporting."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.bench_report import _group_name, _stats_p95, main


def _make_payload(
    *,
    system: str = "Windows",
    cpu: str = "TestCPU",
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
        "machine_info": {"system": system, "cpu": {"brand_raw": cpu}},
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
        tmp_path / "base.json", _make_payload(system="Linux", cpu="AMD", samples=[0.010] * 10)
    )
    cur_path = _write_json(
        tmp_path / "cur.json", _make_payload(system="Windows", cpu="Intel", samples=[0.010] * 10)
    )
    assert main([cur_path, "--compare", bl_path, "--fail-threshold", "10"]) == 1
    assert (
        main([cur_path, "--compare", bl_path, "--fail-threshold", "10", "--allow-cross-env"])
        == 0
    )


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

