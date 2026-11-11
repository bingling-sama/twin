#!/usr/bin/env python3
"""Post-process pytest-benchmark JSON into a human-readable table.

Usage:
    uv run python benchmarks/bench_report.py bench_results.json
    uv run python benchmarks/bench_report.py bench_results.json --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np


def _fmt_mb(mb: float) -> str:
    """Format megabytes for display."""
    if mb <= 0:
        return "     —"
    if mb < 1:
        return f"{mb:.1f} MB"
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb/1024:.1f} GB"


def _fmt_mb_triple(tm_mb: float, rss_mb: float, gpu_mb: float) -> str:
    """Format tracemalloc + RSS + GPU memory as a single column value.

    tracemalloc is the primary (most accurate) metric and always shown.
    RSS and GPU are supplementary — shown only when > 0.
    """
    parts = []
    if tm_mb > 0:
        parts.append(f"Heap {_fmt_mb(tm_mb).strip()}")
    else:
        parts.append("     —")
    if rss_mb > 0:
        parts.append(f"RSS {_fmt_mb(rss_mb).strip()}")
    if gpu_mb > 0:
        parts.append(f"GPU {_fmt_mb(gpu_mb).strip()}")
    return "  ".join(parts)


def _fmt_ms(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} μs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f} ms"
    return f"{seconds:.3f} s"


def _fmt_pct(part: float, total: float) -> str:
    """Format as percentage of total."""
    if total == 0:
        return "  —"
    return f"{part / total * 100:5.1f}%"


def _group_name(name: str) -> str:
    """Extract a short group name from a benchmark test name."""
    # test_bench_clip_encode_single → clip
    # test_bench_faiss_flat_search_1k → faiss
    # test_bench_hash_dhash_single → hash
    mapping = {
        "cluster": "cluster",
        "clip": "clip",
        "faiss": "faiss",
        "ivfpq": "faiss",
        "ivf": "faiss",
        "hnsw": "faiss",
        "hash": "hash",
        "phash": "hash",
        "dhash": "hash",
        "ssim": "ssim",
        "search": "pipeline",
        "index": "indexing",
        "load": "io",
        "image": "io",
        "comparison": "faiss",
    }
    parts = name.replace("test_bench_", "").split("_")
    for key, group in mapping.items():
        if key in name:
            return group
    return parts[0] if parts else "other"


def _short_name(name: str) -> str:
    """Shorten benchmark name for display."""
    return name.replace("test_bench_", "")


def _load_json(path: str) -> dict[str, Any]:
    """Load a pytest-benchmark JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stats_median(stats: dict[str, Any]) -> float:
    """Return median seconds from pytest-benchmark stats."""
    if stats.get("median") is not None:
        return float(stats["median"])
    data = stats.get("data") or []
    return float(np.median(data)) if data else float(stats.get("mean", 0.0))


def _stats_p95(stats: dict[str, Any]) -> float:
    """Return P95 seconds from pytest-benchmark raw round samples."""
    data = stats.get("data") or []
    if data:
        return float(np.percentile(data, 95))
    return float(stats.get("max", stats.get("mean", 0.0)))


def _machine_os_label(machine: dict[str, Any]) -> str:
    sys_name = str(machine.get("system") or "").strip()
    os_val = machine.get("os")
    distro = str(os_val.get("distro") or "").strip() if isinstance(os_val, dict) else ""
    return distro or sys_name


def evaluate_regression(
    results: dict[str, Any],
    baseline: dict[str, Any],
    fail_threshold: float,
    allow_cross_env: bool = False,
) -> list[str]:
    """Evaluate median & p95 regression and environment compatibility against baseline.

    Returns a list of error strings (empty means regression gate passed).
    """
    errors: list[str] = []
    if fail_threshold < 0:
        return [f"fail_threshold must be non-negative, got {fail_threshold}"]

    res_benchmarks = results.get("benchmarks") or []
    bl_benchmarks = baseline.get("benchmarks") or []
    if not res_benchmarks:
        return ["Current benchmark results contain no benchmarks."]
    if not bl_benchmarks:
        return ["Baseline file contains no benchmarks."]

    res_machine = results.get("machine_info") or {}
    bl_machine = baseline.get("machine_info") or {}
    res_sys = str(res_machine.get("system") or "").strip()
    bl_sys = str(bl_machine.get("system") or "").strip()
    res_cpu = str((res_machine.get("cpu") or {}).get("brand_raw") or "").strip()
    bl_cpu = str((bl_machine.get("cpu") or {}).get("brand_raw") or "").strip()

    env_mismatches: list[str] = []
    if res_sys and bl_sys and res_sys != bl_sys:
        env_mismatches.append(f"OS mismatch ({bl_sys} vs {res_sys})")
    if res_cpu and bl_cpu and res_cpu != bl_cpu:
        env_mismatches.append(f"CPU mismatch ({bl_cpu!r} vs {res_cpu!r})")
    if env_mismatches:
        msg = "Environment mismatch vs baseline: " + "; ".join(env_mismatches)
        if allow_cross_env:
            print(f"WARNING: {msg} (ignored due to --allow-cross-env)")
        else:
            errors.append(msg + " (pass --allow-cross-env to override)")

    res_lookup = {bm["name"]: bm for bm in res_benchmarks}
    bl_lookup = {bm["name"]: bm for bm in bl_benchmarks}

    for bl_name in sorted(bl_lookup):
        if bl_name not in res_lookup:
            errors.append(f"Missing baseline benchmark in current run: {bl_name}")

    for name, bm in res_lookup.items():
        if name not in bl_lookup:
            continue
        stats = bm["stats"]
        bl_stats = bl_lookup[name]["stats"]

        med = _stats_median(stats)
        bl_med = _stats_median(bl_stats)
        p95 = _stats_p95(stats)
        bl_p95 = _stats_p95(bl_stats)

        if bl_med > 0:
            med_delta = (med - bl_med) / bl_med * 100.0
            if med_delta > fail_threshold:
                errors.append(
                    f"Median regression > {fail_threshold:.1f}% on {name}: "
                    f"{_fmt_ms(bl_med)} -> {_fmt_ms(med)} ({med_delta:+.1f}%)"
                )

        if bl_p95 > 0:
            p95_delta = (p95 - bl_p95) / bl_p95 * 100.0
            if p95_delta > fail_threshold:
                errors.append(
                    f"P95 regression > {fail_threshold:.1f}% on {name}: "
                    f"{_fmt_ms(bl_p95)} -> {_fmt_ms(p95)} ({p95_delta:+.1f}%)"
                )

    return errors


def _print_system_header(machine: dict) -> None:
    """Print a detailed system info header block."""
    cpu = machine.get("cpu", {})
    cpu_brand = cpu.get("brand_raw", "unknown")
    cpu_count = cpu.get("count", 0)
    cpu_freq = cpu.get("hz_advertised_friendly", "")

    mem = machine.get("memory", {})

    gpu = machine.get("gpu", {})
    os_info = machine.get("os", {})
    pkgs = machine.get("packages", {})
    disk = machine.get("disk", {})
    py_ver = machine.get("python_version", "unknown")

    lines: list[str] = []
    lines.append("")
    lines.append("═" * 110)
    lines.append("  Twin Benchmark Report")
    lines.append("═" * 110)

    # ── Hardware ──
    lines.append("")
    lines.append("  ── Hardware ──")
    cpu_line = f"  CPU:  {cpu_brand}  ·  {cpu_count} cores"
    if cpu_freq:
        cpu_line += f"  ·  {cpu_freq}"
    lines.append(cpu_line)

    l3 = cpu.get("l3_cache_size", 0)
    l2 = cpu.get("l2_cache_size", 0)
    if l3 or l2:
        cache_parts = []
        if l3:
            cache_parts.append(f"L3 {l3/1024:.0f} MB")
        if l2:
            cache_parts.append(f"L2 {l2/1024:.0f} KB")
        lines.append(f"  Cache:  {'  ·  '.join(cache_parts)}")

    if mem:
        mem_parts = [f"Total {mem['total_gb']:.1f} GB"]
        if mem.get("available_gb", 0) > 0:
            mem_parts.append(f"Available {mem['available_gb']:.1f} GB")
        if mem.get("used_gb", 0) > 0:
            mem_parts.append(f"Used {mem['used_gb']:.1f} GB")
        swap = mem.get("swap_total_gb", 0)
        if swap > 0:
            mem_parts.append(f"Swap {swap:.1f} GB")
            if mem.get("swap_used_gb", 0) > 0:
                mem_parts.append(f"SwapUsed {mem['swap_used_gb']:.1f} GB")
        lines.append(f"  RAM:  {'  ·  '.join(mem_parts)}")

    if gpu:
        gpu_parts = [gpu.get("name", "unknown GPU")]
        vram = gpu.get("vram_total_mb", 0)
        if vram:
            gpu_parts.append(f"VRAM {vram/1024:.1f} GB")
        cc = gpu.get("compute_capability", "")
        if cc:
            gpu_parts.append(f"CC {cc}")
        cuda = gpu.get("cuda_version", "")
        if cuda:
            gpu_parts.append(f"CUDA {cuda}")
        drv = gpu.get("driver_version", "")
        if drv:
            gpu_parts.append(f"Driver {drv}")
        lines.append(f"  GPU:  {'  ·  '.join(gpu_parts)}")

    if disk:
        disk_parts = []
        if disk.get("project_root"):
            disk_parts.append(f"Root {disk['project_root']}")
        if disk.get("disk_total_gb"):
            disk_parts.append(f"Total {disk['disk_total_gb']:.0f} GB")
        if disk.get("disk_free_gb"):
            disk_parts.append(f"Free {disk['disk_free_gb']:.0f} GB")
        if disk.get("filesystem"):
            disk_parts.append(f"FS {disk['filesystem']}")
        if disk_parts:
            lines.append(f"  Disk: {'  ·  '.join(disk_parts)}")

    # ── Software ──
    lines.append("")
    lines.append("  ── Software ──")
    os_parts = []
    if os_info.get("distro"):
        os_parts.append(os_info["distro"])
    os_parts.append(f"Kernel {os_info.get('kernel', '?')}")
    lines.append(f"  OS:    {'  ·  '.join(os_parts)}")

    lines.append(f"  Python:  {py_ver} ({machine.get('python_implementation', 'CPython')})")

    # pytest-benchmark version (extract from __version__ if possible)
    try:
        import pytest_benchmark
        pb_ver = getattr(pytest_benchmark, "__version__", "?")
    except ImportError:
        pb_ver = "?"
    lines.append(f"  pytest-benchmark:  {pb_ver}")

    if pkgs:
        pkg_order = ["torch", "open_clip", "numpy", "PIL", "imagehash", "skimage",
                     "fastapi", "uvicorn"]
        pkg_parts = []
        for p in pkg_order:
            v = pkgs.get(p)
            if v:
                pkg_parts.append(f"{p} {v}")

        # Faiss: show version + GPU/CPU variant
        faiss_v = pkgs.get("faiss", "")
        if faiss_v:
            is_gpu = pkgs.get("faiss_gpu", "False") == "True"
            variant = "gpu" if is_gpu else "cpu"
            pkg_parts.append(f"faiss-{variant} {faiss_v}")

        lines.append(f"  Packages:  {'  ·  '.join(pkg_parts)}")

    lines.append("")
    lines.append("═" * 110)
    lines.append("")

    for line in lines:
        print(line)


def print_report(results: dict, baseline: dict | None = None) -> None:
    """Print a formatted benchmark report table."""
    machine = results.get("machine_info", {})
    benchmarks = results.get("benchmarks", [])

    if not benchmarks:
        print("No benchmarks found in results.")
        return

    # Build baseline lookup
    baseline_lookup: dict[str, dict] = {}
    if baseline:
        for bm in baseline.get("benchmarks", []):
            baseline_lookup[bm["name"]] = bm

    # Group benchmarks
    groups: dict[str, list[dict]] = defaultdict(list)
    for bm in benchmarks:
        groups[_group_name(bm["name"])].append(bm)

    # Print detailed system header
    _print_system_header(machine)

    # Group display order
    group_order = ["cluster", "clip", "faiss", "hash", "ssim", "pipeline", "indexing", "io"]

    for group_key in group_order:
        if group_key not in groups:
            continue

        group_benchmarks = groups[group_key]
        group_title = {
            "cluster": "Cluster Discovery",
            "clip": "CLIP Encoding",
            "faiss": "Faiss Index",
            "hash": "Perceptual Hashing",
            "ssim": "SSIM",
            "pipeline": "Search Pipeline",
            "indexing": "Indexing",
            "io": "Image I/O",
        }.get(group_key, group_key)

        print(f"  ── {group_title} ──")
        hdr = (
            f"  {'Benchmark':<44} {'Median':>9} {'P95':>9} {'Mean':>9} {'Min':>9} {'Max':>9}  "
            f"{'Mem (Peak)':>22}  {'Rounds':>6}"
        )
        print(hdr)
        print(
            f"  {'─' * 44} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9}  "
            f"{'─' * 22}  {'─' * 6}"
        )

        for bm in group_benchmarks:
            full_name = bm["name"]
            name = _short_name(full_name)
            stats = bm["stats"]
            med = _stats_median(stats)
            p95 = _stats_p95(stats)
            med_str = _fmt_ms(med)
            p95_str = _fmt_ms(p95)
            mean_str = _fmt_ms(stats["mean"])
            min_str = _fmt_ms(stats["min"])
            max_str = _fmt_ms(stats["max"])
            rounds = stats["rounds"]

            # Stats + memory (tracemalloc primary, RSS + GPU supplementary)
            extras = bm.get("extra_info") or {}
            peak_tm = extras.get("peak_tracemalloc_mb", 0)
            peak_rss = extras.get("peak_rss_mb", 0)
            peak_gpu = extras.get("peak_gpu_mb", 0)
            mem_str = _fmt_mb_triple(peak_tm, peak_rss, peak_gpu)

            # Show regression if baseline available
            reg_str = ""
            if full_name in baseline_lookup:
                bl_stats = baseline_lookup[full_name]["stats"]
                bl_med = _stats_median(bl_stats)
                bl_p95 = _stats_p95(bl_stats)
                med_delta = ((med - bl_med) / bl_med * 100) if bl_med > 0 else 0.0
                p95_delta = ((p95 - bl_p95) / bl_p95 * 100) if bl_p95 > 0 else 0.0
                if abs(med_delta) > 5 or abs(p95_delta) > 5:
                    arrow = "▲" if max(med_delta, p95_delta) > 0 else "▼"
                    reg_str = f"  {arrow} med:{med_delta:+.1f}% p95:{p95_delta:+.1f}%"

            print(
                f"  {name:<44} {med_str:>9} {p95_str:>9} {mean_str:>9} {min_str:>9} {max_str:>9}  "
                f"{mem_str:>22}  {rounds:>6}{reg_str}"
            )

            # Show stage breakdown for pipeline benchmarks
            stages = extras.get("stages", {})
            if stages:
                total = extras.get("total_ms", 1)
                for stage_name in ["faiss", "dhash", "phash", "ssim"]:
                    stage = stages.get(stage_name, {})
                    if stage:
                        elapsed = stage.get("elapsed_ms", 0)
                        pct = _fmt_pct(elapsed, total) if total else ""
                        io_str = f"  in={stage.get('in','?')} → out={stage.get('out','?')}"
                        print(
                            f"    ├─ {stage_name:<8} {_fmt_ms(elapsed/1000):>8}  {pct}"
                            f"  {io_str}"
                        )

        print()

    # Summary
    total_benchmarks = len(benchmarks)
    total_time = sum(
        bm["stats"]["mean"] * bm["stats"]["iterations"] * bm["stats"]["rounds"]
        for bm in benchmarks
    )
    print(f"  {total_benchmarks} benchmarks  ·  ~{_fmt_ms(total_time)} cumulative mean runtime")
    print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Format pytest-benchmark JSON and optionally enforce median/p95 regression gate."
        )
    )
    parser.add_argument("results_json", help="Path to pytest-benchmark JSON results")
    parser.add_argument("--compare", type=str, default=None, help="Path to baseline JSON file")
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Fail with non-zero exit code if median or p95 regresses more than PCT%% vs baseline",
    )
    parser.add_argument(
        "--allow-cross-env",
        action="store_true",
        help="Allow comparison against a baseline captured on a different OS or CPU",
    )
    args = parser.parse_args(argv)

    results = _load_json(args.results_json)
    baseline = _load_json(args.compare) if args.compare else None
    print_report(results, baseline)

    if args.fail_threshold is not None:
        if baseline is None:
            print("ERROR: --fail-threshold requires --compare <baseline.json>", file=sys.stderr)
            return 2
        errors = evaluate_regression(
            results,
            baseline,
            fail_threshold=args.fail_threshold,
            allow_cross_env=args.allow_cross_env,
        )
        if errors:
            print("REGRESSION GATE FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print(
            f"REGRESSION GATE PASSED (median & p95 within +{args.fail_threshold:.1f}% vs baseline)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
