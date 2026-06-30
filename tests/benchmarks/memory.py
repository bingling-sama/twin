"""Memory sampling utilities for benchmarks.

Three measurement layers, each tracking true PEAK (not net delta):

1. tracemalloc  — Python-level peak allocated bytes (accurate, per-allocation)
2. RSS sampler  — OS-level peak RSS via background thread (1 kHz polling)
3. GPU VRAM     — CUDA peak allocated minus pre-existing baseline

All three track the maximum observed across all invocations of the
wrapped function (warmup rounds included).

Graceful degradation: tracemalloc unavailable → 0; no /proc → 0; no CUDA → 0.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Callable


# ── tracemalloc (Python-level heap peak) ──────────────────────────────────────

_tracemalloc_started: bool = False


def _ensure_tracemalloc() -> bool:
    """Start tracemalloc if not already running. Returns True if active."""
    global _tracemalloc_started
    if _tracemalloc_started:
        return True
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()
        _tracemalloc_started = True
        return True
    except Exception:
        return False


def _reset_tracemalloc_peak() -> None:
    """Reset tracemalloc peak stats to current allocation level."""
    try:
        import tracemalloc

        tracemalloc.reset_peak()
    except Exception:
        pass


def _get_tracemalloc_peak_mb() -> float:
    """Return tracemalloc peak in MB since last reset."""
    try:
        import tracemalloc

        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)
    except Exception:
        return 0.0


# ── RSS (OS-level, continuous background sampling) ────────────────────────────


def _get_rss_mb() -> float:
    """Return current process RSS in megabytes, or 0 on failure."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            fields = f.read().split()
            if len(fields) >= 2:
                pages = int(fields[1])
                return pages * 4 / 1024  # → MB
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


class _RssSampler:
    """Background thread that samples RSS at ~1 kHz, tracking peak.

    The sampler records the absolute RSS at start (baseline) and the
    maximum absolute RSS observed during the sampling window.  The
    reported delta is peak − baseline, clamped to ≥ 0.
    """

    def __init__(self, interval_s: float = 0.001):
        self._interval = interval_s
        self._baseline_mb: float = 0.0
        self._peak_mb: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def peak_delta_mb(self) -> float:
        """Peak RSS increase above baseline (MB), ≥ 0."""
        with self._lock:
            return max(0.0, self._peak_mb - self._baseline_mb)

    @property
    def absolute_peak_mb(self) -> float:
        """Absolute peak RSS observed (MB)."""
        with self._lock:
            return self._peak_mb

    @property
    def baseline_mb(self) -> float:
        """Baseline RSS at sampler start (MB)."""
        with self._lock:
            return self._baseline_mb

    def start(self) -> None:
        """Capture baseline RSS and begin sampling."""
        self._baseline_mb = _get_rss_mb()
        self._peak_mb = self._baseline_mb
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and join the background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_loop(self) -> None:
        while self._running:
            current = _get_rss_mb()
            with self._lock:
                if current > self._peak_mb:
                    self._peak_mb = current
            time.sleep(self._interval)


# ── GPU VRAM ──────────────────────────────────────────────────────────────────


def _gpu_available() -> bool:
    """Check CUDA availability without importing torch unnecessarily."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_gpu_allocated_mb() -> float:
    """Return currently allocated GPU memory in MB."""
    try:
        import torch

        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def _get_gpu_peak_mb() -> float:
    """Return peak GPU memory allocated since last reset, in MB."""
    try:
        import torch

        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def _reset_gpu_peak() -> None:
    """Reset CUDA peak memory stats."""
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── Function wrapper ──────────────────────────────────────────────────────────


def wrap_with_memory(
    fn: Callable[..., Any],
) -> tuple[Callable[..., Any], dict[str, float]]:
    """Wrap a callable to track PEAK memory across all invocations.

    Returns ``(wrapped_fn, peaks)`` where *peaks* is a dict with::

        peak_tracemalloc_mb  — Python heap peak (tracemalloc, per-allocation)
        peak_rss_mb          — OS RSS net peak via 1 kHz background sampler
        peak_gpu_mb          — GPU VRAM net peak (baseline subtracted)

    All values track the maximum observed across every call to
    *wrapped_fn* (including warmup rounds run by pytest-benchmark).
    """
    peaks: dict[str, float] = {
        "peak_tracemalloc_mb": 0.0,
        "peak_rss_mb": 0.0,
        "peak_gpu_mb": 0.0,
    }

    _tm_ok = _ensure_tracemalloc()
    _gpu_ok = _gpu_available()

    @functools.wraps(fn)
    def _tracked(*args: Any, **kwargs: Any) -> Any:
        # 1. Snapshot GPU baseline BEFORE reset (model already on GPU)
        gpu_baseline_mb = _get_gpu_allocated_mb() if _gpu_ok else 0.0

        # 2. Reset tracemalloc peak
        if _tm_ok:
            _reset_tracemalloc_peak()

        # 3. Start RSS background sampler (captures baseline internally)
        sampler = _RssSampler(interval_s=0.001)
        sampler.start()

        # 4. Reset GPU peak stats (peak tracked from current allocated level)
        if _gpu_ok:
            _reset_gpu_peak()

        try:
            result = fn(*args, **kwargs)
        finally:
            # 5. Stop RSS sampler
            sampler.stop()

            # 6. Collect tracemalloc peak (Python heap)
            if _tm_ok:
                tm_mb = _get_tracemalloc_peak_mb()
                if tm_mb > peaks["peak_tracemalloc_mb"]:
                    peaks["peak_tracemalloc_mb"] = round(tm_mb, 2)

            # 7. Collect RSS peak delta (OS-level, continuous sampling)
            rss_delta = sampler.peak_delta_mb
            if rss_delta > peaks["peak_rss_mb"]:
                peaks["peak_rss_mb"] = round(rss_delta, 2)

            # 8. Collect GPU peak (net of pre-existing baseline)
            if _gpu_ok:
                gpu_net = _get_gpu_peak_mb() - gpu_baseline_mb
                if gpu_net < 0:
                    gpu_net = 0.0  # clamp: model already counted
                if gpu_net > peaks["peak_gpu_mb"]:
                    peaks["peak_gpu_mb"] = round(gpu_net, 2)

        return result

    return _tracked, peaks
