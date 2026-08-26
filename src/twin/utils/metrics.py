"""Progress metrics utilities — shared by index_service and sync."""

import time


def compute_progress_metrics(
    processed: int,
    total: int,
    started_at: float | None,
    samples: list[tuple[float, int]] | None = None,
    *,
    window_seconds: float = 5.0,
    work_count: int | None = None,
) -> dict:
    """Compute rate, ETA, and progress percentage from rolling samples.

    Args:
        processed: Total items processed so far (including skipped).
        total: Total items to process.
        started_at: perf_counter timestamp of when processing started (or None).
        samples: List of (perf_counter, cumulative_work) tuples.
        window_seconds: Rolling window duration for recent rate (default 5s).
        work_count: Actual work done (e.g. processed - skipped). Defaults to processed.

    Returns:
        Dict with keys: elapsed_ms, rate_img_per_s, eta_ms, progress_pct.
    """
    if started_at is None or total <= 0:
        return {
            "elapsed_ms": 0.0,
            "rate_img_per_s": 0.0,
            "eta_ms": 0.0,
            "progress_pct": 0.0,
        }

    now = time.perf_counter()
    elapsed_ms = (now - started_at) * 1000
    elapsed_s = max(elapsed_ms / 1000, 0.001)

    effective_work = processed if work_count is None else work_count

    # Recent rate (rolling window)
    rate = 0.0
    if samples:
        cutoff = now - window_seconds
        recent = [(t, w) for t, w in samples if t >= cutoff]
        if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
            dw = recent[-1][1] - recent[0][1]
            dt = recent[-1][0] - recent[0][0]
            rate = round(dw / dt, 1) if dt > 0 else 0.0
        elif effective_work > 0:
            rate = round(effective_work / elapsed_s, 1)
    elif effective_work > 0:
        rate = round(effective_work / elapsed_s, 1)

    # ETA uses cumulative rate (more stable for long-running jobs)
    eta_ms = 0.0
    if effective_work > 0:
        cum_rate = effective_work / elapsed_s
        remaining = total - processed
        eta_ms = (remaining / cum_rate) * 1000 if cum_rate > 0 else 0

    progress_pct = round(processed * 100 / max(total, 1), 1)

    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "rate_img_per_s": rate,
        "eta_ms": eta_ms,
        "progress_pct": progress_pct,
    }
