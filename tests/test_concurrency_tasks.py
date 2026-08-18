"""Tests for concurrent async batch indexing, task isolation, and LRU task eviction."""

import tempfile
import time
from pathlib import Path

from PIL import Image

from twin.models.clip_model import load as load_model
from twin.services.index_service import (
    MAX_TASKS,
    _tasks,
    _tasks_lock,
    get_task_status,
    index_batch_async,
)
from twin.services.indexer import indexer

FIXTURES = Path(__file__).parent / "fixtures"


def test_concurrent_tasks_state_isolation():
    """Multiple concurrent async tasks maintain distinct progress and state."""
    load_model()
    indexer.clear()

    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        d1 = Path(dir1)
        d2 = Path(dir2)

        # Create images in d1 and d2
        img = Image.open(FIXTURES / "red.png")
        img.save(d1 / "img1.png")
        img.save(d1 / "img2.png")
        img.save(d2 / "img3.png")

        t1 = index_batch_async(d1)
        t2 = index_batch_async(d2)

        assert t1 != t2

        # Poll both tasks to completion
        for _ in range(50):
            s1 = get_task_status(t1)
            s2 = get_task_status(t2)
            if s1 and s2 and s1.get("status") == "completed" and s2.get("status") == "completed":
                break
            time.sleep(0.1)

        s1 = get_task_status(t1)
        s2 = get_task_status(t2)

        assert s1 is not None
        assert s2 is not None
        assert s1["task_id"] == t1
        assert s2["task_id"] == t2
        assert s1["status"] == "completed"
        assert s2["status"] == "completed"
        assert s1["total"] == 2
        assert s2["total"] == 1
        assert s1["indexed"] == 2
        assert s2["indexed"] == 1


def test_task_lru_pruning():
    """_tasks dictionary prunes oldest completed tasks beyond MAX_TASKS limit."""
    with _tasks_lock:
        _tasks.clear()
        for i in range(MAX_TASKS + 10):
            _tasks[f"old_task_{i}"] = {
                "task_id": f"old_task_{i}",
                "status": "completed",
                "total": 1,
                "indexed": 1,
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.open(FIXTURES / "red.png")
        img.save(Path(tmpdir) / "test.png")

        new_tid = index_batch_async(tmpdir)
        with _tasks_lock:
            assert len(_tasks) <= MAX_TASKS + 1
            assert new_tid in _tasks
            # Oldest tasks should have been evicted
            assert "old_task_0" not in _tasks
