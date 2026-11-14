"""Isolation verification tests for seedless discovery benchmarks (Gate 1).

Verifies that benchmark execution never mutates process-global settings, environment
variables, host disk artifacts, production data directories, the global indexer
singleton, or non-empty caller-owned Indexer instances.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.cluster.dataset import BenchmarkItem, create_base_skin
from benchmarks.cluster.evaluator import (
    _assert_safe_sandbox_path,
    _prod_data_dir,
    evaluate_cluster_scenario,
    populate_cluster_index,
)
from twin.core.config import settings
from twin.services.embedding import get_embedding_dim
from twin.services.indexer import Indexer, indexer


def _make_dup_items(count: int = 8, seed: int = 42) -> list[BenchmarkItem]:
    base = create_base_skin(seed=seed)
    return [
        BenchmarkItem(filename=f"dup_{i:04d}.png", image=base.copy(), cluster_id="cluster_dup_1")
        for i in range(count)
    ]


_MAX_HASH_BYTES = 16 * 1024 * 1024


def _snapshot_files(root: Path) -> dict[Path, tuple[Any, ...]]:
    """Snapshot file sizes and hashes under root without reading multi-GB archives."""
    if not root.exists():
        return {}

    state: dict[Path, tuple[Any, ...]] = {}

    def _record_file(p: Path) -> None:
        st = p.stat()
        rel = p.relative_to(root)
        if st.st_size <= _MAX_HASH_BYTES:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            state[rel] = (st.st_size, digest)
        else:
            state[rel] = (st.st_size, st.st_mtime_ns, "stat-only")

    for entry in sorted(root.iterdir()):
        if entry.is_file():
            _record_file(entry)
        elif entry.is_dir():
            if entry.name == "images":
                st = entry.stat()
                state[entry.relative_to(root)] = (0, st.st_mtime_ns, "dir:images")
                continue
            for p in sorted(entry.rglob("*")):
                if p.is_file():
                    _record_file(p)
    return state


def test_cluster_memory_and_disk_isolation_sentinel(tmp_path: Path) -> None:
    """Benchmark execution preserves host disk files, settings/env, and global singleton."""
    orig_path = settings.index_path
    orig_env = os.environ.get("TWIN_INDEX_PATH")
    orig_dir = settings.index_dir.resolve()
    orig_files = _snapshot_files(orig_dir)

    host_dir = tmp_path / "host_data"
    sentinel_files = {
        host_dir / "sentinel.txt": "CRITICAL_HOST_INDEX_DATA",
        host_dir / "skin" / "index.flat.faiss": "SKIN_FAISS_BINARY_CONTENT",
        host_dir / "skin" / "index.binary.faiss": "SKIN_BINARY_HASH_CONTENT",
        host_dir / "skin" / "metadata.flat.json": '[{"filename": "host_skin_0.png"}]',
        host_dir / "skin" / "index.flat.count": "10",
        host_dir / "cape" / "index.flat.faiss": "CAPE_FAISS_BINARY_CONTENT",
        host_dir / "cape" / "index.binary.faiss": "CAPE_BINARY_HASH_CONTENT",
        host_dir / "cape" / "metadata.flat.json": '[{"filename": "host_cape_0.png"}]',
        host_dir / "cape" / "index.flat.count": "5",
    }
    for path, content in sentinel_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    host_snapshot_before = _snapshot_files(host_dir)

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
        # Seed the global singleton across both datasources without touching host_dir
        settings.index_path = str(empty_stage)
        dummy_vec = np.zeros(get_embedding_dim(), dtype=np.float32)
        for i in range(6):
            indexer.add_item(
                dummy_vec,
                {"filename": f"host_singleton_skin_{i}.png", "path": f"/host/skin_{i}.png"},
                datasource="skin",
            )
        for i in range(4):
            indexer.add_item(
                dummy_vec,
                {"filename": f"host_singleton_cape_{i}.png", "path": f"/host/cape_{i}.png"},
                datasource="cape",
            )

        with indexer._lock:
            snap_indices = dict(indexer._indices)
            snap_binary = dict(indexer._binary_indices)
            snap_meta = {k: [dict(m) for m in v] for k, v in indexer._metadatas.items()}
            snap_dirty = dict(indexer._dirty_flags)

        # Point active settings/env to host_dir with sentinel artifacts
        settings.index_path = str(host_dir)
        os.environ["TWIN_INDEX_PATH"] = str(host_dir)

        items = _make_dup_items(count=8)
        res = evaluate_cluster_scenario(
            scenario_name="Sentinel Host Protection Test",
            items=items,
            ground_truth_counts={"cluster_dup_1": 8},
            mode="confirmed",
            max_batches=5,
            batch_size=5,
        )
        assert res.discovered_batches == 2

        # Also verify sandbox runs safely even when settings.index_path points directly to data/
        prod_dir = _prod_data_dir()
        settings.index_path = str(prod_dir)
        os.environ["TWIN_INDEX_PATH"] = str(prod_dir)
        res_prod = evaluate_cluster_scenario(
            scenario_name="Host Pointing To Prod Protection Test",
            items=items,
            ground_truth_counts={"cluster_dup_1": 8},
            mode="confirmed",
            max_batches=5,
            batch_size=5,
        )
        assert res_prod.discovered_batches == 2
        assert settings.index_path == str(prod_dir)
        assert os.environ.get("TWIN_INDEX_PATH") == str(prod_dir)

        # Assert all host sentinel files and pre-existing index files are byte-identical
        assert _snapshot_files(host_dir) == host_snapshot_before
        assert _snapshot_files(orig_dir) == orig_files

        # Assert global in-memory singleton remained field-identical across all datasources
        with indexer._lock:
            for ds in indexer.KNOWN_DATASOURCES:
                assert indexer._indices.get(ds) is snap_indices.get(ds)
                assert indexer._binary_indices.get(ds) is snap_binary.get(ds)
                assert indexer._metadatas.get(ds) == snap_meta.get(ds)
                assert indexer._dirty_flags.get(ds) == snap_dirty.get(ds)
        assert indexer.get_count("skin") == 6
        assert indexer.get_count("cape") == 4
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


def test_sandbox_path_guards_reject_forbidden_and_nested_and_ancestor_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative isolation tests: reject prod data, host index, and host image trees."""
    prod_dir = _prod_data_dir()
    active_host = tmp_path / "active_host" / "nested_root"
    active_host.mkdir(parents=True)
    active_images = tmp_path / "active_images" / "nested_root"
    active_images.mkdir(parents=True)

    monkeypatch.setattr(settings, "index_path", str(active_host))
    monkeypatch.setattr(settings, "images_dir", str(active_images))

    forbidden_cases = [
        (prod_dir, "production data directory"),
        (prod_dir / "skin" / "nested", "production data directory"),
        (prod_dir.parent, "production data directory"),
        (active_host, "active host settings.index_dir"),
        (active_host / "skin" / "nested", "active host settings.index_dir"),
        (active_host.parent, "active host settings.index_dir"),
        (active_images, "active settings.images_path"),
        (active_images / "cape" / "nested", "active settings.images_path"),
        (active_images.parent, "active settings.images_path"),
    ]

    items = _make_dup_items(count=2)
    for bad_path, match_msg in forbidden_cases:
        with pytest.raises(ValueError, match=match_msg):
            _assert_safe_sandbox_path(bad_path, label="test sandbox")
        with pytest.raises(ValueError, match=match_msg):
            populate_cluster_index(items, tmp_dir=bad_path, indexer_instance=Indexer())


def test_sandbox_rejects_singleton_non_empty_indexer_or_duplicates(tmp_path: Path) -> None:
    """Reject global singleton, populated caller-owned Indexer, and duplicate filenames."""
    items = _make_dup_items(count=2)
    safe_dir = tmp_path / "safe_sandbox"

    with pytest.raises(ValueError, match="non-singleton Indexer"):
        populate_cluster_index(items, tmp_dir=safe_dir / "a", indexer_instance=None)

    with pytest.raises(ValueError, match="non-singleton Indexer"):
        populate_cluster_index(items, tmp_dir=safe_dir / "b", indexer_instance=indexer)

    with pytest.raises(ValueError, match="caller-owned Indexer instances are rejected"):
        evaluate_cluster_scenario(
            scenario_name="Reject Singleton",
            items=items,
            ground_truth_counts={"cluster_dup_1": 2},
            indexer_instance=indexer,
        )

    # Empty caller-owned Indexer must also be rejected and left untouched
    empty_caller_idx = Indexer()
    with pytest.raises(ValueError, match="caller-owned Indexer instances are rejected"):
        evaluate_cluster_scenario(
            scenario_name="Reject Empty Caller",
            items=items,
            ground_truth_counts={"cluster_dup_1": 2},
            indexer_instance=empty_caller_idx,
        )
    assert empty_caller_idx.get_count("skin") == 0
    assert empty_caller_idx.get_count("cape") == 0

    dup_items = [items[0], items[0]]
    with pytest.raises(ValueError, match="Duplicate benchmark filename"):
        populate_cluster_index(dup_items, tmp_dir=safe_dir / "c", indexer_instance=Indexer())

    # Caller-owned Indexer populated in cape partition must be rejected and left untouched
    caller_idx = Indexer()
    vec = np.zeros(get_embedding_dim(), dtype=np.float32)
    caller_meta = {"filename": "caller_cape.png", "path": "/caller/cape.png"}
    caller_idx.add_item(vec, caller_meta, datasource="cape")

    with pytest.raises(ValueError, match="non-empty caller-owned Indexer"):
        populate_cluster_index(items, tmp_dir=safe_dir / "d", indexer_instance=caller_idx)

    with pytest.raises(ValueError, match="caller-owned Indexer instances are rejected"):
        evaluate_cluster_scenario(
            scenario_name="Reject Populated Caller",
            items=items,
            ground_truth_counts={"cluster_dup_1": 2},
            indexer_instance=caller_idx,
        )

    assert caller_idx.get_count("cape") == 1
    assert caller_idx.get_count("skin") == 0
    assert caller_idx.get_metadata(0, datasource="cape") == caller_meta

    # Also reject Indexer containing metadata entries even without vectors
    meta_only_idx = Indexer()
    meta_only_idx._metadatas["skin"].append({"filename": "orphan_meta.png"})
    with pytest.raises(ValueError, match="caller-owned Indexer instances are rejected"):
        evaluate_cluster_scenario(
            scenario_name="Reject Meta-Only Caller",
            items=items,
            ground_truth_counts={"cluster_dup_1": 2},
            indexer_instance=meta_only_idx,
        )

