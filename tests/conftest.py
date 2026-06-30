"""Pytest configuration — prevent OpenMP thread contention between Torch and Faiss."""

import os
import tempfile


def pytest_configure(config):
    """Set environment variables before any imports to avoid thread pool conflicts."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Use a temp directory for TWIN_IMAGES_DIR so the startup sync in
    # TestClient lifespan doesn't try to index the real data/images/
    # (which may contain hundreds of thousands of images).
    if "TWIN_IMAGES_DIR" not in os.environ:
        tmpdir = tempfile.mkdtemp(prefix="twin_test_images_")
        os.environ["TWIN_IMAGES_DIR"] = tmpdir
        config._twin_test_images_dir = tmpdir  # save for cleanup


def pytest_unconfigure(config):
    """Clean up temp directories created during pytest_configure."""
    tmpdir = getattr(config, "_twin_test_images_dir", None)
    if tmpdir and os.path.isdir(tmpdir):
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
