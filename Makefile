.PHONY: dev dev-backend dev-frontend install test lint clean

CONDA_ENV := twin-gpu
CONDA_PYTHON := $(HOME)/miniforge3/envs/$(CONDA_ENV)/bin/python
TWIN_ENV := OMP_NUM_THREADS=8 KMP_DUPLICATE_LIB_OK=TRUE

# Start both backend and frontend
dev:
	@echo "Starting backend (port 8000) + frontend (port 5173)..."
	@echo "Press Ctrl+C to stop both."
	@trap 'echo ""; echo "Shutting down..."; \
	  kill -TERM $$BACKEND_PID 2>/dev/null; \
	  kill -TERM $$FRONTEND_PID 2>/dev/null; \
	  sleep 5; \
	  kill -KILL $$BACKEND_PID 2>/dev/null; \
	  kill -KILL $$FRONTEND_PID 2>/dev/null; \
	  lsof -ti:8000 2>/dev/null | xargs -r kill -KILL; \
	  lsof -ti:5173 2>/dev/null | xargs -r kill -KILL; \
	  exit 0' INT TERM; \
	$(TWIN_ENV) conda run --no-capture-output -n $(CONDA_ENV) uvicorn twin.main:app --reload --reload-dir src/twin --host 0.0.0.0 --port 8000 & \
	BACKEND_PID=$$!; \
	cd src/example/web && pnpm dev --host & \
	FRONTEND_PID=$$!; \
	echo "Backend PID: $$BACKEND_PID  Frontend PID: $$FRONTEND_PID"; \
	wait $$BACKEND_PID $$FRONTEND_PID

# Backend only
dev-backend:
	@echo "Starting backend with GPU Faiss..."
	@trap 'kill -TERM $$BACKEND_PID 2>/dev/null; sleep 5; lsof -ti:8000 2>/dev/null | xargs -r kill -KILL' INT TERM; \
	$(TWIN_ENV) conda run --no-capture-output -n $(CONDA_ENV) uvicorn twin.main:app --reload --reload-dir src/twin --host 0.0.0.0 --port 8000 & \
	BACKEND_PID=$$!; \
	wait $$BACKEND_PID

# Backend with uv (CPU-only faiss, for quick dev without GPU)
dev-backend-cpu:
	OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run uvicorn twin.main:app --reload --reload-dir src/twin --host 0.0.0.0 --port 8000

# Frontend only
dev-frontend:
	cd src/example/web && pnpm dev --host

# Install all dependencies (GPU Faiss via conda, CUDA hosts).
# Requires conda env '$(CONDA_ENV)' with faiss-gpu pre-installed.
install:
	@echo "Installing Python deps into conda env '$(CONDA_ENV)'..."
	conda run -n $(CONDA_ENV) pip install -e ".[dev]" 2>/dev/null || \
		(echo "Run 'conda create -n $(CONDA_ENV) python=3.12 faiss-gpu -c pytorch -c nvidia' first" && exit 1)
	cd src/example/web && pnpm install

# Install Python deps via uv (CPU-only, portable).
# Uses faiss-cpu from the [cpu] optional dep group.
install-cpu:
	uv sync --extra cpu
	cd src/example/web && pnpm install

# Install Python deps via conda with GPU Faiss (CUDA hosts).
# Requires conda env '$(CONDA_ENV)' with faiss-gpu pre-installed.
install-gpu:
	@echo "Installing GPU Faiss into conda env '$(CONDA_ENV)'..."
	conda run -n $(CONDA_ENV) pip install -e ".[dev]"
	cd src/example/web && pnpm install

# Run all tests
test:
	$(TWIN_ENV) conda run -n $(CONDA_ENV) pytest tests/ -v

# Run tests with uv (CPU-only)
test-cpu:
	OMP_NUM_THREADS=1 uv run pytest tests/ -v

# Lint backend
lint:
	uv run ruff check src/ tests/

# Format backend
format:
	uv run ruff format src/ tests/

# Train the IVF index (call after indexing enough images)
train:
	curl -s -X POST http://localhost:8000/api/v1/index/train | python3 -m json.tool

# ───────────────────────────────────────────────────────
# Benchmarks
# ───────────────────────────────────────────────────────

BENCH_DIR := tests/benchmarks
BENCH_BASELINE := benchmarks/baseline/baseline.json
BENCH_JSON := bench_results.json
BENCH_FLAGS := --benchmark-only --benchmark-json=$(BENCH_JSON)
BENCH_QUICK_FLAGS := $(BENCH_FLAGS) --benchmark-min-rounds=5

# Quick benchmark run (smoke markers only, fewer rounds)
bench:
	OMP_NUM_THREADS=1 uv run pytest $(BENCH_DIR)/ -m smoke $(BENCH_QUICK_FLAGS) -v

# Benchmark with GPU (conda, direct python to avoid output buffering)
bench-gpu:
	$(TWIN_ENV) $(CONDA_PYTHON) -m pytest $(BENCH_DIR)/ -m "smoke or gpu" $(BENCH_QUICK_FLAGS) -v -s

# Full benchmark run (all benchmarks including slow)
bench-full:
	OMP_NUM_THREADS=1 uv run pytest $(BENCH_DIR)/ $(BENCH_FLAGS) -v --benchmark-min-rounds=10

# Compare against stored baseline and fail on >10% regression
bench-check:
	OMP_NUM_THREADS=1 uv run pytest $(BENCH_DIR)/ -m smoke $(BENCH_FLAGS) \
		--benchmark-compare=$(BENCH_BASELINE) \
		--benchmark-compare-fail=mean:10% -v

# Generate human-readable report from JSON
bench-report:
	uv run python benchmarks/bench_report.py $(BENCH_JSON)

# Save current results as the new baseline
bench-save-baseline:
	cp $(BENCH_JSON) $(BENCH_BASELINE)
	@echo "Baseline saved to $(BENCH_BASELINE)"

# Clean build artifacts
clean:
	rm -rf src/example/web/dist
	rm -rf src/example/web/node_modules
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
