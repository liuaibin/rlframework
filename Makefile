.PHONY: uv-venv sync lint format format-check typecheck check test test-fast test-cov clean clean-venv clean-all help require-uv

# Project root
ROOTDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Virtual environment directory
VENV_DIR ?= .venv

# Keep uv cache inside the workspace by default so checks work in restricted
# environments. Users can still override this with UV_CACHE_DIR=/path.
UV_CACHE_DIR ?= $(ROOTDIR)/.uv-cache
export UV_CACHE_DIR

# Check if uv is available
HAS_UV := $(shell command -v uv 2>/dev/null)

require-uv:
	@if [ -z "$(HAS_UV)" ]; then \
		echo "Error: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi

# ---------------------------------------------------------------------------
# Setup (uv-only)
# ---------------------------------------------------------------------------
help:
	@echo "rlframework development targets:"
	@echo ""
	@echo "  uv-venv      Create virtual environment with uv"
	@echo "  sync         Install dependencies using uv sync"
	@echo "                (set UV_SYNC_OPTS=... to pass extra args, e.g. --index-url)"
	@echo ""
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with ruff"
	@echo "  format-check Check formatting with ruff"
	@echo "  typecheck    Type-check with mypy"
	@echo "  check        Run all checks (lint + format-check + typecheck)"
	@echo ""
	@echo "  test         Run tests"
	@echo "  test-fast    Run tests (skip slow)"
	@echo "  test-cov     Run tests with coverage report"
	@echo ""
	@echo "  clean        Remove build artifacts"
	@echo "  clean-venv   Remove virtual environment"
	@echo "  clean-all    Remove everything"

# uv-based setup (recommended if uv is installed)
uv-venv: require-uv
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		uv venv $(VENV_DIR) && echo "✓ Virtual environment created at $(VENV_DIR) using uv"; \
	fi
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

sync: require-uv | uv-venv
	uv sync --dev $(UV_SYNC_OPTS)
	@echo "✓ Dependencies synced with uv"

# ---------------------------------------------------------------------------
# Lint / format  (fall back to python -m if bare binary not on PATH)
# ---------------------------------------------------------------------------
lint: require-uv
	uv run ruff check . tests

format: require-uv
	uv run ruff format . tests

format-check: require-uv
	uv run ruff format --check . tests

typecheck: require-uv
	uv run mypy .

# Run all checks (CI gate)
check: lint format-check typecheck

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: require-uv
	uv run pytest tests/ -v

# Fast: skip slow integration tests
test-fast: require-uv
	uv run pytest tests/ -v -m "not slow"

# With coverage report
test-cov: require-uv
	uv run pytest tests/ --cov=rlframework --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf dist build .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache

clean-venv:
	rm -rf $(VENV_DIR)

clean-all: clean clean-venv
	@echo "✓ All cleaned"
