.PHONY: venv uv-venv sync install install-all lint format typecheck test test-fast test-cov clean help

# Project root
ROOTDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Virtual environment directory
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

# Check if uv is available
HAS_UV := $(shell command -v uv 2>/dev/null)

# ---------------------------------------------------------------------------
# Setup - choose between uv and venv
# ---------------------------------------------------------------------------
help:
	@echo "rlframework development targets:"
	@echo ""
	@echo "  uv-venv      Create virtual environment with uv"
	@echo "  sync         Install dependencies using uv sync"
	@echo "  venv         Create virtual environment with python venv"
	@echo "  install      Create venv and install with pip (default)"
	@echo "  install-all  Create venv and install all dependencies"
	@echo ""
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with ruff"
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
uv-venv:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		uv venv $(VENV_DIR) && echo "✓ Virtual environment created at $(VENV_DIR) using uv"; \
	fi
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

sync: | uv-venv
	uv sync --dev
	@echo "✓ Dependencies synced with uv"

# pip-based setup (fallback)
venv:
	python3 -m venv $(VENV_DIR)
	@echo "✓ Virtual environment created at $(VENV_DIR)"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install: | venv
	$(VENV_PIP) install -e ".[dev]"
	@echo "✓ Development dependencies installed"

install-all: | venv
	$(VENV_PIP) install -e ".[all]"
	@echo "✓ All dependencies installed"

install-editable:
	pip install -e ".[dev]"

install-editable-all:
	pip install -e ".[all]"

# ---------------------------------------------------------------------------
# Lint / format  (fall back to python -m if bare binary not on PATH)
# ---------------------------------------------------------------------------
lint:
	python -m ruff check . tests

format:
	python -m ruff format . tests

format-check:
	python -m ruff format --check . tests

typecheck:
	python -m mypy .

# Run all checks (CI gate)
check: lint format-check typecheck

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:
	python -m pytest tests/ -v

# Fast: skip slow integration tests
test-fast:
	python -m pytest tests/ -v -m "not slow"

# With coverage report
test-cov:
	python -m pytest tests/ --cov=rlframework --cov-report=html --cov-report=term-missing
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
