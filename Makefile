
.PHONY: help install test test-cov test-unit test-integration lint format format-check typecheck check clean

.DEFAULT_GOAL := help

PYTHON := python
PYTEST := $(PYTHON) -m pytest

# Installation
install:
	$(PYTHON) -m pip install -e ".[dev]"

# Testing
test:
	$(PYTEST)

test-cov:
	$(PYTEST) --cov=omr_pipeline --cov-report=term-missing --cov-report=html

test-unit:
	$(PYTEST) tests/unit -v

test-integration:
	$(PYTEST) tests/integration -v

# Code quality
lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src/

# Complete quality checks
check: lint format-check typecheck test

# Cleanup
clean:
	python -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in [Path('.pytest_cache'), Path('.mypy_cache'), Path('.ruff_cache'), Path('htmlcov'), Path('build'), Path('dist'), Path('.coverage')]]"
