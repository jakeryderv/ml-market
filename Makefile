.PHONY: help install test lint format clean train

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

test:  ## Run all tests
	uv run pytest -v

test-fast:  ## Run tests, skip integration tests
	uv run pytest -v -m "not integration"

lint:  ## Run linters (ruff, mypy)
	uv run ruff check .
	uv run mypy src/

format:  ## Format code with ruff
	uv run ruff format .

clean:  ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

train:  ## Run training script
	uv run python scripts/test.py

check: lint test  ## Run lint and test
