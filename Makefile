.PHONY: help install-lint lint lint-fix format test test-judge clean check uv-sync uv-install uv-test uv-run

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "UV (Fast Package Manager):"
	@echo "  uv-sync         - Sync test dependencies (fast, no GPU deps)"
	@echo "  uv-sync-full    - Sync ALL dependencies (includes GPU/model deps)"
	@echo "  uv-install      - Install project with uv"
	@echo "  uv-test         - Run tests using uv"
	@echo "  uv-run          - Run command with uv (e.g., make uv-run CMD='python script.py')"
	@echo ""
	@echo "Traditional Commands:"
	@echo "  install-lint    - Install linting and formatting tools"
	@echo "  lint            - Run all linters (ruff, black check, mypy)"
	@echo "  lint-fix        - Auto-fix linting issues where possible"
	@echo "  format          - Format code with black"
	@echo "  test            - Run all tests"
	@echo "  test-judge      - Run LLM judge script tests"
	@echo "  dry-run         - Verify everything works (no LLM needed)"
	@echo "  check           - Run linters and tests"
	@echo "  clean           - Remove cache and temporary files"

# Install linting tools
install-lint:
	@echo "📦 Installing linting tools..."
	pip install ruff black mypy types-tqdm
	@echo "✅ Linting tools installed!"

# Run all linters
lint:
	@echo "🔍 Running linters..."
	@echo "\n=== Running Ruff ==="
	ruff check . --exclude "*.ipynb" || true
	@echo "\n=== Checking code formatting with Black ==="
	black --check --exclude "\.ipynb$$" . || true
	@echo "\n=== Running MyPy type checking ==="
	mypy reasoning_eval/llm_judge_script.py --ignore-missing-imports || true
	@echo "✅ Linting complete!"

# Auto-fix linting issues
lint-fix:
	@echo "🔧 Auto-fixing linting issues..."
	ruff check . --exclude "*.ipynb" --fix
	black --exclude "\.ipynb$$" .
	@echo "✅ Auto-fix complete!"

# Format code with black
format:
	@echo "🎨 Formatting code with Black..."
	black --exclude "\.ipynb$$" .
	@echo "✅ Formatting complete!"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v || python -m unittest discover -s tests -p "test_*.py" -v
	@echo "✅ Tests complete!"

# Run LLM judge script tests specifically
test-judge:
	@echo "🧪 Running LLM judge script tests..."
	python tests/test_llm_judge_script.py
	@echo "✅ Tests complete!"

# Run both linting and tests
check: lint test
	@echo "✅ All checks passed!"

# Clean up cache and temporary files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# ============================================
# UV Commands (Fast Package Manager)
# ============================================

# Sync dependencies with uv (test dependencies only, fast)
uv-sync:
	@echo "🔄 Setting up virtual environment and test dependencies..."
	@test -d .venv || uv venv
	@uv pip install -r requirements-test.txt
	@echo "✅ Test dependencies synced!"

# Install full dependencies (includes GPU/model dependencies)
uv-sync-full:
	@echo "🔄 Setting up virtual environment and FULL dependencies..."
	@echo "⚠️  Warning: This includes GPU dependencies and may take a while..."
	@test -d .venv || uv venv
	@uv pip install -r requirements.txt
	@echo "✅ Full dependencies synced!"

# Install project with uv
uv-install:
	@echo "📦 Installing project with uv..."
	@test -d .venv || uv venv
	@uv pip install -e .
	@echo "✅ Project installed!"

# Run tests with uv (lightweight dependencies only)
uv-test:
	@echo "🧪 Running tests with uv..."
	@test -d .venv || (echo "Creating venv..." && uv venv && uv pip install -r requirements-test.txt)
	@.venv/bin/python tests/test_llm_judge_script.py
	@echo "✅ Tests complete!"

# Run dry-run test (verifies everything works without running LLM)
dry-run:
	@echo "🚀 Running dry-run test (no LLM needed)..."
	@test -d .venv || (echo "Creating venv..." && uv venv && uv pip install -r requirements-test.txt)
	@.venv/bin/python tests/test_dry_run.py
	@echo "✅ Dry-run complete!"

# Run any command with venv (usage: make uv-run CMD="python script.py")
uv-run:
	@echo "🚀 Running with venv..."
	@test -d .venv || (echo "Creating venv..." && uv venv && uv pip install -r requirements-test.txt)
	@.venv/bin/$(CMD)

