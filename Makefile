.PHONY: help install install-dev test test-fast test-all test-cards test-engine test-cfr test-integration coverage format lint typecheck clean train-quick train train-distributed play benchmark docs serve-docs docker-build docker-run profile

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)Mindbug Deep CFR Development Commands$(NC)"
	@echo "$(BLUE)=====================================$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  install          Install package in development mode"
	@echo "  install-dev      Install with all development dependencies"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  test             Run test suite (fast tests only)"
	@echo "  test-fast        Run fast tests with early exit"
	@echo "  test-all         Run all tests including slow ones"
	@echo "  test-cards       Test card implementations"
	@echo "  test-engine      Test game engine"
	@echo "  test-cfr         Test Deep CFR algorithm"
	@echo "  test-integration Run integration tests"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  format           Format code with black and isort"
	@echo "  lint             Run linting checks (ruff)"
	@echo "  typecheck        Run type checking (mypy)"
	@echo ""
	@echo "$(GREEN)Training:$(NC)"
	@echo "  train-quick      Quick training test (100 iterations)"
	@echo "  train            Standard training (10k iterations)"
	@echo "  train-distributed Distributed training setup"
	@echo ""
	@echo "$(GREEN)Running:$(NC)"
	@echo "  play             Play interactive game"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  clean            Clean all build artifacts"
	@echo "  profile          Profile code performance"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker container"

# Setup commands
install:
	pip install -e ".[dev]"

install-dev:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"
	pre-commit install

# Testing commands
test:
	python -m pytest tests/ -v -m "not slow"

test-fast:
	python -m pytest tests/ -v -m "not slow" -x --tb=short

test-all:
	python -m pytest tests/ -v --tb=short

test-cards:
	python -m pytest tests/test_cards.py -v

test-engine:
	python -m pytest tests/test_engine.py -v

test-cfr:
	python -m pytest tests/test_cfr.py -v

test-integration:
	python -m pytest tests/test_integration.py -v

coverage:
	python -m pytest tests/ -v --cov=mindbug --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

# Code quality commands
format:
	black mindbug/ tests/ *.py
	isort mindbug/ tests/ *.py

lint:
	ruff check mindbug/ tests/ *.py

typecheck:
	mypy mindbug/ --ignore-missing-imports

# Training commands
train-quick:
	python train.py --config quick --iterations 100

train:
	python train.py --config performance --iterations 10000

train-distributed:
	@echo "$(BLUE)Setting up distributed training...$(NC)"
	@echo "Run on each GPU node:"
	@echo "  torchrun --nproc_per_node=NUM_GPUS train.py --config distributed"

# Running commands
play:
	python play.py

play-trained:
	@echo "$(BLUE)Looking for latest checkpoint...$(NC)"
	@LATEST=$(find checkpoints -name "final_checkpoint.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" "); \
	if [ -z "$LATEST" ]; then \
		echo "$(RED)No trained model found. Train one with 'make train'$(NC)"; \
	else \
		echo "$(GREEN)Found checkpoint: $LATEST$(NC)"; \
		python play.py --checkpoint "$LATEST"; \
	fi

benchmark:
	python benchmark.py --all

benchmark-quick:
	python benchmark.py --validate --state --network

# Documentation
docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html

serve-docs:
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && python -m http.server 8000

# Profiling
profile:
	python -m cProfile -o profile.stats train.py --config debug --iterations 10
	python -m pstats profile.stats

profile-view:
	snakeviz profile.stats

# Docker support
docker-build:
	docker build -t mindbug-deep-cfr .

docker-run:
	docker run --gpus all -it --rm \
		-v $(pwd)/checkpoints:/app/checkpoints \
		-v $(pwd)/runs:/app/runs \
		mindbug-deep-cfr

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf htmlcov/ .coverage .coverage.* .pytest_cache/
	rm -rf checkpoints/ runs/ logs/ tensorboard/
	rm -rf profile.stats
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".DS_Store" -delete

clean-checkpoints:
	@echo "$(RED)Warning: This will delete all trained models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $REPLY =~ ^[Yy]$ ]]; then \
		rm -rf checkpoints/; \
		echo "$(GREEN)Checkpoints deleted.$(NC)"; \
	fi

# Development shortcuts
dev: format lint typecheck test-fast

check: lint typecheck test

all: clean install test-all coverage

# CI/CD helpers
ci-test:
	python -m pytest tests/ -v --cov=mindbug --cov-report=xml --junitxml=junit.xml

ci-lint:
	ruff check mindbug/ tests/ --format=github
	mypy mindbug/ --ignore-missing-imports --junit-xml mypy.xml

# Utility targets
watch:
	@echo "$(BLUE)Watching for changes...$(NC)"
	watchmedo auto-restart --directory=./mindbug --pattern=*.py --recursive -- python train.py --config debug --iterations 10

tensorboard:
	tensorboard --logdir checkpoints/ --bind_all

gpu-check:
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

requirements:
	pip freeze > requirements-freeze.txt
	@echo "$(GREEN)Frozen requirements saved to requirements-freeze.txt$(NC)"