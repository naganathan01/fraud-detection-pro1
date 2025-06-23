# Makefile
.PHONY: help install test lint format clean build run docker-build docker-run deploy

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  build       Build application"
	@echo "  run         Run application locally"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run Docker container"
	@echo "  deploy      Deploy to Kubernetes"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

build:
	python -m build

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -f infrastructure/docker/Dockerfile.api -t fraud-detection:latest .

docker-run:
	docker-compose -f infrastructure/docker/docker-compose.yml up -d

deploy:
	kubectl apply -f infrastructure/kubernetes/
	kubectl rollout status deployment/fraud-detection-api -n fraud-detection

train-model:
	python training/pipeline/model_training.py

validate-model:
	python training/pipeline/model_validation.py

create-model:
	python save_model.py