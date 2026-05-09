.PHONY: install lint test run train mlflow

install:
	pip install -e ".[dev]"

train:
	python -m src.training.train

lint:
	python -m ruff check src/ tests/

test:
	python -m pytest tests/ -v

run:
	python -m uvicorn src.api.api:app --reload

mlflow:
	python -m mlflow ui

