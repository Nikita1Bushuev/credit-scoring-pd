.PHONY: install data features train api docker-build docker-run test lint format clean

install:
	pip install -r requirements.txt
	pre-commit install

data:
	python -m src.data.make_dataset data/raw/UCI_Credit_Card.csv data/interim

features:
	python -m src.features.build_features data/interim data/processed

train:
	python -m src.models.train --model-type gbc --threshold 0.3

api:
	uvicorn src.api.app:app --reload

docker-build:
	docker build -t credit-pd-api .

docker-run:
	docker run -p 8000:8000 credit-pd-api

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	dvc destroy -f
	find . -type d -name "__pycache__" -exec rm -rf {} +
