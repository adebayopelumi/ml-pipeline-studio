install:
	pip install -r requirements.txt

validate:
	python -m src.data.validate_data

train:
	python -m src.models.train

tune:
	python -m src.models.tune

predict:
	python -m src.models.predict $(file)

serve:
	uvicorn src.api.main:app --reload

app:
	streamlit run app/main.py

test:
	pytest

mlflow-ui:
	mlflow ui

docker-build:
	docker build -t production-ml-pipeline .

docker-run:
	docker run -p 8000:8000 production-ml-pipeline

format:
	black src/ tests/
	ruff check src/ tests/ --fix

lint:
	ruff check src/ tests/
