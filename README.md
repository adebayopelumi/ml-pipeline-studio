# Production ML Pipeline: Telco Churn Prediction

A complete end-to-end machine learning pipeline for predicting customer churn in the telecommunications industry. This project demonstrates production-ready ML practices including data validation, experiment tracking, hyperparameter tuning, model registry, API serving, and monitoring.

## Features

- **Data Validation**: Schema validation using Pandera
- **Experiment Tracking**: MLflow integration for tracking experiments and models
- **Hyperparameter Tuning**: Optuna-based automated hyperparameter optimization
- **Model Registry**: MLflow model registry with versioning and staging
- **REST API**: FastAPI-based prediction service with automatic documentation
- **Monitoring**: Prediction logging, drift detection, and performance tracking
- **CI/CD**: GitHub Actions workflow for automated testing and linting
- **Docker Support**: Containerized deployment

## Architecture

```
├── configs/          # Configuration files
├── data/            
│   ├── raw/         # Raw dataset
│   └── processed/   # Processed data
├── models/          # Saved models and preprocessors
├── logs/            # Prediction and metrics logs
├── src/
│   ├── data/        # Data loading, validation, preprocessing
│   ├── models/      # Training, evaluation, tuning, registry
│   ├── monitoring/  # Logging and drift detection
│   ├── api/         # FastAPI application
│   └── utils/       # Configuration utilities
└── tests/           # Unit tests
```

## Installation

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd production-ml-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data

```bash
python -m src.data.generate_data
```

This creates a synthetic telco churn dataset with 5,000 samples in `data/raw/churn.csv`.

### 2. Validate Data

```bash
make validate
# or
python -m src.data.validate_data
```

Validates the dataset schema and data quality constraints.

### 3. Train Model

```bash
make train
# or
python -m src.models.train
```

Trains a Random Forest classifier and logs the experiment to MLflow. The model and preprocessor are saved to the `models/` directory.

### 4. Hyperparameter Tuning

```bash
make tune
# or
python -m src.models.tune
```

Runs Optuna hyperparameter optimization (default: 30 trials) and logs the best parameters to MLflow.

### 5. View Experiments

```bash
make mlflow-ui
# or
mlflow ui
```

Opens the MLflow UI at http://localhost:5000 to view experiment metrics, parameters, and model artifacts.

### 6. Start API Server

```bash
make serve
# or
uvicorn src.api.main:app --reload
```

Starts the FastAPI server at http://localhost:8000. 

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 7. Make Predictions

Example API request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "monthly_charges": 99.0,
    "total_charges": 200.0,
    "contract": "Month-to-month"
  }'
```

Response:

```json
{
  "prediction": 1,
  "prediction_label": "churn",
  "probability": 0.73,
  "model_version": "v1"
}
```

### 8. Run Tests

```bash
make test
# or
pytest
```

Runs the full test suite with coverage.

## API Endpoints

### Health Check
- **GET** `/health` - Check API and model status

### Model Information
- **GET** `/model-info` - Get model type and parameters

### Predictions
- **POST** `/predict` - Make churn prediction

**Request Body:**
```json
{
  "tenure": 12,
  "monthly_charges": 75.5,
  "total_charges": 900.0,
  "contract": "One year"
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "no churn",
  "probability": 0.15,
  "model_version": "v1"
}
```

## Configuration

Edit `configs/config.yaml` to customize:

- Data paths and preprocessing settings
- Model type and hyperparameters
- Training metrics and test split
- MLflow tracking configuration
- Tuning parameters

## Monitoring

### Prediction Logging

All predictions are automatically logged to `logs/predictions/` with:
- Timestamp
- Input features
- Prediction and probability
- Model version

### Data Drift Detection

Check for distribution drift in production data:

```python
from src.monitoring.drift_detection import DataDriftDetector
from src.data.load_data import load_raw_data
from src.monitoring.logger import PredictionLogger

# Load reference data
reference_data = load_raw_data("data/raw/churn.csv")

# Load recent predictions
logger = PredictionLogger()
recent_preds = logger.get_recent_predictions(days=7)

# Detect drift
detector = DataDriftDetector(reference_data)
drift_results = detector.detect_drift(recent_preds)
summary = detector.get_drift_summary(drift_results)

print(f"Drifted features: {summary['drifted_feature_names']}")
```

### Performance Monitoring

Track model performance degradation:

```python
from src.monitoring.drift_detection import ModelPerformanceMonitor

baseline_metrics = {"f1": 0.85, "accuracy": 0.88, "roc_auc": 0.92}
monitor = ModelPerformanceMonitor(baseline_metrics, threshold=0.05)

current_metrics = {"f1": 0.78, "accuracy": 0.83, "roc_auc": 0.89}
degradation = monitor.check_degradation(current_metrics)
alert_summary = monitor.get_alert_summary(degradation)

if alert_summary['requires_retraining']:
    print(f"⚠️ Model retraining required!")
    print(f"Degraded metrics: {alert_summary['alert_metrics']}")
```

## Docker Deployment

### Build Image

```bash
make docker-build
# or
docker build -t production-ml-pipeline .
```

### Run Container

```bash
make docker-run
# or
docker run -p 8000:8000 production-ml-pipeline
```

## Development

### Code Formatting

```bash
make format
```

Formats code using Black and fixes linting issues with Ruff.

### Linting

```bash
make lint
```

Checks code quality with Ruff.

## Model Registry

### Promote Model to Production

```python
from src.models.registry import promote_to_production

promote_to_production(model_name="churn_model")
```

### Get Model Information

```python
from src.models.registry import get_model_info

info = get_model_info(model_name="churn_model")
print(f"Latest version: {info['version']}")
print(f"Status: {info['status']}")
```

## Continuous Integration

GitHub Actions workflow automatically:
- Runs linting checks
- Executes test suite
- Reports test coverage

Triggered on push to `main` and pull requests.

## Project Structure Details

### Data Module (`src/data/`)
- `generate_data.py` - Synthetic data generation
- `load_data.py` - Data loading utilities
- `validate_data.py` - Schema validation with Pandera
- `preprocess.py` - Feature engineering and preprocessing

### Models Module (`src/models/`)
- `train.py` - Model training with MLflow tracking
- `evaluate.py` - Metrics calculation
- `tune.py` - Hyperparameter tuning with Optuna
- `registry.py` - Model versioning and promotion

### Monitoring Module (`src/monitoring/`)
- `logger.py` - Prediction and metrics logging
- `drift_detection.py` - Data drift and performance monitoring

### API Module (`src/api/`)
- `main.py` - FastAPI application with prediction endpoint

## Dataset Features

The synthetic telco churn dataset includes:

- **tenure**: Number of months as a customer (0-72)
- **monthly_charges**: Monthly bill amount ($20-$120)
- **total_charges**: Total amount charged over customer lifetime
- **contract**: Contract type (Month-to-month, One year, Two year)
- **churn**: Target variable (0 = no churn, 1 = churn)

Realistic correlations:
- Higher monthly charges → higher churn risk
- Longer contracts → lower churn risk
- Longer tenure → lower churn risk

## Performance Metrics

The pipeline tracks:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

## Future Enhancements

- [ ] Add model explainability (SHAP values)
- [ ] Implement A/B testing framework
- [ ] Add feature importance tracking
- [ ] Implement automated retraining pipeline
- [ ] Add real-time monitoring dashboard
- [ ] Implement canary deployments
- [ ] Add batch prediction endpoint
- [ ] Integrate with cloud providers (AWS, GCP, Azure)

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.
