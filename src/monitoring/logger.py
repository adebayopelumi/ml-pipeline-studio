import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class PredictionLogger:
    """Log predictions for monitoring and drift detection."""

    def __init__(self, log_dir: str = "logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        input_features: Dict[str, Any],
        prediction: int,
        prediction_proba: float = None,
        model_version: str = "v1"
    ) -> None:
        """Log a single prediction with timestamp."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "input_features": input_features,
            "prediction": prediction,
            "prediction_proba": prediction_proba,
        }

        log_file = self.log_dir / f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_recent_predictions(self, days: int = 1) -> pd.DataFrame:
        """Load recent predictions for analysis."""
        predictions = []
        for log_file in sorted(self.log_dir.glob("predictions_*.jsonl"))[-days:]:
            with open(log_file, "r") as f:
                for line in f:
                    predictions.append(json.loads(line))

        if not predictions:
            return pd.DataFrame()

        df = pd.json_normalize(predictions)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


class MetricsTracker:
    """Track model performance metrics over time."""

    def __init__(self, metrics_dir: str = "logs/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        dataset: str = "test",
        target_column: str = "",
        problem_type: str = "",
    ) -> None:
        """Log evaluation metrics with timestamp and run context."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset,
            "target_column": target_column,
            "problem_type": problem_type,
            **metrics,
        }

        log_file = self.metrics_dir / f"metrics_{dataset}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_metrics_history(self, dataset: str = "test") -> pd.DataFrame:
        """Load metrics history for a dataset."""
        log_file = self.metrics_dir / f"metrics_{dataset}.jsonl"
        if not log_file.exists():
            return pd.DataFrame()

        metrics = []
        with open(log_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line))

        df = pd.DataFrame(metrics)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
