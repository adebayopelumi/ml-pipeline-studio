import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


class DataDriftDetector:
    """Detect data drift in production predictions."""

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize with reference (training) data.

        Args:
            reference_data: DataFrame with reference distribution
        """
        self.reference_data = reference_data
        self.numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = reference_data.select_dtypes(include=['object']).columns

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Detect drift using statistical tests.

        Args:
            current_data: Recent production data
            threshold: P-value threshold for drift detection

        Returns:
            Dictionary with drift results per feature
        """
        drift_results = {}

        # Kolmogorov-Smirnov test for numeric features
        for col in self.numeric_cols:
            if col not in current_data.columns:
                continue

            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()

            statistic, p_value = stats.ks_2samp(ref_values, curr_values)

            drift_results[col] = {
                "test": "ks_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < threshold,
                "ref_mean": float(ref_values.mean()),
                "curr_mean": float(curr_values.mean()),
                "ref_std": float(ref_values.std()),
                "curr_std": float(curr_values.std()),
            }

        # Chi-squared test for categorical features
        for col in self.categorical_cols:
            if col not in current_data.columns:
                continue

            ref_counts = self.reference_data[col].value_counts()
            curr_counts = current_data[col].value_counts()

            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

            # Normalize to proportions
            ref_prop = np.array(ref_freq) / sum(ref_freq)
            curr_prop = np.array(curr_freq) / sum(curr_freq)

            # Chi-squared test
            statistic, p_value = stats.chisquare(curr_freq, f_exp=ref_prop * sum(curr_freq))

            drift_results[col] = {
                "test": "chi_squared",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < threshold,
                "ref_distribution": dict(zip(all_categories, ref_prop)),
                "curr_distribution": dict(zip(all_categories, curr_prop)),
            }

        return drift_results

    def get_drift_summary(self, drift_results: Dict) -> Dict:
        """Summarize drift detection results."""
        total_features = len(drift_results)
        drifted_features = sum(1 for r in drift_results.values() if r["drift_detected"])

        return {
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": drifted_features / total_features if total_features > 0 else 0,
            "drifted_feature_names": [
                col for col, r in drift_results.items() if r["drift_detected"]
            ],
        }


class ModelPerformanceMonitor:
    """Monitor model performance degradation."""

    def __init__(self, baseline_metrics: Dict[str, float], threshold: float = 0.05):
        """
        Initialize with baseline metrics.

        Args:
            baseline_metrics: Initial model performance metrics
            threshold: Acceptable degradation threshold (e.g., 0.05 = 5%)
        """
        self.baseline_metrics = baseline_metrics
        self.threshold = threshold

    def check_degradation(self, current_metrics: Dict[str, float]) -> Dict:
        """
        Check if model performance has degraded.

        Args:
            current_metrics: Recent model performance metrics

        Returns:
            Dictionary with degradation status per metric
        """
        degradation_results = {}

        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]
            degradation = baseline_value - current_value
            degradation_pct = degradation / baseline_value if baseline_value > 0 else 0

            degradation_results[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "degradation": degradation,
                "degradation_pct": degradation_pct,
                "alert": degradation_pct > self.threshold,
            }

        return degradation_results

    def get_alert_summary(self, degradation_results: Dict) -> Dict:
        """Summarize performance degradation alerts."""
        alerts = [
            metric for metric, r in degradation_results.items() if r["alert"]
        ]

        return {
            "alerts_triggered": len(alerts),
            "alert_metrics": alerts,
            "requires_retraining": len(alerts) > 0,
        }
