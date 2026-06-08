import optuna
import mlflow

from src.utils.config import load_config
from src.data.load_data import load_raw_data, clean_raw_data
from src.data.validate_data import validate
from src.data.preprocess import preprocess
from src.core.problem_detector import detect_problem_type
from src.core.algorithm_registry import get_algorithm
from src.core.metrics import compute_metrics, DEFAULT_METRICS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(
    config: dict = None,
    dataset_path: str = None,
    target_column: str = None,
    algorithm: str = "auto",
    metrics: list = None,
):
    """
    Auto-tune hyperparameters using Optuna for any algorithm and dataset.

    Args:
        config:        Config dict (loaded from yaml if not provided)
        dataset_path:  Path to CSV (overrides config)
        target_column: Target column name (overrides config)
        algorithm:     Algorithm name or 'auto'
        metrics:       Metric to optimise (first in list is used). None = auto.
    """
    if config is None:
        config = load_config()

    raw_path = dataset_path or config["data"]["raw_path"]
    target   = target_column or config["data"]["target_column"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_raw_data(raw_path)
    df = clean_raw_data(df)
    df = validate(df, target_column=target)

    problem_type = detect_problem_type(df[target])
    algo_entry   = get_algorithm(algorithm, problem_type)

    X_train, X_test, y_train, y_test, _, _le = preprocess(
        df,
        target_column=target,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Pick optimisation metric — first user metric or sensible default
    opt_metrics = metrics or DEFAULT_METRICS[problem_type]
    opt_metric  = opt_metrics[0]

    # Lower is better for error metrics
    direction = "minimize" if opt_metric in ("rmse", "mae", "mape") else "maximize"

    def objective(trial):
        params = algo_entry["tune_space"](trial)
        model  = algo_entry["model_class"](**{**algo_entry["default_params"], **params})
        model.fit(X_train, y_train)
        result = compute_metrics(model, X_test, y_test, problem_type, [opt_metric])
        return result[opt_metric]

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=config["tuning"]["n_trials"])

    best_params = study.best_params
    best_value  = study.best_value

    print(f"\nBest {opt_metric}: {best_value:.4f}")
    print(f"Best params: {best_params}")

    with mlflow.start_run(run_name=f"optuna_{algorithm}"):
        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_{opt_metric}", best_value)

    return best_params, best_value


if __name__ == "__main__":
    tune()
