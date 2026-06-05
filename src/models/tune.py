import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.utils.config import load_config
from src.data.load_data import load_raw_data
from src.data.validate_data import validate
from src.data.preprocess import preprocess

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(config: dict = None):
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_raw_data(config["data"]["raw_path"])
    df = validate(df)

    X_train, X_test, y_train, y_test, _ = preprocess(
        df,
        target_column=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        model = RandomForestClassifier(**params, random_state=config["data"]["random_state"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, zero_division=0)

    study = optuna.create_study(direction=config["tuning"]["direction"])
    study.optimize(objective, n_trials=config["tuning"]["n_trials"])

    best_params = study.best_params
    best_value = study.best_value
    print(f"\nBest F1: {best_value:.4f}")
    print(f"Best params: {best_params}")

    with mlflow.start_run(run_name="optuna_best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1", best_value)

    return best_params, best_value


if __name__ == "__main__":
    tune()
