import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import load_config
from src.data.load_data import load_raw_data
from src.data.validate_data import validate
from src.data.preprocess import preprocess, save_preprocessor
from src.models.evaluate import compute_metrics


def get_model(config: dict):
    params = config["model"]["parameters"]
    model_type = config["model"]["type"]
    if model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=config["data"]["random_state"])
    raise ValueError(f"Unsupported model type: {model_type}")


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)


def train(config: dict = None, params_override: dict = None):
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_raw_data(config["data"]["raw_path"])
    df = validate(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess(
        df,
        target_column=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    model_params = {**config["model"]["parameters"]}
    if params_override:
        model_params.update(params_override)

    model = get_model({**config, "model": {**config["model"], "parameters": model_params}})

    with mlflow.start_run():
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("test_size", config["data"]["test_size"])

        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="churn_model")

        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    save_model(model, config["model"]["save_path"])
    save_preprocessor(preprocessor)

    return model, preprocessor, metrics


if __name__ == "__main__":
    train()
