import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import load_config


def get_latest_model_version(model_name: str, client: MlflowClient) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model: {model_name}")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    return latest.version


def promote_to_production(model_name: str = "churn_model", config: dict = None) -> dict:
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    version = get_latest_model_version(model_name, client)
    client.set_registered_model_alias(model_name, "production", version)

    print(f"Model '{model_name}' version {version} promoted to production.")
    return {"model_name": model_name, "version": version, "stage": "production"}


def get_model_info(model_name: str = "churn_model", config: dict = None) -> dict:
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    version = get_latest_model_version(model_name, client)
    model_version = client.get_model_version(model_name, version)

    return {
        "model_name": model_name,
        "version": version,
        "status": model_version.status,
        "run_id": model_version.run_id,
    }


if __name__ == "__main__":
    info = get_model_info()
    print(info)
