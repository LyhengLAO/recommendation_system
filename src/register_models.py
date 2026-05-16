"""Register pre-trained models into the MLflow Model Registry."""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1]
MLRUNS_URI = "sqlite:///" + str(BASE / "mlflow.db").replace("\\", "/")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def register_isolation_forest() -> None:
    model_path = BASE / "models" / "best_isolation_forest.pkl"
    params_path = BASE / "models" / "best_isolation_forest_params.json"

    model = load_pickle(model_path)
    params = load_json(params_path) if params_path.exists() else {}

    mlflow.set_tracking_uri(MLRUNS_URI)
    mlflow.set_experiment("anomaly-isolation-forest")

    with mlflow.start_run(run_name="register-pretrained-iforest"):
        mlflow.log_params(params)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="anomaly-isolation-forest",
        )
        logger.info("IsolationForest registered in MLflow Model Registry")


def register_als() -> None:
    model_path = BASE / "models" / "best_model.pkl"
    params_path = BASE / "models" / "best_params.json"
    mappings_path = BASE / "models" / "mappings.json"

    model = load_pickle(model_path)
    params = load_json(params_path) if params_path.exists() else {}

    mlflow.set_tracking_uri(MLRUNS_URI)
    mlflow.set_experiment("recommender-als")

    with mlflow.start_run(run_name="register-pretrained-als"):
        mlflow.log_params(params)
        # ALS (implicit) n'est pas sklearn → on logue le pkl comme artifact
        mlflow.log_artifact(str(model_path), artifact_path="model")
        if mappings_path.exists():
            mlflow.log_artifact(str(mappings_path), artifact_path="model")
        mlflow.set_tag("model_type", "implicit-ALS")
        logger.info("ALS model artifact logged under experiment 'recommender-als'")


if __name__ == "__main__":
    register_isolation_forest()
    register_als()
    logger.info("Done. Open MLflow UI → Models tab to see registered models.")
