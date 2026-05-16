"""Train an Isolation Forest anomaly detector on preprocessed API logs."""

import logging
import pickle
import sys
from pathlib import Path

import json
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import optuna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
FEATURES_FILE = BASE / "data" / "processed" / "logs_features.csv"
MODEL_FILE    = BASE / "models" / "best_isolation_forest.pkl"
BEST_PARAMS_FILE = BASE / "models" / "best_isolation_forest_params.json"

FEATURE_COLS = [
    "hour_scaled",
    "latency_ms_scaled",
    "status_code_scaled",
    "user_request_freq_scaled",
    "user_error_ratio_scaled",
]

IF_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.05,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}


# --- data --------------------------------------------------------------------


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading features from %s", path)
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing} — run preprocess_logs.py first")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["is_anomaly"].astype(int).values   # ground truth (1 = anomaly)

    logger.info(
        "Loaded %d rows | %d features | %d anomalies (%.1f%%)",
        len(df), X.shape[1], y.sum(), y.mean() * 100,
    )
    return X, y


# --- evaluation --------------------------------------------------------------


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.where(y_pred == -1, 0, 1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    metrics = {"precision": precision, "recall": recall, "f1_score": f1}

    logger.info("--- Evaluation metrics ---")
    logger.info("  Precision : %.4f", precision)
    logger.info("  Recall    : %.4f", recall)
    logger.info("  F1-score  : %.4f", f1)

    report = classification_report(
        y_true, y_pred,
        target_names=["normal", "anomaly"],
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report)
    return metrics

def objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 50, 300, step=50),
        "contamination": trial.suggest_float("contamination", 0.01, 0.2),
        "max_features":  trial.suggest_float("max_features", 0.5, 1.0),
        "bootstrap":     trial.suggest_categorical("bootstrap", [True, False]),
        "random_state":  42,
        "n_jobs":        -1,
    }
    model = IsolationForest(**params)
    model.fit(X)
    y_pred = model.predict(X)
    metrics = evaluate(y, y_pred)
    trial.set_user_attr("precision", metrics["precision"])
    trial.set_user_attr("recall", metrics["recall"])
    return metrics["f1_score"]

# --- model -------------------------------------------------------------------


def train(X: np.ndarray, y: np.ndarray) -> IsolationForest:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info(
        "Training IsolationForest — n_estimators=%d, contamination=%.2f",
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X)
    logger.info("Training complete")

    return best_model, best_params, study


def predict_labels(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Convert sklearn convention (-1/+1) to binary labels (1=anomaly, 0=normal)."""
    raw = model.predict(X)          # -1 = anomaly, +1 = normal
    return (raw == -1).astype(int)

def log_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    logger.info("Confusion matrix — TP:%d FP:%d FN:%d TN:%d", tp, fp, fn, tn)
    mlflow.log_metrics({"tp": tp, "fp": fp, "fn": fn, "tn": tn})


# --- I/O ---------------------------------------------------------------------


def save_model(model: IsolationForest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = path.stat().st_size / 1024
    logger.info("Saved model → %s (%.1f KB)", path, size_kb)

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved JSON → %s", path)

# --- entry point -------------------------------------------------------------


def main() -> None:
    if not FEATURES_FILE.exists():
        logger.error("Features not found: %s — run preprocess_logs.py first", FEATURES_FILE)
        sys.exit(1)

    X, y_true = load_features(FEATURES_FILE)

    mlflow.set_tracking_uri("sqlite:///" + str(BASE / "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("anomaly-isolation-forest")
    with mlflow.start_run(run_name="iforest-api-logs"):

        model, best_params, study = train(X, y_true)

        for trial in study.trials:
            mlflow.log_metrics(
                {
                    "trial_f1": trial.value,
                    "trial_precision": trial.user_attrs.get("precision", 0.0),
                    "trial_recall": trial.user_attrs.get("recall", 0.0),
                },
                step=trial.number,
            )

        mlflow.log_params(best_params)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("feature_cols", FEATURE_COLS)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("anomaly_rate_true", float(y_true.mean()))

        y_pred = predict_labels(model, X)

        metrics = evaluate(y_true, y_pred)
        mlflow.log_metrics(metrics)
        log_confusion(y_true, y_pred)

        save_model(model, MODEL_FILE)
        save_json(best_params, BEST_PARAMS_FILE)
        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(BEST_PARAMS_FILE))

        logger.info(
            "MLflow run %s — F1=%.4f precision=%.4f recall=%.4f",
            mlflow.active_run().info.run_id,
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
