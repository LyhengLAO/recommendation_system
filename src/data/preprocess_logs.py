"""Preprocess API logs into a normalized feature matrix for anomaly detection."""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "logs" / "api_logs.csv"
OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "logs_features.csv"
SCALER_FILE = Path(__file__).resolve().parents[2] / "models" / "scaler.pkl"

FEATURE_COLS = [
    "hour",
    "latency_ms",
    "status_code",
    "user_request_freq",
    "user_error_ratio",
]


# --- load --------------------------------------------------------------------


def load_logs(path: Path) -> pd.DataFrame:
    logger.info("Loading logs from %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


# --- feature engineering -----------------------------------------------------


def extract_hour(df: pd.DataFrame) -> pd.Series:
    """Hour of day (0–23) captures request time patterns."""
    return df["timestamp"].dt.hour.astype(np.float32)


def extract_user_request_freq(df: pd.DataFrame) -> pd.Series:
    """Total number of requests per user_id — high counts signal burst activity."""
    freq = df["user_id"].map(df["user_id"].value_counts())
    return freq.astype(np.float32)


def extract_user_error_ratio(df: pd.DataFrame) -> pd.Series:
    """Fraction of 5xx responses per user_id — high ratio flags problematic clients."""
    is_error = (df["status_code"] >= 500).astype(int)
    user_errors = is_error.groupby(df["user_id"]).transform("sum")
    user_total = df["user_id"].map(df["user_id"].value_counts())
    ratio = (user_errors / user_total).astype(np.float32)
    return ratio


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features …")
    features = pd.DataFrame(index=df.index)
    features["hour"] = extract_hour(df)
    features["latency_ms"] = df["latency_ms"].astype(np.float32)
    features["status_code"] = df["status_code"].astype(np.float32)
    features["user_request_freq"] = extract_user_request_freq(df)
    features["user_error_ratio"] = extract_user_error_ratio(df)

    # carry through metadata for traceability
    features["timestamp"] = df["timestamp"]
    features["user_id"] = df["user_id"]
    features["endpoint"] = df["endpoint"]
    features["is_anomaly"] = df["is_anomaly"]

    logger.info("Engineered %d features over %d rows", len(FEATURE_COLS), len(features))
    return features


# --- validation --------------------------------------------------------------


def validate(df: pd.DataFrame) -> None:
    nulls = df[FEATURE_COLS].isnull().sum()
    if nulls.any():
        logger.warning("Null values detected before scaling:\n%s", nulls[nulls > 0])
    infs = np.isinf(df[FEATURE_COLS].values).sum()
    if infs:
        logger.warning("%d infinite values detected — clipping", infs)


# --- normalisation -----------------------------------------------------------


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    logger.info("Fitting StandardScaler on features: %s", FEATURE_COLS)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[FEATURE_COLS])
    scaled_names = [f"{c}_scaled" for c in FEATURE_COLS]
    scaled_df = pd.DataFrame(scaled_values, columns=scaled_names, index=df.index)

    result = pd.concat([df, scaled_df], axis=1)

    for col, mean, std in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
        logger.info("  %-22s  mean=%8.3f  std=%8.3f", col, mean, std)

    return result, scaler


# --- I/O ---------------------------------------------------------------------


def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_kb = path.stat().st_size / 1024
    logger.info("Saved features → %s (%d rows, %.1f KB)", path, len(df), size_kb)


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler → %s", path)


# --- stats -------------------------------------------------------------------


def log_stats(df: pd.DataFrame) -> None:
    n_anomaly = df["is_anomaly"].sum()
    logger.info("--- Feature stats ---")
    logger.info("  Total rows   : %d", len(df))
    logger.info("  Anomalies    : %d (%.1f%%)", n_anomaly, n_anomaly / len(df) * 100)
    logger.info("  Latency p50  : %.1f ms", df["latency_ms"].median())
    logger.info("  Latency p95  : %.1f ms", df["latency_ms"].quantile(0.95))
    logger.info("  Latency p99  : %.1f ms", df["latency_ms"].quantile(0.99))
    logger.info("  Max freq/user: %d requests", int(df["user_request_freq"].max()))
    logger.info("  Max err ratio: %.3f", df["user_error_ratio"].max())


# --- entry point -------------------------------------------------------------


def main() -> None:
    if not RAW_FILE.exists():
        logger.error("Raw file not found: %s — run generate_logs.py first", RAW_FILE)
        sys.exit(1)

    df = load_logs(RAW_FILE)
    df = engineer_features(df)
    validate(df)
    log_stats(df)
    df, scaler = normalize(df)
    save_features(df, OUT_FILE)
    save_scaler(scaler, SCALER_FILE)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)
