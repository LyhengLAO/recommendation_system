"""Real-time anomaly detection on individual API log entries."""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_FILE  = BASE / "models" / "isolation_forest.pkl"
SCALER_FILE = BASE / "models" / "scaler.pkl"

# Must match the order used during training (preprocess_logs.py)
FEATURE_COLS = ["hour", "latency_ms", "status_code", "user_request_freq", "user_error_ratio"]

# Sensible defaults for aggregate features that may be absent on a single log entry
FEATURE_DEFAULTS: dict[str, float] = {
    "hour": 12.0,
    "latency_ms": 200.0,
    "status_code": 200.0,
    "user_request_freq": 10.0,
    "user_error_ratio": 0.03,
}

# Human-readable labels for each feature used in the reason string
FEATURE_LABELS: dict[str, str] = {
    "hour": "unusual hour",
    "latency_ms": "high latency",
    "status_code": "error status code",
    "user_request_freq": "abnormal request frequency",
    "user_error_ratio": "high error ratio",
}


# --- exceptions --------------------------------------------------------------


class ModelNotLoadedError(RuntimeError):
    """Raised when detection is attempted before artefacts are on disk."""


class MissingFeaturesError(ValueError):
    """Raised when required fields cannot be resolved from the log entry."""


# --- result ------------------------------------------------------------------


@dataclass
class DetectionResult:
    is_anomaly: bool
    score: float       # anomaly score: more negative = more anomalous
    reason: str

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "score": round(self.score, 6),
            "reason": self.reason,
        }


# --- model store (lazy singleton) --------------------------------------------


class _DetectorStore:
    def __init__(self) -> None:
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> None:
        for path in (MODEL_FILE, SCALER_FILE):
            if not path.exists():
                raise ModelNotLoadedError(
                    f"Artefact not found: {path} — run the relevant train script first"
                )

        logger.info("Loading IsolationForest from %s", MODEL_FILE)
        with open(MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)

        logger.info("Loading StandardScaler from %s", SCALER_FILE)
        with open(SCALER_FILE, "rb") as f:
            self.scaler = pickle.load(f)

        logger.info("Detector ready")


_store = _DetectorStore()


def _ensure_loaded() -> None:
    if not _store.is_loaded:
        _store.load()


# --- feature extraction ------------------------------------------------------


def _extract_hour(entry: dict[str, Any]) -> float:
    ts = entry.get("timestamp")
    if ts is None:
        return FEATURE_DEFAULTS["hour"]
    if isinstance(ts, (int, float)):
        return float(datetime.fromtimestamp(ts).hour)
    if isinstance(ts, datetime):
        return float(ts.hour)
    try:
        return float(datetime.fromisoformat(str(ts)).hour)
    except ValueError:
        logger.warning("Cannot parse timestamp %r — using default hour", ts)
        return FEATURE_DEFAULTS["hour"]


def _extract_features(entry: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """
    Build a (1, n_features) array from a raw log entry dict.

    Returns the feature vector and a list of field names that fell back to
    their defaults (used to build the reason string).
    """
    values: list[float] = []
    defaults_used: list[str] = []

    for col in FEATURE_COLS:
        if col == "hour":
            val = _extract_hour(entry)
            if "timestamp" not in entry:
                defaults_used.append(col)
        elif col in entry and entry[col] is not None:
            try:
                val = float(entry[col])
            except (TypeError, ValueError) as exc:
                raise MissingFeaturesError(
                    f"Cannot convert field '{col}' to float: {entry[col]!r}"
                ) from exc
        else:
            val = FEATURE_DEFAULTS[col]
            defaults_used.append(col)

        values.append(val)

    return np.array(values, dtype=np.float32).reshape(1, -1), defaults_used


# --- reason generation -------------------------------------------------------


def _build_reason(
    raw_values: np.ndarray,
    scaled_values: np.ndarray,
    is_anomaly: bool,
    defaults_used: list[str],
) -> str:
    if not is_anomaly:
        return "all features within normal range"

    # rank features by absolute deviation from the training mean (in std units)
    abs_z = np.abs(scaled_values[0])
    ranking = np.argsort(abs_z)[::-1]

    parts: list[str] = []
    for idx in ranking[:2]:  # top-2 contributing features
        col = FEATURE_COLS[idx]
        z = scaled_values[0, idx]
        raw = raw_values[0, idx]
        direction = "high" if z > 0 else "low"
        label = FEATURE_LABELS[col]
        if col == "latency_ms":
            parts.append(f"{label} ({raw:.0f} ms)")
        elif col == "user_error_ratio":
            parts.append(f"{label} ({raw:.2%})")
        elif col == "status_code":
            parts.append(f"{label} ({int(raw)})")
        elif col == "hour":
            parts.append(f"{label} ({int(raw)}h, {direction})")
        else:
            parts.append(f"{label} ({raw:.0f}, {direction})")

    reason = "; ".join(parts) if parts else "anomalous combination of features"

    if defaults_used:
        reason += f" [defaults used for: {', '.join(defaults_used)}]"

    return reason


# --- public API --------------------------------------------------------------


def load_model() -> None:
    """Explicitly pre-load artefacts (call at API startup)."""
    _store.load()


def detect(log_entry: dict[str, Any]) -> dict:
    """
    Score a single API log entry for anomalies.

    Parameters
    ----------
    log_entry : dict
        Raw log fields. Required: ``latency_ms``, ``status_code``.
        Optional but recommended: ``timestamp``, ``user_request_freq``,
        ``user_error_ratio``. Missing fields fall back to training-set means.

    Returns
    -------
    dict with keys:
        - ``is_anomaly`` (bool)
        - ``score`` (float, more negative = more anomalous)
        - ``reason`` (str, human-readable explanation)

    Raises
    ------
    ModelNotLoadedError
        If model artefacts are missing from disk.
    MissingFeaturesError
        If a required field cannot be parsed.
    """
    _ensure_loaded()

    raw_vec, defaults_used = _extract_features(log_entry)

    if defaults_used:
        logger.debug("Fields resolved to defaults: %s", defaults_used)

    scaled_vec = _store.scaler.transform(raw_vec)

    # score_samples returns negative values; more negative = more anomalous
    score = float(_store.model.score_samples(scaled_vec)[0])
    # predict returns -1 (anomaly) or +1 (normal)
    is_anomaly = bool(_store.model.predict(scaled_vec)[0] == -1)

    reason = _build_reason(raw_vec, scaled_vec, is_anomaly, defaults_used)

    logger.debug(
        "detect → is_anomaly=%s score=%.4f reason=%r",
        is_anomaly, score, reason,
    )

    return DetectionResult(is_anomaly=is_anomaly, score=score, reason=reason).to_dict()


def detect_batch(entries: list[dict[str, Any]]) -> list[dict]:
    """
    Score a list of log entries. Entries that fail feature extraction are
    returned with is_anomaly=False, score=0.0 and the error as the reason.
    """
    _ensure_loaded()
    results: list[dict] = []
    for entry in entries:
        try:
            results.append(detect(entry))
        except MissingFeaturesError as exc:
            logger.warning("Skipping malformed entry: %s", exc)
            results.append({"is_anomaly": False, "score": 0.0, "reason": f"error: {exc}"})
    return results
