"""Generate 10 000 synthetic API log lines with ~5% injected anomalies."""

import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

N_ROWS = 10_000
ANOMALY_RATE = 0.05
ANOMALY_N = int(N_ROWS * ANOMALY_RATE)   # ~500 rows

ENDPOINTS = ["/recommend", "/anomaly/detect", "/health"]
ENDPOINT_WEIGHTS = [0.55, 0.35, 0.10]

NORMAL_STATUS_CODES = [200, 400, 500]
NORMAL_STATUS_WEIGHTS = [0.92, 0.05, 0.03]

LATENCY_MEAN = 200.0   # ms
LATENCY_STD = 40.0
LATENCY_MIN = 10.0

ANOMALY_LATENCY_MIN = 3_000.0
ANOMALY_LATENCY_MAX = 8_000.0

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "logs"
OUT_FILE = OUT_DIR / "api_logs.csv"

RNG_SEED = 42

def _random_ip(rng: np.random.Generator) -> str:
    octets = rng.integers(1, 255, size=4)
    return ".".join(str(o) for o in octets)


def _timestamps(rng: np.random.Generator, n: int, now: datetime) -> list[datetime]:
    """Uniform-random timestamps over the last 30 days."""
    window = 30 * 24 * 3600  # seconds
    offsets = rng.integers(0, window, size=n)
    return [now - timedelta(seconds=int(s)) for s in offsets]


def build_normal_logs(rng: np.random.Generator, n: int, now: datetime) -> pd.DataFrame:
    latencies = rng.normal(LATENCY_MEAN, LATENCY_STD, size=n).clip(min=LATENCY_MIN)
    return pd.DataFrame(
        {
            "timestamp": _timestamps(rng, n, now),
            "user_id": rng.integers(1, 1001, size=n),
            "endpoint": rng.choice(ENDPOINTS, size=n, p=ENDPOINT_WEIGHTS),
            "latency_ms": np.round(latencies, 1),
            "status_code": rng.choice(NORMAL_STATUS_CODES, size=n, p=NORMAL_STATUS_WEIGHTS),
            "ip_address": [_random_ip(rng) for _ in range(n)],
            "is_anomaly": False,
        }
    )

def inject_high_latency(rng: np.random.Generator, n: int, now: datetime) -> pd.DataFrame:
    """Spike latency anomalies: latency_ms > 3 000, still mostly 200."""
    logger.info("Injecting %d high-latency anomalies …", n)
    latencies = rng.uniform(ANOMALY_LATENCY_MIN, ANOMALY_LATENCY_MAX, size=n)
    return pd.DataFrame(
        {
            "timestamp": _timestamps(rng, n, now),
            "user_id": rng.integers(1, 1001, size=n),
            "endpoint": rng.choice(ENDPOINTS, size=n, p=ENDPOINT_WEIGHTS),
            "latency_ms": np.round(latencies, 1),
            "status_code": rng.choice([200, 500], size=n, p=[0.6, 0.4]),
            "ip_address": [_random_ip(rng) for _ in range(n)],
            "is_anomaly": True,
        }
    )

def inject_500_bursts(rng: np.random.Generator, n: int, now: datetime) -> pd.DataFrame:
    """Burst anomalies: clusters of HTTP 500 from the same IP in quick succession."""
    logger.info("Injecting %d 500-burst anomalies …", n)
    burst_size = 10
    n_bursts = max(1, n // burst_size)
    rows: list[dict] = []

    for _ in range(n_bursts):
        burst_ip = _random_ip(rng)
        burst_start = now - timedelta(seconds=int(rng.integers(0, 30 * 24 * 3600)))
        for j in range(burst_size):
            rows.append(
                {
                    "timestamp": burst_start + timedelta(seconds=j * int(rng.integers(1, 5))),
                    "user_id": int(rng.integers(1, 1001)),
                    "endpoint": rng.choice(ENDPOINTS, p=ENDPOINT_WEIGHTS),
                    "latency_ms": round(float(np.clip(rng.normal(LATENCY_MEAN, LATENCY_STD), LATENCY_MIN, None)), 1),
                    "status_code": 500,
                    "ip_address": burst_ip,
                    "is_anomaly": True,
                }
            )

    df = pd.DataFrame(rows)
    # trim or pad to exactly n rows
    if len(df) > n:
        df = df.iloc[:n]
    elif len(df) < n:
        extra = inject_high_latency(rng, n - len(df), now)
        df = pd.concat([df, extra], ignore_index=True)
    return df


def generate(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    now = datetime.now().replace(microsecond=0)

    n_normal = N_ROWS - ANOMALY_N
    # split anomalies evenly between the two anomaly types
    n_latency = ANOMALY_N // 2
    n_burst = ANOMALY_N - n_latency

    df_normal = build_normal_logs(rng, n_normal, now)
    df_latency = inject_high_latency(rng, n_latency, now)
    df_burst = inject_500_bursts(rng, n_burst, now)

    df = pd.concat([df_normal, df_latency, df_burst], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_kb = path.stat().st_size / 1024
    logger.info("Saved %d rows → %s (%.1f KB)", len(df), path, size_kb)


def report(df: pd.DataFrame) -> None:
    n_anomaly = df["is_anomaly"].sum()
    logger.info(
        "Dataset summary: %d rows | %d normal | %d anomalies (%.1f%%)",
        len(df),
        len(df) - n_anomaly,
        n_anomaly,
        n_anomaly / len(df) * 100,
    )
    logger.info(
        "Latency — mean: %.1f ms | p95: %.1f ms | max: %.1f ms",
        df["latency_ms"].mean(),
        df["latency_ms"].quantile(0.95),
        df["latency_ms"].max(),
    )
    status_counts = df["status_code"].value_counts().to_dict()
    logger.info("Status codes: %s", status_counts)
    

def main() -> None:
    logger.info("Starting synthetic log generation (n=%d, anomaly_rate=%.0f%%)", N_ROWS, ANOMALY_RATE * 100)
    df = generate()
    report(df)
    save(df, OUT_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        sys.exit(1)