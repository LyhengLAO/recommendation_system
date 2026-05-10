"""Preprocess MovieLens 100k ratings into a sparse user-item matrix."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "movielens" / "u.data"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MATRIX_FILE = OUT_DIR / "movielens_matrix.npz"
USER_MAP_FILE = OUT_DIR / "user_mapping.json"
ITEM_MAP_FILE = OUT_DIR / "item_mapping.json"

MIN_RATINGS_PER_USER = 50

def load_ratings(path: Path) -> pd.DataFrame:
    logger.info("Loading ratings from %s", path)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    logger.info("Loaded %d raw interactions", len(df))
    return df

def filter_inactive_users(df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    counts = df["user_id"].value_counts()
    active = counts[counts >= min_ratings].index
    filtered = df[df["user_id"].isin(active)].copy()
    dropped = len(counts) - len(active)
    logger.info(
        "User filter (min %d ratings): kept %d users, dropped %d users, %d → %d rows",
        min_ratings,
        len(active),
        dropped,
        len(df),
        len(filtered),
    )
    return filtered

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map raw user/item IDs to contiguous 0-based indices."""
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["item_idx"] = df["item_id"].map(item_to_idx)

    logger.info("Encoded %d unique users, %d unique items", len(unique_users), len(unique_items))
    return df, user_to_idx, item_to_idx

def create_sparse_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    logger.info("Building sparse user-item matrix (%d × %d) …", n_users, n_items)
    matrix = csr_matrix(
        (df["rating"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    return matrix

def save_matrix(matrix: csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(path), matrix)
    size_kb = path.stat().st_size / 1024
    logger.info("Saved sparse matrix → %s (%.1f KB)", path, size_kb)


def save_mapping(mapping: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON keys must be strings
    path.write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))
    logger.info("Saved mapping (%d entries) → %s", len(mapping), path)

def log_stats(matrix: csr_matrix) -> None:
    n_users, n_items = matrix.shape
    n_interactions = matrix.nnz
    density = n_interactions / (n_users * n_items) * 100
    ratings = matrix.data
    logger.info("--- Matrix stats ---")
    logger.info("  Users        : %d", n_users)
    logger.info("  Items        : %d", n_items)
    logger.info("  Interactions : %d", n_interactions)
    logger.info("  Density      : %.4f%%", density)
    logger.info(
        "  Rating dist  : min=%.1f  mean=%.2f  max=%.1f",
        ratings.min(),
        ratings.mean(),
        ratings.max(),
    )

def main() -> None:
    if not RAW_FILE.exists():
        logger.error("Raw file not found: %s — run download.py first", RAW_FILE)
        sys.exit(1)

    df = load_ratings(RAW_FILE)
    df = filter_inactive_users(df, MIN_RATINGS_PER_USER)
    df, user_to_idx, item_to_idx = encode_ids(df)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    matrix = create_sparse_matrix(df, n_users, n_items)

    log_stats(matrix)

    save_matrix(matrix, MATRIX_FILE)
    save_mapping(user_to_idx, USER_MAP_FILE)
    save_mapping(item_to_idx, ITEM_MAP_FILE)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)