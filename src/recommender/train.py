"""Train an ALS recommender on the MovieLens user-item matrix and log to MLflow."""

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MATRIX_FILE   = BASE / "data" / "processed" / "movielens_matrix.npz"
USER_MAP_FILE = BASE / "data" / "processed" / "user_mapping.json"
ITEM_MAP_FILE = BASE / "data" / "processed" / "item_mapping.json"
MODEL_FILE    = BASE / "models" / "als_model.pkl"
MAPPINGS_FILE = BASE / "models" / "mappings.json"

# ALS hyper-parameters
ALS_PARAMS = {
    "factors": 50,
    "iterations": 20,
    "regularization": 0.01,
    "alpha": 40.0,       # confidence scaling: c_ui = 1 + alpha * r_ui
    "random_state": 42,
}

EVAL_K = 10
TEST_FRACTION = 0.2


def load_matrix(path: Path) -> csr_matrix:
    logger.info("Loading user-item matrix from %s", path)
    matrix = load_npz(str(path))
    logger.info("Matrix shape: %s | nnz: %d", matrix.shape, matrix.nnz)
    return matrix.astype(np.float32)


def load_mappings(user_path: Path, item_path: Path) -> tuple[dict, dict]:
    user_map = json.loads(user_path.read_text())
    item_map = json.loads(item_path.read_text())
    logger.info("Loaded %d user IDs, %d item IDs", len(user_map), len(item_map))
    return user_map, item_map


def train_test_split(
    matrix: csr_matrix, test_fraction: float, seed: int = 42
) -> tuple[csr_matrix, csr_matrix]:
    """
    Hold out `test_fraction` of non-zero entries per user as a test mask.
    Returns (train_matrix, test_matrix) — both sparse, same shape.
    """
    rng = np.random.default_rng(seed)
    matrix = matrix.tocsr()
    train = matrix.copy().tolil()
    test  = matrix.copy().tolil()

    for user in range(matrix.shape[0]):
        row = matrix.getrow(user).indices
        if len(row) < 2:
            test[user, :] = 0
            continue
        n_test = max(1, int(len(row) * test_fraction))
        test_items = rng.choice(row, size=n_test, replace=False)
        train_items = np.setdiff1d(row, test_items)
        # zero out the complementary side
        for col in test_items:
            train[user, col] = 0
        for col in train_items:
            test[user, col] = 0

    train_csr = train.tocsr()
    test_csr  = test.tocsr()
    train_csr.eliminate_zeros()
    test_csr.eliminate_zeros()
    logger.info(
        "Split → train nnz: %d | test nnz: %d",
        train_csr.nnz, test_csr.nnz,
    )
    return train_csr, test_csr


def build_confidence_matrix(matrix: csr_matrix, alpha: float) -> csr_matrix:
    """Convert explicit ratings to implicit confidence: c_ui = 1 + alpha * r_ui."""
    conf = matrix.copy().astype(np.float32)
    conf.data = 1.0 + alpha * conf.data
    return conf


def train(train_matrix: csr_matrix, params: dict) -> AlternatingLeastSquares:
    logger.info("Training ALS — factors=%d, iterations=%d", params["factors"], params["iterations"])
    conf = build_confidence_matrix(train_matrix, params["alpha"])
    # implicit expects item-user format
    item_user = conf.T.tocsr()

    model = AlternatingLeastSquares(
        factors=params["factors"],
        iterations=params["iterations"],
        regularization=params["regularization"],
        random_state=params["random_state"],
        use_gpu=False,
    )
    model.fit(item_user, show_progress=True)
    logger.info("Training complete")
    return model


def precision_recall_at_k(
    model: AlternatingLeastSquares,
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    k: int,
    max_users: int = 500,
) -> tuple[float, float]:
    """
    Compute mean precision@k and recall@k over a sample of users.
    Filters out users with no test items.
    """
    rng = np.random.default_rng(0)
    n_users = train_matrix.shape[0]
    candidates = np.where(np.diff(test_matrix.indptr) > 0)[0]  # users with test items
    sample = rng.choice(candidates, size=min(max_users, len(candidates)), replace=False)

    precisions, recalls = [], []
    # user_items passed to recommend must be the user-item training row (items already seen)
    user_items = train_matrix.tocsr()

    for user in sample:
        test_items = set(test_matrix.getrow(user).indices)
        if not test_items:
            continue
        recs = model.recommend(
            user,
            user_items[user],
            N=k,
            filter_already_liked_items=True,
        )
        rec_ids = set(recs[0].tolist())
        hits = len(rec_ids & test_items)
        precisions.append(hits / k)
        recalls.append(hits / len(test_items))

    precision = float(np.mean(precisions))
    recall    = float(np.mean(recalls))
    logger.info("Precision@%d: %.4f | Recall@%d: %.4f", k, precision, k, recall)
    return precision, recall

def save_model(model: AlternatingLeastSquares, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = path.stat().st_size / 1024
    logger.info("Saved model → %s (%.1f KB)", path, size_kb)


def save_mappings(user_map: dict, item_map: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"user_to_idx": user_map, "item_to_idx": item_map}
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved mappings → %s", path)

def main() -> None:
    for path in (MATRIX_FILE, USER_MAP_FILE, ITEM_MAP_FILE):
        if not path.exists():
            logger.error("Missing file: %s — run preprocess_movielens.py first", path)
            sys.exit(1)

    matrix = load_matrix(MATRIX_FILE)
    user_map, item_map = load_mappings(USER_MAP_FILE, ITEM_MAP_FILE)
    train_matrix, test_matrix = train_test_split(matrix, TEST_FRACTION)

    mlflow.set_experiment("recommender-als")
    with mlflow.start_run(run_name="als-movielens-100k"):
        mlflow.log_params(ALS_PARAMS)
        mlflow.log_param("eval_k", EVAL_K)
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("n_users", matrix.shape[0])
        mlflow.log_param("n_items", matrix.shape[1])

        model = train(train_matrix, ALS_PARAMS)

        precision, recall = precision_recall_at_k(
            model, train_matrix, test_matrix, k=EVAL_K
        )
        mlflow.log_metric("precision_at_10", precision)
        mlflow.log_metric("recall_at_10", recall)

        save_model(model, MODEL_FILE)
        save_mappings(user_map, item_map, MAPPINGS_FILE)

        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(MAPPINGS_FILE))

        logger.info(
            "MLflow run %s — precision@10=%.4f recall@10=%.4f",
            mlflow.active_run().info.run_id,
            precision,
            recall,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
