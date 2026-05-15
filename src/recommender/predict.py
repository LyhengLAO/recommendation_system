"""Inference layer for the ALS recommender — loads once, serves many calls."""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_FILE    = BASE / "models" / "als_model.pkl"
MAPPINGS_FILE = BASE / "models" / "mappings.json"
MATRIX_FILE   = BASE / "data" / "processed" / "movielens_matrix.npz"
ITEMS_FILE    = BASE / "data" / "raw" / "movielens" / "u.item"


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model artefacts are ready."""


class UnknownUserError(ValueError):
    """Raised when a user_id has no entry in the training mappings."""


@dataclass
class Recommendation:
    item_id: int
    title: str
    score: float

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "title": self.title, "score": round(self.score, 6)}


@dataclass
class _ModelStore:
    model: AlternatingLeastSquares | None = field(default=None, repr=False)
    user_to_idx: dict[str, int] = field(default_factory=dict)
    item_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_item: dict[int, int] = field(default_factory=dict)   # 0-based idx → original item_id
    item_titles: dict[int, str] = field(default_factory=dict)   # original item_id → title
    user_items: csr_matrix | None = field(default=None, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        for path in (MODEL_FILE, MAPPINGS_FILE, MATRIX_FILE):
            if not path.exists():
                raise ModelNotLoadedError(
                    f"Artefact not found: {path} — run src/recommender/train.py first"
                )

        logger.info("Loading ALS model from %s", MODEL_FILE)
        with open(MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)

        logger.info("Loading mappings from %s", MAPPINGS_FILE)
        payload = json.loads(MAPPINGS_FILE.read_text())
        self.user_to_idx = payload["user_to_idx"]
        self.item_to_idx = payload["item_to_idx"]
        self.idx_to_item = {idx: int(iid) for iid, idx in self.item_to_idx.items()}

        logger.info("Loading user-item matrix from %s", MATRIX_FILE)
        self.user_items = load_npz(str(MATRIX_FILE)).astype(np.float32).tocsr()

        self.item_titles = _load_item_titles(ITEMS_FILE)
        logger.info(
            "Model store ready — %d users, %d items, %d titles",
            len(self.user_to_idx),
            len(self.item_to_idx),
            len(self.item_titles),
        )


_store = _ModelStore()


def _load_item_titles(path: Path) -> dict[int, str]:
    """Parse u.item (pipe-separated, latin-1) and return {item_id: title}."""
    if not path.exists():
        logger.warning("u.item not found at %s — titles will be empty", path)
        return {}
    titles: dict[int, str] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                try:
                    titles[int(parts[0])] = parts[1]
                except ValueError:
                    continue
    logger.info("Loaded %d item titles", len(titles))
    return titles


def _ensure_loaded() -> None:
    if not _store.is_loaded:
        _store.load()

def load_model() -> None:
    """Explicitly pre-load artefacts (e.g. at API startup)."""
    _store.load()


def recommend(user_id: int, top_n: int = 10) -> list[dict]:
    """
    Return top-N recommendations for a given user.

    Parameters
    ----------
    user_id : int
        Original MovieLens user ID (1-based).
    top_n : int
        Number of items to return.

    Returns
    -------
    list of {"item_id": int, "title": str, "score": float}

    Raises
    ------
    ModelNotLoadedError
        If artefacts are missing from disk.
    UnknownUserError
        If user_id was not seen during training.
    """
    _ensure_loaded()

    key = str(user_id)
    if key not in _store.user_to_idx:
        known_range = f"1–{max(int(k) for k in _store.user_to_idx)}"
        raise UnknownUserError(
            f"user_id={user_id} not found in training data (known range: {known_range})"
        )

    user_idx = _store.user_to_idx[key]
    user_row = _store.user_items[user_idx]

    item_indices, raw_scores = _store.model.recommend(
        user_idx,
        user_row,
        N=top_n,
        filter_already_liked_items=True,
    )

    results: list[Recommendation] = []
    for item_idx, score in zip(item_indices.tolist(), raw_scores.tolist()):
        original_item_id = _store.idx_to_item.get(item_idx, item_idx)
        title = _store.item_titles.get(original_item_id, f"item_{original_item_id}")
        results.append(Recommendation(item_id=original_item_id, title=title, score=score))

    logger.debug("recommend(user_id=%d, top_n=%d) → %d results", user_id, top_n, len(results))
    return [r.to_dict() for r in results]


def recommend_batch(user_ids: list[int], top_n: int = 10) -> dict[int, list[dict]]:
    """
    Recommend for multiple users at once.
    Unknown users are returned with an empty list and a warning — not an exception.
    """
    _ensure_loaded()
    output: dict[int, list[dict]] = {}
    for uid in user_ids:
        try:
            output[uid] = recommend(uid, top_n)
        except UnknownUserError as exc:
            logger.warning("%s", exc)
            output[uid] = []
    return output
