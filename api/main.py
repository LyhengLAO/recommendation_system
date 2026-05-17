"""FastAPI application — recommender + anomaly detection endpoints."""

import csv
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# --- path bootstrap ----------------------------------------------------------
# Allow `src.*` imports when the app is launched from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.detect import (  # noqa: E402
    MissingFeaturesError,
    ModelNotLoadedError as AnomalyModelNotLoadedError,
    detect,
    load_model as load_anomaly_model,
)
from src.recommender.predict import (  # noqa: E402
    ModelNotLoadedError as RecommenderModelNotLoadedError,
    UnknownUserError,
    load_model as load_recommender_model,
    recommend,
)

# --- logging -----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

LOGS_DIR = ROOT / "data" / "raw" / "logs"
REQUEST_LOG_FILE = LOGS_DIR / "requests.csv"
_CSV_HEADER = ["timestamp", "method", "path", "status_code", "latency_ms", "client_ip"]


def _init_request_log() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not REQUEST_LOG_FILE.exists():
        with open(REQUEST_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)
        logger.info("Request log initialised at %s", REQUEST_LOG_FILE)


def _append_request_log(row: dict[str, Any]) -> None:
    with open(REQUEST_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([row[k] for k in _CSV_HEADER])


# --- lifespan ----------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_request_log()
    errors: list[str] = []

    logger.info("Loading recommender model …")
    try:
        load_recommender_model()
    except Exception as exc:
        logger.warning("Recommender model not available: %s", exc)
        errors.append(f"recommender: {exc}")

    logger.info("Loading anomaly model …")
    try:
        load_anomaly_model()
    except Exception as exc:
        logger.warning("Anomaly model not available: %s", exc)
        errors.append(f"anomaly: {exc}")

    if errors:
        logger.warning("Started with %d missing model(s) — some endpoints will return 503", len(errors))
    else:
        logger.info("All models loaded — API ready")

    yield

    logger.info("Shutting down API")


# --- app ---------------------------------------------------------------------

app = FastAPI(
    title="ML Platform API",
    description="Recommender system + anomaly detection",
    version="0.1.0",
    lifespan=lifespan,
)


# --- middleware --------------------------------------------------------------


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "%s %s → %d (%.1f ms) [%s]",
        request.method, request.url.path, response.status_code, latency_ms, client_ip,
    )

    _append_request_log({
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
    })

    return response


# --- global error handler ----------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# --- schemas -----------------------------------------------------------------


class AnomalyRequest(BaseModel):
    latency_ms: float = Field(..., gt=0, description="Request latency in milliseconds")
    status_code: int  = Field(..., ge=100, le=599)
    endpoint: str     = Field(..., description="API endpoint path e.g. /recommend")
    user_id: int      = Field(..., ge=1)
    timestamp: str | None = Field(default=None, description="ISO-8601 timestamp; defaults to now")
    user_request_freq: float | None = Field(default=None, ge=0)
    user_error_ratio: float | None  = Field(default=None, ge=0, le=1)


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float
    reason: str


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


# --- endpoints ---------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, tags=["ops"], include_in_schema=False)
async def homepage() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Platform API</title>
    <style>
        body { font-family: sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px; color: #222; }
        h1   { font-size: 1.8rem; margin-bottom: 4px; }
        p    { line-height: 1.6; color: #444; }
        table { width: 100%; border-collapse: collapse; margin-top: 24px; }
        th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #e5e5e5; font-size: 0.95rem; }
        th   { background: #f5f5f5; font-weight: 600; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9rem; }
        a    { color: #0070f3; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .get  { background: #dff0d8; color: #3a763a; }
        .post { background: #d9edf7; color: #2a6496; }
    </style>
</head>
<body>
    <h1>ML Platform API</h1>
    <p>
        This platform exposes two machine-learning services built on the
        <strong>MovieLens</strong> dataset:
    </p>
    <ul>
        <li><strong>Recommender</strong> — ALS (Alternating Least Squares) collaborative filtering
            that returns personalised movie suggestions for a given user.</li>
        <li><strong>Anomaly Detection</strong> — Isolation Forest model that scores API log entries
            and flags unusual traffic patterns (high latency, error spikes, abnormal request frequency).</li>
    </ul>
    <p>
        Every request is automatically logged to <code>data/raw/logs/requests.csv</code>
        and can be fed back into the anomaly-detection pipeline for continuous monitoring.
    </p>

    <table>
        <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
        <tr>
            <td><span class="badge get">GET</span></td>
            <td><code>/health</code></td>
            <td>Check API status and model availability</td>
        </tr>
        <tr>
            <td><span class="badge get">GET</span></td>
            <td><code>/recommend/{user_id}</code></td>
            <td>Top-N movie recommendations for a user</td>
        </tr>
        <tr>
            <td><span class="badge post">POST</span></td>
            <td><code>/anomaly/detect</code></td>
            <td>Score a single API log entry for anomalies</td>
        </tr>
    </table>

    <p style="margin-top:28px;">
        Interactive documentation: <a href="/docs">/docs</a>
    </p>
</body>
</html>
""")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    from src.recommender.predict import _store as rec_store
    from src.anomaly.detect import _store as det_store

    return HealthResponse(
        status="ok",
        models_loaded={
            "recommender": rec_store.is_loaded,
            "anomaly": det_store.is_loaded,
        },
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    tags=["recommender"],
    summary="Top-N recommendations for a user",
)
async def get_recommendations(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=100, description="Number of items to return"),
) -> RecommendResponse:
    try:
        recs = recommend(user_id, top_n=top_n)
    except (RecommenderModelNotLoadedError,) as exc:
        logger.error("Recommender model not ready: %s", exc)
        raise HTTPException(status_code=503, detail="Recommender model not loaded — run train.py first")
    except UnknownUserError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /recommend/%d: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Recommendation failed")

    return RecommendResponse(
        user_id=user_id,
        recommendations=[RecommendationItem(**r) for r in recs],
    )


@app.post(
    "/anomaly/detect",
    response_model=AnomalyResponse,
    tags=["anomaly"],
    summary="Score a single API log entry for anomalies",
)
async def anomaly_detect(body: AnomalyRequest) -> AnomalyResponse:
    entry: dict[str, Any] = body.model_dump(exclude_none=True)

    if "timestamp" not in entry:
        entry["timestamp"] = datetime.utcnow().isoformat(timespec="seconds")

    try:
        result = detect(entry)
    except (AnomalyModelNotLoadedError,) as exc:
        logger.error("Anomaly model not ready: %s", exc)
        raise HTTPException(status_code=503, detail="Anomaly model not loaded — run train.py first")
    except MissingFeaturesError as exc:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {exc}")
    except Exception as exc:
        logger.error("Unexpected error in /anomaly/detect: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Detection failed")

    return AnomalyResponse(**result)
