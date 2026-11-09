"""FastAPI service for airline passenger satisfaction predictions.

Run locally:   uvicorn api.main:app --reload --port 8000
Docs (Swagger): http://localhost:8000/docs
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_loader import ModelNotLoaded, store
from api.monitoring_logger import log_prediction
from api.schemas import (BatchRequest, BatchResponse, HealthResponse, PassengerFeatures,
                         PredictionResponse)

MAX_BATCH = 1000
_counters = {"requests": 0, "errors": 0, "total_latency_ms": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store.load()
    except ModelNotLoaded as exc:
        print(f"[startup] WARNING: {exc}")
    yield


app = FastAPI(title="Airline Passenger Satisfaction API", version="1.0.0", lifespan=lifespan)


def _score(records):
    start = time.perf_counter()
    try:
        results = store.predict(records)
    except ModelNotLoaded as exc:
        _counters["errors"] += 1
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        _counters["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    latency = (time.perf_counter() - start) * 1000
    _counters["requests"] += 1
    _counters["total_latency_ms"] += latency
    for r in results:
        log_prediction(r, latency / len(results))
    return results


@app.get("/")
def root():
    return {"service": "airline-satisfaction-api", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if store.loaded else "degraded", model_loaded=store.loaded,
                          model_version=store.version)


@app.get("/model-info")
def model_info():
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return store.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerFeatures):
    return _score([passenger.model_dump(by_alias=True)])[0]


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if len(req.passengers) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Batch too large (max {MAX_BATCH})")
    results = _score([p.model_dump(by_alias=True) for p in req.passengers])
    return BatchResponse(predictions=results, count=len(results))


@app.post("/reload")
def reload_model():
    try:
        store.load()
    except ModelNotLoaded as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "reloaded", "model_version": store.version}


@app.get("/metrics")
def metrics():
    n = _counters["requests"]
    return {"requests": n, "errors": _counters["errors"],
            "avg_latency_ms": round(_counters["total_latency_ms"] / n, 2) if n else None}
