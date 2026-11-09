"""Latency check of the in-process scoring path."""
import time

import pandas as pd
import pytest

from api.model_loader import DEPLOY_DIR, ModelStore
from src.common import p


def test_single_prediction_latency_and_batch_throughput():
    if not (DEPLOY_DIR / "model.pkl").exists():
        pytest.skip("run the pipeline first")
    store = ModelStore(); store.load()
    recs = pd.read_csv(p("data", "processed", "X_test_raw.csv")).drop(columns=["TotalDelay", "CompositeServiceScore"]).head(500).to_dict("records")
    store.predict(recs[:1])  # warm-up
    t = time.perf_counter(); store.predict(recs[:1]); single_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); store.predict(recs); batch_s = time.perf_counter() - t
    assert single_ms < 200, f"single prediction too slow: {single_ms:.1f} ms"
    assert batch_s < 5, f"batch of 500 too slow: {batch_s:.2f} s"
