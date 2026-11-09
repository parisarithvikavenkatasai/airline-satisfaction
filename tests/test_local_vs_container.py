"""Compare in-process predictions with a running container/API.

Usage: CONTAINER_URL=http://localhost:8000 pytest tests/test_local_vs_container.py
Skipped when CONTAINER_URL is not set.
"""
import os

import pandas as pd
import pytest

from api.model_loader import DEPLOY_DIR, ModelStore
from src.common import p

URL = os.environ.get("CONTAINER_URL")


@pytest.mark.skipif(not URL, reason="CONTAINER_URL not set")
def test_container_matches_local_predictions():
    import requests
    if not (DEPLOY_DIR / "model.pkl").exists():
        pytest.skip("run the pipeline first")
    store = ModelStore(); store.load()
    recs = pd.read_csv(p("data", "processed", "X_test_raw.csv")).drop(columns=["satisfaction", "TotalDelay", "CompositeServiceScore"], errors="ignore").head(20).to_dict("records")
    local = store.predict(recs)
    for rec, loc in zip(recs, local):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        remote = requests.post(f"{URL}/predict", json=rec, timeout=10).json()
        assert remote["prediction"] == loc["prediction"]
        assert abs(remote["probability_satisfied"] - loc["probability_satisfied"]) < 1e-4
