"""API tests (FastAPI TestClient). Needs: pip install fastapi httpx. Run pipeline first."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from api.model_loader import DEPLOY_DIR  # noqa: E402

if not (DEPLOY_DIR / "model.pkl").exists():
    pytest.skip("deployment/ not prepared - run the pipeline first", allow_module_level=True)

VALID = {"Gender": "Female", "Customer Type": "Loyal Customer", "Age": 41, "Type of Travel": "Business travel",
         "Class": "Business", "Flight Distance": 1850, "Inflight wifi service": 4,
         "Departure/Arrival time convenient": 4, "Ease of Online booking": 4, "Gate location": 3,
         "Food and drink": 4, "Online boarding": 5, "Seat comfort": 5, "Inflight entertainment": 5,
         "On-board service": 4, "Leg room service": 4, "Baggage handling": 4, "Checkin service": 4,
         "Inflight service": 4, "Cleanliness": 5, "Departure Delay in Minutes": 5,
         "Arrival Delay in Minutes": 3}


def test_health_and_predict():
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        r = c.post("/predict", json=VALID)
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in ("satisfied", "neutral or dissatisfied")
        assert 0.0 <= body["probability_satisfied"] <= 1.0


def test_missing_arrival_delay_is_accepted():
    with TestClient(app) as c:
        payload = {k: v for k, v in VALID.items() if k != "Arrival Delay in Minutes"}
        assert c.post("/predict", json=payload).status_code == 200


def test_invalid_rating_rejected():
    with TestClient(app) as c:
        assert c.post("/predict", json={**VALID, "Cleanliness": 9}).status_code == 422


def test_unknown_category_rejected():
    with TestClient(app) as c:
        assert c.post("/predict", json={**VALID, "Class": "First"}).status_code == 422


def test_missing_field_rejected():
    with TestClient(app) as c:
        bad = {k: v for k, v in VALID.items() if k != "Age"}
        assert c.post("/predict", json=bad).status_code == 422


def test_batch():
    with TestClient(app) as c:
        r = c.post("/predict_batch", json={"passengers": [VALID, VALID]})
        assert r.status_code == 200 and r.json()["count"] == 2


def test_empty_batch_rejected():
    with TestClient(app) as c:
        assert c.post("/predict_batch", json={"passengers": []}).status_code == 422
