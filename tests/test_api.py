import joblib
import pytest
from fastapi.testclient import TestClient
from forecasting.data import generate_synthetic, to_series_dict, get_dates
from forecasting.model import LGBMForecaster


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = generate_synthetic(n_items=3, n_stores=2, weeks=130)
    s, d = to_series_dict(df), get_dates(df)
    m = LGBMForecaster(horizon=8, params=dict(n_estimators=30, verbose=-1)).fit(s, d, train_end=130)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"model": m, "series": s, "dates": d, "horizon": 8, "config": {},
                 "trained_at": "test"}, path)
    import os
    os.environ["MODEL_PATH"] = str(path)
    from forecasting.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["series"] == 6


def test_forecast_ok(client):
    r = client.get("/forecast", params={"store": 1, "item": 2})
    assert r.status_code == 200
    j = r.json()
    assert len(j["forecast"]) == 8 and all(p["forecast"] >= 0 for p in j["forecast"])
    assert j["forecast"][0]["week_ending"] > j["last_observed_week"]


def test_unknown_series_404(client):
    assert client.get("/forecast", params={"store": 99, "item": 1}).status_code == 404


def test_missing_param_422(client):
    assert client.get("/forecast", params={"store": 1}).status_code == 422
