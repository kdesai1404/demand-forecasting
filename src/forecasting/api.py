"""FastAPI service: GET /forecast?store=1&item=1 -> next-8-week forecast.
Run: uvicorn forecasting.api:app --port 8000   (docs at /docs)"""
import os
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from .data import extend_dates

STATE = {}


class Point(BaseModel):
    week_ending: str
    forecast: float


class ForecastResponse(BaseModel):
    store: int
    item: int
    horizon_weeks: int
    last_observed_week: str
    recent_actuals: list[float]
    forecast: list[Point]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
    if not os.path.exists(path):
        raise RuntimeError(f"Model bundle not found at {path}. Run scripts/train_final.py first.")
    STATE["bundle"] = joblib.load(path)
    yield
    STATE.clear()


app = FastAPI(title="Demand Forecasting API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    b = STATE["bundle"]
    return {"status": "ok", "series": len(b["series"]), "weeks": len(b["dates"]),
            "horizon": b["horizon"], "trained_at": b["trained_at"]}


@app.get("/items")
def items():
    keys = sorted(STATE["bundle"]["series"])
    return {"count": len(keys), "stores": sorted({k[0] for k in keys}),
            "items": sorted({k[1] for k in keys})}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(store: int = Query(..., ge=0), item: int = Query(..., ge=0)):
    b = STATE["bundle"]
    key = (store, item)
    if key not in b["series"]:
        raise HTTPException(404, f"Unknown series: store={store}, item={item}")
    H, dates = b["horizon"], b["dates"]
    n = len(dates)
    preds = b["model"].predict({key: b["series"][key]}, extend_dates(dates, H), origin=n)[key]
    future = extend_dates(dates, H)[n:]
    return ForecastResponse(
        store=store, item=item, horizon_weeks=H,
        last_observed_week=str(dates[-1].date()),
        recent_actuals=[float(x) for x in b["series"][key][-12:]],
        forecast=[Point(week_ending=str(d.date()), forecast=round(float(p), 2))
                  for d, p in zip(future, np.asarray(preds))],
    )
