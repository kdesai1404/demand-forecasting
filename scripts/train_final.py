"""Train the final model on ALL available data and save a bundle for the API."""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import pandas as pd
from forecasting.data import load, to_series_dict, get_dates
from forecasting.model import LGBMForecaster, DEFAULT_PARAMS

H = 8
CONFIG = {**DEFAULT_PARAMS, "num_leaves": 63}   # chosen on the selection folds
out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/model.joblib")
out.parent.mkdir(parents=True, exist_ok=True)

df = load("data/train.csv")
series, dates = to_series_dict(df), get_dates(df)
n = min(len(v) for v in series.values())
series = {k: v[:n] for k, v in series.items()}
print(f"Training on {len(series)} series x {n} weeks ...")
model = LGBMForecaster(horizon=H, params=CONFIG).fit(series, dates[:n], train_end=n)
joblib.dump({"model": model, "series": series, "dates": dates[:n], "horizon": H,
             "config": CONFIG, "trained_at": pd.Timestamp.now().isoformat()}, out)
print(f"Saved {out}")
