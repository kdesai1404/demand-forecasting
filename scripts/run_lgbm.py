import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from forecasting.data import load, to_series_dict, get_dates
from forecasting.baselines import BASELINES
from forecasting.validation import evaluate, evaluate_global
from forecasting.model import LGBMForecaster

H = 8
df = load("data/train.csv")
series, dates = to_series_dict(df), get_dates(df)
rows = []
for name, f in BASELINES.items():
    r = evaluate(series, f, horizon=H)
    rows.append({"model": name, "WAPE": r.wape.mean(), "MAPE": r.mape.mean(), "Bias": r.bias.mean()})
r = evaluate_global(series, dates, lambda: LGBMForecaster(horizon=H), horizon=H)
rows.append({"model": "lightgbm", "WAPE": r.wape.mean(), "MAPE": r.mape.mean(), "Bias": r.bias.mean()})
res = pd.DataFrame(rows).sort_values("WAPE")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
base = res[res.model == "seasonal_naive_52"].WAPE.iloc[0]
best = res[res.model == "lightgbm"].WAPE.iloc[0]
print(f"\nLightGBM vs seasonal naive: {(base - best) / base:+.1%} WAPE reduction")
res.to_csv("data/model_comparison.csv", index=False)
