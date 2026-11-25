import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from forecasting.data import load, to_series_dict
from forecasting.baselines import BASELINES
from forecasting.validation import evaluate

df = load("data/train.csv")  # drop Kaggle train.csv here to use real data
series = to_series_dict(df)
print(f"{len(series)} series, {len(next(iter(series.values())))} weeks each\n")
rows = []
for name, f in BASELINES.items():
    r = evaluate(series, f, horizon=8, n_folds=4)
    rows.append({"model": name, "WAPE": r.wape.mean(), "MAPE": r.mape.mean(), "Bias": r.bias.mean()})
res = pd.DataFrame(rows).sort_values("WAPE")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
res.to_csv("data/baseline_results.csv", index=False)
