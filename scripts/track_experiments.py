"""Compare baselines and LightGBM configs with MLflow, using a clean selection/holdout protocol.

Folds (oldest -> newest): the first N-1 are SELECTION folds used to pick the best config;
the newest fold is a FINAL HOLDOUT, only reported for the chosen config. This avoids
reporting a number you tuned directly against.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import mlflow
from forecasting.data import load, to_series_dict, get_dates
from forecasting.baselines import BASELINES
from forecasting.metrics import wape, mape, bias
from forecasting.validation import rolling_origin_splits
from forecasting.model import prepare_fold, fit_predict_prepared

H, N_FOLDS = 8, 5
CONFIGS = {
    "lgbm_default":        {},
    "lgbm_leaves15":       {"num_leaves": 15},
    "lgbm_leaves63":       {"num_leaves": 63},
    "lgbm_lr0.1_n150":     {"learning_rate": 0.1, "n_estimators": 150},
    "lgbm_l2_objective":   {"objective": "regression"},
    "lgbm_minchild100":    {"min_child_samples": 100},
}

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("demand-forecasting")

df = load("data/train.csv")
series, dates = to_series_dict(df), get_dates(df)
n = min(len(v) for v in series.values())
splits = rolling_origin_splits(n, H, N_FOLDS)
print(f"{len(series)} series, {n} weeks, {len(splits)} folds (last = holdout)")


def score(pred_fn):
    rows = []
    for tr, te in splits:
        y = np.concatenate([series[k][tr:te] for k in series])
        yh = np.concatenate([pred_fn(tr, k) for k in series])
        rows.append((wape(y, yh), mape(y, yh), bias(y, yh)))
    return np.array(rows)


results = {}
for name, f in BASELINES.items():
    results[name] = score(lambda tr, k, f=f: f(series[k][:tr], H))

print("Preparing fold features (one-time)...")
preps = {tr: prepare_fold(series, dates, tr, H) for tr, _ in splits}
for name, cfg in CONFIGS.items():
    cache = {tr: fit_predict_prepared(preps[tr], cfg) for tr, _ in splits}
    results[name] = score(lambda tr, k, c=cache: c[tr][k])
    print(f"  done {name}")

rows = []
for name, r in results.items():
    sel, hold = r[:-1], r[-1]
    cfg = CONFIGS.get(name, {})
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": "lightgbm" if name in CONFIGS else name, "horizon": H, **cfg})
        mlflow.log_metrics({"selection_wape": sel[:, 0].mean(), "selection_mape": sel[:, 1].mean(),
                            "selection_bias": sel[:, 2].mean(), "holdout_wape": hold[0]})
    rows.append({"model": name, "selection_WAPE": sel[:, 0].mean(), "holdout_WAPE": hold[0]})

res = pd.DataFrame(rows).sort_values("selection_WAPE")
print("\n" + res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
best = res[res.model.isin(CONFIGS)].iloc[0]
base_name = res[res.model.isin(BASELINES)].iloc[0]
red = (base_name.holdout_WAPE - best.holdout_WAPE) / base_name.holdout_WAPE
print(f"\nChosen on selection folds: {best.model}")
print(f"FINAL HOLDOUT: {best.model} WAPE {best.holdout_WAPE:.4f} vs best baseline "
      f"({base_name.model}) {base_name.holdout_WAPE:.4f} -> {red:+.1%}")
res.to_csv("data/experiment_results.csv", index=False)
print("\nView runs:  mlflow ui --backend-store-uri sqlite:///mlruns.db")
