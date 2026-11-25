"""Time-based rolling-origin validation. Never shuffles; train always precedes test."""
import numpy as np
import pandas as pd
from .metrics import wape, mape, bias


def rolling_origin_splits(n_weeks, horizon=8, n_folds=4, min_train=104):
    """Return [(train_end, test_end)] index pairs; test = [train_end, test_end)."""
    last_start = n_weeks - horizon
    if last_start < min_train:
        raise ValueError("Series too short for requested min_train/horizon")
    starts = [last_start - i * horizon for i in range(n_folds)][::-1]
    return [(s, s + horizon) for s in starts if s >= min_train]


def evaluate(series: dict, forecaster, horizon=8, n_folds=4, min_train=104) -> pd.DataFrame:
    """Pooled metrics per fold across all series."""
    n = min(len(v) for v in series.values())
    out = []
    for fold, (tr, te) in enumerate(rolling_origin_splits(n, horizon, n_folds, min_train)):
        ys, yhs = [], []
        for v in series.values():
            ys.append(v[tr:te])
            yhs.append(forecaster(v[:tr], horizon))
        y, yh = np.concatenate(ys), np.concatenate(yhs)
        out.append({"fold": fold, "wape": wape(y, yh), "mape": mape(y, yh), "bias": bias(y, yh)})
    return pd.DataFrame(out)


def evaluate_global(series: dict, dates, make_model, horizon=8, n_folds=4, min_train=104) -> pd.DataFrame:
    """Same folds as `evaluate`, for global models with fit/predict. Refits per fold."""
    n = min(len(v) for v in series.values())
    out = []
    for fold, (tr, te) in enumerate(rolling_origin_splits(n, horizon, n_folds, min_train)):
        model = make_model().fit(series, dates, train_end=tr)
        preds = model.predict(series, dates, origin=tr)
        y = np.concatenate([series[k][tr:te] for k in series])
        yh = np.concatenate([preds[k] for k in series])
        out.append({"fold": fold, "wape": wape(y, yh), "mape": mape(y, yh), "bias": bias(y, yh)})
    return pd.DataFrame(out)
