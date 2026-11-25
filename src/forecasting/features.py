"""Leakage-safe feature building for a direct multi-horizon global model.

For a forecast origin t (history = y[:t]) and horizon step h (1..H), the target is
y[t+h-1]. Every feature uses only y[:t] plus calendar info of the *target* week
(known in advance). Values are scaled by a recent level so one model can serve
SKUs of very different sizes.
"""
import numpy as np

FEATURES = ["h", "last", "mean4", "mean8", "mean13", "std13", "trend",
            "ly", "ly_nb", "woy", "month", "store", "item"]


def row_features(y, t, h, target_date, store, item):
    hist = y[:t]
    level = hist[-13:].mean() + 1.0
    tgt = t + h - 1
    ly_idx = tgt - 52                      # same week last year (< t whenever h <= 52)
    ly = y[ly_idx] / level
    ly_nb = y[max(ly_idx - 1, 0): ly_idx + 2].mean() / level
    return level, [
        h,
        hist[-1] / level, hist[-4:].mean() / level, hist[-8:].mean() / level,
        1.0, hist[-13:].std() / level,
        hist[-13:].mean() / (hist[-26:-13].mean() + 1.0),
        ly, ly_nb,
        target_date.isocalendar().week, target_date.month, store, item,
    ]


def build_rows(series, dates, origins, horizon, with_target=True):
    """Return X, y_ratio, levels, keys, targets for the given origins."""
    X, Y, L, K = [], [], [], []
    for (store, item), y in series.items():
        for t in origins:
            for h in range(1, horizon + 1):
                tgt = t + h - 1
                if with_target and tgt >= len(y):
                    continue
                level, f = row_features(y, t, h, dates[tgt], store, item)
                X.append(f); L.append(level); K.append((store, item, h))
                Y.append(y[tgt] / level if with_target else np.nan)
    return np.array(X, float), np.array(Y), np.array(L), K
