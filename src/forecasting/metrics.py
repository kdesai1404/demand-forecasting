import numpy as np


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|. Robust to zeros."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def mape(y, yhat) -> float:
    """MAPE ignoring points where actual == 0."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m]))) if m.any() else float("nan")


def bias(y, yhat) -> float:
    """Positive = over-forecasting."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / max(np.abs(y).sum(), 1e-9))
