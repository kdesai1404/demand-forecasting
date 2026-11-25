"""Baselines every real model must beat. Each: f(history, horizon) -> np.ndarray."""
import numpy as np


def naive(history, horizon):
    return np.repeat(history[-1], horizon)


def moving_average(history, horizon, window=4):
    return np.repeat(np.mean(history[-window:]), horizon)


def seasonal_naive(history, horizon, season=52):
    """Value from the same week last year; falls back to naive if too short."""
    if len(history) < season:
        return naive(history, horizon)
    return np.array([history[-season + (h % season)] for h in range(horizon)])


BASELINES = {
    "naive": naive,
    "moving_avg_4": moving_average,
    "seasonal_naive_52": seasonal_naive,
}
