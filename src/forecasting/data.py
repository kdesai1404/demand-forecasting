"""Data loading. Works with synthetic data out of the box, or the Kaggle
'Store Item Demand Forecasting Challenge' train.csv (daily) aggregated to weekly."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic(n_items=30, n_stores=3, weeks=156, seed=42) -> pd.DataFrame:
    """Weekly demand for fabric/garment SKUs: trend + yearly seasonality +
    festive (Diwali-like) spike + noise. Columns: date, store, item, sales."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-03", periods=weeks, freq="W-SUN")
    t = np.arange(weeks)
    week_of_year = dates.isocalendar().week.to_numpy().astype(int)
    festive = np.exp(-0.5 * ((week_of_year - 43) / 2.5) ** 2)  # peak around late Oct
    rows = []
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            base = rng.uniform(20, 200)
            trend = rng.uniform(-0.05, 0.25) * t / 52
            season = rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * (t / 52 + rng.uniform()))
            spike = rng.uniform(0.2, 1.0) * festive
            mean = base * (1 + trend + season + spike)
            sales = rng.poisson(np.clip(mean, 1, None) * rng.uniform(0.9, 1.1, weeks))
            rows.append(pd.DataFrame({"date": dates, "store": store, "item": item, "sales": sales}))
    return pd.concat(rows, ignore_index=True)


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to weeks (weeks end Sunday), per store/item."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    out = (df.set_index("date").groupby(["store", "item"])["sales"]
             .resample("W-SUN").sum().reset_index())
    return out[["date", "store", "item", "sales"]]


def load_kaggle_store_item(path: str) -> pd.DataFrame:
    return to_weekly(pd.read_csv(path))


def load(path: str | None = None) -> pd.DataFrame:
    """Load real data if path exists, else synthetic."""
    if path and Path(path).exists():
        return load_kaggle_store_item(path)
    return generate_synthetic()


def to_series_dict(df: pd.DataFrame) -> dict:
    """{(store, item): np.ndarray of weekly sales sorted by date}."""
    df = df.sort_values("date")
    return {k: g["sales"].to_numpy(float) for k, g in df.groupby(["store", "item"])}


def get_dates(df: pd.DataFrame):
    return pd.DatetimeIndex(sorted(df["date"].unique()))


def extend_dates(dates: pd.DatetimeIndex, extra: int) -> pd.DatetimeIndex:
    """Weekly dates extended `extra` weeks into the future (needed to forecast beyond the data)."""
    return pd.date_range(dates[0], periods=len(dates) + extra, freq="W-SUN")
