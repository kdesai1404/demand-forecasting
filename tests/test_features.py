import numpy as np
import pandas as pd
from forecasting.features import row_features
from forecasting.data import generate_synthetic, to_series_dict, get_dates
from forecasting.model import LGBMForecaster


def test_features_ignore_future():
    rng = np.random.default_rng(0)
    y = rng.poisson(50, 120).astype(float)
    d = pd.date_range("2021-01-03", periods=120, freq="W-SUN")
    t = 100
    _, f1 = row_features(y, t, 8, d[t + 7], 1, 1)
    y2 = y.copy(); y2[t:] = 9999          # corrupt everything from the origin on
    _, f2 = row_features(y2, t, 8, d[t + 7], 1, 1)
    assert f1 == f2


def test_model_trains_and_predicts():
    df = generate_synthetic(n_items=3, n_stores=1, weeks=130)
    s, d = to_series_dict(df), get_dates(df)
    m = LGBMForecaster(horizon=4, params=dict(n_estimators=30, verbose=-1)).fit(s, d, train_end=110)
    p = m.predict(s, d, origin=110)
    assert len(p) == 3 and all(len(v) == 4 and (v >= 0).all() for v in p.values())
