import numpy as np
import lightgbm as lgb
from .features import build_rows, FEATURES

MIN_ORIGIN = 56  # need >52 weeks of history for last-year features


class LGBMForecaster:
    def __init__(self, horizon=8, params=None):
        self.horizon = horizon
        self.params = params or dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                     min_child_samples=20, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1, random_state=42)
        self.model = None

    def fit(self, series, dates, train_end):
        """Train only on targets strictly before train_end (no leakage)."""
        train = {k: v[:train_end] for k, v in series.items()}
        origins = range(MIN_ORIGIN, train_end - self.horizon + 1)
        X, y, _, _ = build_rows(train, dates, origins, self.horizon)
        self.model = lgb.LGBMRegressor(objective="regression_l1", **self.params)
        self.model.fit(X, y, feature_name=FEATURES, categorical_feature=["store", "item"])
        return self

    def predict(self, series, dates, origin):
        """Forecast y[origin : origin+horizon] for every series -> {key: array}."""
        hist = {k: v[:origin] for k, v in series.items()}
        # pad so features can index last-year values; targets are never read
        padded = {k: np.concatenate([h, np.zeros(self.horizon)]) for k, h in hist.items()}
        X, _, L, K = build_rows(padded, dates, [origin], self.horizon, with_target=False)
        pred = np.clip(self.model.predict(X), 0, None) * L
        out = {}
        for (store, item, h), p in zip(K, pred):
            out.setdefault((store, item), np.zeros(self.horizon))[h - 1] = p
        return out


DEFAULT_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)


def prepare_fold(series, dates, train_end, horizon):
    """Build train and predict matrices ONCE per fold so many configs can reuse them."""
    train = {k: v[:train_end] for k, v in series.items()}
    X, y, _, _ = build_rows(train, dates, range(MIN_ORIGIN, train_end - horizon + 1), horizon)
    padded = {k: np.concatenate([v[:train_end], np.zeros(horizon)]) for k, v in series.items()}
    Xp, _, Lp, Kp = build_rows(padded, dates, [train_end], horizon, with_target=False)
    return dict(X=X, y=y, Xp=Xp, Lp=Lp, Kp=Kp, horizon=horizon, train_end=train_end)


def fit_predict_prepared(prep, config):
    """config: LightGBM params (may include 'objective'). Returns {key: forecast array}."""
    params = {**DEFAULT_PARAMS, **config}
    objective = params.pop("objective", "regression_l1")
    m = lgb.LGBMRegressor(objective=objective, **params)
    m.fit(prep["X"], prep["y"], feature_name=FEATURES, categorical_feature=["store", "item"])
    pred = np.clip(m.predict(prep["Xp"]), 0, None) * prep["Lp"]
    out = {}
    for (store, item, h), p in zip(prep["Kp"], pred):
        out.setdefault((store, item), np.zeros(prep["horizon"]))[h - 1] = p
    return out
