import numpy as np
import pytest
from forecasting.metrics import wape, mape, bias
from forecasting.baselines import naive, seasonal_naive, moving_average
from forecasting.validation import rolling_origin_splits
from forecasting.data import generate_synthetic, to_weekly


def test_wape_perfect_and_known():
    assert wape([1, 2, 3], [1, 2, 3]) == 0
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)


def test_mape_ignores_zero_actuals():
    assert mape([0, 10], [5, 5]) == pytest.approx(0.5)


def test_bias_sign():
    assert bias([10, 10], [12, 12]) > 0


def test_naive_and_ma():
    h = np.array([1., 2., 3., 4., 5.])
    assert list(naive(h, 3)) == [5, 5, 5]
    assert moving_average(h, 2, window=2)[0] == 4.5


def test_seasonal_naive_repeats_last_year():
    h = np.arange(104, dtype=float)
    f = seasonal_naive(h, 3, season=52)
    assert list(f) == [52, 53, 54]


def test_splits_never_leak():
    for tr, te in rolling_origin_splits(156, horizon=8, n_folds=4, min_train=104):
        assert tr < te <= 156 and tr >= 104
    assert rolling_origin_splits(156)[-1][1] == 156


def test_splits_too_short():
    with pytest.raises(ValueError):
        rolling_origin_splits(50)


def test_synthetic_shape_and_weekly_agg():
    df = generate_synthetic(n_items=2, n_stores=2, weeks=60)
    assert len(df) == 2 * 2 * 60 and (df.sales >= 0).all()
    daily = df.assign(date=df.date - __import__("pandas").Timedelta(days=3))
    assert len(to_weekly(daily)) >= len(df) - 4
