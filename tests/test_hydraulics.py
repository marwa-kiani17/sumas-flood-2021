import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodml.hydraulics import (annual_maxima, bootstrap_ci, fit_gev, flow_to_level, quantile,
                                rating_curve, return_period)


def test_gev_recovers_quantiles():
    true = (-0.1, 100.0, 30.0)
    peaks = stats.genextreme.rvs(*true, size=2000, random_state=1)
    est = quantile(fit_gev(peaks), [2, 10, 25])
    ref = quantile(true, [2, 10, 25])
    assert np.allclose(est, ref, rtol=0.05)


def test_return_period_roundtrip():
    p = (-0.1, 100.0, 30.0)
    assert abs(return_period(quantile(p, 10), p) - 10) < 1e-6


def test_bootstrap_ci_brackets_estimate():
    peaks = stats.genextreme.rvs(-0.1, 100, 30, size=40, random_state=2)
    lo, hi = bootstrap_ci(peaks, [10], n=100)
    assert lo[0] < quantile(fit_gev(peaks), [10])[0] < hi[0]


def test_annual_maxima_skips_incomplete_years():
    idx = pd.date_range("2000-01-01", "2001-06-30")
    s = pd.Series(1.0, index=idx)
    assert list(annual_maxima(s).index) == [2000]


def test_rating_curve_recovery_and_inverse():
    rng = np.random.default_rng(0)
    h = rng.uniform(1.0, 4.0, 300)
    q = 20 * (h - 0.5) ** 2.0
    (a, b, h0), res = rating_curve(h, q)
    assert abs(b - 2.0) < 0.05 and abs(h0 - 0.5) < 0.1
    assert abs(flow_to_level(q[0], (a, b, h0)) - h[0]) < 0.05
