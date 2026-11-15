"""Flood frequency (GEV return periods) and rating curves.

Hydrology terms:
- Annual maximum: the highest daily flow of each year.
- GEV (Generalized Extreme Value): a curve fitted to those yearly maxima. It lets us say how large
  a flood is expected once every T years (the "T-year flood", Q_T), e.g. Q10 is exceeded
  on average once in 10 years (10% chance in any year).
- Rating curve: Q = a * (h - h0)^b, links water level h (m) to flow Q (m3/s). h0 is the level
  at which flow stops.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize

RETURN_PERIODS = [2, 5, 10, 25]


def annual_maxima(flow, min_days=300, months=None):
    """Highest daily flow per calendar year. Years with < min_days valid days are skipped
    (an incomplete year may have missed its true peak).

    months=(m0, m1): use only that season and require 90% of its days to be present. This is the
    fallback for seasonal gauges that do not run in winter (ice), e.g. Kettle River."""
    f = flow.dropna()
    if months is None:
        g = f.groupby(f.index.year)
        return g.max()[g.count() >= min_days]
    f = f[(f.index.month >= months[0]) & (f.index.month <= months[1])]
    n_season = int(sum(pd.Period(f"2001-{m:02d}").days_in_month for m in range(months[0], months[1] + 1)))
    g = f.groupby(f.index.year)
    return g.max()[g.count() >= 0.9 * n_season]


def fit_gev(peaks, xi_bounds=(-0.1, 0.4)):
    """Fit GEV to annual maxima by maximum likelihood; returns (c, loc, scale) in scipy's
    convention (c = -xi).

    With only ~20-30 peaks an unconstrained fit often picks an absurd tail shape (Q25 in the
    millions, or a hard upper limit the 2021 flood exceeds). So the shape xi is bounded to
    [-0.1, 0.4], the range usually seen for river floods (xi = 0 is Gumbel; larger xi = heavier tail).
    """
    x = np.asarray(peaks, float)
    s = x.mean()  # work on a ~1 scale so the optimiser behaves
    xs = x / s
    g_loc, g_scale = stats.gumbel_r.fit(xs)

    def nll(th):
        c, loc, scale = th
        v = -stats.genextreme.logpdf(xs, c, loc, scale).sum()
        return v if np.isfinite(v) else 1e10

    best = None
    for c0 in (-0.1, -0.02):  # start near Gumbel
        r = minimize(nll, [c0, g_loc, g_scale], method="L-BFGS-B",
                     bounds=[(-xi_bounds[1], -xi_bounds[0]), (None, None), (1e-6, None)])
        if best is None or r.fun < best.fun:
            best = r
    c, loc, scale = best.x
    return c, loc * s, scale * s


def quantile(params, T):
    """Flow with return period T years."""
    return stats.genextreme.ppf(1 - 1.0 / np.asarray(T, float), *params)


def return_period(q, params):
    """Return period (years) of a flow q under the fitted GEV."""
    p = stats.genextreme.cdf(q, *params)
    return 10000.0 if p >= 1 - 1e-4 else 1.0 / (1.0 - p)  # capped: beyond ~10000 yr is meaningless


def bootstrap_ci(peaks, Ts=RETURN_PERIODS, n=300, seed=42, level=0.90):
    """Bootstrap confidence interval of Q_T: refit GEV on resampled peaks n times."""
    rng = np.random.default_rng(seed)
    peaks = np.asarray(peaks, float)
    qs = []
    for _ in range(n):
        try:
            qs.append(quantile(fit_gev(rng.choice(peaks, len(peaks), replace=True)), Ts))
        except Exception:
            continue
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    qs = np.array(qs)
    return np.percentile(qs, lo, axis=0), np.percentile(qs, hi, axis=0)


def fit_gumbel(peaks):
    """Gumbel (a simpler extreme-value curve) used as a cross-check of the GEV."""
    return stats.gumbel_r.fit(np.asarray(peaks, float))


def rating_curve(h, q):
    """Fit Q = a*(h-h0)^b with a>0, 1<b<3.5, h0<min(h). Returns (a, b, h0), residuals (m3/s)."""
    h, q = np.asarray(h, float), np.asarray(q, float)
    f = lambda h, a, b, h0: a * np.clip(h - h0, 1e-9, None) ** b
    hmin = h.min()
    p0 = [max(q.max() / max((h.max() - hmin + 0.1) ** 2, 1e-6), 1e-3), 2.0, hmin - 0.1]
    lo = [1e-9, 1.0, hmin - 20]
    hi = [np.inf, 3.5, hmin - 1e-6]
    params, _ = curve_fit(f, h, q, p0=p0, bounds=(lo, hi), maxfev=20000)
    return params, q - f(h, *params)


def flow_to_level(q, params):
    """Invert the rating curve: water level for a given flow."""
    a, b, h0 = params
    return h0 + (np.asarray(q, float) / a) ** (1.0 / b)
