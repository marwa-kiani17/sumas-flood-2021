"""Baselines and LightGBM quantile models.

Quantile model: instead of one number, predict the 10%, 50% and 90% levels of tomorrow's flow
(a forecast range). The 0.9 model is "pessimistic": the flow should stay below it 90% of the time.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*")  # harmless LightGBM renaming notice


def persistence(df, k):
    """Q(t+k) = Q(t): 'tomorrow looks like today'."""
    return df["flow0"].values


def climatology(df, flow_df, train_range, k, window=15):
    """Mean flow for the target's day of year, from TRAINING years only, smoothed circularly."""
    tr0, tr1 = train_range
    out = np.full(len(df), np.nan)
    f = flow_df.copy()
    f = f[(f["date"] >= tr0) & (f["date"] <= tr1)]
    for sid, g in f.groupby("station"):
        m = g.assign(doy=g["date"].dt.dayofyear.clip(upper=365)).groupby("doy")["flow_m3s"].mean()
        m = m.reindex(range(1, 366)).interpolate(limit_direction="both")
        ext = pd.concat([m.iloc[-window:], m, m.iloc[:window]])
        sm = ext.rolling(window, center=True).mean().iloc[window:-window]
        rows = (df["station"] == sid).values
        tdoy = (pd.to_datetime(df.loc[rows, "date"]) + pd.Timedelta(days=k)).dt.dayofyear.clip(upper=365)
        out[rows] = sm.reindex(tdoy.values).values
    return out


def train_quantile_models(Xtr, ytr, Xva, yva, quantiles, seed, params=None):
    """One LightGBM model per quantile (pinball loss), early-stopped on the validation years."""
    p = dict(n_estimators=2000, learning_rate=0.06, num_leaves=31, min_child_samples=50,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=seed, verbose=-1, n_jobs=-1)
    p.update(params or {})
    models = {}
    for a in quantiles:
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        models[a] = m
    return models


def predict_flow(models, X, meanq):
    """Quantile predictions in m3/s. Sorted across quantiles so the 10-50-90 never cross."""
    qs = sorted(models)
    P = np.column_stack([models[a].predict(X) for a in qs])
    P = np.sort(P, axis=1)
    return np.clip(np.expm1(P), 0, None) * np.asarray(meanq)[:, None], qs
