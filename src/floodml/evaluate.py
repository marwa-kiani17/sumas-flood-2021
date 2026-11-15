"""Forecast scores.

NSE (Nash-Sutcliffe): 1 = perfect, 0 = no better than always predicting the mean flow.
KGE (Kling-Gupta): combines correlation, bias and variability; 1 = perfect.
POD / FAR / CSI: for the yes/no warning "flow will exceed the 5-year flood":
  POD = share of real exceedances that were warned, FAR = share of warnings that were false,
  CSI = hits / (hits + misses + false alarms).
"""
import numpy as np
import pandas as pd


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2 + (sim.mean() / obs.mean() - 1) ** 2)


def contingency(obs_event, warn):
    obs_event, warn = np.asarray(obs_event, bool), np.asarray(warn, bool)
    return int((obs_event & warn).sum()), int((~obs_event & warn).sum()), int((obs_event & ~warn).sum())


def pod_far_csi(obs_event, warn):
    hits, false, miss = contingency(obs_event, warn)
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = false / (hits + false) if hits + false else np.nan
    csi = hits / (hits + miss + false) if hits + miss + false else np.nan
    return pod, far, csi


def coverage(obs, lo, hi):
    """Share of observations inside [lo, hi]; ~0.80 expected for a 10-90% range."""
    obs, lo, hi = map(np.asarray, (obs, lo, hi))
    return float(np.mean((obs >= lo) & (obs <= hi)))


def pinball(obs, pred, alpha):
    d = np.asarray(obs) - np.asarray(pred)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def flood_events(dates, obs, thr, gap=3):
    """Group days with obs > thr into events (days closer than `gap` days are merged).
    Returns list of (start_idx, end_idx) into the arrays."""
    idx = np.where(np.asarray(obs) > thr)[0]
    if len(idx) == 0:
        return []
    events, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            events.append((s, prev))
            s = i
        prev = i
    events.append((s, prev))
    return events


def peak_errors(dates, obs, pred, thr, pad=3):
    """Per flood event (obs > thr): peak size error % and timing error (days) of the forecast.
    Forecast peak is searched within +-pad days of the event."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    rows = []
    for a, b in flood_events(dates, obs, thr):
        lo, hi = max(a - pad, 0), min(b + pad + 1, len(obs))
        io = lo + int(np.argmax(obs[lo:hi]))
        ip = lo + int(np.argmax(pred[lo:hi]))
        rows.append(dict(peak_date=pd.Timestamp(np.asarray(dates)[io]), obs_peak=obs[io],
                         pred_peak=pred[ip], peak_err_pct=100 * (pred[ip] - obs[io]) / obs[io],
                         timing_err_days=ip - io))
    return pd.DataFrame(rows)
