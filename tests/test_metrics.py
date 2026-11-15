import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodml.evaluate import coverage, flood_events, kge, nse, peak_errors, pod_far_csi
from floodml.features import split_of


def test_nse_kge_perfect_and_mean():
    o = np.array([1.0, 2, 3, 4, 5])
    assert nse(o, o) == 1 and abs(kge(o, o) - 1) < 1e-12
    assert abs(nse(o, np.full(5, o.mean()))) < 1e-12


def test_pod_far_csi():
    obs = [1, 1, 1, 0, 0, 0]
    warn = [1, 1, 0, 1, 0, 0]  # 2 hits, 1 miss, 1 false alarm
    pod, far, csi = pod_far_csi(obs, warn)
    assert abs(pod - 2 / 3) < 1e-9 and abs(far - 1 / 3) < 1e-9 and abs(csi - 0.5) < 1e-9


def test_coverage():
    assert coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 0.5


def test_events_and_peak_timing():
    dates = pd.date_range("2020-01-01", periods=12)
    obs = np.array([1, 1, 5, 9, 6, 1, 1, 1, 1, 1, 1, 1.0])
    pred = np.array([1, 1, 1, 4, 8, 3, 1, 1, 1, 1, 1, 1.0])  # peak one day late, 11% low
    assert flood_events(dates, obs, 4) == [(2, 4)]
    r = peak_errors(dates, obs, pred, 4).iloc[0]
    assert r["timing_err_days"] == 1 and abs(r["peak_err_pct"] - (-100 / 9)) < 1e-6


def test_split_has_no_overlap_gap():
    cfg = {"dates": {"train": ["1981-01-01", "2012-12-31"], "valid": ["2013-01-01", "2016-12-31"],
                     "test": ["2017-01-01", "2023-12-31"]}}
    d = pd.to_datetime(["2012-12-28", "2012-12-29", "2013-01-01", "2017-01-01"])
    assert list(split_of(d, cfg)) == ["train", None, "valid", "test"]
