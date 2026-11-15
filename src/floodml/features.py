"""Feature table for river forecasting. Every feature on issue day t only uses information
available on day t (past flow/weather), except the optional "future weather" columns
fw_* which are a perfect forecast of days t+1..t+3 (ERA5 reanalysis, an upper bound on skill).

Hydrology terms:
- Rain-on-snow: warm rain falling on snow melts it and adds extra water -> bigger floods.
- AR-like storm: proxy for an atmospheric river (heavy, warm 3-day rain in Oct-Feb).
- Antecedent wetness: wet soil/high flow before a storm makes the river respond more.
"""
import numpy as np
import pandas as pd

LEADS = [1, 2, 3]
STORM_FEATS = ["ros", "ros_3d", "ar_flag", "ar_rain3"]  # removed in the ablation run
REGIME = {"rain": 0, "mixed": 1, "snow": 2}


def station_features(flow, wx, meta, ar_cfg, meanq):
    """One river -> DataFrame indexed by issue date t.

    flow: Series (m3/s) daily; wx: daily weather DataFrame; meanq: mean flow on TRAINING years only
    (targets and flow features are log1p(Q/meanq) so all rivers can share one model).
    """
    d = pd.DataFrame({"flow": flow}).join(wx, how="left")
    q = np.log1p(d["flow"] / meanq)
    X = pd.DataFrame(index=d.index)
    for lag in range(8):
        X[f"q_lag{lag}"] = q.shift(lag)
    X["q_mean7"] = q.rolling(7).mean()

    rain, snow, tm = d["rain_sum"], d["snowfall_sum"], d["temperature_2m_mean"]
    X["rain"] = rain
    for n in (3, 7, 30):
        X[f"rain_{n}d"] = rain.rolling(n).sum()
    X["snowfall"] = snow
    X["snowfall_7d"] = snow.rolling(7).sum()
    X["t_mean"], X["t_max"] = tm, d["temperature_2m_max"]
    X["degday_7d"] = tm.clip(lower=0).rolling(7).sum()      # warmth available to melt snow
    X["snow_depth"] = d["snow_depth"]
    X["snow_depth_chg7"] = d["snow_depth"].diff(7)           # negative = melting
    X["soil_0_7"] = d["soil_moisture_0_to_7cm"]

    ros = rain * ((d["snow_depth"] > 0.05) & (tm > 0))
    X["ros"], X["ros_3d"] = ros, ros.rolling(3).sum()
    flag = (X["rain_3d"] >= ar_cfg["rain_3day_mm"]) & (tm >= ar_cfg["min_temp_c"]) & d.index.month.isin(ar_cfg["months"])
    X["ar_flag"], X["ar_rain3"] = flag.astype(float), X["rain_3d"] * flag

    doy = d.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["log_area"], X["regime"], X["log_meanq"] = np.log(meta["area_km2"]), REGIME[meta["regime"]], np.log(meanq)

    for j in LEADS:                                          # perfect-forecast weather
        X[f"fw_rain_{j}"], X[f"fw_t_{j}"] = rain.shift(-j), tm.shift(-j)
        X[f"y{j}"] = q.shift(-j)                              # target (log scale)
        X[f"obs{j}"] = d["flow"].shift(-j)                    # target (m3/s)
    X["flow0"] = d["flow"]
    X["meanq"] = meanq
    return X


def build_all(flow_df, wx_df, stations, cfg):
    """All rivers stacked. Drops rows with a missing flow lag or target (long gaps)."""
    tr0, tr1 = cfg["dates"]["train"]
    out = []
    for _, s in stations.iterrows():
        sid = s["id"]
        f = flow_df[flow_df["station"] == sid].set_index("date")["flow_m3s"]
        w = wx_df[wx_df["station"] == sid].set_index("date").drop(columns="station")
        meanq = f.loc[tr0:tr1].mean()                         # training years only -> no leakage
        X = station_features(f, w, s, cfg["features"]["ar_storm"], meanq)
        X.insert(0, "station", sid)
        out.append(X)
    X = pd.concat(out)
    X.index.name = "date"
    need = [f"q_lag{l}" for l in range(8)] + [f"y{j}" for j in LEADS]
    return X.dropna(subset=need).reset_index()


def split_of(dates, cfg):
    """'train' / 'valid' / 'test' / None by issue date. The last 3 days of train and valid are
    dropped so their targets never reach into the next split."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    lab = np.full(len(d), None, dtype=object)
    for name, (a, b) in (("train", cfg["dates"]["train"]), ("valid", cfg["dates"]["valid"]),
                         ("test", cfg["dates"]["test"])):
        end = pd.Timestamp(b) - (pd.Timedelta(days=3) if name != "test" else pd.Timedelta(0))
        lab[(d >= pd.Timestamp(a)) & (d <= end)] = name
    return lab


def feature_columns(df, k, future=True, storm=True):
    """Columns used for lead time k. Future weather is limited to days t+1..t+k."""
    drop_prefix = ("y", "obs", "station", "date", "flow0", "meanq", "fw_", "split")
    cols = [c for c in df.columns if not c.startswith(drop_prefix)]
    if future:
        cols += [f"fw_{v}_{j}" for j in range(1, k + 1) for v in ("rain", "t")]
    if not storm:
        cols = [c for c in cols if c not in STORM_FEATS]
    return cols
