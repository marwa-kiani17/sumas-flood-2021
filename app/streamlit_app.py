"""Small Streamlit viewer for the Sumas Prairie 2021 project (reads the saved figures and tables).

Install once (inside the floodml env):   pip install streamlit
Run:                                     streamlit run app/streamlit_app.py
Nothing is recomputed here; it only shows outputs of scripts 01-09.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FIG, REP, PROC, INTERIM = ROOT / "reports" / "figures", ROOT / "reports", ROOT / "data" / "processed", ROOT / "data" / "interim"

st.set_page_config(page_title="Sumas Prairie 2021", layout="wide")


def show_image(path, caption=None):
    if Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing file: {Path(path).name} (run the matching script first)")


def show_table(path, **kw):
    if Path(path).exists():
        st.dataframe(pd.read_csv(path, **kw), use_container_width=True, hide_index=True)
    else:
        st.info(f"Missing file: {Path(path).name}")


names = {}
if (INTERIM / "stations.csv").exists():
    names = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str}).set_index("id")["name"].to_dict()

page = st.sidebar.radio("Page", ["Overview", "Nov 2021 forecast replay", "Flood maps", "Flood levels",
                                 "Model scores", "Limitations"])

if page == "Overview":
    st.title("Mapping and Forecasting the November 2021 BC Floods")
    st.write("An atmospheric river hit southern British Columbia on 14-15 Nov 2021. The Nooksack River overflowed "
             "into the Sumas River basin and flooded Sumas Prairie. This project asks two questions: "
             "**(1)** could a machine-learning model have warned a few days ahead, and "
             "**(2)** where was the water, as seen by satellite radar?")
    c1, c2 = st.columns(2)
    with c1:
        show_image(FIG / "hydrographs.png", "Daily river flow, Nov 2021 in red")
    with c2:
        show_image(FIG / "sumas_flood_maps.png", "Sumas Prairie before / during / after")

elif page == "Nov 2021 forecast replay":
    st.header("Replay: what the model would have forecast")
    st.write("Each dot is a forecast issued on an earlier day. The blue band is the 10-90% range. "
             "A warning fires when the top of the band passes the 5-year flood level (Q5).")
    ids = sorted(p.stem.replace("nov2021_replay_", "") for p in FIG.glob("nov2021_replay_*.png") if "_linear" not in p.stem)
    if ids:
        sid = st.selectbox("River", ids, format_func=lambda i: f"{i}  {names.get(i, '')}")
        variant = st.radio("Model", ["Standard LightGBM", "Linear-leaf LightGBM"], horizontal=True)
        suffix = "" if variant.startswith("Standard") else "_linear"
        show_image(FIG / f"nov2021_replay_{sid}{suffix}.png")
        show_image(FIG / f"shap_nov2021_{sid}.png", "Why the model forecast the peak (SHAP)")
    st.subheader("First day the warning fired")
    show_table(REP / "tables_nov2021_warning.csv", dtype={"river": str})

elif page == "Flood maps":
    st.header("Sumas Prairie flood maps (Sentinel-1 radar + U-Net)")
    show_image(FIG / "sumas_flood_maps.png")
    st.subheader("Flooded area by date")
    show_table(PROC / "sumas_area.csv")
    show_image(FIG / "flooded_area_timeseries.png")
    show_image(FIG / "flooded_area_vs_level.png")
    st.caption("Descriptive only: 8-9 satellite dates, not a hydraulic model.")

elif page == "Flood levels":
    st.header("How rare was Nov 2021?")
    show_image(FIG / "return_periods.png")
    show_table(PROC / "thresholds.csv", dtype={"id": str})
    st.caption("Return periods above 100 years come from a 21-32 year fit and are only indicative.")

elif page == "Model scores":
    st.header("Scores")
    st.subheader("Forecasting (test years 2017-2023), per river")
    show_table(REP / "tables_skill_per_river.csv", dtype={"station": str})
    st.subheader("Warnings and 10-90% coverage")
    show_table(REP / "tables_warning.csv")
    st.subheader("Water mapping (Sen1Floods11 test chips)")
    show_table(REP / "tables_cv.csv")
    seeds = [p for p in REP.glob("tables_cv_seed*.csv")]
    if seeds:
        st.caption("Other training seeds:")
        for p in sorted(seeds):
            st.write(p.stem)
            show_table(p)
    guide = ROOT / "docs" / "METRICS_EXPLAINED.md"
    if guide.exists():
        with st.expander("What do these scores mean? (plain words)"):
            st.markdown(guide.read_text(encoding="utf-8"))

else:
    st.header("Limitations")
    st.markdown("""
- Weather is one ERA5 point per basin and has no atmospheric-river measure (IVT); a rain + warmth proxy is used.
- "Future weather" is observed reanalysis (a perfect forecast); real forecasts would be less accurate.
- Tree models under-predict record peaks such as Nov 2021.
- The U-Net is trained on global flood chips and under-maps flooded fields on Sumas Prairie.
- Flooded area vs level rests on a few satellite dates.
""")
