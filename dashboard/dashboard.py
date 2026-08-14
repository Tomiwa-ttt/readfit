"""
ReadFit dashboard - Streamlit.
OWNER: Innocent Amos Mchechesi (Track 5)

CRITICAL: this reads from the DEPLOYED API over HTTP. Never from a local file.
Course outcome 5 is "connect visualisation tools to DEPLOYED models" - the whole
point is that this box is talking to the GCP box.

Run local:  streamlit run dashboard/dashboard.py
Point at prod: API_URL=http://<GCP_IP>:5000 streamlit run dashboard/dashboard.py
"""
import os
import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:5000")
INK, ACCENT, MUTED = "#1a1c1e", "#2b5fa8", "#9aa4b0"

st.set_page_config(page_title="ReadFit Dashboard", layout="wide")
st.title("ReadFit - Reading Fit Analytics")

api = st.sidebar.text_input("Deployed API endpoint", API)
st.sidebar.caption("Point this at the GCP VM's public IP to prove outcome 5.")


def get(path, **params):
    try:
        r = requests.get(f"{api}{path}", params=params, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.sidebar.error(f"{path}: {e}")
        return None


health = get("/health")
if not health:
    st.error(f"Cannot reach the API at {api}. Is the container running?")
    st.stop()

st.sidebar.success(f"Connected - model {health.get('model_version')}")

# ---------- KPI row ----------
m = get("/metrics") or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Passages analysed", m.get("total_predictions", 0))
c2.metric("Mean readability", m.get("mean_readability", "n/a"))
c3.metric("Mean decodability", f"{m.get('mean_decodability', 0)}%")
c4.metric("Mean FK grade", m.get("mean_fk_grade", "n/a"))

# ---------- Model benchmarking ----------
st.subheader("Model benchmarking")
b = get("/benchmark")
if b:
    bench = pd.DataFrame(b["benchmark"])
    st.caption(f"{b['dataset']} - {b['n_train']} train / {b['n_test']} held out")
    chart = (alt.Chart(bench)
             .mark_bar(cornerRadiusEnd=3)
             .encode(
                 x=alt.X("rmse:Q", title="RMSE (lower is better)"),
                 y=alt.Y("model:N", sort="x", title=None),
                 color=alt.condition(alt.datum.rmse == bench.rmse.min(),
                                     alt.value(ACCENT), alt.value(MUTED)),
                 tooltip=["model", "rmse", "mae", "r2"])
             .properties(height=210))
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(bench, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Hyper-parameter tuning (5-fold CV)**")
        tc = pd.DataFrame(b["tuning_curve"])
        st.altair_chart(alt.Chart(tc).mark_line(point=True, color=ACCENT).encode(
            x=alt.X("alpha:Q", scale=alt.Scale(type="log"), title="Ridge alpha"),
            y=alt.Y("cv_rmse:Q", scale=alt.Scale(zero=False), title="CV RMSE"),
            tooltip=["alpha", "cv_rmse"]).properties(height=240),
            use_container_width=True)
    with right:
        st.markdown("**What drives readability**")
        fi = pd.DataFrame(b["feature_importance"]).head(10)
        st.altair_chart(alt.Chart(fi).mark_bar(cornerRadiusEnd=3, color=ACCENT).encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", "importance"]).properties(height=240),
            use_container_width=True)

# ---------- Live traffic ----------
st.subheader("Live predictions from the deployed model")
h = get("/history", n=200)
if h and h["count"]:
    df = pd.DataFrame(h["rows"])
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    left, right = st.columns(2)
    with left:
        st.markdown("**Readability vs decodability**")
        st.altair_chart(alt.Chart(df).mark_circle(size=90, opacity=.75, color=ACCENT).encode(
            x=alt.X("readability:Q", title="Predicted readability (higher = easier)"),
            y=alt.Y("decodability:Q", title="Decodability %"),
            tooltip=["snippet", "readability", "decodability", "band"]
        ).properties(height=280), use_container_width=True)
    with right:
        st.markdown("**Grade band distribution**")
        bd = pd.DataFrame(m.get("band_distribution", []))
        if not bd.empty:
            st.altair_chart(alt.Chart(bd).mark_bar(cornerRadiusEnd=3, color=ACCENT).encode(
                x=alt.X("count:Q", title="Passages"),
                y=alt.Y("band:N", sort="-x", title=None),
                tooltip=["band", "count"]).properties(height=280),
                use_container_width=True)
    st.dataframe(df[["time", "snippet", "readability", "band", "decodability",
                     "fk_grade", "word_count"]].head(25),
                 use_container_width=True, hide_index=True)
else:
    st.info("No predictions logged yet. Submit a passage on the API to populate this.")

# ---------- Live scoring against prod ----------
st.subheader("Score a passage against the deployed model")
txt = st.text_area("Passage", height=140,
                   value="Tim the cat sat on a mat. A big dog ran at Tim.")
lessons = (get("/lessons") or {}).get("lessons", [])
opts = {l["label"]: l["lesson"] for l in lessons} or {"Lesson 34": 34}
lab = st.selectbox("Phonics lesson", list(opts.keys()))
if st.button("Analyse", type="primary"):
    try:
        r = requests.post(f"{api}/predict",
                          json={"text": txt, "lesson": opts[lab]}, timeout=15).json()
        a, bb, cc = st.columns(3)
        a.metric("Readability", r.get("readability"), r.get("band"))
        bb.metric("Decodability", f"{r.get('decodability')}%")
        cc.metric("FK grade", r.get("fk_grade"))
        if r.get("illegal_words"):
            st.warning("Cannot decode: " +
                       ", ".join(w["word"] for w in r["illegal_words"]))
        else:
            st.success("Fully decodable at this lesson.")
    except Exception as e:
        st.error(str(e))
