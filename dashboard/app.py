"""
LLM Inference Pipeline — Observability Dashboard
Real-time metrics: latency (P50/P95/P99), throughput, batch sizes.
Run: streamlit run dashboard/app.py
"""

import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
from datetime import datetime

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_BASE      = "http://localhost:5000"
REFRESH_SEC   = 2
HISTORY_LEN   = 60   # data points to keep per series

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Inference Monitor",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  code, .metric-label { font-family: 'JetBrains Mono', monospace; }

  .stMetric { background: #0f1117; border: 1px solid #1e2130;
              border-radius: 8px; padding: 16px; }
  .stMetric label { color: #7c85a2 !important; font-size: 11px !important;
                    text-transform: uppercase; letter-spacing: 1px; }
  .stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important;
                                             font-size: 28px !important; }

  h1 { color: #38bdf8 !important; font-weight: 700; letter-spacing: -1px; }
  h2, h3 { color: #94a3b8 !important; font-weight: 500; }

  .status-ok   { color: #34d399; font-weight: 700; }
  .status-down { color: #f87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state — rolling history
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = {
        "timestamps":   deque(maxlen=HISTORY_LEN),
        "p50":          deque(maxlen=HISTORY_LEN),
        "p95":          deque(maxlen=HISTORY_LEN),
        "p99":          deque(maxlen=HISTORY_LEN),
        "rps":          deque(maxlen=HISTORY_LEN),
        "avg_batch":    deque(maxlen=HISTORY_LEN),
    }

hist = st.session_state.history


# ──────────────────────────────────────────────
# Fetch helpers
# ──────────────────────────────────────────────
def fetch_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=2)
        return r.json() if r.ok else None
    except Exception:
        return None


def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.ok
    except Exception:
        return False


# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────
st.title("⚡ LLM Inference Pipeline")
st.caption("Real-time observability · GPT-2 · Async Batching · Mixed Precision")

placeholder = st.empty()

while True:
    m      = fetch_metrics()
    online = fetch_health()

    if m:
        ts = datetime.now().strftime("%H:%M:%S")
        hist["timestamps"].append(ts)
        hist["p50"].append(m.get("latency_p50_ms", 0))
        hist["p95"].append(m.get("latency_p95_ms", 0))
        hist["p99"].append(m.get("latency_p99_ms", 0))
        hist["rps"].append(m.get("requests_per_sec", 0))
        hist["avg_batch"].append(m.get("avg_batch_size", 0))

    with placeholder.container():

        # ── Status bar ──────────────────────────
        status = (
            '<span class="status-ok">● ONLINE</span>'
            if online else
            '<span class="status-down">● OFFLINE — start the API first</span>'
        )
        st.markdown(f"**API Status:** {status}", unsafe_allow_html=True)
        st.divider()

        if not m:
            st.warning("Waiting for API data… make sure `python -m app.api` is running.")
            time.sleep(REFRESH_SEC)
            continue

        # ── KPI row ─────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Requests",  f"{m['total_requests']:,}")
        c2.metric("Req / sec",       f"{m['requests_per_sec']}")
        c3.metric("Latency P50",     f"{m['latency_p50_ms']} ms")
        c4.metric("Latency P95",     f"{m['latency_p95_ms']} ms")
        c5.metric("Latency P99",     f"{m['latency_p99_ms']} ms")
        c6.metric("Errors",          f"{m['total_errors']}")

        st.markdown("###")

        # ── Latency chart ───────────────────────
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Latency over time (ms)")
            if len(hist["timestamps"]) > 1:
                df = pd.DataFrame({
                    "time": list(hist["timestamps"]),
                    "P50":  list(hist["p50"]),
                    "P95":  list(hist["p95"]),
                    "P99":  list(hist["p99"]),
                })
                fig = go.Figure()
                for col, color in [("P50","#38bdf8"), ("P95","#fb923c"), ("P99","#f87171")]:
                    fig.add_trace(go.Scatter(
                        x=df["time"], y=df[col],
                        name=col, mode="lines",
                        line=dict(color=color, width=2),
                        fill="tozeroy",
                        fillcolor="rgba(56,189,248,0.08)",
                    ))
                fig.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="#94a3b8", legend=dict(orientation="h"),
                    margin=dict(l=0, r=0, t=10, b=0), height=280,
                    xaxis=dict(showgrid=False, color="#334155"),
                    yaxis=dict(gridcolor="#1e2130", color="#334155"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sending requests to populate the chart…")

        with col_right:
            st.subheader("Throughput (req/s)")
            if len(hist["rps"]) > 1:
                df2 = pd.DataFrame({"time": list(hist["timestamps"]), "rps": list(hist["rps"])})
                fig2 = px.area(df2, x="time", y="rps",
                               color_discrete_sequence=["#34d399"])
                fig2.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="#94a3b8", showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0), height=280,
                    xaxis=dict(showgrid=False, color="#334155"),
                    yaxis=dict(gridcolor="#1e2130", color="#334155"),
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── Stats row ───────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Batch Size",   f"{m['avg_batch_size']}")
        s2.metric("Latency Min",      f"{m['latency_min_ms']} ms")
        s3.metric("Latency Avg",      f"{m['latency_avg_ms']} ms")
        s4.metric("Uptime",           f"{m['uptime_seconds']} s")

        st.caption(f"Auto-refreshing every {REFRESH_SEC}s · window = last {m['window_samples']} samples")

    time.sleep(REFRESH_SEC)
