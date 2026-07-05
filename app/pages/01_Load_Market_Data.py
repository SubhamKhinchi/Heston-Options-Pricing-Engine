"""
Step 1 — Load Market Data.

Loads a live option chain (services/market_service.load_live_chain) and the
SOFR/OIS discount curve (config/market_config), then stashes the raw chain and
rate curve in session state for the rest of the pipeline. First page in the flow;
feeds Step 2 (Filter Options).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.market_config import get_ois_curve, interpolate_rate, curve_summary
from services.market_service import (
    load_live_chain,
    extract_dividend_yields,
    parse_tickers,
)

st.set_page_config(page_title="Load Market Data", layout="wide")
st.title("Load Market Data")
st.caption("Pull a live option chain from Yahoo Finance. This is the starting point of the pipeline.")

ss = st.session_state


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_ois_curve() -> dict[float, float]:
    return get_ois_curve(force_refresh=True)


# ── SOFR/OIS rate curve (auto-fetched) ────────────────────────────────────────
_sofr_ok = True
try:
    _rate_curve = _cached_ois_curve()
    r = interpolate_rate(_rate_curve, 0.25)
    st.caption(f"**Risk-free (SOFR/OIS)** — {curve_summary(_rate_curve)}")
except Exception:
    _rate_curve = {}
    r = 0.045
    _sofr_ok = False
    st.warning(
        f"⚠️ SOFR/OIS rates unavailable — interest rate switched to "
        f"3M SOFR fallback ({r*100:.2f}%)"
    )

# Always persist the latest rate_curve in session state for downstream pages
ss["rate_curve"] = _rate_curve
ss["r_scalar"] = r   # representative scalar (3M point or fallback)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.subheader("Data source inputs")

tickers_input = st.text_input(
    "Tickers (comma-separated)",
    value=ss.get("fetch_params", {}).get("tickers", "NVDA"),
    help="e.g. NVDA, AAPL, TSLA",
)

pull_clicked = st.button("Pull Options Data", type="primary")

# ── Fetch ─────────────────────────────────────────────────────────────────────
if pull_clicked:
    tickers = parse_tickers(tickers_input)
    with st.spinner(f"Fetching option chain for {', '.join(tickers)} from Yahoo Finance …"):
        try:
            raw_df = load_live_chain(tickers)
        except Exception as exc:
            st.error(f"Fetch failed: {exc}")
            st.stop()

    div_yields = extract_dividend_yields(raw_df)
    # Average q across all tickers for downstream use
    auto_q = float(sum(div_yields.values()) / len(div_yields)) if div_yields else 0.0

    ss["raw_df"] = raw_df
    ss["fetch_params"] = {"tickers": tickers_input, "r": r, "q": auto_q}
    ss["_div_yields"] = div_yields  # per-ticker, for display
    # rate_curve already in ss from top-of-page SOFR fetch; no need to re-store

    # Invalidate all downstream results
    for key in ("filtered_df", "filter_stats", "filter_params", "calibration"):
        ss.pop(key, None)

    st.success(f"Fetched {len(raw_df):,} raw contracts across {raw_df['maturity'].nunique()} expiries.")
    st.rerun()

# ── Display ───────────────────────────────────────────────────────────────────
if "raw_df" not in ss:
    st.info("Set your inputs above and click **Pull Options Data** to begin.")
    st.stop()

raw_df: pd.DataFrame = ss["raw_df"]
params: dict = ss["fetch_params"]

# ── Per-ticker summary ────────────────────────────────────────────────────────
st.subheader("Fetched data summary")

tickers_list = parse_tickers(params["tickers"])

def _first(series_df, col, default=None):
    if col in series_df.columns and not series_df.empty:
        return series_df[col].iloc[0]
    return default


def _forward_summary(tkr_df: pd.DataFrame) -> str:
    """How dividends/carry were sourced across this ticker's expiries."""
    if "dividend_source" not in tkr_df.columns or tkr_df.empty:
        return "—"
    per_exp = tkr_df.groupby("maturity")["dividend_source"].first()
    n_total = len(per_exp)
    n_implied = int((per_exp == "implied_forward").sum())
    if n_implied == 0:
        return "trailing-yield fallback"
    return f"implied forward ({n_implied}/{n_total} expiries)"


ticker_rows = []
for tkr in tickers_list:
    tkr_df = raw_df[raw_df["ticker"] == tkr] if "ticker" in raw_df.columns else raw_df
    spot = tkr_df["spot"].iloc[0] if not tkr_df.empty and "spot" in tkr_df.columns else None
    inst_type = _first(tkr_df, "instrument_type", "—")
    exercise = _first(tkr_df, "ExerciseStyle", "—")
    ticker_rows.append({
        "Ticker": tkr,
        "Type": inst_type,
        "Exercise": str(exercise).capitalize(),
        "Spot": f"${spot:.2f}" if spot is not None else "n/a",
        "r (risk-free)": f"{params['r']*100:.3f}%",
        "Carry / dividends": _forward_summary(tkr_df),
        "Contracts": f"{len(tkr_df):,}",
        "Expiries": tkr_df["maturity"].nunique() if not tkr_df.empty else 0,
    })

summary_df = pd.DataFrame(ticker_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Downstream status (applies to both tabs)
if "filtered_df" in ss:
    st.success(
        f"Downstream: {len(ss['filtered_df']):,} contracts currently filtered. "
        "Pulling new data will clear the filter results."
    )

# Below this T (~1 month) the carry diagnostic q(T) is 1/T-amplified and unstable.
SHORT_T = 0.08

_TYPE_COLORS = {"call": "#2196F3", "put": "#EF5350"}


def _contracts_per_expiry_fig(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(["maturity", "type"]).size().reset_index(name="n")
    fig = px.bar(
        g, x="maturity", y="n", color="type", barmode="group",
        color_discrete_map=_TYPE_COLORS,
        labels={"maturity": "Expiry", "n": "Contracts", "type": ""},
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend_title_text="", xaxis_tickangle=-45)
    return fig


def _oi_by_moneyness_fig(df: pd.DataFrame) -> go.Figure | None:
    if "openInterest" not in df.columns or "moneyness" not in df.columns:
        return None
    sub = df.dropna(subset=["moneyness"]).copy()
    sub["openInterest"] = sub["openInterest"].fillna(0)
    if sub.empty or sub["openInterest"].sum() == 0:
        return None
    fig = px.histogram(
        sub, x="moneyness", y="openInterest", color="type", nbins=60,
        histfunc="sum", color_discrete_map=_TYPE_COLORS,
        labels={"moneyness": "Moneyness  K / S", "openInterest": "Open interest", "type": ""},
    )
    fig.add_vline(x=1.0, line_dash="dot", line_color="#9E9E9E")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend_title_text="", bargap=0.02)
    return fig


def _forward_curve_fig(fc: pd.DataFrame, spot: float, rate_curve: dict) -> go.Figure:
    T = fc["T"].to_numpy(dtype=float)
    fwd = fc["forward"].to_numpy(dtype=float)
    rates = (np.array([interpolate_rate(rate_curve, float(t)) for t in T])
             if rate_curve else np.zeros_like(T))
    financed = spot * np.exp(rates * T)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=financed, name="Financed spot  S·e^(rT)",
                             mode="lines", line=dict(color="#9E9E9E", dash="dash")))
    fig.add_trace(go.Scatter(x=T, y=fwd, name="Implied forward  F(T)",
                             mode="lines+markers", line=dict(color="#2196F3")))
    fig.add_hline(y=spot, line_dash="dot", line_color="#BDBDBD",
                  annotation_text="Spot", annotation_position="bottom right")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      xaxis_title="T (years)", yaxis_title="Price")
    return fig


def _carry_term_fig(fc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["T"], y=fc["implied_q"] * 100, mode="lines+markers",
                             line=dict(color="#7E57C2"), name="Implied carry q(T)"))
    fig.add_hline(y=0, line_dash="dot", line_color="#BDBDBD")
    fig.add_vrect(x0=0, x1=SHORT_T, fillcolor="#FF9800", opacity=0.10, line_width=0,
                  annotation_text="1/T-amplified", annotation_position="top left")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      showlegend=False, xaxis_title="T (years)",
                      yaxis_title="Implied carry q(T)  (%)")
    return fig


tab_raw, tab_fwd = st.tabs(["📄 Raw option chain", "📈 Implied forward curve  F(T)"])

# ── Tab 1: Raw option chain — insights, then the full table ───────────────────
with tab_raw:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total contracts", f"{len(raw_df):,}")
    m2.metric("Total expiries", raw_df["maturity"].nunique())
    if "type" in raw_df.columns:
        m3.metric("Calls / Puts",
                  f"{(raw_df['type']=='call').sum():,} / {(raw_df['type']=='put').sum():,}")
    if "rel_spread" in raw_df.columns:
        med_spread = raw_df["rel_spread"].median()
        m4.metric("Median rel. spread",
                  f"{med_spread*100:.1f}%" if pd.notna(med_spread) else "—")

    st.markdown("**Chain at a glance**")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Contracts per expiry (calls vs puts)")
        st.plotly_chart(_contracts_per_expiry_fig(raw_df), use_container_width=True)
    with g2:
        st.caption("Open interest by moneyness — where liquidity concentrates")
        fig_oi = _oi_by_moneyness_fig(raw_df)
        if fig_oi is not None:
            st.plotly_chart(fig_oi, use_container_width=True)
        else:
            st.info("No open-interest / moneyness data to chart (common outside market hours).")

    st.subheader("Raw option chain")
    _COLS = ["ticker", "instrument_type", "ExerciseStyle", "type", "maturity", "strike", "spot",
             "bid", "ask", "mid_price", "lastPrice",
             "volume", "openInterest", "rel_spread", "T", "moneyness",
             "forward", "dividend_yield", "dividend_source"]
    show_cols = [c for c in _COLS if c in raw_df.columns]
    st.dataframe(raw_df[show_cols], use_container_width=True, hide_index=True)

    with st.expander("Expiry breakdown", expanded=False):
        expiry_table = (
            raw_df.groupby(["ticker", "maturity"])
            .agg(contracts=("strike", "count"), T=("T", "first"))
            .reset_index()
            .sort_values(["ticker", "T"])
        )
        expiry_table["T"] = expiry_table["T"].round(4)
        st.dataframe(expiry_table, use_container_width=True, hide_index=True)

# ── Tab 2: Implied forward curve — the curve, then the table ──────────────────
with tab_fwd:
    st.caption(
        "Dividends/carry enter pricing through the **implied forward F(T)** recovered "
        "per expiry from near-ATM put–call parity — not a single scalar dividend yield. "
        "The implied carry q(T) = r − ln(F/S)/T is shown **only as a diagnostic**, and it "
        "is a *carry* number, not a dividend yield: for a low-dividend name it is dominated "
        "by the financing basis, so small **negative** values are normal (forward sitting "
        "just above S·e^(rT)). It is also amplified by 1/T and unstable at short maturities "
        "— which is exactly why the engine carries F(T), not q."
    )
    for tkr in tickers_list:
        tkr_df = raw_df[raw_df["ticker"] == tkr] if "ticker" in raw_df.columns else raw_df
        if tkr_df.empty or "forward" not in tkr_df.columns:
            continue
        spot = tkr_df["spot"].iloc[0]
        fc = (
            tkr_df.groupby("maturity")
            .agg(
                T=("T", "first"),
                forward=("forward", "first"),
                implied_q=("dividend_yield", "first"),
                source=("dividend_source", "first"),
            )
            .reset_index()
            .sort_values("T")
        )
        if len(tickers_list) > 1:
            st.markdown(f"**{tkr}**")

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.caption("Forward curve vs financed spot — the gap is net carry")
            st.plotly_chart(_forward_curve_fig(fc, spot, _rate_curve), use_container_width=True)
        with fcol2:
            st.caption("Implied carry term structure q(T) [diagnostic]")
            st.plotly_chart(_carry_term_fig(fc), use_container_width=True)

        disp = pd.DataFrame({
            "Maturity": fc["maturity"],
            "T (yrs)": fc["T"].round(4),
            "Forward F(T)": fc["forward"].round(2),
            "Forward factor F/S": (fc["forward"] / spot).round(4),
            "Forward − Spot": (fc["forward"] - spot).round(2),
            "Implied carry q(T) [diagnostic]": (fc["implied_q"] * 100).round(3).map(lambda v: f"{v}%"),
            "Source": fc["source"].astype(str).str.replace("_", " "),
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/02_Filter_Options.py", label="Next: Filter Options →", icon="🔍")
