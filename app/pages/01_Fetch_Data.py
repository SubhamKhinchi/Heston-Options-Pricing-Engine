from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import streamlit as st

from config.market_config import get_ois_curve, interpolate_rate, curve_summary
from services.market_service import (
    load_live_chain,
    extract_dividend_yields,
    parse_tickers,
)

st.set_page_config(page_title="Fetch Options Data", layout="wide")
st.title("Step 1 — Fetch Options Data")
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
div_yields: dict = ss.get("_div_yields", {})

# ── Per-ticker summary ────────────────────────────────────────────────────────
st.subheader("Fetched data summary")

tickers_list = parse_tickers(params["tickers"])

ticker_rows = []
for tkr in tickers_list:
    tkr_df = raw_df[raw_df["ticker"] == tkr] if "ticker" in raw_df.columns else raw_df
    spot = tkr_df["spot"].iloc[0] if not tkr_df.empty and "spot" in tkr_df.columns else None
    q_val = div_yields.get(tkr, params.get("q", 0.0))
    ticker_rows.append({
        "Ticker": tkr,
        "Spot": f"${spot:.2f}" if spot is not None else "n/a",
        "Div yield (q)": f"{q_val*100:.3f}%",
        "r (risk-free)": f"{params['r']*100:.3f}%",
        "Contracts": f"{len(tkr_df):,}",
        "Expiries": tkr_df["maturity"].nunique() if not tkr_df.empty else 0,
    })

summary_df = pd.DataFrame(ticker_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Small caption line matching user's requested style
caption_parts = []
for row in ticker_rows:
    caption_parts.append(
        f"**{row['Ticker']}**: spot {row['Spot']}  |  r = {params['r']*100:.3f}%  |  q = {row['Div yield (q)']}"
    )
for part in caption_parts:
    st.caption(part)

# Total contracts metric
c1, c2 = st.columns(2)
c1.metric("Total contracts", f"{len(raw_df):,}")
c2.metric("Total expiries", raw_df["maturity"].nunique())

# Downstream status
if "filtered_df" in ss:
    st.success(
        f"Downstream: {len(ss['filtered_df']):,} contracts currently filtered. "
        "Pulling new data will clear the filter results."
    )

# Raw chain table
st.subheader("Raw option chain")
_COLS = ["ticker", "type", "maturity", "strike", "spot",
         "bid", "ask", "mid_price", "lastPrice",
         "volume", "openInterest", "rel_spread", "T", "moneyness"]
show_cols = [c for c in _COLS if c in raw_df.columns]
st.dataframe(raw_df[show_cols], use_container_width=True, hide_index=True)

# Expiry breakdown
with st.expander("Expiry breakdown", expanded=False):
    expiry_table = (
        raw_df.groupby(["ticker", "maturity"])
        .agg(contracts=("strike", "count"), T=("T", "first"))
        .reset_index()
        .sort_values(["ticker", "T"])
    )
    expiry_table["T"] = expiry_table["T"].round(4)
    st.dataframe(expiry_table, use_container_width=True, hide_index=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/02_Filter_Options.py", label="Next: Filter Options →", icon="🔍")
