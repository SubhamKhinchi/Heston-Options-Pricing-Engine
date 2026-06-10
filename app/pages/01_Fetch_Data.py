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

st.caption(
    "Dividends/carry enter pricing through the **implied forward F(T)** recovered "
    "per expiry from near-ATM put–call parity — not a single scalar dividend yield. "
    "The implied yield q(T) = r − ln(F/S)/T is shown in the curve below **only as a "
    "diagnostic**: it is amplified by 1/T and unstable at short maturities, which is "
    "exactly why the engine carries F(T) rather than q."
)

# ── Implied forward curve F(T) ────────────────────────────────────────────────
st.subheader("Implied forward curve  F(T)")
st.caption(
    "The forward factor F/S is the actual carry applied at each maturity "
    "(F/S > 1 ⇒ positive net carry, < 1 ⇒ dividend drag)."
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
    disp = pd.DataFrame({
        "Maturity": fc["maturity"],
        "T (yrs)": fc["T"].round(4),
        "Forward F(T)": fc["forward"].round(2),
        "Forward factor F/S": (fc["forward"] / spot).round(4),
        "Implied q(T) [diagnostic]": (fc["implied_q"] * 100).round(3).map(lambda v: f"{v}%"),
        "Source": fc["source"].astype(str).str.replace("_", " "),
    })
    if len(tickers_list) > 1:
        st.markdown(f"**{tkr}**")
    st.dataframe(disp, use_container_width=True, hide_index=True)

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
_COLS = ["ticker", "instrument_type", "ExerciseStyle", "type", "maturity", "strike", "spot",
         "bid", "ask", "mid_price", "lastPrice",
         "volume", "openInterest", "rel_spread", "T", "moneyness",
         "forward", "dividend_yield", "dividend_source"]
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
