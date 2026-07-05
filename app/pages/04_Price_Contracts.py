"""
Step 4 — Price Contracts.

Prices every filtered contract under a chosen calibration's parameters using the
European Heston model (closed form), then adds the de-Americanized market IV, model
IV, BS Greeks, and mispricing metrics — the same European-equivalent basis used for
calibration and the vol surface. Upstream: Step 3 (Calibrate Heston).
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
import streamlit as st

from services.analytics_service import build_chain_analytics
from services.market_service import parse_tickers
from services.pricing_service import HestonParameters

st.set_page_config(page_title="Price Contracts", layout="wide")
st.title("Price Contracts")
st.caption(
    "Apply calibrated Heston parameters to price every filtered contract under the "
    "European Heston model. Adds model price, de-Americanized market IV, model IV, "
    "BS Greeks, and mispricing metrics to the chain."
)

ss = st.session_state

_METHOD_LABELS = {
    "european_proxy": "Characteristic-Function",
}


def _completed_methods(cal: dict) -> dict[str, str]:
    # Prefer the label the calibration page stored (e.g. "Characteristic-Function")
    # so the source dropdown matches; fall back to the static method label.
    return {
        code: cal[code]["meta"].get("method_label", _METHOD_LABELS[code])
        for code in _METHOD_LABELS
        if code in cal and "meta" in cal[code]
    }


# ── Prerequisite checks ───────────────────────────────────────────────────────
if "raw_df" not in ss:
    st.warning("No data loaded. Go to **Load Market Data** first.")
    st.page_link("pages/01_Load_Market_Data.py", label="← Go to Load Market Data", icon="📥")
    st.stop()

if "filtered_df" not in ss or ss["filtered_df"].empty:
    st.warning("No filtered data. Go to **Filter Options** first.")
    st.page_link("pages/02_Filter_Options.py", label="← Go to Filter Options", icon="🔍")
    st.stop()

cal: dict = ss.get("calibration", {})
done = _completed_methods(cal)

if not done:
    st.warning("No calibration results yet. Go to **Calibrate Heston** first.")
    st.page_link("pages/03_Calibrate_Heston.py", label="← Go to Calibrate Heston", icon="⚙️")
    st.stop()

filtered_df: pd.DataFrame = ss["filtered_df"]
params: dict = ss.get("fetch_params", {})
div_yields: dict = ss.get("_div_yields", {})
rate_curve: dict = ss.get("rate_curve", {})
r = ss.get("r_scalar", params.get("r", 0.045))
q = params.get("q", 0.0)

if not rate_curve:
    st.warning(
        f"⚠️ SOFR/OIS rates unavailable — using {r*100:.2f}% flat rate. "
        "Go to Load Market Data and refresh rates to reload the curve."
    )

# Per-ticker caption
for tkr in parse_tickers(params.get("tickers", "NVDA")):
    tkr_df = filtered_df[filtered_df["ticker"] == tkr] if "ticker" in filtered_df.columns else filtered_df
    spot = tkr_df["spot"].iloc[0] if not tkr_df.empty and "spot" in tkr_df.columns else None
    q_val = div_yields.get(tkr, q)
    st.caption(
        f"**{tkr}**: spot {'${:.2f}'.format(spot) if spot else 'n/a'}  |  "
        f"r = {r*100:.3f}%  |  q = {q_val*100:.3f}%  |  "
        f"{len(tkr_df):,} filtered contracts"
    )

# ── Inputs ────────────────────────────────────────────────────────────────────
st.subheader("Pricing inputs")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Calibration parameters**")
    cal_code = st.selectbox(
        "Use parameters from",
        options=list(done.keys()),
        format_func=lambda c: done[c],
        help="Selects which calibration run's Heston parameters are used for pricing.",
    )
    selected_meta = cal[cal_code]["meta"]
    hp = HestonParameters(
        v0=selected_meta["v0"],
        kappa=selected_meta["kappa"],
        theta=selected_meta["theta"],
        sigma=selected_meta["sigma"],
        rho=selected_meta["rho"],
    )

    # Show selected params as a compact table
    feller = 2 * hp.kappa * hp.theta - hp.sigma ** 2
    p_df = pd.DataFrame({
        "Param": ["v₀", "κ", "θ̄", "σ", "ρ", "Feller"],
        "Value": [
            f"{hp.v0:.6f}",
            f"{hp.kappa:.4f}",
            f"{hp.theta:.6f}",
            f"{hp.sigma:.4f}",
            f"{hp.rho:.4f}",
            f"{feller:+.4f} ({'✓' if feller > 0 else '✗'})",
        ],
    })
    st.dataframe(p_df, hide_index=True, use_container_width=True)

with col_right:
    st.markdown("**Pricing**")
    st.caption(
        "Contracts are priced with the **European Heston** closed form — consistent "
        "with the de-Americanized calibration and vol surface. American pricing "
        "(PDE / LSMC) is not used here."
    )
    n_chain = len(filtered_df)
    pricing_limit = st.number_input(
        "Max contracts to price",
        min_value=1, max_value=n_chain,
        value=n_chain,
        step=50,
        help=(
            "Defaults to the full filtered chain. Lower it to price only a subset — "
            "contracts are sorted by ATM distance + spread + volume, so the most "
            "liquid are priced first."
        ),
    )

price_clicked = st.button(
    f"Price Contracts  ({len(filtered_df):,} in chain, limit {int(pricing_limit)})",
    type="primary",
)

# ── Price ─────────────────────────────────────────────────────────────────────
if price_clicked:
    with st.spinner(f"Pricing up to {int(pricing_limit)} contracts (European Heston) …"):
        try:
            # build_chain_analytics prices the European Heston model, de-Americanizes
            # the market quotes, and computes IVs, Greeks, and mispricing metrics.
            analytics_df = build_chain_analytics(
                filtered_df,
                r=r,
                q=q,
                rate_curve=rate_curve,
                heston_params=hp,
                compute_model_prices=True,
                pricing_limit=int(pricing_limit),
            )

            ss["analytics_df"] = analytics_df
            ss["pricing_params"] = {
                "cal_method": cal_code,
                "pricing_limit": int(pricing_limit),
            }

            n_priced = int(analytics_df["model_price"].notna().sum()) if "model_price" in analytics_df.columns else 0
            st.success(
                f"Priced {n_priced:,} of {len(filtered_df):,} contracts "
                f"using **{done[cal_code]}** params (European Heston)."
            )
        except Exception as exc:
            st.error(f"Pricing failed: {exc}")
            st.stop()

    st.rerun()

# ── Display results ───────────────────────────────────────────────────────────
if "analytics_df" not in ss:
    st.info(
        "Configure inputs above and click **Price Contracts** to run pricing. "
        "This may take a while for large contract sets or slow pricing methods."
    )
    st.stop()

analytics_df: pd.DataFrame = ss["analytics_df"]
pp: dict = ss.get("pricing_params", {})

st.caption(
    f"Priced with: **{done.get(pp.get('cal_method', cal_code), '?')} params**  |  "
    f"Model: **European Heston**  |  "
    f"Limit: {pp.get('pricing_limit', '?')}"
)

# ── Summary metrics ───────────────────────────────────────────────────────────
n_total   = len(analytics_df)
n_priced  = int(analytics_df["model_price"].notna().sum()) if "model_price" in analytics_df.columns else 0
n_nan     = n_total - n_priced

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total contracts", f"{n_total:,}")
m2.metric("Model priced", f"{n_priced:,}")
m3.metric("Not priced (NaN)", f"{n_nan:,}")

if "price_error" in analytics_df.columns:
    avg_abs_err = analytics_df["price_error"].abs().mean()
    m4.metric("Avg |price error|", f"${avg_abs_err:.4f}" if pd.notna(avg_abs_err) else "n/a")
if "iv_error" in analytics_df.columns:
    avg_iv_err = analytics_df["iv_error"].abs().mean()
    m5.metric("Avg |IV error|", f"{avg_iv_err*100:.3f}%" if pd.notna(avg_iv_err) else "n/a")

# ── Enriched contract table ───────────────────────────────────────────────────
st.subheader("Priced option chain")

_DISPLAY_COLS = [
    "ticker", "type", "maturity", "strike", "spot",
    "mid_price", "model_price", "price_error", "relative_price_error",
    "market_iv", "model_iv", "iv_error",
    "market_delta", "market_gamma", "market_vega",
    "model_delta",  "model_gamma",  "model_vega",
    "liquidity_score", "mispricing_score", "mispricing_bias",
    "moneyness", "T",
]
show_cols = [c for c in _DISPLAY_COLS if c in analytics_df.columns]

# Column selector
selected_cols = st.multiselect(
    "Visible columns",
    options=show_cols,
    default=show_cols,
    key="pricing_col_selector",
)

# Highlight rows with large pricing errors
def _highlight_error(row: pd.Series):
    err = row.get("price_error", 0)
    if pd.isna(err):
        return ["background-color: #3a3a3a"] * len(row)
    if abs(err) > 1.0:
        return ["background-color: #4a1a1a"] * len(row)
    return [""] * len(row)

if selected_cols:
    st.dataframe(
        analytics_df[selected_cols],
        use_container_width=True,
        hide_index=True,
    )

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
csv = analytics_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download pricing results (CSV)",
    data=csv,
    file_name="heston_pricing_results.csv",
    mime="text/csv",
)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/03_Calibrate_Heston.py", label="← Back to Calibrate Heston", icon="⚙️")
with col_fwd:
    if "analytics_df" in ss:
        st.page_link("pages/05_Volatility_Surface.py", label="Next: Volatility Surface →", icon="📈")
