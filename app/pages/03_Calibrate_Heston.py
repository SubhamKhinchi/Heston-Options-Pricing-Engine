"""
Step 3 — Calibrate Heston.

Fits the five Heston parameters to the filtered chain via the single
characteristic-function method (services/calibration_service.calibrate_option_chain):
select the OTM, near-ATM calibration universe, then vega-weighted Levenberg-Marquardt
over the fast CF pricer. Quotes are already de-Americanized upstream (Step 2 Filter).
Stashes the result for downstream pricing. Upstream: Step 2.
Downstream: Step 4 (Price Contracts) and the surface/screener pages.
"""

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

from services.calibration_service import (
    calibrate_option_chain,
    DEFAULT_BOUNDS,
    DEFAULT_INITIAL_GUESS,
)
from services.market_service import parse_tickers
from services.pricing_service import HestonParameters

st.set_page_config(page_title="Calibrate Heston", layout="wide")
st.title("Step 3 — Calibrate Heston Model")
st.caption(
    "Fit the five Heston parameters (v₀, κ, θ̄, σ, ρ) to the de-Americanized option "
    "chain using the Levenberg-Marquardt optimiser with the analytic Cui et al. (2016) "
    "Jacobian. Set the initial guess and search bounds below."
)

ss = st.session_state

# ── Prerequisite checks ───────────────────────────────────────────────────────
if "raw_df" not in ss:
    st.warning("No data loaded. Go to **Fetch Data** first.")
    st.page_link("pages/01_Fetch_Data.py", label="← Go to Fetch Data", icon="📥")
    st.stop()

if "filtered_df" not in ss:
    st.warning("No filtered data. Go to **Filter Options** first.")
    st.page_link("pages/02_Filter_Options.py", label="← Go to Filter Options", icon="🔍")
    st.stop()

filtered_df: pd.DataFrame = ss["filtered_df"]
params: dict = ss.get("fetch_params", {})
rate_curve: dict = ss.get("rate_curve", {})
r = ss.get("r_scalar", params.get("r", 0.045))
q = params.get("q", 0.0)

if not rate_curve:
    st.warning(
        f"⚠️ SOFR/OIS rates unavailable — using {r*100:.2f}% flat rate. "
        "Go to Fetch Data and refresh rates to reload the curve."
    )

if filtered_df.empty:
    st.error(
        "Filtered dataset is empty. Go back to **Filter Options** and relax the filters."
    )
    st.page_link("pages/02_Filter_Options.py", label="← Back to Filter Options", icon="🔍")
    st.stop()

# Per-ticker status caption
for tkr in parse_tickers(params.get("tickers", "NVDA")):
    tkr_df = filtered_df[filtered_df["ticker"] == tkr] if "ticker" in filtered_df.columns else filtered_df
    spot = tkr_df["spot"].iloc[0] if not tkr_df.empty and "spot" in tkr_df.columns else None
    st.caption(
        f"**{tkr}**: spot {'${:.2f}'.format(spot) if spot else 'n/a'}  |  "
        f"r = {r*100:.3f}%  |  carry via implied forward F(T)  |  "
        f"{len(tkr_df):,} filtered contracts"
    )

# ── Calibration universe ──────────────────────────────────────────────────────
st.subheader("Calibration universe")
st.caption(
    "The calibration set is deliberately *tighter* than the filtered (pricing) chain — "
    "**calibrate tight, price broad**. We fit the **out-of-the-money** leg per strike off "
    "the implied forward F (OTM put for K<F, OTM call for K>F), vega-weighted. Tune the "
    "selection below; pricing/screening still use the full filtered chain."
)
uni_c1, uni_c2, uni_c3 = st.columns(3)
with uni_c1:
    mny_lo = st.number_input(
        "Forward-moneyness K/F — low", min_value=0.50, max_value=1.00, value=0.85, step=0.01,
        help="Keep strikes with K/F ≥ this. Tighter = more ATM-concentrated.",
    )
    mny_hi = st.number_input(
        "Forward-moneyness K/F — high", min_value=1.00, max_value=1.50, value=1.15, step=0.01,
        help="Keep strikes with K/F ≤ this.",
    )
with uni_c2:
    min_oi = st.number_input(
        "Min open interest", min_value=0, max_value=100000, value=0, step=100,
        help="Calibration-specific OI floor (on top of the filter). OI is the liquidity proxy.",
    )
    atm_band = st.number_input(
        "ATM zone band  |ln(K/F)|", min_value=0.0, max_value=0.20, value=0.02, step=0.01,
        help="Within this band a strike is ATM; the marginally-OTM leg is kept.",
    )
with uni_c3:
    min_mat_days = st.number_input(
        "Min days to expiry", min_value=0, max_value=120, value=7, step=1,
        help="Drop very short-dated contracts (microstructure noise).",
    )
    max_mat_years = st.number_input(
        "Max years to expiry  (0 = no cap)", min_value=0.0, max_value=5.0, value=0.0, step=0.25,
        help="Drop long-dated / stale LEAPS, e.g. 1.5.",
    )

exp_c1, exp_c2 = st.columns(2)
with exp_c1:
    max_expiries = st.number_input(
        "Max expiries to use  (0 = all)", min_value=0, max_value=50, value=0, step=1,
        help="Limit to the nearest N expiries to speed up calibration.",
    )
with exp_c2:
    contracts_per_expiry = st.number_input(
        "Max contracts per expiry  (0 = all)", min_value=0, max_value=100, value=0, step=2,
        help="Cap near-ATM contracts per expiry so no single dense expiry dominates the fit.",
    )

max_exp = int(max_expiries) if max_expiries > 0 else None
cpe = int(contracts_per_expiry) if contracts_per_expiry > 0 else None
min_maturity = float(min_mat_days) / 365.0
max_maturity = float(max_mat_years) if max_mat_years > 0 else None

# Parameter metadata and the global feasible box (mirrors calibrate_heston._DEFAULT_BOUNDS).
_PARAM_META = [
    ("v₀", "init variance"),
    ("κ",  "mean-reversion speed"),
    ("θ̄",  "long-run variance"),
    ("σ",  "vol of vol"),
    ("ρ",  "spot-vol correlation"),
]
_GLOBAL_LB = [1e-4, 1e-4, 1e-4, 1e-4, -0.999]
_GLOBAL_UB = [2.0, 10.0, 2.0, 5.0, 0.999]


def _params_from_editor(edited_df: pd.DataFrame) -> tuple[HestonParameters, list[tuple[float, float]]]:
    """Read the shared editable table into an initial guess + bounds list.

    Clamps each row to the global feasible box and enforces lower ≤ guess ≤ upper.
    """
    guess_vals: list[float] = []
    bounds_list: list[tuple[float, float]] = []
    for i in range(5):
        g = float(edited_df["Initial guess"].iloc[i])
        lo = max(float(edited_df["Lower bound"].iloc[i]), _GLOBAL_LB[i])
        hi = min(float(edited_df["Upper bound"].iloc[i]), _GLOBAL_UB[i])
        if lo > hi:
            lo, hi = hi, lo
        g = min(max(g, lo), hi)
        guess_vals.append(g)
        bounds_list.append((lo, hi))
    return HestonParameters.from_iterable(guess_vals), bounds_list


def _run_calibration(
    method_code: str,
    guess: HestonParameters,
    bounds: list[tuple[float, float]],
    Ns: int, Nv: int, Nt: int,
):
    return calibrate_option_chain(
        filtered_df,
        r=r, q=q,
        rate_curve=rate_curve,
        initial_guess=guess,
        bounds=bounds,
        Ns=Ns, Nv=Nv, Nt=Nt,
        max_expiries=max_exp,
        contracts_per_expiry=cpe,
        american_method=method_code,
        mny_lo=mny_lo,
        mny_hi=mny_hi,
        min_open_interest=int(min_oi),
        min_maturity=min_maturity,
        max_maturity=max_maturity,
        atm_band=atm_band,
    )


# ── Initial parameters & search bounds (user-set; seeded from Cui et al. 2016) ──
st.divider()
st.subheader("Initial parameters & search bounds")
st.caption(
    "Defaults are the fixed ranges from **Cui et al. (2016), Table 5** — the paper "
    "calibrates *without presuming parameter values*. **Edit any cell** before "
    "calibrating; the search box is entirely user-controlled. "
    "Note: Table 5's σ ceiling (0.95) suits the paper's validation set — for "
    "high vol-of-vol single names (e.g. NVDA) raise the σ upper bound."
)

_ig = DEFAULT_INITIAL_GUESS.as_tuple()
_bnds = DEFAULT_BOUNDS

_editor_seed = pd.DataFrame({
    "Parameter":     [m[0] for m in _PARAM_META],
    "Initial guess": [round(float(_ig[i]), 4) for i in range(5)],
    "Lower bound":   [round(float(_bnds[i][0]), 4) for i in range(5)],
    "Upper bound":   [round(float(_bnds[i][1]), 4) for i in range(5)],
    "Meaning":       [m[1] for m in _PARAM_META],
})
edited_params = st.data_editor(
    _editor_seed,
    key="param_editor",
    hide_index=True,
    use_container_width=True,
    disabled=["Parameter", "Meaning"],
    column_config={
        "Initial guess": st.column_config.NumberColumn(format="%.4f", step=0.01),
        "Lower bound":   st.column_config.NumberColumn(format="%.4f", step=0.01),
        "Upper bound":   st.column_config.NumberColumn(format="%.4f", step=0.01),
    },
)
shared_guess, shared_bounds = _params_from_editor(edited_params)

def _render_result(meta: dict, cal_df: pd.DataFrame | None) -> None:
    """Render calibration result inside a column."""
    feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
    color = "green" if feller > 0 else "red"

    res_df = pd.DataFrame({
        "Param": ["v₀", "κ", "θ̄", "σ", "ρ"],
        "Value": [
            f"{meta['v0']:.6f}",
            f"{meta['kappa']:.4f}",
            f"{meta['theta']:.6f}",
            f"{meta['sigma']:.4f}",
            f"{meta['rho']:.4f}",
        ],
        "Implied": [
            f"init vol {meta['v0']**0.5*100:.2f}%",
            "mean-reversion speed",
            f"long-run vol {meta['theta']**0.5*100:.2f}%",
            "vol of vol",
            "spot-vol correlation",
        ],
    })
    st.dataframe(res_df, hide_index=True, use_container_width=True)
    st.markdown(
        f"**Feller** 2κθ−σ²: :{color}[{feller:+.4f}  "
        f"({'satisfied ✓' if feller > 0 else 'violated ✗'})]"
    )
    st.caption(
        f"Loss: {meta['loss']:.4e}  |  "
        f"Contracts used: {int(meta['contract_count'])}  |  "
        f"Runtime: {meta['runtime_seconds']:.1f}s"
    )
    if cal_df is not None and not cal_df.empty:
        with st.expander("Calibration universe", expanded=False):
            # When de-Americanized, show the raw American quote alongside the
            # European-equivalent target and the de-Americanized IV (σ*).
            show_cols = [c for c in ["maturity", "type", "strike", "T", "moneyness",
                                      "mid_price_market", "mid_price", "deam_iv", "market_iv"]
                         if c in cal_df.columns]
            display_df = cal_df[show_cols].sort_values(["maturity", "strike"])
            if "mid_price_market" in show_cols:
                display_df = display_df.rename(columns={
                    "mid_price_market": "amer_quote",
                    "mid_price": "euro_equiv",
                    "deam_iv": "deam_iv (σ*)",
                    "market_iv": "euro_iv",
                })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

# ── Calibration ───────────────────────────────────────────────────────────────
st.divider()
cal: dict = ss.get("calibration", {})

col_main, _spacer = st.columns([3, 2])
with col_main:
    st.markdown("### Characteristic-Function Calibration")
    st.caption(
        "Quotes are de-Americanized to European-equivalent prices upstream (in the "
        "**Filter** step, via a Black-Scholes binomial tree that strips the early-exercise "
        "premium), so the fit targets the European-equivalent surface directly. The five "
        "Heston parameters are then fit with the fast characteristic-function pricer and "
        "the Levenberg-Marquardt optimiser (Cui et al. 2016), vega-weighted — no American "
        "pricer in the loop."
    )

    if st.button("Calibrate", type="primary", key="btn_ep"):
        with st.spinner("Calibrating …"):
            try:
                result, cal_df = _run_calibration(
                    "european_proxy", shared_guess, shared_bounds, Ns=40, Nv=20, Nt=40,
                )
                meta = result.as_dict()
                meta["method_label"] = "Characteristic-Function"
                if "calibration" not in ss:
                    ss["calibration"] = {}
                ss["calibration"]["european_proxy"] = {"meta": meta, "df": cal_df}
                st.success("Done.")
                st.rerun()
            except Exception as e:
                st.error(f"Calibration failed: {e}")

    if "european_proxy" in cal:
        _render_result(cal["european_proxy"]["meta"], cal["european_proxy"].get("df"))
    else:
        st.info("Not yet calibrated.")

st.caption(
    "American-option pricing (PDE / LSMC) lives on the **Price Contracts** page — "
    "calibration is decoupled from pricing."
)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/02_Filter_Options.py", label="← Back to Filter Options", icon="🔍")
with col_fwd:
    st.page_link("pages/04_Price_Contracts.py", label="Next: Price Contracts →", icon="💰")
