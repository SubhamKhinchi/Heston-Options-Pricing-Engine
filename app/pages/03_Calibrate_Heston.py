"""
Step 3 — Calibrate Heston.

Fits (v0, theta, sigma, rho) to the filtered chain — kappa is FIXED to the chain's
own ATM term-structure estimate (the surface does not identify kappa; see
calibration/data_driven_bounds.estimate_kappa0_from_chain) — via
services/calibration_service.calibrate_option_chain: select the OTM, near-ATM
calibration universe, then vega-weighted Levenberg-Marquardt over the fast CF
pricer. v0/theta search bounds are dynamic guard rails scaled to the chain's
observed deam_iv range. Quotes are already de-Americanized upstream (Step 2 Filter).
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

import numpy as np
import pandas as pd
import streamlit as st

from calibration.data_driven_bounds import (
    dynamic_v0_theta_bounds,
    estimate_kappa0_from_chain,
)
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
    "Fit four Heston parameters (v₀, θ̄, σ, ρ) to the de-Americanized option chain "
    "using the Levenberg-Marquardt optimiser with the analytic Cui et al. (2016) "
    "Jacobian. κ is **fixed** from the chain's own ATM term structure — see below."
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

# ── κ fixed from the chain + dynamic v0/θ guard rails ────────────────────────
# Same estimators the calibration service uses internally, run here so the page
# can PREVIEW the κ₀ and the default box the fit will actually use.
_kappa_info = estimate_kappa0_from_chain(filtered_df)
_var_lo, _var_hi = dynamic_v0_theta_bounds(filtered_df)
_k0 = float(_kappa_info["kappa0"])


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


# ── κ — fixed from the option chain ──────────────────────────────────────────
st.divider()
st.subheader("κ — fixed from the option chain")
_kc1, _kc2, _kc3 = st.columns(3)
_kc1.metric("κ₀ (mean-reversion speed)", f"{_k0:.2f}")
_kc2.metric("Variance half-life", f"{_kappa_info['half_life_months']:.1f} mo")
_kc3.metric(
    "Source",
    "ATM term structure" if _kappa_info["trusted"] else "fallback (κ₀ = 2.0)",
)
st.caption(
    "κ is **not optimised** — the option surface doesn't identify it (κ and σ trade "
    "off along a near-flat valley, so a free κ just drifts to a bound). Instead, κ₀ "
    "is read off this chain's ATM variance term structure: one ATM implied variance "
    "per expiry, fit to the Heston curve  w(T)/T = θ̄ + (v₀−θ̄)(1−e^{−κT})/(κT), "
    "where κ is the speed at which short-dated variance bends to the long-run level. "
    "Risk-neutral by construction — no historical data involved."
)
if not _kappa_info["trusted"]:
    st.warning(
        "This chain's ATM term structure is too flat or too short to pin κ "
        "(no method can recover κ from a flat term structure) — using the "
        "conventional fallback κ₀ = 2.0."
    )

# ── Initial parameters & search bounds (v₀, θ̄, σ, ρ) ─────────────────────────
st.divider()
st.subheader("Initial parameters & search bounds")
_iv_series = filtered_df["deam_iv"].dropna() if "deam_iv" in filtered_df.columns else pd.Series(dtype=float)
_iv_series = _iv_series[_iv_series > 0]
_iv_note = (
    f"observed IV range {_iv_series.quantile(0.01):.0%}–{_iv_series.quantile(0.99):.0%} "
    f"→ variance box [{_var_lo:.4f}, {_var_hi:.3f}]"
    if len(_iv_series) else "no de-Americanized IVs found — wide static box"
)
st.caption(
    f"v₀/θ̄ default bounds are **dynamic guard rails** scaled to this chain's "
    f"de-Americanized IV level ({_iv_note}); σ/ρ defaults are fixed wide ranges. "
    "In the spirit of Cui et al. (2016), bounds exist to keep the optimiser in "
    "sane territory, **never to steer the fit** — a fitted parameter sitting at a "
    "bound signals a data problem, not a market view. All fields are editable."
)

# Seed guesses: v₀/θ̄ from the term-structure pre-fit when available (same surface,
# closer start), else the stock defaults clamped into the box.
_g = DEFAULT_INITIAL_GUESS
_v0_seed = _kappa_info.get("v0_ts", float("nan"))
_th_seed = _kappa_info.get("theta_ts", float("nan"))
_v0_seed = float(_v0_seed) if np.isfinite(_v0_seed) and _v0_seed > 0 else _g.v0
_th_seed = float(_th_seed) if np.isfinite(_th_seed) and _th_seed > 0 else _g.theta

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))

_param_spec = [
    # key, label, meaning, (guess, lo, hi), widget min/max/step/format
    ("v0",    "v₀ — init variance",      (_clamp(_v0_seed, _var_lo, _var_hi), _var_lo, _var_hi),
     dict(min_value=0.0001, max_value=4.0, step=0.005, format="%.4f")),
    ("theta", "θ̄ — long-run variance",   (_clamp(_th_seed, _var_lo, _var_hi), _var_lo, _var_hi),
     dict(min_value=0.0001, max_value=4.0, step=0.005, format="%.4f")),
    ("sigma", "σ — vol of vol",          (_clamp(_g.sigma, *DEFAULT_BOUNDS[3]), *DEFAULT_BOUNDS[3]),
     dict(min_value=0.01, max_value=5.0, step=0.05, format="%.2f")),
    ("rho",   "ρ — spot-vol correlation", (_clamp(_g.rho, *DEFAULT_BOUNDS[4]), *DEFAULT_BOUNDS[4]),
     dict(min_value=-0.999, max_value=0.0, step=0.05, format="%.3f")),
]

_vals: dict[str, tuple[float, float, float]] = {}
for _col, (key, label, (g0, lo0, hi0), kw) in zip(st.columns(4), _param_spec):
    with _col:
        st.markdown(f"**{label}**")
        g = st.number_input("Initial guess", value=float(round(g0, 4)), key=f"{key}_guess", **kw)
        lo = st.number_input("Lower bound", value=float(round(lo0, 4)), key=f"{key}_lo", **kw)
        hi = st.number_input("Upper bound", value=float(round(hi0, 4)), key=f"{key}_hi", **kw)
        if lo > hi:
            lo, hi = hi, lo
        _vals[key] = (_clamp(g, lo, hi), lo, hi)

shared_guess = HestonParameters(
    v0=_vals["v0"][0], kappa=_k0, theta=_vals["theta"][0],
    sigma=_vals["sigma"][0], rho=_vals["rho"][0],
)
# κ slot is a placeholder — calibrate_option_chain pinches it to κ₀ (fix_kappa=True).
shared_bounds = [
    (_vals["v0"][1], _vals["v0"][2]),
    DEFAULT_BOUNDS[1],
    (_vals["theta"][1], _vals["theta"][2]),
    (_vals["sigma"][1], _vals["sigma"][2]),
    (_vals["rho"][1], _vals["rho"][2]),
]

def _render_result(meta: dict, cal_df: pd.DataFrame | None) -> None:
    """Render calibration result inside a column."""
    feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
    color = "green" if feller > 0 else "red"

    if meta.get("kappa_fixed"):
        half_life = meta.get("kappa_half_life_months", float("nan"))
        hl_txt = f", half-life {half_life:.1f} mo" if pd.notna(half_life) else ""
        kappa_note = (
            f"fixed — κ₀ from ATM term structure{hl_txt}"
            if meta.get("kappa_source") == "chain_term_structure"
            else f"fixed — fallback κ₀ (flat term structure){hl_txt}"
        )
    else:
        kappa_note = "mean-reversion speed (optimised)"

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
            kappa_note,
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
    # Headline fit quality is IV-space (vol points) — interpretable and comparable
    # across tickers. The raw vega-weighted price loss scales with notional
    # (~1e5-1e6 for an index even for an excellent fit), so it is not shown here.
    iv_rmse = meta.get("iv_rmse")
    iv_mae = meta.get("iv_mae")
    if pd.notna(iv_rmse):
        quality = (
            f"IV-RMSE: {iv_rmse * 100:.2f} vol pts  |  "
            f"IV-MAE: {iv_mae * 100:.2f} vol pts"
        )
    else:  # back-compat for results cached before IV metrics existed
        quality = f"Loss: {meta['loss']:.4e}"
    st.caption(
        f"{quality}  |  "
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
        "premium), so the fit targets the European-equivalent surface directly. The four "
        "free parameters (v₀, θ̄, σ, ρ) are then fit — with κ held at the chain-implied "
        "κ₀ above — using the fast characteristic-function pricer and the "
        "Levenberg-Marquardt optimiser (Cui et al. 2016), vega-weighted — no American "
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
