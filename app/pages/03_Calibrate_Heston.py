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

from services.calibration_service import calibrate_option_chain
from services.market_service import parse_tickers
from services.pricing_service import HestonParameters

st.set_page_config(page_title="Calibrate Heston", layout="wide")
st.title("Step 3 — Calibrate Heston Model")
st.caption(
    "Fit the five Heston parameters (v₀, κ, θ̄, σ, ρ) to the filtered option chain "
    "using the Levenberg-Marquardt optimiser (Cui et al. 2016). "
    "Run each method independently or compare all three side by side."
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
div_yields: dict = ss.get("_div_yields", {})
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
    q_val = div_yields.get(tkr, q)
    st.caption(
        f"**{tkr}**: spot {'${:.2f}'.format(spot) if spot else 'n/a'}  |  "
        f"r = {r*100:.3f}%  |  q = {q_val*100:.3f}%  |  "
        f"{len(tkr_df):,} filtered contracts"
    )

# ── Shared settings ───────────────────────────────────────────────────────────
st.subheader("Shared settings")
sh_col1, sh_col2 = st.columns(2)
with sh_col1:
    max_expiries = st.number_input(
        "Max expiries to use  (0 = all)",
        min_value=0, max_value=50, value=0, step=1,
        help="Limit to the nearest N expiries to speed up calibration.",
    )
with sh_col2:
    contracts_per_expiry = st.number_input(
        "Max contracts per expiry  (0 = all)",
        min_value=0, max_value=100, value=0, step=5,
        help="Selects near-ATM contracts per expiry when > 0.",
    )

max_exp = int(max_expiries) if max_expiries > 0 else None
cpe = int(contracts_per_expiry) if contracts_per_expiry > 0 else None

_DEFAULT_GUESS = "0.04, 2.0, 0.04, 0.5, -0.7"

def _parse_guess(text: str) -> HestonParameters:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 5:
        raise ValueError("Expected 5 comma-separated values: v0, kappa, theta, sigma, rho")
    return HestonParameters.from_iterable(vals)

def _run_calibration(method_code: str, guess: HestonParameters, Ns: int, Nv: int, Nt: int):
    return calibrate_option_chain(
        filtered_df,
        r=r, q=q,
        rate_curve=rate_curve,
        initial_guess=guess,
        Ns=Ns, Nv=Nv, Nt=Nt,
        max_expiries=max_exp,
        contracts_per_expiry=cpe,
        american_method=method_code,
    )

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
            show_cols = [c for c in ["maturity", "type", "strike", "T",
                                      "moneyness", "mid_price", "market_iv"]
                         if c in cal_df.columns]
            st.dataframe(
                cal_df[show_cols].sort_values(["maturity", "strike"]),
                hide_index=True, use_container_width=True,
            )

# ── Three method columns ──────────────────────────────────────────────────────
st.divider()
col_a, col_b, col_c = st.columns(3)
cal: dict = ss.get("calibration", {})

# ── Column A: European Proxy ──────────────────────────────────────────────────
with col_a:
    st.markdown("### European Proxy")
    st.caption("Prices American options as European. Fast — good for initial calibration.")

    guess_a = st.text_input(
        "Initial guess  (v₀, κ, θ̄, σ, ρ)",
        value=_DEFAULT_GUESS,
        key="guess_european_proxy",
        help="Starting point for the optimiser. Format: v0, kappa, theta, sigma, rho",
    )

    if st.button("Calibrate — European Proxy", type="primary", key="btn_ep"):
        try:
            hp = _parse_guess(guess_a)
        except Exception as e:
            st.error(f"Invalid initial guess: {e}")
        else:
            with st.spinner("Calibrating with European Proxy …"):
                try:
                    result, cal_df = _run_calibration("european_proxy", hp, Ns=40, Nv=20, Nt=40)
                    meta = result.as_dict()
                    meta["method_label"] = "European Proxy"
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

# ── Column B: PDE Solver ──────────────────────────────────────────────────────
with col_b:
    st.markdown("### PDE Solver")
    st.caption("Solves the Heston PDE on a finite-difference grid. Accurate for early-exercise premium.")

    guess_b = st.text_input(
        "Initial guess  (v₀, κ, θ̄, σ, ρ)",
        value=_DEFAULT_GUESS,
        key="guess_pde",
        help="Starting point for the optimiser.",
    )

    b1, b2, b3 = st.columns(3)
    Ns_pde = b1.number_input("Ns (stock steps)", min_value=5, max_value=200, value=40, step=5, key="Ns_pde",
                              help="Number of stock-price grid points in the PDE.")
    Nv_pde = b2.number_input("Nv (var steps)",   min_value=5, max_value=200, value=20, step=5, key="Nv_pde",
                              help="Number of variance grid points in the PDE.")
    Nt_pde = b3.number_input("Nt (time steps)",  min_value=5, max_value=200, value=40, step=5, key="Nt_pde",
                              help="Number of time steps in the PDE.")

    if st.button("Calibrate — PDE Solver", type="primary", key="btn_pde"):
        try:
            hp = _parse_guess(guess_b)
        except Exception as e:
            st.error(f"Invalid initial guess: {e}")
        else:
            with st.spinner("Calibrating with PDE Solver — this may take several minutes …"):
                try:
                    result, cal_df = _run_calibration(
                        "pde", hp, Ns=int(Ns_pde), Nv=int(Nv_pde), Nt=int(Nt_pde)
                    )
                    meta = result.as_dict()
                    meta["method_label"] = "PDE Solver"
                    if "calibration" not in ss:
                        ss["calibration"] = {}
                    ss["calibration"]["pde"] = {"meta": meta, "df": cal_df}
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

    if "pde" in cal:
        _render_result(cal["pde"]["meta"], cal["pde"].get("df"))
    else:
        st.info("Not yet calibrated.")

# ── Column C: LSMC ───────────────────────────────────────────────────────────
with col_c:
    st.markdown("### LSMC Simulation")
    st.caption("Longstaff-Schwartz Monte Carlo. Most flexible; highest variance per run.")

    guess_c = st.text_input(
        "Initial guess  (v₀, κ, θ̄, σ, ρ)",
        value=_DEFAULT_GUESS,
        key="guess_lsmc",
        help="Starting point for the optimiser.",
    )

    c1, c2 = st.columns(2)
    Ns_lsmc = c1.number_input("Paths (Ns)", min_value=100, max_value=50000, value=1000, step=100,
                               key="Ns_lsmc",
                               help="Number of Monte Carlo paths. More paths = more accurate but slower.")
    Nt_lsmc = c2.number_input("Time steps (Nt)", min_value=10, max_value=500, value=50, step=10,
                               key="Nt_lsmc",
                               help="Number of time discretisation steps per path.")

    if st.button("Calibrate — LSMC", type="primary", key="btn_lsmc"):
        try:
            hp = _parse_guess(guess_c)
        except Exception as e:
            st.error(f"Invalid initial guess: {e}")
        else:
            with st.spinner("Calibrating with LSMC — this may take several minutes …"):
                try:
                    result, cal_df = _run_calibration(
                        "lsmc", hp, Ns=int(Ns_lsmc), Nv=20, Nt=int(Nt_lsmc)
                    )
                    meta = result.as_dict()
                    meta["method_label"] = "LSMC Simulation"
                    if "calibration" not in ss:
                        ss["calibration"] = {}
                    ss["calibration"]["lsmc"] = {"meta": meta, "df": cal_df}
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

    if "lsmc" in cal:
        _render_result(cal["lsmc"]["meta"], cal["lsmc"].get("df"))
    else:
        st.info("Not yet calibrated.")

# ── Comparison table (shown when ≥ 2 methods done) ───────────────────────────
if len(cal) >= 2:
    st.divider()
    st.subheader("Parameter comparison across methods")
    _LABELS = {
        "european_proxy": "European Proxy",
        "pde": "PDE Solver",
        "lsmc": "LSMC Simulation",
    }
    rows = []
    for code, label in _LABELS.items():
        if code not in cal:
            continue
        meta = cal[code]["meta"]
        feller = 2 * meta["kappa"] * meta["theta"] - meta["sigma"] ** 2
        rows.append({
            "Method":        label,
            "v₀":            round(meta["v0"],    6),
            "κ":             round(meta["kappa"], 4),
            "θ̄":             round(meta["theta"], 6),
            "σ":             round(meta["sigma"], 4),
            "ρ":             round(meta["rho"],   4),
            "Feller 2κθ−σ²": round(feller,        4),
            "Loss":          f"{meta['loss']:.4e}",
            "Runtime (s)":   round(meta["runtime_seconds"], 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.divider()
col_back, col_fwd = st.columns([1, 1])
with col_back:
    st.page_link("pages/02_Filter_Options.py", label="← Back to Filter Options", icon="🔍")
with col_fwd:
    st.page_link("pages/04_Price_Contracts.py", label="Next: Price Contracts →", icon="💰")
