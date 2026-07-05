"""
Calibration service: orchestrates fitting the five Heston parameters to a chain.

`select_calibration_universe()` picks the out-of-the-money leg per strike relative
to the implied forward (OTM put for K<F, OTM call for K>F — industry practice; drops
the near-intrinsic ITM mirror and avoids double-counting), ranks near-ATM per expiry,
and on the European-proxy path de-Americanizes the quotes (calibration/de_americanize.py)
so the early-exercise premium is stripped before fitting. `calibrate_option_chain()`
then runs the Levenberg-Marquardt optimiser (calibration/calibrate_heston.py, Cui
et al. 2016) over the fast characteristic-function pricer — vega-weighting the price
residuals so the fit behaves like an IV-space fit — no American pricer in the loop —
and caches results under results/calibrations/.

Position in the pipeline: MarketService -> [CalibrationService] -> Heston params ->
AnalyticsService / PricingService. Downstream UI: app/pages/03_Calibrate_Heston.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import re

import numpy as np
import pandas as pd

from analytics.schema import ensure_option_frame
from calibration.calibrate_heston import calibrate_heston
from calibration.data_driven_bounds import (
    dynamic_v0_theta_bounds,
    estimate_kappa0_from_chain,
)
from calibration.de_americanize import add_deamericanized_columns
from calibration.heston_loss_function import iv_error_metrics
from config.market_config import interpolate_rate
from services.pricing_service import HestonParameters


# Default (fallback) search box. In the spirit of Cui et al. (2016) — whose point
# is that the analytic-gradient LM needs neither a good start nor a tight box —
# bounds here are guard rails that should never bind, not steering constraints.
# The v0/θ slots below are STATIC FALLBACKS only: by default calibrate_option_chain
# replaces them with dynamic_v0_theta_bounds() scaled to the chain's observed
# deam_iv range (a fixed variance floor of 0.05 = 22.4% vol sits above the entire
# SPX surface and pins 4 of 5 parameters — see calibration/data_driven_bounds.py).
# The κ slot is likewise superseded: κ is fixed to the chain's ATM term-structure
# κ₀ (fix_kappa=True) because the full surface does not identify it.
DEFAULT_BOUNDS: list[tuple[float, float]] = [
    (0.001, 2.00),    # v0    (fallback; dynamic per-chain box used by default)
    (0.05, 10.00),    # kappa (superseded by fixed κ₀ when fix_kappa=True)
    (0.001, 2.00),    # theta (fallback; dynamic per-chain box used by default)
    (0.05, 3.00),     # sigma (vol-of-vol; Cui ceiling raised for high-vol names)
    (-0.95, -0.05),   # rho
]
DEFAULT_INITIAL_GUESS = HestonParameters(v0=0.20, kappa=1.20, theta=0.20, sigma=0.30, rho=-0.60)
CALIBRATION_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "calibrations"


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParameters
    loss: float
    contract_count: int
    objective: str
    pricing_mode: str
    calibration_style: str
    runtime_seconds: float
    weight_scheme: str = "vega"
    # IV-space fit quality (fractions; ×100 for vol points). These are the
    # interpretable, cross-ticker-comparable headline numbers — the raw `loss`
    # is a vega-weighted price SSE that scales with notional and is kept only
    # for internal/back-compat use. See heston_loss_function.iv_error_metrics.
    iv_rmse: float = float("nan")
    iv_mae: float = float("nan")
    # κ handling: κ is fixed (not optimised) by default because the surface does
    # not identify it. kappa_source records where the fixed value came from:
    # "chain_term_structure" (trusted ATM term-structure fit), "fallback_default"
    # (term structure uninformative — conventional κ₀), "caller" (explicit kappa0
    # argument) or "free" (fix_kappa=False; κ optimised as before).
    kappa_fixed: bool = False
    kappa_source: str = "free"
    kappa_se: float = float("nan")
    kappa_half_life_months: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "v0": self.params.v0,
            "kappa": self.params.kappa,
            "theta": self.params.theta,
            "sigma": self.params.sigma,
            "rho": self.params.rho,
            "loss": self.loss,
            "iv_rmse": self.iv_rmse,
            "iv_mae": self.iv_mae,
            "kappa_fixed": self.kappa_fixed,
            "kappa_source": self.kappa_source,
            "kappa_se": self.kappa_se,
            "kappa_half_life_months": self.kappa_half_life_months,
            "contract_count": self.contract_count,
            "objective": self.objective,
            "pricing_mode": self.pricing_mode,
            "calibration_style": self.calibration_style,
            "runtime_seconds": self.runtime_seconds,
            "weight_scheme": self.weight_scheme,
        }


def calibration_scope_id(
    *,
    source: str,
    tickers_text: str,
    r: float,
    q: float,
    calibration_style: str,
) -> str:
    raw = f"{source}_{tickers_text}_{r:.6f}_{q:.6f}_{calibration_style}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def calibration_cache_path(scope_id: str) -> Path:
    CALIBRATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CALIBRATION_CACHE_DIR / f"{scope_id}.json"


def save_calibration_result(scope_id: str, calibration_meta: dict[str, float]) -> Path:
    path = calibration_cache_path(scope_id)
    path.write_text(json.dumps(calibration_meta, indent=2))
    return path


def load_saved_calibration(scope_id: str) -> dict[str, float] | None:
    path = calibration_cache_path(scope_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_triple_calibration(
    scope_id: str,
    meta_a: dict[str, float],
    meta_b: dict[str, float],
    meta_c: dict[str, float],
) -> Path:
    """Persist three calibration results (one per slot) under a single scope file."""
    path = calibration_cache_path(scope_id)
    path.write_text(json.dumps({"slot_a": meta_a, "slot_b": meta_b, "slot_c": meta_c}, indent=2))
    return path


def load_triple_calibration(
    scope_id: str,
) -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float] | None]:
    """Load three calibration results.  Returns (meta_a, meta_b, meta_c), any may be None."""
    path = calibration_cache_path(scope_id)
    if not path.exists():
        return None, None, None
    data = json.loads(path.read_text())
    if "slot_a" in data:
        return data.get("slot_a"), data.get("slot_b"), data.get("slot_c")
    return None, None, None


def select_calibration_universe(
    options_df: pd.DataFrame,
    *,
    max_expiries: int | None = None,
    contracts_per_expiry: int | None = None,
    r: float = 0.0,
    q: float = 0.0,
    american_method: str = "european_proxy",
    otm_only: bool = True,
    atm_band: float = 0.02,
    mny_lo: float | None = None,
    mny_hi: float | None = None,
    min_open_interest: int = 0,
    min_maturity: float = 0.0,
    max_maturity: float | None = None,
) -> pd.DataFrame:
    """Pick the OTM, near-ATM calibration set off the (already-europeanized) chain.

    The calibration universe is deliberately *tighter* than the broad pricing/screening
    universe ("calibrate tight, price broad"): forward-moneyness band, OI floor, maturity
    window, OTM-only leg per strike, and per-expiry near-ATM caps. The market side is
    expected to already carry European-equivalent prices (`euro_mid`/`deam_iv`, stamped
    by services.market_service.europeanize_chain after filtering); they are computed here
    only if absent, so the function is still correct when called on a raw frame.
    """
    df = ensure_option_frame(options_df)
    df = df[(df["T"] > 0) & df["mid_price"].notna()].copy()
    if df.empty:
        return df

    # Maturity window (calibration-specific; drops noisy front-week / stale long-dated).
    if min_maturity:
        df = df[df["T"] >= min_maturity]
    if max_maturity is not None:
        df = df[df["T"] <= max_maturity]

    # Forward-moneyness band — calibration-specific, applied on top of the broad filter.
    # Uses K/F (implied-forward moneyness) when available, else spot K/S.
    fm = df["forward_moneyness"] if "forward_moneyness" in df.columns else df["moneyness"]
    df = df.assign(_fm=fm)
    df = df[df["_fm"].notna() & (df["_fm"] > 0)]
    if mny_lo is not None:
        df = df[df["_fm"] >= mny_lo]
    if mny_hi is not None:
        df = df[df["_fm"] <= mny_hi]
    if min_open_interest and "openInterest" in df.columns:
        df = df[df["openInterest"].fillna(0.0) >= min_open_interest]
    if df.empty:
        return df.drop(columns=["_fm"], errors="ignore")

    # OTM-only selection (industry practice): per strike, keep the out-of-the-money
    # leg relative to the implied forward F — the put when K < F, the call when K > F
    # (in the |ln(K/F)| ≤ atm_band ATM zone, keep the marginally-OTM leg). This drops
    # the ITM mirror (wide-spread, near-intrinsic, low vol-information, early-exercise
    # noise) and avoids double-counting a call+put at the same strike (same IV).
    if otm_only:
        log_fm = np.log(df["_fm"])
        is_put = df["type"].eq("put")
        is_call = df["type"].eq("call")
        atm_zone = log_fm.abs() <= atm_band
        keep = (
            ((log_fm < -atm_band) & is_put)
            | ((log_fm > atm_band) & is_call)
            | (atm_zone & (((log_fm <= 0) & is_put) | ((log_fm > 0) & is_call)))
        )
        df = df[keep.fillna(False)].copy()
        if df.empty:
            return df.drop(columns=["_fm"], errors="ignore")

    # European-equivalent engine: always calibrate on European contracts so the
    # analytic Cui gradient is available. Quotes are de-Americanized upstream, and
    # the American PDE/LSMC pricers have been removed (see _graveyard.py), so the
    # `american_method` argument is retained only for backward-compatible call
    # sites — "european_proxy" is the sole supported path.
    df["ExerciseStyle"] = "european"

    expiry_key = "maturity" if "maturity" in df.columns else "T"
    selected_groups: list[pd.DataFrame] = []

    for _, group in df.sort_values("T").groupby(expiry_key, sort=False):
        ranked = group.copy()
        ranked["atm_distance"] = ranked["atm_distance"].fillna(np.inf)
        ranked["rel_spread"] = ranked.get("rel_spread", pd.Series(1.0, index=ranked.index)).fillna(1.0)
        ranked["volume"] = ranked.get("volume", pd.Series(0.0, index=ranked.index)).fillna(0.0)
        ranked = ranked.sort_values(
            ["atm_distance", "rel_spread", "volume"],
            ascending=[True, True, False],
        )
        selected_groups.append(ranked.head(contracts_per_expiry))
        if max_expiries is not None and len(selected_groups) >= max_expiries:
            break

    if not selected_groups:
        return df.head(0).drop(columns=["_fm"], errors="ignore")

    calibration_df = pd.concat(selected_groups, ignore_index=True)
    calibration_df = calibration_df.drop(columns=["_fm"], errors="ignore")

    # European-equivalent prices are the calibration target. They are normally stamped
    # upstream (europeanize_chain); compute on the fly only if a caller passed a raw
    # frame, so this function is self-contained. `mid_price_market` preserves the raw
    # American quote; `deam_iv` (σ*) is the European-equivalent market IV used for
    # vega-weighting downstream.
    if "euro_mid" not in calibration_df.columns or "deam_iv" not in calibration_df.columns:
        calibration_df = add_deamericanized_columns(calibration_df, r=r, q=q)
    calibration_df["mid_price_market"] = calibration_df["mid_price"]
    calibration_df["mid_price"] = calibration_df["euro_mid"].fillna(calibration_df["mid_price"])
    calibration_df["market_iv"] = calibration_df["deam_iv"]
    return calibration_df


def calibrate_option_chain(
    options_df: pd.DataFrame,
    *,
    r: float = 0.0,
    q: float = 0.0,
    rate_curve: dict | None = None,
    initial_guess: HestonParameters = DEFAULT_INITIAL_GUESS,
    bounds: list[tuple[float, float]] | None = None,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
    max_expiries: int | None = None,
    contracts_per_expiry: int | None = None,
    objective: str | None = None,
    american_method: str = "european_proxy",
    otm_only: bool = True,
    atm_band: float = 0.02,
    mny_lo: float | None = None,
    mny_hi: float | None = None,
    min_open_interest: int = 0,
    min_maturity: float = 0.0,
    max_maturity: float | None = None,
    weight_scheme: str = "vega",
    fix_kappa: bool = True,
    kappa0: float | None = None,
) -> tuple[CalibrationResult, pd.DataFrame]:
    """Calibrate Heston parameters to the chain.

    κ handling (fix_kappa, default True): κ is FIXED, not optimised — the full
    surface does not identify it (the κ–σ degeneracy valley is flat, so a free κ
    drifts to whatever bound the box imposes). The fixed value is `kappa0` when
    given, else the chain's own ATM term-structure estimate clipped to a sane
    range (calibration.data_driven_bounds.estimate_kappa0_from_chain), else a
    conventional fallback when the chain has too few expiries to fit. Only
    (v0, θ, σ, ρ) are optimised. Pass fix_kappa=False for the legacy free-κ fit.

    Bounds (when `bounds` is None): v0/θ boxes are DYNAMIC guard rails scaled to
    the chain's observed deam_iv range (dynamic_v0_theta_bounds); σ/ρ use the
    fixed DEFAULT_BOUNDS slots. Caller-supplied `bounds` are respected as given,
    except the κ slot, which is pinched to the fixed κ₀ while fix_kappa=True.
    """
    # LM always minimises price residuals (Cui et al., 2016).
    # The objective parameter is retained for API compatibility only.
    if objective is None:
        objective = "price"
    # pricing_mode tells heston_residuals how to price American contracts.
    pricing_mode = american_method

    calibration_df = select_calibration_universe(
        options_df,
        max_expiries=max_expiries,
        contracts_per_expiry=contracts_per_expiry,
        r=r,
        q=q,
        american_method=american_method,
        otm_only=otm_only,
        atm_band=atm_band,
        mny_lo=mny_lo,
        mny_hi=mny_hi,
        min_open_interest=min_open_interest,
        min_maturity=min_maturity,
        max_maturity=max_maturity,
    )

    if calibration_df.empty:
        raise ValueError("No valid contracts available for calibration.")

    calibration_df = calibration_df.copy()
    if "r" not in calibration_df.columns:
        if rate_curve:
            calibration_df["r"] = calibration_df["T"].map(
                lambda T: interpolate_rate(rate_curve, T)
            )
        else:
            calibration_df["r"] = r

    # κ anchor + term-structure diagnostics. Estimated from the broad chain when
    # possible (more expiries pin the term structure better than the maturity-
    # windowed calibration universe); falls back to the universe itself when the
    # caller passed a frame without deam_iv/forward columns.
    kappa_info: dict | None = None
    if fix_kappa:
        kappa_info = estimate_kappa0_from_chain(options_df)
        if kappa_info["n_expiries"] == 0:
            kappa_info = estimate_kappa0_from_chain(calibration_df)
        if kappa0 is not None:
            kappa_info = dict(kappa_info)
            kappa_info["kappa0"] = float(kappa0)
            kappa_info["source"] = "caller"
            kappa_info["half_life_months"] = float(np.log(2.0) / kappa0 * 12)
        else:
            kappa_info["source"] = (
                "chain_term_structure" if kappa_info["trusted"] else "fallback_default"
            )

    # Search box. Caller bounds are respected as given; otherwise v0/θ get the
    # dynamic per-chain guard rails (variance-level is the only genuinely
    # ticker-sensitive box — see dynamic_v0_theta_bounds) and σ/ρ the fixed
    # DEFAULT_BOUNDS slots. Either way the box is meant to never bind.
    if bounds is not None:
        effective_bounds = [tuple(b) for b in bounds]
    else:
        var_box = dynamic_v0_theta_bounds(
            options_df if "deam_iv" in getattr(options_df, "columns", []) else calibration_df
        )
        effective_bounds = [
            var_box,                # v0
            DEFAULT_BOUNDS[1],      # kappa (pinched below when fix_kappa)
            var_box,                # theta
            DEFAULT_BOUNDS[3],      # sigma
            DEFAULT_BOUNDS[4],      # rho
        ]

    ig_list = list(initial_guess.as_tuple())

    # Fix κ by pinching its box to κ₀·(1±1e-6): scipy's trf requires lb < ub
    # strictly, and a width-zero direction is equivalent to removing κ from the
    # optimisation. Applied on top of caller bounds too — fixed κ is the engine
    # default, not a bounds preference.
    if fix_kappa and kappa_info is not None:
        k0 = float(kappa_info["kappa0"])
        effective_bounds[1] = (k0 * (1.0 - 1e-6), k0 * (1.0 + 1e-6))
        ig_list[1] = k0
        # When the caller left the stock initial guess, seed v0/θ from the same
        # term-structure fit — same surface, closer start, fewer LM iterations.
        if initial_guess is DEFAULT_INITIAL_GUESS:
            for slot, key in ((0, "v0_ts"), (2, "theta_ts")):
                val = kappa_info.get(key, float("nan"))
                if np.isfinite(val) and val > 0:
                    ig_list[slot] = float(val)

    # Guarantee the initial guess lies inside the search box (scipy's
    # least_squares rejects an out-of-bounds start). Clamp each component.
    ig_list = [min(max(g, lo), hi) for g, (lo, hi) in zip(ig_list, effective_bounds)]

    start_time = perf_counter()
    params_opt, loss_val = calibrate_heston(
        r=r,
        q=q,
        option_df=calibration_df,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
        initial_guess=ig_list,
        bounds=effective_bounds,
        objective=objective,
        pricing_mode=pricing_mode,
        weight_scheme=weight_scheme,
    )
    runtime_seconds = perf_counter() - start_time

    # Interpretable IV-space fit quality (vol points), independent of the raw
    # notional-scaled price loss. Reporting only — does not affect the fit.
    iv_rmse, iv_mae = iv_error_metrics(params_opt, r, q, calibration_df)

    result = CalibrationResult(
        params=HestonParameters.from_iterable(params_opt),
        loss=float(loss_val),
        contract_count=len(calibration_df),
        objective=objective,
        pricing_mode=pricing_mode,
        calibration_style=american_method,
        runtime_seconds=float(runtime_seconds),
        weight_scheme=weight_scheme,
        iv_rmse=float(iv_rmse),
        iv_mae=float(iv_mae),
        kappa_fixed=bool(fix_kappa and kappa_info is not None),
        kappa_source=(kappa_info["source"] if (fix_kappa and kappa_info) else "free"),
        kappa_se=float(kappa_info.get("se_kappa", float("nan"))) if (fix_kappa and kappa_info) else float("nan"),
        kappa_half_life_months=float(kappa_info.get("half_life_months", float("nan"))) if (fix_kappa and kappa_info) else float("nan"),
    )
    return result, calibration_df
