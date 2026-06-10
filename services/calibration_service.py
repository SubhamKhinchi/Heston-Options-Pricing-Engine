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
from calibration.calibrate_heston_lbfgsb import calibrate_heston_lbfgsb
from calibration.data_driven_bounds import compute_data_driven_bounds
from calibration.implied_vol import implied_volatility
from config.market_config import interpolate_rate
from services.pricing_service import HestonParameters


def _loosen_data_driven_bounds(
    bounds: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Widen the tight per-chain bounds from compute_data_driven_bounds before
    handing them to the optimizer.

    The data-driven estimates are a good *centre* for the search, but the
    underlying short-time smile heuristics (σ≈√(8c), ρ≈2b/σ) are only
    approximate, so the raw ±0.20 ρ window / [0.5σ*, 2σ*] σ box can trap the
    optimizer near a biased guess. We keep the data-driven v0/κ/θ boxes (well
    identified by ATM level and term structure) but loosen the two parameters
    the heuristics estimate least reliably:
      • σ: raise the ceiling to at least 3.0 so vol-of-vol is never pre-capped
      • ρ: widen to a ±0.40 window, clipped to (-0.999, 0.999)
    """
    v0_b, kappa_b, theta_b, sigma_b, rho_b = bounds
    sigma_b = (sigma_b[0], max(sigma_b[1], 3.0))
    rho_lo = max(rho_b[0] - 0.20, -0.999)
    rho_hi = min(rho_b[1] + 0.20, 0.999)
    rho_b = (rho_lo, rho_hi)
    return [v0_b, kappa_b, theta_b, sigma_b, rho_b]


DEFAULT_INITIAL_GUESS = HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
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

    def as_dict(self) -> dict[str, float]:
        return {
            "v0": self.params.v0,
            "kappa": self.params.kappa,
            "theta": self.params.theta,
            "sigma": self.params.sigma,
            "rho": self.params.rho,
            "loss": self.loss,
            "contract_count": self.contract_count,
            "objective": self.objective,
            "pricing_mode": self.pricing_mode,
            "calibration_style": self.calibration_style,
            "runtime_seconds": self.runtime_seconds,
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
) -> pd.DataFrame:
    df = ensure_option_frame(options_df)
    df = df[(df["T"] > 0) & df["mid_price"].notna()].copy()

    if df.empty:
        return df

    if american_method == "european_proxy":
        # Force European so the analytical Jacobian is always available.
        df["ExerciseStyle"] = "european"
    # For "pde" and "lsmc": keep the original ExerciseStyle from market data (american).

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
        return df.head(0).copy()

    calibration_df = pd.concat(selected_groups, ignore_index=True)
    calibration_df["market_iv"] = calibration_df.apply(
        lambda row: implied_volatility(
            row["mid_price"],
            row["spot"],
            row["strike"],
            float(row.get("r", r)),
            row["T"],
            row["type"],
            float(row.get("q", q)),
        ),
        axis=1,
    )
    return calibration_df


def calibrate_option_chain(
    options_df: pd.DataFrame,
    *,
    r: float = 0.0,
    q: float = 0.0,
    rate_curve: dict | None = None,
    initial_guess: HestonParameters = DEFAULT_INITIAL_GUESS,
    bounds: list[tuple[float, float]] | None = None,
    use_data_driven: bool = True,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
    max_expiries: int | None = None,
    contracts_per_expiry: int | None = None,
    objective: str | None = None,
    american_method: str = "european_proxy",
) -> tuple[CalibrationResult, pd.DataFrame]:
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

    # Derive a per-chain initial guess and search bounds from the shape of the
    # market IV surface, instead of always starting from the static default.
    # Only when the caller did not supply explicit bounds.
    ig_list = list(initial_guess.as_tuple())
    effective_bounds = bounds
    if use_data_driven and bounds is None:
        dd = compute_data_driven_bounds(calibration_df, r=r, q=q)
        warn = dd["diagnostics"].get("warning")
        if warn:
            # Not enough liquid maturities/strikes to read the surface —
            # compute_data_driven_bounds already fell back to static defaults.
            print(f"[calibration] data-driven bounds unavailable: {warn} "
                  f"Using static default guess/bounds.")
        else:
            effective_bounds = _loosen_data_driven_bounds(dd["bounds"])
            # Adopt the data-driven guess only if the caller kept the default.
            if initial_guess == DEFAULT_INITIAL_GUESS:
                ig_list = dd["initial_guess"]
            diag = dd["diagnostics"]
            print(
                "[calibration] using data-driven guess/bounds "
                f"({diag.get('n_liquid_maturities')} liquid maturities): "
                f"v0={ig_list[0]:.4f} kappa={ig_list[1]:.4f} theta={ig_list[2]:.4f} "
                f"sigma={ig_list[3]:.4f} rho={ig_list[4]:+.4f}"
            )

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
    )
    runtime_seconds = perf_counter() - start_time

    result = CalibrationResult(
        params=HestonParameters.from_iterable(params_opt),
        loss=float(loss_val),
        contract_count=len(calibration_df),
        objective=objective,
        pricing_mode=pricing_mode,
        calibration_style=american_method,
        runtime_seconds=float(runtime_seconds),
    )
    return result, calibration_df


def calibrate_option_chain_lbfgsb(
    options_df: pd.DataFrame,
    *,
    r: float,
    q: float,
    initial_guess: HestonParameters = DEFAULT_INITIAL_GUESS,
    bounds: list[tuple[float, float]] | None = None,
    max_expiries: int | None = None,
    contracts_per_expiry: int | None = None,
) -> tuple[CalibrationResult, pd.DataFrame]:
    """
    Calibrate Heston parameters using the L-BFGS-B optimizer with scalar MSE
    price loss.  Uses the same Gauss-Legendre pricing and the same full contract
    universe (calls + puts, European proxy) as the LM method, so the two methods
    are directly comparable.
    """
    calibration_df = select_calibration_universe(
        options_df,
        max_expiries=max_expiries,
        contracts_per_expiry=contracts_per_expiry,
        r=r,
        q=q,
    )

    if calibration_df.empty:
        raise ValueError("No valid contracts available for calibration.")

    calibration_df = calibration_df.copy()
    calibration_df["r"] = r

    start_time = perf_counter()
    params_opt, loss_val = calibrate_heston_lbfgsb(
        r=r,
        q=q,
        option_df=calibration_df,
        initial_guess=list(initial_guess.as_tuple()),
        bounds=bounds,
    )
    runtime_seconds = perf_counter() - start_time

    result = CalibrationResult(
        params=HestonParameters.from_iterable(params_opt),
        loss=float(loss_val),
        contract_count=len(calibration_df),
        objective="price_mse",
        pricing_mode="european_proxy",
        calibration_style="lbfgsb",
        runtime_seconds=float(runtime_seconds),
    )
    return result, calibration_df
