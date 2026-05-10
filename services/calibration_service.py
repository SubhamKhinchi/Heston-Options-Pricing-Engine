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
from calibration.implied_vol import implied_volatility
from services.pricing_service import HestonParameters


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


def select_calibration_universe(
    options_df: pd.DataFrame,
    *,
    max_expiries: int = 6,
    contracts_per_expiry: int = 6,
    r: float = 0.05,
    q: float = 0.0,
    calibration_style: str = "fast",
) -> pd.DataFrame:
    df = ensure_option_frame(options_df)
    df = df[(df["T"] > 0) & df["mid_price"].notna()].copy()

    if df.empty:
        return df

    if calibration_style == "fast":
        if abs(q) <= 1e-12 and (df["type"] == "call").any():
            df = df[df["type"] == "call"].copy()
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
        if len(selected_groups) >= max_expiries:
            break

    if not selected_groups:
        return df.head(0).copy()

    calibration_df = pd.concat(selected_groups, ignore_index=True)
    calibration_df["market_iv"] = calibration_df.apply(
        lambda row: implied_volatility(
            row["mid_price"],
            row["spot"],
            row["strike"],
            r,
            row["T"],
            row["type"],
            q,
        ),
        axis=1,
    )
    return calibration_df


def calibrate_option_chain(
    options_df: pd.DataFrame,
    *,
    r: float,
    q: float,
    initial_guess: HestonParameters = DEFAULT_INITIAL_GUESS,
    bounds: list[tuple[float, float]] | None = None,
    Ns: int = 40,
    Nv: int = 20,
    Nt: int = 40,
    max_expiries: int = 6,
    contracts_per_expiry: int = 6,
    calibration_style: str = "fast",
    objective: str | None = None,
    pricing_mode: str | None = None,
) -> tuple[CalibrationResult, pd.DataFrame]:
    if objective is None:
        objective = "price" if calibration_style == "fast" else "iv"
    if pricing_mode is None:
        pricing_mode = "european_proxy" if calibration_style == "fast" else "auto"

    calibration_df = select_calibration_universe(
        options_df,
        max_expiries=max_expiries,
        contracts_per_expiry=contracts_per_expiry,
        r=r,
        q=q,
        calibration_style=calibration_style,
    )

    if calibration_df.empty:
        raise ValueError("No valid contracts available for calibration.")

    calibration_df = calibration_df.copy()
    calibration_df["r"] = r

    start_time = perf_counter()
    params_opt, loss_val = calibrate_heston(
        r=r,
        q=q,
        option_df=calibration_df,
        Ns=Ns,
        Nv=Nv,
        Nt=Nt,
        initial_guess=list(initial_guess.as_tuple()),
        bounds=bounds,
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
        calibration_style=calibration_style,
        runtime_seconds=float(runtime_seconds),
    )
    return result, calibration_df
