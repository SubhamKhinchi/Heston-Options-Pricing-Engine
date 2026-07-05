"""
Data-driven initial guess and parameter bounds for Heston model calibration.

Two generations of tooling live here:

**Active (used by calibration_service by default):**
- ``estimate_kappa0_from_chain`` — κ₀ from the chain's ATM average-variance term
  structure. κ is weakly identified by the full surface (the κ–σ degeneracy valley
  lets the optimizer drift to a bound), so the service *fixes* κ = κ₀ and
  calibrates only (v0, θ, σ, ρ). Q-measure, no historical data.
- ``dynamic_v0_theta_bounds`` — wide guard-rail boxes for v0/θ scaled to the
  chain's observed deam_iv range. The variance-level parameters are the only
  truly ticker-sensitive bounds (a fixed 0.05 variance floor = 22.4% vol sits
  above the entire SPX surface); reading the scale from the data keeps the box
  from ever binding, in the spirit of Cui et al.'s unconstrained calibration.

**Legacy (kept as sanity/starting-point tools, not on the default path):**
``compute_data_driven_bounds`` — full 5-parameter smile-shape estimator (ATM IVs,
skew slope, curvature). Produces *tight* per-chain boxes, which steer the fit;
see project history for why tight boxing was retired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from calibration.implied_vol import implied_volatility


# ── κ₀ from the ATM term structure (active path) ─────────────────────────────

def estimate_kappa0_from_chain(
    chain: pd.DataFrame,
    *,
    atm_band: float = 0.10,
    fallback: float = 2.0,
    kappa_fit_bounds: tuple[float, float] = (0.05, 20.0),
    kappa_range: tuple[float, float] = (0.5, 12.0),
) -> dict:
    """
    Estimate the Heston mean-reversion speed κ₀ from the option chain itself.

    Under Heston, the ATM total implied variance is (to leading order) the
    integrated expected variance, so the *average* variance term structure is

        w(T)/T ≈ θ + (v0 − θ) · (1 − e^{−κT}) / (κT)

    a 3-parameter curve where v0 is the short-end limit, θ the long-end limit
    and κ the bend speed between them. One ATM variance per expiry (OTM call +
    OTM put nearest the forward, ``deam_iv`` averaged — cancels parity noise),
    fit in average-variance space so every expiry carries equal weight.

    This is a Q-measure estimate — no historical data, no P→Q risk-premium
    assumption — and it is the anchor used to FIX κ in calibration (κ is not
    identified by the full surface; see module docstring).

    Policy — deliberately simple (cf. Bloomberg OVML, which fixes mean reversion
    to a conventional value outright because of the κ–σ "interplay of opposing
    roles"): whenever the curve fit can run (≥ 4 usable expiries), use the fitted
    κ **clipped to ``kappa_range``**. κ is weakly identified and the full
    calibration barely reacts inside that range, so a bounded chain-consistent
    estimate always beats switching to an arbitrary constant — earlier binary
    trust gates (slope / relative-se thresholds) flipped marginal chains between
    κ≈6 and the 2.0 fallback on day-to-day quote noise. ``fallback`` is used only
    when the fit cannot run at all (missing columns or < 4 expiries). ``se_kappa``
    is reported as a display-only diagnostic, not a gate; the clip handles the
    flat-term-structure artifact case (where the fitted κ is an arbitrary point
    in a flat valley) by bounding the damage instead of pretending to detect it.

    Requires columns: T, strike, forward, type, deam_iv, atm_distance
    (all present on any chain out of market_service.filter_chain_with_stats).

    Returns dict with: kappa0, kappa0_raw, trusted (fit ran), clipped, v0_ts,
    theta_ts, se_kappa, n_expiries, half_life_months, ts_points [(T, atm_var)].
    """
    needed = {"T", "strike", "forward", "type", "deam_iv", "atm_distance"}
    if chain is None or chain.empty or not needed.issubset(chain.columns):
        return {"kappa0": fallback, "kappa0_raw": float("nan"), "trusted": False,
                "v0_ts": float("nan"), "theta_ts": float("nan"),
                "se_kappa": float("nan"), "n_expiries": 0, "clipped": False,
                "half_life_months": float(np.log(2.0) / fallback * 12),
                "ts_points": [],
                "warning": "chain missing deam_iv/forward columns — fallback κ₀ used"}

    pts = []
    for T, g in chain.groupby("T"):
        ivs = []
        for side in (
            g[(g["type"] == "call") & (g["strike"] >= g["forward"])],
            g[(g["type"] == "put") & (g["strike"] <= g["forward"])],
        ):
            side = side[(side["atm_distance"] < atm_band) & (side["deam_iv"] > 0)]
            if len(side):
                ivs.append(float(side.loc[side["atm_distance"].idxmin(), "deam_iv"]))
        if ivs:
            pts.append((float(T), float(np.mean(ivs)) ** 2))

    out = {"kappa0": fallback, "kappa0_raw": float("nan"), "trusted": False,
           "clipped": False, "v0_ts": float("nan"), "theta_ts": float("nan"),
           "se_kappa": float("nan"), "n_expiries": len(pts), "ts_points": pts,
           "half_life_months": float(np.log(2.0) / fallback * 12)}
    if len(pts) < 4:
        out["warning"] = f"only {len(pts)} usable expiries — the term-structure fit needs ≥4; fallback κ₀ used"
        return out

    Ts = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    def avg_var(p):
        v0, th, k = p
        return th + (v0 - th) * (1.0 - np.exp(-k * Ts)) / (k * Ts)

    p0 = [y[np.argmin(Ts)], y[np.argmax(Ts)], 2.0]
    lo_k, hi_k = kappa_fit_bounds
    res = least_squares(lambda p: avg_var(p) - y, p0,
                        bounds=([1e-4, 1e-4, lo_k], [4.0, 4.0, hi_k]))
    v0_ts, theta_ts, k = (float(x) for x in res.x)

    # se(κ) from the fit covariance: legit here (one independent point per expiry,
    # unlike the overlapping-window AR(1) estimator this design replaced).
    dof = max(len(Ts) - 3, 1)
    s2 = 2.0 * res.cost / dof
    try:
        cov = s2 * np.linalg.pinv(res.jac.T @ res.jac)
        se_k = float(np.sqrt(max(cov[2, 2], 0.0)))
    except Exception:
        se_k = float("nan")

    # The fit ran: use its κ, clipped. No trust gates — see docstring.
    kappa0 = float(np.clip(k, *kappa_range))
    out.update({
        "kappa0": kappa0, "kappa0_raw": k, "trusted": True,
        "clipped": bool(kappa0 != k),
        "v0_ts": v0_ts, "theta_ts": theta_ts, "se_kappa": se_k,
        "half_life_months": float(np.log(2.0) / kappa0 * 12),
    })
    return out


# ── Dynamic v0/θ guard rails from the observed IV range (active path) ────────

def dynamic_v0_theta_bounds(
    chain: pd.DataFrame,
    *,
    lo_mult: float = 0.25,
    hi_mult: float = 4.0,
    var_floor: float = 1e-4,
    var_cap: float = 4.0,
) -> tuple[float, float]:
    """
    Wide (lo, hi) variance box for v0 and θ from the chain's deam_iv range:

        [lo_mult · q01(deam_iv)²,  hi_mult · q99(deam_iv)²]

    1%/99% quantiles so a single junk quote cannot distort the box; ×4 margins
    so the box is a guard rail, never a steering constraint (a fitted v0/θ at
    one of these edges indicates a data problem, not a tight market). Falls
    back to (var_floor, var_cap) — effectively unbounded — without deam_iv.
    """
    if chain is None or chain.empty or "deam_iv" not in chain.columns:
        return (var_floor, var_cap)
    iv = chain["deam_iv"].dropna()
    iv = iv[iv > 0]
    if len(iv) < 5:
        return (var_floor, var_cap)
    lo_iv, hi_iv = float(iv.quantile(0.01)), float(iv.quantile(0.99))
    lo = float(np.clip(lo_mult * lo_iv ** 2, var_floor, var_cap))
    hi = float(np.clip(hi_mult * hi_iv ** 2, lo + 1e-6, var_cap))
    return (lo, hi)


# ── IV helper ─────────────────────────────────────────────────────────────────

def _compute_iv_column(df: pd.DataFrame, r: float, q: float) -> pd.Series:
    def _row_iv(row):
        return implied_volatility(
            heston_model_price=row["mid_price"],
            S=row["spot"],
            K=row["strike"],
            r=r,
            T=row["T"],
            option_type=row["type"],
            q=q,
        )
    return df.apply(_row_iv, axis=1)


# ── Smile fitting helpers ─────────────────────────────────────────────────────

def _select_liquid_maturities(df: pd.DataFrame, min_contracts: int = 5) -> list[float]:
    """Sorted list of maturities that have at least *min_contracts* valid IV rows."""
    counts = df.groupby("T").size()
    return sorted(counts[counts >= min_contracts].index.tolist())


def _fit_smile(k: np.ndarray, iv: np.ndarray, degree: int = 2):
    """
    Fit σ_imp ≈ a + b*k + c*k² by least squares.

    Returns coefficients [a, b, c] (lowest-degree first), or None on failure.
    """
    mask = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[mask], iv[mask]
    n = len(k)
    if n < 2:
        return None
    deg = min(degree, n - 1)
    try:
        poly = np.polyfit(k, iv, deg=deg)   # highest-degree first
        coeffs = poly[::-1]                  # → [a, b, c, ...]
        while len(coeffs) < 3:
            coeffs = np.append(coeffs, 0.0) # pad with zeros if deg < 2
        return coeffs
    except Exception:
        return None


def _atm_iv_from_slice(slice_df: pd.DataFrame, atm_threshold: float) -> float:
    """
    Estimate the ATM implied vol (k = 0) for one maturity slice.

    Prefers a polynomial fit evaluated at k = 0; falls back to the
    nearest-to-ATM contract if there are too few points.
    """
    sub = slice_df[
        (slice_df["log_moneyness"].abs() < atm_threshold)
        & slice_df["market_iv"].notna()
        & (slice_df["market_iv"] > 0.01)
    ]

    if sub.empty:
        idx = slice_df["log_moneyness"].abs().idxmin()
        return float(slice_df.loc[idx, "market_iv"]) if not pd.isna(slice_df.loc[idx, "market_iv"]) else np.nan

    k  = sub["log_moneyness"].values
    iv = sub["market_iv"].values

    coeffs = _fit_smile(k, iv, degree=min(2, len(sub) - 1))
    if coeffs is None:
        return float(np.nanmedian(iv))

    # Evaluate polynomial at k = 0: value is the intercept (coeffs[0])
    return float(np.clip(coeffs[0], 0.01, 5.0))


def _estimate_smile_shape(
    slice_df: pd.DataFrame, fit_threshold: float
) -> tuple[float, float, float]:
    """
    Fit σ_imp ≈ a + b*k + c*k² and return (atm_iv=a, slope=b, curvature=c).

    Uses a wider moneyness window than _atm_iv_from_slice so there are
    enough contracts on both sides of ATM for a reliable regression.
    """
    sub = slice_df[
        (slice_df["log_moneyness"].abs() < fit_threshold)
        & slice_df["market_iv"].notna()
        & (slice_df["market_iv"] > 0.01)
    ]

    if len(sub) < 3:
        atm_iv = _atm_iv_from_slice(slice_df, fit_threshold)
        return (atm_iv if not np.isnan(atm_iv) else 0.30), 0.0, 0.01

    k      = sub["log_moneyness"].values
    iv     = sub["market_iv"].values
    coeffs = _fit_smile(k, iv, degree=2)

    if coeffs is None:
        return float(np.nanmedian(iv)), 0.0, 0.01

    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a, b, c


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_data_driven_bounds(
    df: pd.DataFrame,
    r: float = 0.0,
    q: float = 0.0,
    atm_threshold: float = 0.10,
    fit_threshold: float = 0.15,
    min_contracts: int = 5,
) -> dict:
    """
    Derive data-driven initial guess and search bounds for Heston calibration.

    Parameters
    ----------
    df : DataFrame
        Option chain after ensure_option_frame(). Must contain:
        mid_price, spot, strike, T, type.
        market_iv is computed internally if absent.
    r, q : float
        Risk-free rate and continuous dividend yield.
    atm_threshold : float
        |log(K/S)| cutoff for ATM IV estimation (default 10%).
    fit_threshold : float
        |log(K/S)| cutoff for the skew/curvature regression (default 15%).
    min_contracts : int
        Minimum valid-IV contracts per maturity to count as liquid.

    Returns
    -------
    dict with keys:
        'initial_guess' : [v0, kappa, theta, sigma, rho]
        'bounds'        : [(lo, hi), ...] for each parameter, same order
        'diagnostics'   : dict of intermediate derived quantities
    """
    df = df.copy()

    # Log-moneyness: k = log(K / S)
    df["log_moneyness"] = np.log(df["strike"] / df["spot"])

    # Ensure market_iv is present
    if "market_iv" not in df.columns or df["market_iv"].isna().all():
        df["market_iv"] = _compute_iv_column(df, r, q)
    else:
        missing = df["market_iv"].isna()
        if missing.any():
            df.loc[missing, "market_iv"] = _compute_iv_column(df[missing], r, q)

    # Working set: valid IV rows only
    iv_df = df[df["market_iv"].notna() & (df["market_iv"] > 0.01)].copy()

    liquid_T = _select_liquid_maturities(iv_df, min_contracts=min_contracts)

    if len(liquid_T) < 2:
        return _default_output("Fewer than 2 liquid maturities — static defaults used.")

    T_short = liquid_T[0]
    T_long  = liquid_T[-1]

    short_slice = iv_df[iv_df["T"] == T_short]
    long_slice  = iv_df[iv_df["T"] == T_long]

    # ── 1. v0: initial variance ───────────────────────────────────────────────
    # v0* = σ_ATM(T_short)²  using nearest-to-ATM strike at shortest maturity
    sigma_atm_short = _atm_iv_from_slice(short_slice, atm_threshold)
    if np.isnan(sigma_atm_short) or sigma_atm_short <= 0:
        sigma_atm_short = 0.20

    v0_star = sigma_atm_short ** 2
    v0_lo   = max(0.7  * v0_star, 1e-4)
    v0_hi   = min(1.3  * v0_star, 2.0)

    # ── 2. theta: long-run variance ───────────────────────────────────────────
    # θ* = σ_ATM(T_long)²  using ATM strike at longest maturity
    sigma_atm_long = _atm_iv_from_slice(long_slice, atm_threshold)
    if np.isnan(sigma_atm_long) or sigma_atm_long <= 0:
        sigma_atm_long = sigma_atm_short

    theta_star = sigma_atm_long ** 2
    theta_lo   = max(0.5  * theta_star, 1e-4)
    theta_hi   = min(2.0  * theta_star, 2.0)

    # ── 3 & 4. sigma (vol-of-vol) and rho (correlation) ──────────────────────
    # Fit σ_imp ≈ a + b*k + c*k²  at the short-maturity slice (richest in skew)
    _, slope_b, curv_c = _estimate_smile_shape(short_slice, fit_threshold)

    # sigma from curvature: Heston short-time approx → c ≈ σ² / 8
    c_safe     = max(curv_c, 1e-4)
    sigma_star = float(np.clip(np.sqrt(8.0 * c_safe), 0.10, 2.0))
    sigma_lo   = max(0.5 * sigma_star, 1e-4)
    sigma_hi   = min(2.0 * sigma_star, 2.0)

    # rho from skew: Heston short-time approx → b ≈ ρ * σ / 2
    # → ρ = 2 * b / σ
    rho_star = float(np.clip(2.0 * slope_b / sigma_star, -0.95, 0.95))
    rho_lo   = max(rho_star - 0.20, -0.999)
    rho_hi   = min(rho_star + 0.20,  0.999)

    # ── 5. kappa: mean reversion from ATM term structure ─────────────────────
    # slope_T = (σ_ATM(T_long) − σ_ATM(T_short)) / (T_long − T_short)
    #
    # Heston term structure for variance:
    #   E[v_T] ≈ θ + (v0 − θ) e^{−κT}
    # Differentiating and applying at T_short → 0:
    #   d(σ²)/dT ≈ −κ (v0 − θ)
    # Using d(σ²)/dT ≈ 2 σ_mid · slope_T:
    #   κ* = −2 σ_mid · slope_T / (v0* − θ*)
    slope_T   = (sigma_atm_long - sigma_atm_short) / (T_long - T_short)
    sigma_mid = (sigma_atm_short + sigma_atm_long) / 2.0
    dv        = v0_star - theta_star

    if abs(dv) < 5e-4:
        # Near-flat term structure: κ is weakly identified; use a neutral value
        kappa_star = 2.0
    else:
        kappa_star = float(np.clip(-2.0 * sigma_mid * slope_T / dv, 0.1, 10.0))

    kappa_lo = max(0.5, 0.5 * kappa_star)
    kappa_hi = min(3.0 * kappa_star, 10.0)

    # ── Assemble result ───────────────────────────────────────────────────────
    initial_guess = [v0_star, kappa_star, theta_star, sigma_star, rho_star]
    bounds = [
        (v0_lo,    v0_hi),
        (kappa_lo, kappa_hi),
        (theta_lo, theta_hi),
        (sigma_lo, sigma_hi),
        (rho_lo,   rho_hi),
    ]

    diagnostics = {
        "T_short":             T_short,
        "T_long":              T_long,
        "sigma_atm_short":     sigma_atm_short,
        "sigma_atm_long":      sigma_atm_long,
        "slope_T":             slope_T,
        "smile_slope_b":       slope_b,
        "smile_curvature_c":   curv_c,
        "n_liquid_maturities": len(liquid_T),
        "liquid_maturities":   liquid_T,
    }

    return {
        "initial_guess": initial_guess,
        "bounds":        bounds,
        "diagnostics":   diagnostics,
    }


def _default_output(reason: str = "") -> dict:
    """Fallback when data is insufficient to estimate bounds."""
    return {
        "initial_guess": [0.04, 2.0, 0.04, 0.5, -0.70],
        "bounds": [
            (1e-4, 2.0),
            (1e-4, 10.0),
            (1e-4, 2.0),
            (1e-4, 2.0),
            (-0.999, 0.999),
        ],
        "diagnostics": {"warning": reason or "Static defaults used."},
    }
