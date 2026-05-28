"""
Heston calibration via L-BFGS-B with scalar MSE price loss.

Classical quasi-Newton approach: minimise mean-squared price residuals
using scipy's L-BFGS-B with finite-difference gradients.  Uses the same
fast Gauss-Legendre pricing as the LM method so that the comparison between
the two calibrators isolates optimizer + formulation, not pricing speed.

Differences from the Cui et al. (2016) LM approach:
  - Loss function : scalar MSE  vs  NLS residual vector
  - Gradient      : numerical finite differences (~5–10 loss evaluations/step)
                    vs analytical Jacobian (~2n integrals, same as pricing)
  - Hessian approx: BFGS quasi-Newton  vs  Gauss-Newton (J^T J)
  - Constraints   : Feller penalty added to scalar loss
"""

import numpy as np
from scipy.optimize import minimize

from pricing.european_gl import heston_call_gl, heston_put_gl
from calibration.heston_loss_function import FELLER_WEIGHT


# Same σ cap as the LM method for numerical stability in pricing
_DEFAULT_BOUNDS = [
    (1e-4, 2.0),      # v0
    (1e-4, 10.0),     # kappa
    (1e-4, 2.0),      # theta
    (1e-4, 1.0),      # sigma ≤ 1.0 (prevents CF overflow at long maturities)
    (-0.999, 0.999),  # rho
]


def _scalar_mse_loss(params, r, q, options_df):
    """
    Scalar MSE price loss + soft Feller penalty.

    loss = mean_i[(C_model_i - C_market_i)²] + Feller_penalty

    The Feller penalty is scaled to be in the same units as the MSE term:
    it equals (FELLER_WEIGHT * gap)² / n, consistent with adding the Feller
    residual to the LM residual vector before squaring.
    """
    v0, kappa, theta, sigma, rho = params
    if any(p <= 0 for p in [v0, kappa, theta, sigma]) or not (-1 < rho < 1):
        return 1e10

    sq_errors = []
    for row in options_df.itertuples(index=False):
        market = row.mid_price
        if market <= 0:
            continue
        S0, K, T, opt_type = float(row.spot), float(row.strike), float(row.T), row.type
        if T <= 0:
            continue
        try:
            price = (
                heston_call_gl(S0, K, r, T, v0, kappa, theta, sigma, rho)
                if opt_type == "call"
                else heston_put_gl(S0, K, r, T, v0, kappa, theta, sigma, rho)
            )
        except Exception:
            continue
        sq_errors.append((price - market) ** 2)

    if not sq_errors:
        return 1e10

    n = len(sq_errors)
    mse = float(np.mean(sq_errors))

    # Feller soft penalty — same weight as LM for a consistent comparison
    feller_gap = max(0.0, sigma ** 2 - 2.0 * kappa * theta)
    feller_penalty = (FELLER_WEIGHT * feller_gap) ** 2 / n

    return mse + feller_penalty


def calibrate_heston_lbfgsb(
    r,
    q,
    option_df,
    initial_guess,
    bounds=None,
    **_kwargs,   # absorb unused keyword args for API symmetry with LM caller
):
    """
    Calibrate Heston parameters via L-BFGS-B with scalar MSE price loss.

    Parameters
    ----------
    r, q          : float — risk-free rate, dividend yield
    option_df     : DataFrame — calibration universe (same format as LM caller)
    initial_guess : list/array length 5 — [v0, kappa, theta, sigma, rho]
    bounds        : optional list of (lo, hi) tuples length 5

    Returns
    -------
    params_opt : np.ndarray shape (5,)
    loss_val   : float — MSE at optimum
    """
    bounds_list = (
        [(b[0], b[1]) for b in bounds] if bounds is not None else _DEFAULT_BOUNDS
    )

    result = minimize(
        _scalar_mse_loss,
        x0=np.array(initial_guess, dtype=float),
        args=(r, q, option_df),
        method="L-BFGS-B",
        bounds=bounds_list,
        options={
            "maxiter": 500,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    return result.x, float(result.fun)
