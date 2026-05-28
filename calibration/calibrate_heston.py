"""
Heston calibration via the Levenberg-Marquardt method (Cui et al., 2016).

Replaces the previous L-BFGS-B / scalar-MSE approach with:
  • Nonlinear least-squares formulation  r(θ) = C_model − C_market
  • Analytical Jacobian from Theorem 1 of the paper (10–16× faster
    than finite-difference gradient, same number of integrals as pricing)
  • scipy.optimize.least_squares with method='trf' (Trust-Region
    Reflective) — scipy's bounded equivalent of LM; uses the same
    Gauss-Newton / damped-Newton step structure as Algorithm 4.1.

Stopping tolerances mirror the paper: ε₁ = ε₂ = ε₃ = 1e-10.
"""

import numpy as np
from scipy.optimize import least_squares

from calibration.heston_loss_function import heston_residuals, heston_jacobian


# Default parameter bounds matching the paper's validation ranges
# (Table 5) extended slightly for equity options.
_DEFAULT_BOUNDS = (
    [1e-4, 1e-4, 1e-4, 1e-4, -0.999],   # lower: [v0, kappa, theta, sigma, rho]
    [2.0,  10.0, 2.0,  1.0,   0.999],   # upper: σ ≤ 1.0 keeps the CF numerically stable
)


def calibrate_heston(
    r,
    q,
    option_df,
    Ns,
    Nv,
    Nt,
    initial_guess,
    bounds=None,
    objective="price",        # kept for API compatibility; LM always uses price residuals
    pricing_mode="european_proxy",
):
    """
    Calibrate Heston parameters via Levenberg-Marquardt (Cui et al., 2016).

    Parameters
    ----------
    r, q            : float  — risk-free rate, dividend yield
    option_df       : DataFrame with columns expected by heston_residuals
    Ns, Nv, Nt      : PDE grid sizes (used only for American contracts)
    initial_guess   : list/array of length 5 — [v0, kappa, theta, sigma, rho]
    bounds          : optional list of (lower, upper) tuples length 5,
                      or None to use the defaults above
    objective       : ignored (kept for backward-compatible call sites)
    pricing_mode    : "european_proxy" (default) or "auto"

    Returns
    -------
    params_opt : np.ndarray shape (5,)  — [v0, kappa, theta, sigma, rho]
    loss_val   : float — ½‖r(θ*)‖² (NLS objective at optimum)
    """
    if bounds is None:
        lb, ub = _DEFAULT_BOUNDS
    else:
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

    # Shared args tuple passed to residuals and Jacobian
    extra = (r, q, option_df, Ns, Nv, Nt, pricing_mode)

    def residuals_fn(params):
        return heston_residuals(params, *extra)

    # Use the analytical Jacobian only when every contract is European
    # (pricing_mode == "european_proxy" guarantees this).  For full mode
    # with American options, fall back to scipy's 2-point finite differences
    # so gradient accuracy is not silently lost.
    all_european = (
        pricing_mode == "european_proxy"
        or (
            "ExerciseStyle" in option_df.columns
            and (option_df["ExerciseStyle"].str.lower() == "european").all()
        )
    )

    if all_european:
        def jacobian_fn(params):
            return heston_jacobian(params, *extra)
        jac_arg = jacobian_fn
    else:
        jac_arg = "2-point"   # scipy finite-difference fallback for American options

    result = least_squares(
        residuals_fn,
        x0=np.array(initial_guess, dtype=float),
        jac=jac_arg,
        bounds=(lb, ub),
        method="trf",          # Trust-Region Reflective — bounded LM equivalent
        ftol=1e-10,            # ε₁: stop on small ‖r‖ improvement  (Eq. 36a)
        gtol=1e-10,            # ε₂: stop on small gradient norm    (Eq. 36b)
        xtol=1e-10,            # ε₃: stop on small parameter update (Eq. 36c)
        max_nfev=200,
    )

    params_opt = result.x
    loss_val = 0.5 * float(np.dot(result.fun, result.fun))   # ½‖r‖²

    return params_opt, loss_val
