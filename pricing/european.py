"""
European Heston call/put prices — the engine's standard European pricer.

Delegates to the Gauss-Legendre / Cui et al. (2016) pricer (pricing/european_gl),
which uses the numerically stable Cui characteristic function. This avoids the
"little Heston trap" in the legacy Heston (1993) CF (models/heston_european), whose
e^{dT} term overflows to NaN for long maturities / high vol-of-vol. It is also the
same pricer calibration uses, so pricing and calibration are fully consistent
(same CF, same continuous carry q).

Downstream: services/pricing_service.py (European exercise) and synthetic/test code.
"""

from pricing.european_gl import heston_call_gl, heston_put_gl


def heston_european_call_option(S0, K, r, T, v0, kappa, theta, sigma, rho, q=0.0):
    return heston_call_gl(S0, K, r, T, v0, kappa, theta, sigma, rho, q)


def heston_european_put_option(S0, K, r, T, v0, kappa, theta, sigma, rho, q=0.0):
    return heston_put_gl(S0, K, r, T, v0, kappa, theta, sigma, rho, q)
