"""
Semi-closed-form European Heston prices via Fourier inversion (scipy.integrate.quad)
over the legacy characteristic function (models/Heston_cf).

This is the adaptive-quadrature pricer. The fast, vectorised Gauss-Legendre
pricer in pricing/european_gl.py (Cui CF, with analytic gradient) is what both
pricing and calibration use now, so this module is retained for reference /
cross-checking only and is no longer on the live pricing path.

Downstream: none (legacy; superseded by pricing/european_gl.py).
"""

import numpy as np
from scipy.integrate import quad
from models.Heston_cf import heston_cf



def P_integral(params, S0, K, r, T, j, q=0.0):
    integrand = lambda u: np.real(np.exp(-1j*u*np.log(K)) * heston_cf(u, params, S0, r, T, j, q=q) / (1j*u))
    # The Fourier-inversion upper limit must scale with maturity. The integrand's
    # decay length in u is ~1/sqrt(v*T), so at short T (e.g. a few hours) it decays
    # slowly and a fixed cap (the old value 100) chops off real mass — giving wrong
    # P1/P2 and even negative prices. Scale the cap ~1/sqrt(T); the upper bound 20000
    # keeps runtime sane for near-expiry contracts, the lower bound 200 is ample for
    # normal maturities (and stays within the legacy CF's stable range at large T).
    upper = min(20000.0, max(200.0, 200.0 / np.sqrt(max(T, 1e-12))))
    return 0.5 + (1/np.pi) * quad(integrand, 0, upper, limit=200)[0]


def heston_european_call_price(S0, K, r, T, params, q=0.0):
    P1 = P_integral(params, S0, K, r, T, 1, q=q)
    P2 = P_integral(params, S0, K, r, T, 2, q=q)
    return S0*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2

def heston_european_put_price(S0, K, r, T, params, q=0.0):
    P1 = P_integral(params, S0, K, r, T, 1, q=q)
    P2 = P_integral(params, S0, K, r, T, 2, q=q)
    return K*np.exp(-r*T)*(1-P2) - S0*np.exp(-q*T)*(1-P1)