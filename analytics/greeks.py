from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_greeks(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> dict[str, float]:
    """
    Return Black-Scholes greeks with continuous dividend yield.

    Invalid or expired inputs produce NaNs so the analytics layer can filter
    them without raising.
    """
    try:
        S = float(S)
        K = float(K)
        T = float(T)
        sigma = float(sigma)
    except Exception:
        return {name: np.nan for name in ("delta", "gamma", "vega", "theta", "rho")}

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {name: np.nan for name in ("delta", "gamma", "vega", "theta", "rho")}

    option_type = option_type.lower()
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -(S * disc_q * pdf * sigma) / (2 * sqrt_t)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -(S * disc_q * pdf * sigma) / (2 * sqrt_t)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        return {name: np.nan for name in ("delta", "gamma", "vega", "theta", "rho")}

    gamma = disc_q * pdf / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf * sqrt_t

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }

