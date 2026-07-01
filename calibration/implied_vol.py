"""
European Black-Scholes implied volatility by Brent root-finding.

There is no closed form for sigma given a price, so we bracket and solve
BS(sigma) = target_price numerically, with no-arbitrage bound checks. Used to
invert market mid-prices and model prices into IV across the analytics and
calibration layers (and as the European leg of de-Americanization).
"""

import numpy as np
from scipy.optimize import brentq
from models.black_scholes import black_scholes_price

def implied_volatility(heston_model_price, S, K, r, T, option_type, q):
    # Basic validation
    try:
        S = float(S)
        K = float(K)
        T = float(T)
        heston_model_price = float(heston_model_price)
    except Exception:
        return np.nan

    if S <= 0 or K <= 0 or T <= 0 or heston_model_price <= 0:
        return np.nan

    # Arbitrage bounds for European options with continuous dividend yield
    forward_spot = S * np.exp(-q * T)
    disc_strike = K * np.exp(-r * T)

    if option_type.lower() == 'call':
        lower_bound = max(0.0, forward_spot - disc_strike)
        upper_bound = forward_spot
    elif option_type.lower() == 'put':
        lower_bound = max(0.0, disc_strike - forward_spot)
        upper_bound = disc_strike
    else:
        return np.nan

    tol = 1e-12
    # If price violates arbitrage bounds, it's not a valid BS price
    if (heston_model_price + tol) < lower_bound or (heston_model_price - tol) > upper_bound:
        return np.nan

    def objective(sigma):
        # Protect against bad sigma values
        if sigma <= 0:
            sigma = 1e-12
        try:
            return black_scholes_price(S, K, r, T, sigma, option_type, q) - heston_model_price
        except Exception:
            return np.nan

    # Try to bracket a root. Start with reasonable bounds and expand if necessary.
    sigma_low = 1e-8
    sigma_high = 5.0

    try:
        f_low = objective(sigma_low)
        f_high = objective(sigma_high)

        # If any endpoint evaluation returned NaN, fail gracefully
        if np.isnan(f_low) or np.isnan(f_high):
            return np.nan

        # Expand the upper bound until we find a sign change or hit a max
        max_high = 200.0
        while f_low * f_high > 0 and sigma_high < max_high:
            sigma_high *= 2.0
            f_high = objective(sigma_high)
            if np.isnan(f_high):
                return np.nan

        if f_low * f_high > 0:
            return np.nan

        iv = brentq(objective, sigma_low, sigma_high)
        return float(iv)
    except Exception:
        return np.nan