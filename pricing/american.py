import numpy as np
from typing import Literal

#import functions for american puts/calls
from simulation.heston_path import simulate_heston_paths_american_put_without_dividends, simulate_heston_paths_american_put_with_dividends, simulate_heston_paths_american_call_with_dividends
#american call without dividends is same as european call
from models.heston_european import heston_european_call_price

from simulation.lsmc import american_put_lsmc_vec, american_call_lsmc_vec

def american_put_without_dividends(S0, K, T, r, v0, kappa, theta, sigma, rho, M, N):
    """
    Wrapper function that:
    1. Simulates Heston paths
    2. Applies LSMC
    """
    S, v, dt = simulate_heston_paths_american_put_without_dividends( S0, T, r,v0, kappa, theta, sigma, rho, M, N)

    american_put_price_without_dividends = american_put_lsmc_vec(S, K, r, dt)

    return american_put_price_without_dividends 

def american_put_with_dividends(S0, K, T, r, v0, kappa, theta, sigma, rho, M, N, q):
    """
    Wrapper function that:
    1. Simulates Heston paths
    2. Applies LSMC
    """
    S, v, dt = simulate_heston_paths_american_put_with_dividends( S0, T, r,v0, kappa, theta, sigma, rho, M, N, q)

    american_put_price_with_dividends = american_put_lsmc_vec(S, K, r, dt)

    return american_put_price_with_dividends

#American call options without dividends (same as european call price)
def american_call_without_dividends(S0, K, T, r, v0, kappa, theta, sigma, rho):

    params = (v0 ,kappa, theta, sigma, rho)

    american_call_price_without_dividends = heston_european_call_price(
        S0=S0,
        K=K,
        r=r,
        T=T,
        params=params,
    )
    return american_call_price_without_dividends



def american_call_with_dividends(S0, K, T, r, v0, kappa, theta, sigma, rho, M, N, q):

    # 1️⃣ Simulate paths
    S, v, dt = simulate_heston_paths_american_call_with_dividends(S0, T, r, v0, kappa, theta, sigma, rho, M, N, q
    )
    #pricing from lsmc
    american_call_price_with_dividends = american_call_lsmc_vec(S, K, r, dt)
    return american_call_price_with_dividends
