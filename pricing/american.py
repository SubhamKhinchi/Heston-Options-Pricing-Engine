import numpy as np

from simulation.heston_path import simulate_heston_paths
from models.heston_european import heston_european_call_price
from simulation.lsmc import american_put_lsmc_vec, american_call_lsmc_vec


def american_put_without_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho, M, N):
    S, v, dt = simulate_heston_paths(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q=0.0)
    return american_put_lsmc_vec(S, K, r, dt)


def american_put_with_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho, M, N, q):
    S, v, dt = simulate_heston_paths(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q=q)
    return american_put_lsmc_vec(S, K, r, dt)


def american_call_without_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho):
    # Early exercise of a call with no dividends is never optimal → European price.
    params = (v0, kappa, theta, sigma, rho)
    return heston_european_call_price(S0=S0, K=K, r=r, T=T, params=params)


def american_call_with_dividends(S0, K, r, T, v0, kappa, theta, sigma, rho, M, N, q):
    S, v, dt = simulate_heston_paths(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q=q)
    return american_call_lsmc_vec(S, K, r, dt)
