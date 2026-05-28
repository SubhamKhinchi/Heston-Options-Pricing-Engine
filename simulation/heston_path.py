import numpy as np


def simulate_heston_paths(
    S0: float,
    r: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    M: int,
    N: int,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate Heston model paths under the risk-neutral measure.

    Uses full-truncation Euler for variance and log-Euler for the stock price.

    Parameters
    ----------
    S0, r, T, v0, kappa, theta, sigma, rho : Heston model inputs
    M : number of time steps
    N : number of Monte Carlo paths
    q : continuous dividend yield (default 0)

    Returns
    -------
    S  : stock paths, shape (N, M+1)
    v  : variance paths, shape (N, M+1)
    dt : time step size T/M
    """
    dt = T / M
    S = np.zeros((N, M + 1))
    v = np.zeros((N, M + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    Z1 = np.random.normal(size=(N, M))
    Z2 = np.random.normal(size=(N, M))
    W1 = np.sqrt(dt) * Z1
    W2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

    for t in range(M):
        # Full-truncation Euler for variance (prevents negative variance)
        v[:, t + 1] = np.maximum(
            v[:, t]
            + kappa * (theta - v[:, t]) * dt
            + sigma * np.sqrt(np.maximum(v[:, t], 0)) * W2[:, t],
            0,
        )
        # Log-Euler for stock price
        S[:, t + 1] = S[:, t] * np.exp(
            (r - q - 0.5 * v[:, t]) * dt
            + np.sqrt(np.maximum(v[:, t], 0)) * W1[:, t]
        )

    return S, v, dt


# ---------------------------------------------------------------------------
# Backward-compatible aliases — existing callers can keep using these names.
# All three delegate to simulate_heston_paths with the correct convention:
#   M = time steps, N = paths.
# ---------------------------------------------------------------------------

def simulate_heston_paths_american_put_without_dividends(S0, r, T, v0, kappa, theta, sigma, rho, M, N):
    return simulate_heston_paths(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q=0.0)


def simulate_heston_paths_american_put_with_dividends(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q):
    return simulate_heston_paths(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q=q)


def simulate_heston_paths_american_call_with_dividends(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q):
    # Previously had M and N swapped (M was used as paths, N as time steps).
    # Now fixed: M = time steps, N = paths — consistent with put functions.
    return simulate_heston_paths(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q=q)
