# ---------------------------
# 1️⃣ Heston Path Simulations
# ---------------------------

import numpy as np
def simulate_heston_paths_american_put_without_dividends(S0, r, T, v0, kappa, theta, sigma, rho, M, N):
    """Simulate Heston paths without dividends (American Put pricing)"""
    dt = T / M
    S = np.zeros((N, M+1))
    v = np.zeros((N, M+1))
    S[:,0] = S0
    v[:,0] = v0

    Z1 = np.random.normal(size=(N, M))
    Z2 = np.random.normal(size=(N, M))
    W1 = np.sqrt(dt) * Z1
    W2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
    
    
    for t in range(M):
        #Variance discretization process - Euler discretization
        # Full truncation Euler for variance
        v[:, t+1] = np.maximum(
            v[:, t]
            + kappa * (theta - v[:, t]) * dt
            + sigma * np.sqrt(np.maximum(v[:, t], 0)) * W2[:, t],
            0
        )
        # Log-Euler for stock
        S[:, t+1] = S[:, t] * np.exp(
            (r - 0.5 * v[:, t]) * dt
            + np.sqrt(np.maximum(v[:, t], 0)) * W1[:, t]
        )
    return S, v, dt


def simulate_heston_paths_american_put_with_dividends(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q):
    """
    Simulate Heston paths under risk-neutral measure with continuous dividends.

    Parameters:
        q : continuous dividend yield

    Returns:
        S : stock paths (N x M+1)
        v : variance paths (N x M+1)
        dt : time step
    """
    dt = T / M
    S = np.zeros((N, M+1))
    v = np.zeros((N, M+1))
    S[:,0] = S0
    v[:,0] = v0

    Z1 = np.random.normal(size=(N, M))
    Z2 = np.random.normal(size=(N, M))
    W1 = np.sqrt(dt) * Z1
    W2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
    
    
    for t in range(M):
        #Variance discretization process - Euler discretization
        # Full truncation Euler for variance
        v[:, t+1] = np.maximum(
            v[:, t]
            + kappa * (theta - v[:, t]) * dt
            + sigma * np.sqrt(np.maximum(v[:, t], 0)) * W2[:, t],
            0
        )
        #only stock's drift part will change
        # Log-Euler for stock 
        S[:, t+1] = S[:, t] * np.exp(
            (r - q- 0.5 * v[:, t]) * dt
            + np.sqrt(np.maximum(v[:, t], 0)) * W1[:, t]
        )
    return S, v, dt
    


def simulate_heston_paths_american_call_with_dividends(S0, r, T, v0, kappa, theta, sigma, rho, M, N, q):
    """Simulate Heston paths with dividend yield q (American Call pricing)"""
    dt = T / N
    S = np.zeros((M, N+1))
    v = np.zeros((M, N+1))
    S[:,0] = S0
    v[:,0] = v0

    Z1 = np.random.normal(size=(M, N))
    Z2 = np.random.normal(size=(M, N))
    W1 = np.sqrt(dt) * Z1
    W2 = np.sqrt(dt) * (rho*Z1 + np.sqrt(1 - rho**2)*Z2)

    for t in range(N):
        v[:,t+1] = np.maximum(v[:,t] + kappa*(theta - v[:,t])*dt + sigma*np.sqrt(v[:,t])*W2[:,t], 0)
        S[:,t+1] = S[:,t] * np.exp((r - q - 0.5*v[:,t])*dt + np.sqrt(v[:,t])*W1[:,t])
    return S, v, dt
