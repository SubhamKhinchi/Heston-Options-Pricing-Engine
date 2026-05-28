"""
European Heston option pricing and analytical gradient via Gauss-Legendre
quadrature, using the Cui et al. (2016) characteristic function (Eq. 9 & 22).

Paper recommends 64 GL nodes truncated at ū = 200, which achieves
~10⁻⁸ accuracy for all standard maturities (Figs. 3-5 of the paper).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from models.heston_cf_cui import heston_cf_cui, heston_cf_and_gradient

# Pre-compute fixed GL nodes/weights once at module load.
_N_NODES = 64
_U_MAX = 200.0

_gl_x, _gl_w = leggauss(_N_NODES)            # nodes/weights on [-1, 1]
_U_NODES = (_U_MAX / 2.0) * (_gl_x + 1.0)   # mapped to [0, U_MAX], shape (N,)
_W_SCALED = (_U_MAX / 2.0) * _gl_w           # scaled weights, shape (N,)


def _call_price_from_integrals(S0, K, r, T, I1, I2):
    """Assemble call price from the two Fourier integrals (Eq. 9)."""
    return (0.5 * (S0 - K * np.exp(-r * T))
            + np.exp(-r * T) / np.pi * (I1 - K * I2))


def heston_call_gl(S0, K, r, T, v0, kappa, theta, sigma, rho):
    """
    European call price via 64-point GL quadrature (Eq. 9 + Cui et al. CF).

    Parameters match the project convention: (S0, K, r, T, v0, kappa, theta, sigma, rho).
    """
    u = _U_NODES  # (N,)

    phi_u = heston_cf_cui(u, v0, kappa, theta, sigma, rho, S0, r, T)
    phi_ui = heston_cf_cui(u - 1j, v0, kappa, theta, sigma, rho, S0, r, T)

    K_neg_iu = np.exp(-1j * u * np.log(K))          # K^{-iu}, shape (N,)
    kernel = K_neg_iu / (1j * u)                     # common factor, shape (N,)

    I1 = float(np.sum(_W_SCALED * np.real(kernel * phi_ui)))
    I2 = float(np.sum(_W_SCALED * np.real(kernel * phi_u)))

    return _call_price_from_integrals(S0, K, r, T, I1, I2)


def heston_put_gl(S0, K, r, T, v0, kappa, theta, sigma, rho):
    """European put via put-call parity (gradient is identical to call)."""
    call = heston_call_gl(S0, K, r, T, v0, kappa, theta, sigma, rho)
    return call - S0 + K * np.exp(-r * T)


def heston_call_price_and_gradient(S0, K, r, T, v0, kappa, theta, sigma, rho):
    """
    Return (call_price, grad) where grad has shape (5,) in project order
    [v0, kappa, theta, sigma, rho], computed via the analytical gradient
    of Theorem 1 / Eq. 22 of Cui et al. (2016).

    The vectorised GL quadrature evaluates the price and all 5 gradient
    components in a single pass over the 64 quadrature nodes.
    """
    u = _U_NODES  # (N,)

    # φ and h at u and u−i  — h has shape (5, N)
    phi_u, h_u = heston_cf_and_gradient(u, v0, kappa, theta, sigma, rho, S0, r, T)
    phi_ui, h_ui = heston_cf_and_gradient(u - 1j, v0, kappa, theta, sigma, rho, S0, r, T)

    K_neg_iu = np.exp(-1j * u * np.log(K))      # (N,)
    kernel = K_neg_iu / (1j * u)                 # (N,)

    # ---- price --------------------------------------------------------
    I1 = float(np.sum(_W_SCALED * np.real(kernel * phi_ui)))
    I2 = float(np.sum(_W_SCALED * np.real(kernel * phi_u)))
    price = _call_price_from_integrals(S0, K, r, T, I1, I2)

    # ---- gradient (Eq. 22) -------------------------------------------
    # ∂C/∂θⱼ = exp(−rT)/π · [I1ⱼ − K·I2ⱼ]
    # where I1ⱼ = ∫ Re[K^{−iu}/(iu) · φ(u−i) · hⱼ(u−i)] du
    #       I2ⱼ = ∫ Re[K^{−iu}/(iu) · φ(u)   · hⱼ(u)  ] du
    base_ui = _W_SCALED * kernel * phi_ui   # (N,)
    base_u  = _W_SCALED * kernel * phi_u    # (N,)

    factor = np.exp(-r * T) / np.pi
    grad = np.empty(5)
    for j in range(5):
        I1j = float(np.sum(np.real(h_ui[j] * base_ui)))
        I2j = float(np.sum(np.real(h_u[j]  * base_u)))
        grad[j] = factor * (I1j - K * I2j)

    return price, grad


def heston_put_price_and_gradient(S0, K, r, T, v0, kappa, theta, sigma, rho):
    """
    Put price and gradient.  By put-call parity the gradient is identical
    to the call gradient (the parity terms S0, K, r, T are θ-independent).
    """
    call_price, grad = heston_call_price_and_gradient(
        S0, K, r, T, v0, kappa, theta, sigma, rho
    )
    put_price = call_price - S0 + K * np.exp(-r * T)
    return put_price, grad
