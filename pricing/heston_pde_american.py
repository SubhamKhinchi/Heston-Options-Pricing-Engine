import numpy as np

try:
    from pricing.heston_mcs import heston_pde_american as _heston_pde_american_cpp
    _USE_CPP = True
except ImportError:
    _USE_CPP = False


def _heston_pde_american_python(
    S0, K, r, q, T, v0, kappa, theta, sigma, rho,
    option_type, Ns=40, Nv=20, Nt=40,
):
    """
    American option under Heston via Modified Craig-Sneyd (MCS) ADI.

    Unconditionally stable and 2nd-order in both time and space for all
    correlation values rho in (-1, 1).

    Reference: In 't Hout & Foulon (2010), "ADI finite difference schemes
    for option pricing in the Heston model with correlation."
    Optimal stability parameter: theta_adi = 1 - 1/sqrt(2) ≈ 0.2929.

    Grid layout
    -----------
    S in [0, Smax],  Smax = 4 * max(S0, K)      -- avoids truncation at large spots
    v in [0, vmax],  vmax = clip(5*v0, 0.5, 3.0) -- adaptive variance ceiling

    Boundary conditions
    -------------------
    S=0     : call -> 0,  put  -> K * exp(-r*tau)   (Dirichlet)
    S=Smax  : call -> Smax*exp(-q*tau) - K*exp(-r*tau),  put -> 0  (Dirichlet)
    v=0     : degenerate boundary — evolved only by S-direction terms
    v=vmax  : linear extrapolation (zero curvature Neumann)
    """
    th = 1.0 - 1.0 / np.sqrt(2.0)   # MCS stability parameter

    Smax = 4.0 * max(S0, K)
    vmax = float(np.clip(5.0 * v0, 0.5, 3.0))

    dS = Smax / Ns
    dv = vmax / Nv
    # Enforce at least 50 steps/year so dt ≤ 0.02 for long-dated contracts.
    # MCS is unconditionally stable but the explicit predictor stage can overflow
    # when dt is too large relative to the diffusion coefficients at large S.
    Nt = max(Nt, int(np.ceil(T * 50)))
    dt = T / Nt

    S = np.linspace(0.0, Smax, Ns + 1)   # (Ns+1,)
    v = np.linspace(0.0, vmax, Nv + 1)   # (Nv+1,)

    if option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    V = np.outer(payoff, np.ones(Nv + 1))   # (Ns+1, Nv+1)

    # ── Precompute operator coefficients ────────────────────────────────
    # L_S: acts on interior S nodes i=1..Ns-1 for every v level j.
    # At each (i,j): L_S V = aS*V[i-1,j] + bS*V[i,j] + cS*V[i+1,j]
    Si = S[1:Ns, None]   # (Ns-1, 1)  broadcast over j
    vj = v[None, :]      # (1,  Nv+1) broadcast over i

    aS = 0.5 * vj * Si**2 / dS**2 - (r - q) * Si / (2.0 * dS)   # (Ns-1, Nv+1)
    bS = -vj * Si**2 / dS**2 - 0.5 * r
    cS = 0.5 * vj * Si**2 / dS**2 + (r - q) * Si / (2.0 * dS)

    # L_v: acts on interior v nodes j=1..Nv-1; coefficients are S-independent.
    # At each (i,j): L_v V = av*V[i,j-1] + bv*V[i,j] + cv*V[i,j+1]
    vj_int = v[1:Nv]   # (Nv-1,)
    av = 0.5 * sigma**2 * vj_int / dv**2 - kappa * (theta - vj_int) / (2.0 * dv)
    bv = -sigma**2 * vj_int / dv**2 - 0.5 * r
    cv = 0.5 * sigma**2 * vj_int / dv**2 + kappa * (theta - vj_int) / (2.0 * dv)

    # L_mix: rho*sigma*v*S * d²V/dSdv  (4-point cross stencil)
    mix = (rho * sigma / (4.0 * dS * dv)) * np.outer(S[1:Ns], v[1:Nv])  # (Ns-1, Nv-1)

    # ── Operator application ─────────────────────────────────────────────

    def apply_LS(W):
        out = np.zeros_like(W)
        out[1:Ns, :] = aS * W[:Ns-1, :] + bS * W[1:Ns, :] + cS * W[2:Ns+1, :]
        return out

    def apply_Lv(W):
        out = np.zeros_like(W)
        out[:, 1:Nv] = av * W[:, :Nv-1] + bv * W[:, 1:Nv] + cv * W[:, 2:Nv+1]
        return out

    def apply_Lmix(W):
        out = np.zeros_like(W)
        out[1:Ns, 1:Nv] = mix * (
            W[2:Ns+1, 2:Nv+1] - W[2:Ns+1, :Nv-1]
            - W[:Ns-1, 2:Nv+1] + W[:Ns-1, :Nv-1]
        )
        return out

    def apply_L(W):
        return apply_LS(W) + apply_Lv(W) + apply_Lmix(W)

    # ── Vectorized Thomas algorithm ───────────────────────────────────────

    def thomas_s(lo, di, up, rhs):
        """
        Batch Thomas solver for S-sweeps.
        lo, di, up, rhs: (Ns-1, Nv+1) — one independent tridiagonal per column j.
        Returns solution (Ns-1, Nv+1).
        Each column j is solved independently because aS/bS/cS vary with v[j].
        """
        n = lo.shape[0]
        cp = np.zeros_like(up)
        dp = np.zeros_like(rhs)
        cp[0] = up[0] / di[0]
        dp[0] = rhs[0] / di[0]
        for i in range(1, n):
            denom = di[i] - lo[i] * cp[i - 1]
            cp[i] = up[i] / denom
            dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / denom
        x = np.zeros_like(rhs)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def thomas_v(lo, di, up, rhs):
        """
        Batch Thomas solver for v-sweeps.
        lo, di, up: (Nv-1,) — same tridiagonal for every S-row.
        rhs: (Ns+1, Nv-1) — one RHS per row i.
        Returns solution (Ns+1, Nv-1).
        All rows share the same LU factors; forward-sweep scalars broadcast over i.
        """
        n = len(lo)
        cp = np.zeros(n)
        dp = np.zeros_like(rhs)   # (Ns+1, Nv-1)
        cp[0] = up[0] / di[0]
        dp[:, 0] = rhs[:, 0] / di[0]
        for j in range(1, n):
            denom = di[j] - lo[j] * cp[j - 1]
            cp[j] = up[j] / denom
            dp[:, j] = (rhs[:, j] - lo[j] * dp[:, j - 1]) / denom
        x = np.zeros_like(rhs)
        x[:, -1] = dp[:, -1]
        for j in range(n - 2, -1, -1):
            x[:, j] = dp[:, j] - cp[j] * x[:, j + 1]
        return x

    # ── Implicit sweep helpers ────────────────────────────────────────────

    def s_sweep(RHS, bc_lo, bc_hi):
        """Solve (I - th*dt*L_S) Y = RHS with Dirichlet BCs at i=0 and i=Ns."""
        lo = -th * dt * aS
        di = 1.0 - th * dt * bS
        up = -th * dt * cS
        rhs_int = RHS[1:Ns, :].copy()
        rhs_int[0, :]  -= lo[0, :] * bc_lo   # absorb left boundary
        rhs_int[-1, :] -= up[-1, :] * bc_hi  # absorb right boundary
        Y = RHS.copy()
        Y[1:Ns, :] = thomas_s(lo, di, up, rhs_int)
        Y[0, :] = bc_lo
        Y[Ns, :] = bc_hi
        return Y

    def v_sweep(RHS, bc_bot, bc_top):
        """Solve (I - th*dt*L_v) Y = RHS with BCs at j=0 and j=Nv."""
        lo = -th * dt * av
        di = 1.0 - th * dt * bv
        up = -th * dt * cv
        rhs_int = RHS[:, 1:Nv].copy()
        rhs_int[:, 0]  -= lo[0]  * bc_bot   # absorb bottom boundary
        rhs_int[:, -1] -= up[-1] * bc_top   # absorb top boundary
        Y = RHS.copy()
        Y[:, 1:Nv] = thomas_v(lo, di, up, rhs_int)
        Y[:, 0]  = bc_bot
        Y[:, Nv] = bc_top
        return Y

    def boundary_s(tau):
        """Dirichlet values at S=0 and S=Smax for time-to-maturity tau."""
        disc_r = np.exp(-r * tau)
        disc_q = np.exp(-q * tau)
        if option_type == "call":
            lo = np.zeros(Nv + 1)
            hi = np.full(Nv + 1, max(Smax * disc_q - K * disc_r, 0.0))
        else:
            lo = np.full(Nv + 1, K * disc_r)
            hi = np.zeros(Nv + 1)
        return lo, hi

    def neumann_top(W):
        """Zero-curvature (linear) extrapolation at v=vmax."""
        W[:, Nv] = 2.0 * W[:, Nv - 1] - W[:, Nv - 2]
        return W

    # ── MCS ADI time loop ────────────────────────────────────────────────
    # Six stages per step; rolling backwards from terminal payoff to t=0.
    for n in range(Nt):
        tau = (n + 1) * dt
        bc_lo, bc_hi = boundary_s(tau)

        LV  = apply_L(V)
        LSV = apply_LS(V)
        LvV = apply_Lv(V)

        # Stage 1 — explicit predictor
        Y0 = V + dt * LV

        # Stage 2 — implicit S sweep: (I - th*dt*L_S) Y1 = Y0 - th*dt*L_S*V
        Y1 = s_sweep(Y0 - th * dt * LSV, bc_lo, bc_hi)
        neumann_top(Y1)

        # Stage 3 — implicit v sweep: (I - th*dt*L_v) Y2 = Y1 - th*dt*L_v*V
        # v=0 BC comes from the S sweep result Y1[:,0] (degenerate boundary)
        Y2 = v_sweep(Y1 - th * dt * LvV, Y1[:, 0], Y1[:, Nv])
        Y2[0, :] = bc_lo
        Y2[Ns, :] = bc_hi
        neumann_top(Y2)

        # Stage 4 — corrected predictor: trapezoidal average of L at V and Y2
        LY2  = apply_L(Y2)
        LSY2 = apply_LS(Y2)
        LvY2 = apply_Lv(Y2)
        Y0t = V + 0.5 * dt * (LV + LY2)

        # Stage 5 — implicit S sweep on corrected: (I - th*dt*L_S) Y1t = Y0t - th*dt*L_S*Y2
        Y1t = s_sweep(Y0t - th * dt * LSY2, bc_lo, bc_hi)
        neumann_top(Y1t)

        # Stage 6 — implicit v sweep on corrected (final): (I - th*dt*L_v) Vnew = Y1t - th*dt*L_v*Y2
        Vnew = v_sweep(Y1t - th * dt * LvY2, Y1t[:, 0], Y1t[:, Nv])
        Vnew[0, :] = bc_lo
        Vnew[Ns, :] = bc_hi
        neumann_top(Vnew)

        # American early-exercise constraint: floor at intrinsic value
        np.maximum(Vnew, payoff[:, None], out=Vnew)

        V = Vnew

    # ── Bilinear interpolation at (S0, v0) ───────────────────────────────
    i0 = np.clip(int(np.searchsorted(S, S0)) - 1, 0, Ns - 1)
    j0 = np.clip(int(np.searchsorted(v, v0)) - 1, 0, Nv - 1)
    wS = (S0 - S[i0]) / dS
    wv = (v0 - v[j0]) / dv

    return float(
        (1 - wS) * (1 - wv) * V[i0,     j0    ]
        + wS     * (1 - wv) * V[i0 + 1, j0    ]
        + (1 - wS) * wv     * V[i0,     j0 + 1]
        + wS     * wv       * V[i0 + 1, j0 + 1]
    )


def heston_pde_american(
    S0, K, r, q, T, v0, kappa, theta, sigma, rho,
    option_type, Ns=40, Nv=20, Nt=40,
):
    """Dispatch to C++ MCS ADI solver when available, else fall back to Python."""
    if _USE_CPP:
        return _heston_pde_american_cpp(
            S0=S0, K=K, r=r, q=q, T=T,
            v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            option_type=option_type, Ns=Ns, Nv=Nv, Nt=Nt,
        )
    return _heston_pde_american_python(
        S0, K, r, q, T, v0, kappa, theta, sigma, rho,
        option_type, Ns, Nv, Nt,
    )
