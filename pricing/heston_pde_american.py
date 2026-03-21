import numpy as np

def heston_pde_american(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type, Ns, Nv, Nt):
    Smax = 3 * S0
    vmax = 1.0

    dS = Smax / Ns
    dv = vmax / Nv
    dt = T / Nt

    S_grid = np.linspace(0, Smax, Ns+1)
    v_grid = np.linspace(0, vmax, Nv+1)

    V = np.zeros((Ns+1, Nv+1))

    # terminal payoff
    for i,S in enumerate(S_grid):
        if option_type == "call":
            V[i,:] = np.maximum(S-K,0)
        else:
            V[i,:] = np.maximum(K-S,0)

    for t in range(Nt-1, -1, -1):

        V_new = V.copy()

        for i in range(1, Ns):
            for j in range(1, Nv):

                S = S_grid[i]
                v = v_grid[j]

                dVdS = (V[i+1,j]-V[i-1,j])/(2*dS)
                d2VdS2 = (V[i+1,j]-2*V[i,j]+V[i-1,j])/(dS**2)

                dVdv = (V[i,j+1]-V[i,j-1])/(2*dv)
                d2Vdv2 = (V[i,j+1]-2*V[i,j]+V[i,j-1])/(dv**2)

                d2VdSdv = (
                    V[i+1,j+1] - V[i+1,j-1]
                    - V[i-1,j+1] + V[i-1,j-1]
                )/(4*dS*dv)

                driftS = (r-q)*S*dVdS
                driftV = kappa*(theta-v)*dVdv

                diffS = 0.5*v*S**2*d2VdS2
                diffV = 0.5*sigma**2*v*d2Vdv2

                cross = rho*sigma*v*S*d2VdSdv

                V_new[i,j] = V[i,j] + dt*(
                    driftS + driftV
                    + diffS + diffV + cross
                    - r*V[i,j]
                )

        # American constraint
        for i,S in enumerate(S_grid):
            if option_type=="call":
                payoff = max(S-K,0)
            else:
                payoff = max(K-S,0)

            V_new[i,:] = np.maximum(V_new[i,:], payoff)

        V = V_new

    # interpolate at S0,v0
    i = int(S0/dS)
    j = int(v0/dv)

    return V[i,j]