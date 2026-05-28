/*
 * Heston MCS ADI PDE solver for American options — pybind11 extension.
 *
 * Modified Craig-Sneyd (MCS) ADI, θ = 1 - 1/√2 ≈ 0.2929.
 * Reference: In 't Hout & Foulon (2010).
 *
 * Build:
 *   python setup_cpp.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Row-major 2-D array stored in a flat vector.
// ---------------------------------------------------------------------------
struct G {
    std::vector<double> d;
    int R, C;

    G() : R(0), C(0) {}
    G(int r, int c, double init = 0.0) : d(r * c, init), R(r), C(c) {}

    inline double& operator()(int i, int j)       { return d[i * C + j]; }
    inline double  operator()(int i, int j) const { return d[i * C + j]; }

    G clone() const { G o(R, C); o.d = d; return o; }
};

// Elementwise: dst += src
static void add_into(G& dst, const G& src) {
    for (int k = 0; k < (int)dst.d.size(); ++k) dst.d[k] += src.d[k];
}

// ---------------------------------------------------------------------------
// Operator applications
// ---------------------------------------------------------------------------

// L_S: out[i,j] = aS[i-1,j]*W[i-1,j] + bS[i-1,j]*W[i,j] + cS[i-1,j]*W[i+1,j]
//      for i = 1..Ns-1, all j
static G apply_LS(const G& W, const G& aS, const G& bS, const G& cS, int Ns, int Nv) {
    G out(Ns + 1, Nv + 1, 0.0);
    for (int i = 1; i < Ns; ++i)
        for (int j = 0; j <= Nv; ++j)
            out(i, j) = aS(i-1,j)*W(i-1,j) + bS(i-1,j)*W(i,j) + cS(i-1,j)*W(i+1,j);
    return out;
}

// L_v: out[i,j] = av[j-1]*W[i,j-1] + bv[j-1]*W[i,j] + cv[j-1]*W[i,j+1]
//      for j = 1..Nv-1, all i
static G apply_Lv(const G& W,
                  const std::vector<double>& av,
                  const std::vector<double>& bv,
                  const std::vector<double>& cv,
                  int Ns, int Nv) {
    G out(Ns + 1, Nv + 1, 0.0);
    for (int j = 1; j < Nv; ++j)
        for (int i = 0; i <= Ns; ++i)
            out(i,j) = av[j-1]*W(i,j-1) + bv[j-1]*W(i,j) + cv[j-1]*W(i,j+1);
    return out;
}

// L_mix: out[i,j] = mix[i-1,j-1]*(W[i+1,j+1]-W[i+1,j-1]-W[i-1,j+1]+W[i-1,j-1])
//        for i = 1..Ns-1, j = 1..Nv-1
static G apply_Lmix(const G& W, const G& mix, int Ns, int Nv) {
    G out(Ns + 1, Nv + 1, 0.0);
    for (int i = 1; i < Ns; ++i)
        for (int j = 1; j < Nv; ++j)
            out(i,j) = mix(i-1,j-1) * (W(i+1,j+1) - W(i+1,j-1)
                                      - W(i-1,j+1) + W(i-1,j-1));
    return out;
}

// L = L_S + L_v + L_mix
static G apply_L(const G& W, const G& aS, const G& bS, const G& cS,
                 const std::vector<double>& av,
                 const std::vector<double>& bv,
                 const std::vector<double>& cv,
                 const G& mix, int Ns, int Nv) {
    G out = apply_LS(W, aS, bS, cS, Ns, Nv);
    add_into(out, apply_Lv(W, av, bv, cv, Ns, Nv));
    add_into(out, apply_Lmix(W, mix, Ns, Nv));
    return out;
}

// ---------------------------------------------------------------------------
// Thomas solvers
// ---------------------------------------------------------------------------

// S-sweep: solve (Ns-1) independent tridiagonals, one per column j.
// lo, di, up, rhs: (Ns-1) × (Nv+1)  — coefficients differ across both dims.
// Returns solution of shape (Ns-1) × (Nv+1).
static G thomas_s(const G& lo, const G& di, const G& up, const G& rhs, int Ns, int Nv) {
    int n = Ns - 1, m = Nv + 1;
    G cp(n, m), dp(n, m), x(n, m);

    for (int j = 0; j < m; ++j) {
        cp(0,j) = up(0,j) / di(0,j);
        dp(0,j) = rhs(0,j) / di(0,j);
    }
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            double denom = di(i,j) - lo(i,j) * cp(i-1,j);
            cp(i,j) = up(i,j) / denom;
            dp(i,j) = (rhs(i,j) - lo(i,j) * dp(i-1,j)) / denom;
        }

    for (int j = 0; j < m; ++j) x(n-1,j) = dp(n-1,j);
    for (int i = n-2; i >= 0; --i)
        for (int j = 0; j < m; ++j)
            x(i,j) = dp(i,j) - cp(i,j) * x(i+1,j);
    return x;
}

// v-sweep: all rows share the same LU factors because av/bv/cv are S-independent.
// lo, di, up: (Nv-1,)   rhs: (Ns+1) × (Nv-1)
// Returns solution of shape (Ns+1) × (Nv-1).
static G thomas_v(const std::vector<double>& lo,
                  const std::vector<double>& di,
                  const std::vector<double>& up,
                  const G& rhs, int Ns, int Nv) {
    int n = Nv - 1, m = Ns + 1;
    std::vector<double> cp(n);
    G dp(m, n), x(m, n);

    // Forward sweep — cp scalars are shared; dp broadcasts over all rows.
    cp[0] = up[0] / di[0];
    for (int i = 0; i < m; ++i) dp(i,0) = rhs(i,0) / di[0];

    for (int j = 1; j < n; ++j) {
        double denom = di[j] - lo[j] * cp[j-1];
        cp[j] = up[j] / denom;
        for (int i = 0; i < m; ++i)
            dp(i,j) = (rhs(i,j) - lo[j] * dp(i,j-1)) / denom;
    }

    for (int i = 0; i < m; ++i) x(i,n-1) = dp(i,n-1);
    for (int j = n-2; j >= 0; --j)
        for (int i = 0; i < m; ++i)
            x(i,j) = dp(i,j) - cp[j] * x(i,j+1);
    return x;
}

// ---------------------------------------------------------------------------
// Implicit sweep helpers
// ---------------------------------------------------------------------------

// (I - th*dt*L_S) Y = RHS with Dirichlet BCs at i=0 and i=Ns.
// bc_lo, bc_hi: (Nv+1,)
static G s_sweep(const G& RHS,
                 const std::vector<double>& bc_lo,
                 const std::vector<double>& bc_hi,
                 const G& aS, const G& bS, const G& cS,
                 double th, double dt, int Ns, int Nv) {
    int n = Ns - 1, m = Nv + 1;
    G lo(n, m), di(n, m), up(n, m), rhs_int(n, m);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            lo(i,j)      = -th * dt * aS(i,j);
            di(i,j)      =  1.0 - th * dt * bS(i,j);
            up(i,j)      = -th * dt * cS(i,j);
            rhs_int(i,j) =  RHS(i+1, j);
        }

    // Absorb Dirichlet boundaries into RHS
    for (int j = 0; j < m; ++j) {
        rhs_int(0,   j) -= lo(0,   j) * bc_lo[j];
        rhs_int(n-1, j) -= up(n-1, j) * bc_hi[j];
    }

    G sol = thomas_s(lo, di, up, rhs_int, Ns, Nv);

    G Y = RHS.clone();
    for (int i = 1; i < Ns; ++i)
        for (int j = 0; j < m; ++j)
            Y(i,j) = sol(i-1, j);
    for (int j = 0; j < m; ++j) { Y(0,j) = bc_lo[j]; Y(Ns,j) = bc_hi[j]; }
    return Y;
}

// (I - th*dt*L_v) Y = RHS with BCs at j=0 and j=Nv.
// bc_bot, bc_top: (Ns+1,) — one value per S-row (degenerate v=0 boundary).
static G v_sweep(const G& RHS,
                 const std::vector<double>& bc_bot,
                 const std::vector<double>& bc_top,
                 const std::vector<double>& av,
                 const std::vector<double>& bv,
                 const std::vector<double>& cv,
                 double th, double dt, int Ns, int Nv) {
    int n = Nv - 1, m = Ns + 1;
    std::vector<double> lo(n), di(n), up(n);
    G rhs_int(m, n);

    for (int j = 0; j < n; ++j) {
        lo[j] = -th * dt * av[j];
        di[j] =  1.0 - th * dt * bv[j];
        up[j] = -th * dt * cv[j];
        for (int i = 0; i < m; ++i)
            rhs_int(i,j) = RHS(i, j+1);
    }

    // Absorb v-boundaries
    for (int i = 0; i < m; ++i) {
        rhs_int(i, 0  ) -= lo[0]   * bc_bot[i];
        rhs_int(i, n-1) -= up[n-1] * bc_top[i];
    }

    G sol = thomas_v(lo, di, up, rhs_int, Ns, Nv);

    G Y = RHS.clone();
    for (int j = 1; j < Nv; ++j)
        for (int i = 0; i < m; ++i)
            Y(i,j) = sol(i, j-1);
    for (int i = 0; i < m; ++i) { Y(i,0) = bc_bot[i]; Y(i,Nv) = bc_top[i]; }
    return Y;
}

// Zero-curvature (linear) extrapolation at v = vmax.
static void neumann_top(G& W, int Ns, int Nv) {
    for (int i = 0; i <= Ns; ++i)
        W(i, Nv) = 2.0 * W(i, Nv-1) - W(i, Nv-2);
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

static double heston_pde_american_cpp(
    double S0, double K, double r, double q, double T,
    double v0, double kappa, double theta, double sigma, double rho,
    const std::string& option_type,
    int Ns, int Nv, int Nt)
{
    const double th = 1.0 - 1.0 / std::sqrt(2.0);   // MCS parameter

    double Smax = 4.0 * std::max(S0, K);
    double vmax = std::min(std::max(5.0 * v0, 0.5), 3.0);
    double dS   = Smax / Ns;
    double dv   = vmax / Nv;

    // Enforce ≥ 50 steps/year so dt ≤ 0.02 for all maturities.
    Nt = std::max(Nt, (int)std::ceil(T * 50.0));
    double dt = T / Nt;

    bool is_call = (option_type == "call");

    // Grids
    std::vector<double> S(Ns + 1), v(Nv + 1);
    for (int i = 0; i <= Ns; ++i) S[i] = i * dS;
    for (int j = 0; j <= Nv; ++j) v[j] = j * dv;

    // Payoff (terminal condition)
    std::vector<double> payoff(Ns + 1);
    for (int i = 0; i <= Ns; ++i)
        payoff[i] = is_call ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);

    // Initialise V to payoff (outer product: each col identical)
    G V(Ns + 1, Nv + 1);
    for (int i = 0; i <= Ns; ++i)
        for (int j = 0; j <= Nv; ++j)
            V(i,j) = payoff[i];

    // ---- Precompute operator coefficients --------------------------------
    // aS, bS, cS: (Ns-1) × (Nv+1)
    G aS(Ns-1, Nv+1), bS(Ns-1, Nv+1), cS(Ns-1, Nv+1);
    for (int i = 1; i < Ns; ++i) {
        double Si = S[i], Si2 = Si * Si;
        for (int j = 0; j <= Nv; ++j) {
            double vj = v[j];
            aS(i-1,j) = 0.5*vj*Si2/(dS*dS) - (r-q)*Si/(2.0*dS);
            bS(i-1,j) = -vj*Si2/(dS*dS) - 0.5*r;
            cS(i-1,j) = 0.5*vj*Si2/(dS*dS) + (r-q)*Si/(2.0*dS);
        }
    }

    // av, bv, cv: (Nv-1,)
    std::vector<double> av(Nv-1), bv(Nv-1), cv(Nv-1);
    for (int j = 1; j < Nv; ++j) {
        double vj = v[j];
        av[j-1] = 0.5*sigma*sigma*vj/(dv*dv) - kappa*(theta - vj)/(2.0*dv);
        bv[j-1] = -sigma*sigma*vj/(dv*dv) - 0.5*r;
        cv[j-1] = 0.5*sigma*sigma*vj/(dv*dv) + kappa*(theta - vj)/(2.0*dv);
    }

    // mix: (Ns-1) × (Nv-1)
    G mix(Ns-1, Nv-1);
    for (int i = 1; i < Ns; ++i)
        for (int j = 1; j < Nv; ++j)
            mix(i-1,j-1) = rho*sigma/(4.0*dS*dv) * S[i] * v[j];

    // Boundary value arrays (reused each step)
    std::vector<double> bc_lo(Nv+1), bc_hi(Nv+1);

    // ---- MCS ADI time loop -----------------------------------------------
    for (int n = 0; n < Nt; ++n) {
        double tau    = (n + 1) * dt;
        double disc_r = std::exp(-r * tau);
        double disc_q = std::exp(-q * tau);

        if (is_call) {
            double hi_val = std::max(Smax*disc_q - K*disc_r, 0.0);
            for (int j = 0; j <= Nv; ++j) { bc_lo[j] = 0.0; bc_hi[j] = hi_val; }
        } else {
            for (int j = 0; j <= Nv; ++j) { bc_lo[j] = K*disc_r; bc_hi[j] = 0.0; }
        }

        G LV  = apply_L (V, aS, bS, cS, av, bv, cv, mix, Ns, Nv);
        G LSV = apply_LS(V, aS, bS, cS, Ns, Nv);
        G LvV = apply_Lv(V, av, bv, cv, Ns, Nv);

        // Stage 1 — explicit predictor
        G Y0(Ns+1, Nv+1);
        for (int k = 0; k < (int)Y0.d.size(); ++k)
            Y0.d[k] = V.d[k] + dt * LV.d[k];

        // Stage 2 — implicit S sweep: (I - th*dt*L_S) Y1 = Y0 - th*dt*L_S*V
        {
            G rhs(Ns+1, Nv+1);
            for (int k = 0; k < (int)rhs.d.size(); ++k)
                rhs.d[k] = Y0.d[k] - th*dt * LSV.d[k];
            G Y1 = s_sweep(rhs, bc_lo, bc_hi, aS, bS, cS, th, dt, Ns, Nv);
            neumann_top(Y1, Ns, Nv);

            // Stage 3 — implicit v sweep: (I - th*dt*L_v) Y2 = Y1 - th*dt*L_v*V
            // v=0 BC taken from Y1[:,0] (degenerate boundary)
            std::vector<double> bot(Ns+1), top_bc(Ns+1);
            for (int i = 0; i <= Ns; ++i) { bot[i] = Y1(i,0); top_bc[i] = Y1(i,Nv); }

            G rhs3(Ns+1, Nv+1);
            for (int k = 0; k < (int)rhs3.d.size(); ++k)
                rhs3.d[k] = Y1.d[k] - th*dt * LvV.d[k];
            G Y2 = v_sweep(rhs3, bot, top_bc, av, bv, cv, th, dt, Ns, Nv);
            for (int j = 0; j <= Nv; ++j) { Y2(0,j) = bc_lo[j]; Y2(Ns,j) = bc_hi[j]; }
            neumann_top(Y2, Ns, Nv);

            // Stage 4 — corrected predictor: trapezoidal average of L at V and Y2
            G LY2  = apply_L (Y2, aS, bS, cS, av, bv, cv, mix, Ns, Nv);
            G LSY2 = apply_LS(Y2, aS, bS, cS, Ns, Nv);
            G LvY2 = apply_Lv(Y2, av, bv, cv, Ns, Nv);

            G Y0t(Ns+1, Nv+1);
            for (int k = 0; k < (int)Y0t.d.size(); ++k)
                Y0t.d[k] = V.d[k] + 0.5*dt*(LV.d[k] + LY2.d[k]);

            // Stage 5 — implicit S sweep on corrected
            G rhs5(Ns+1, Nv+1);
            for (int k = 0; k < (int)rhs5.d.size(); ++k)
                rhs5.d[k] = Y0t.d[k] - th*dt * LSY2.d[k];
            G Y1t = s_sweep(rhs5, bc_lo, bc_hi, aS, bS, cS, th, dt, Ns, Nv);
            neumann_top(Y1t, Ns, Nv);

            // Stage 6 — implicit v sweep on corrected (final)
            std::vector<double> bot_t(Ns+1), top_t(Ns+1);
            for (int i = 0; i <= Ns; ++i) { bot_t[i] = Y1t(i,0); top_t[i] = Y1t(i,Nv); }

            G rhs6(Ns+1, Nv+1);
            for (int k = 0; k < (int)rhs6.d.size(); ++k)
                rhs6.d[k] = Y1t.d[k] - th*dt * LvY2.d[k];
            G Vnew = v_sweep(rhs6, bot_t, top_t, av, bv, cv, th, dt, Ns, Nv);
            for (int j = 0; j <= Nv; ++j) { Vnew(0,j) = bc_lo[j]; Vnew(Ns,j) = bc_hi[j]; }
            neumann_top(Vnew, Ns, Nv);

            // American early-exercise constraint
            for (int i = 0; i <= Ns; ++i)
                for (int j = 0; j <= Nv; ++j)
                    Vnew(i,j) = std::max(Vnew(i,j), payoff[i]);

            V = std::move(Vnew);
        }
    }

    // ---- Bilinear interpolation at (S0, v0) ------------------------------
    int i0 = (int)(std::lower_bound(S.begin(), S.end(), S0) - S.begin()) - 1;
    i0 = std::min(std::max(i0, 0), Ns - 1);
    int j0 = (int)(std::lower_bound(v.begin(), v.end(), v0) - v.begin()) - 1;
    j0 = std::min(std::max(j0, 0), Nv - 1);

    double wS = (S0 - S[i0]) / dS;
    double wv = (v0 - v[j0]) / dv;

    return (1-wS)*(1-wv)*V(i0,  j0  )
         +    wS *(1-wv)*V(i0+1,j0  )
         + (1-wS)*   wv *V(i0,  j0+1)
         +    wS *   wv *V(i0+1,j0+1);
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(heston_mcs, m) {
    m.doc() = "Heston MCS ADI PDE solver for American options (C++ backend)";
    m.def("heston_pde_american",
          &heston_pde_american_cpp,
          py::arg("S0"),
          py::arg("K"),
          py::arg("r"),
          py::arg("q"),
          py::arg("T"),
          py::arg("v0"),
          py::arg("kappa"),
          py::arg("theta"),
          py::arg("sigma"),
          py::arg("rho"),
          py::arg("option_type"),
          py::arg("Ns") = 40,
          py::arg("Nv") = 20,
          py::arg("Nt") = 40,
          R"doc(
Price an American option under the Heston model via Modified Craig-Sneyd ADI.

Parameters
----------
S0, K, r, q, T   : spot, strike, risk-free rate, dividend yield, maturity
v0, kappa, theta, sigma, rho : Heston parameters
option_type       : "call" or "put"
Ns, Nv, Nt        : grid sizes (S-nodes, v-nodes, time-steps)

Returns
-------
float — option price
)doc");
}
