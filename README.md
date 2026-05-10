# Heston Options Pricing Engine

A full-stack quantitative finance platform for pricing, calibrating, and analyzing options under the Heston stochastic volatility model. The system covers everything from raw market data ingestion to real-time volatility surface visualization, strategy payoff construction, and risk limit evaluation — all accessible through a multi-page Streamlit web app or headless CLI pipelines.

---

## Table of Contents

1. [The Heston Model](#1-the-heston-model)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Setup](#4-setup)
5. [Data Layer](#5-data-layer)
6. [Schema Normalization](#6-schema-normalization)
7. [Pricing Methods](#7-pricing-methods)
8. [Implied Volatility Inversion](#8-implied-volatility-inversion)
9. [Calibration Pipeline](#9-calibration-pipeline)
10. [Analytics Engine](#10-analytics-engine)
11. [Volatility Surfaces](#11-volatility-surfaces)
12. [Strategies](#12-strategies)
13. [Risk Engine](#13-risk-engine)
14. [Services Layer](#14-services-layer)
15. [Web App](#15-web-app)
16. [CLI Pipelines](#16-cli-pipelines)
17. [Configuration](#17-configuration)
18. [Full End-to-End Workflow](#18-full-end-to-end-workflow)
19. [Key Design Decisions](#19-key-design-decisions)

---

## 1. The Heston Model

### Stochastic Differential Equations

The Heston model describes joint dynamics of the asset price $S_t$ and its instantaneous variance $v_t$ under the risk-neutral measure Q:

```
dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^(1)
dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^(2)
dW_t^(1) dW_t^(2) = rho dt
```

### Parameters

| Parameter | Symbol | Interpretation | Default |
|-----------|--------|----------------|---------|
| Initial variance | v0 | Current instantaneous variance level | 0.04 |
| Mean reversion speed | kappa | Rate at which variance reverts to long-run mean | 2.0 |
| Long-run variance | theta | Unconditional variance; the level v_t reverts to | 0.04 |
| Vol-of-vol | sigma | Volatility of the variance process | 0.5 |
| Correlation | rho | Correlation between asset returns and variance shocks; negative for equity skew | −0.7 |

### Feller Condition

The variance process v_t stays strictly positive whenever `2 * kappa * theta > sigma^2`. The calibration diagnostics module reports whether this condition is satisfied for any estimated parameter set.

### Characteristic Function

The Heston model admits a semi-analytical characteristic function (Heston 1993):

```
phi_j(u) = exp( C_j(u) + D_j(u) * v0 + i*u * ln(S0) )

C_j(u) = r*i*u*T + (kappa*theta / sigma^2) * [
           (b_j - rho*sigma*i*u + d_j) * T
           - 2 * ln( (1 - g_j * exp(d_j*T)) / (1 - g_j) )
         ]

D_j(u) = (b_j - rho*sigma*i*u + d_j) / sigma^2
        * (1 - exp(d_j*T)) / (1 - g_j * exp(d_j*T))

d_j = sqrt( (rho*sigma*i*u - b_j)^2 - sigma^2 * (2*u_bar_j*i*u - u^2) )
g_j = (b_j - rho*sigma*i*u + d_j) / (b_j - rho*sigma*i*u - d_j)
```

For j=1: `b_1 = kappa - rho*sigma`, `u_bar_1 = 0.5`
For j=2: `b_2 = kappa`, `u_bar_2 = -0.5`

Implemented in [models/Heston_cf.py](models/Heston_cf.py).

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Market Data Sources                          │
│       Yahoo Finance (live)        NVDA Excel snapshot (sample)   │
└─────────────────────┬────────────────────────┬───────────────────┘
                      │                        │
                      ▼                        ▼
         ┌────────────────────────────────────────────┐
         │         data/market_data.py                │
         │  Raw option chain fetch + T computation     │
         └──────────────────┬─────────────────────────┘
                            │
                            ▼
         ┌────────────────────────────────────────────┐
         │         analytics/schema.py                │
         │  ensure_option_frame() — normalize cols,   │
         │  compute mid_price, rel_spread, moneyness  │
         └──────────────────┬─────────────────────────┘
                            │
                            ▼
         ┌────────────────────────────────────────────┐
         │         data/option_filters.py             │
         │  11-step filter: arbitrage, spread,        │
         │  moneyness, volume, OI, maturity cap        │
         └──────────────────┬─────────────────────────┘
                            │
            ┌───────────────┴──────────────────┐
            │                                  │
            ▼                                  ▼
┌──────────────────────┐        ┌──────────────────────────────┐
│  calibration/        │        │  services/pricing_service    │
│  calibrate_heston.py │        │  price_option_frame()        │
│  L-BFGS-B optimizer  │        │                              │
│  + loss function     │        │  European: Fourier           │
└──────────┬───────────┘        │  American call (no div):     │
           │                    │    European proxy            │
           │ fitted params      │  American (general):         │
           │                    │    PDE or LSMC               │
           ▼                    └──────────────┬───────────────┘
┌──────────────────────┐                       │
│  services/           │◄──────────────────────┘
│  calibration_service │        model prices
│  CalibrationResult   │
│  + JSON cache        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                analytics/chain_metrics.py                    │
│  enrich_option_chain():                                      │
│   - market IV (Brent inversion of BS)                        │
│   - BS Greeks from market IV (delta, gamma, vega, theta, rho)│
│   - model price -> model IV -> model Greeks                  │
│   - liquidity score (volume + OI + spread + price)           │
│   - mispricing_score, mispricing_bias (buy/sell/hold)        │
└─────────────────────────────────┬────────────────────────────┘
                                  │
            ┌─────────────────────┴──────────────────────┐
            │                                            │
            ▼                                            ▼
┌─────────────────────────┐              ┌───────────────────────────┐
│  analytics/surfaces.py  │              │  strategies/ + risk/      │
│  SurfaceGrid: cubic/    │              │  OptionLeg dataclass,     │
│  linear interpolation   │              │  payoff, portfolio,       │
│  over (T, moneyness)    │              │  scenario table,          │
│  grid for IV & Greeks   │              │  RiskLimits evaluation    │
└────────────┬────────────┘              └────────────┬──────────────┘
             │                                        │
             └──────────────────┬─────────────────────┘
                                │
                                ▼
          ┌──────────────────────────────────────────────────┐
          │                app/  (Streamlit)                 │
          │   Home -> Option Chain -> Vol Surfaces ->        │
          │   Strategy Lab -> Risk Dashboard -> Mispricing   │
          └──────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
Heston-Options-Pricing-Engine/
│
├── models/                       # Pure mathematical models
│   ├── Heston_cf.py              # Heston characteristic function
│   ├── heston_european.py        # Fourier-inversion European pricing
│   └── black_scholes.py          # BS price with continuous dividend yield
│
├── pricing/                      # Pricing wrappers by exercise type
│   ├── european.py               # Heston European call/put wrappers
│   ├── american.py               # American put/call dispatch (LSMC or European proxy)
│   └── heston_pde_american.py    # Explicit finite-difference PDE solver
│
├── simulation/                   # Monte Carlo path generation
│   ├── heston_path.py            # Full-truncation Euler + log-Euler Heston paths
│   └── lsmc.py                   # Longstaff-Schwartz backward regression
│
├── calibration/                  # Parameter estimation
│   ├── calibrate_heston.py       # scipy L-BFGS-B optimizer entry point
│   ├── heston_loss_function.py   # MSE loss (price or IV objective)
│   ├── implied_vol.py            # Brent-method BS IV inversion
│   ├── data_driven_bounds.py     # IV-surface-based initial guess and bounds
│   └── historical_bounds.py      # AR(1)-realized-variance-based P-measure estimates
│
├── analytics/                    # Option chain enrichment
│   ├── schema.py                 # ensure_option_frame() normalization gate
│   ├── greeks.py                 # Closed-form BS Greeks (delta, gamma, vega, theta, rho)
│   ├── chain_metrics.py          # enrich_option_chain() — full enrichment pipeline
│   └── surfaces.py               # SurfaceGrid: scipy griddata interpolation
│
├── data/                         # Market data ingestion and filtering
│   ├── market_data.py            # yfinance option chain fetcher
│   ├── option_filters.py         # 11-step filter pipeline
│   └── cleaning.py               # Data cleaning utilities
│
├── services/                     # Business logic orchestration layer
│   ├── market_service.py         # Load + filter chains; single data entry point
│   ├── analytics_service.py      # Build enriched tables + optional calibration
│   ├── calibration_service.py    # CalibrationResult, caching, universe selection
│   └── pricing_service.py        # HestonParameters dataclass + frame pricer
│
├── strategies/                   # Options strategy construction
│   ├── contracts.py              # OptionLeg dataclass
│   ├── builders.py               # build_leg_from_row() factory
│   ├── payoff.py                 # Leg/portfolio P&L over spot grid
│   ├── portfolio.py              # summarize_strategy() aggregate Greeks + payoff
│   └── screener.py               # Mispricing ranker + relative-value pair builder
│
├── risk/                         # Risk management
│   ├── engine.py                 # evaluate_strategy_risk() — limits + scenarios
│   ├── limits.py                 # RiskLimits dataclass + evaluate_limits()
│   ├── scenarios.py              # Spot x IV x time P&L scenario table
│   └── exposure.py               # exposure_snapshot() Greek aggregation
│
├── app/                          # Streamlit web app
│   ├── Home.py                   # Landing page; raw/filtered chain explorer
│   ├── shared.py                 # Sidebar controls, @st.cache_data wrappers
│   └── pages/
│       ├── 1_Option_Chain.py         # Per-contract metrics + Greeks table
│       ├── 2_Volatility_Surfaces.py  # 3-D IV and Greek surfaces
│       ├── 3_Strategy_Lab.py         # Multi-leg strategy builder + payoff chart
│       ├── 4_Risk_Dashboard.py       # Limit checks + scenario table
│       └── 5_Mispricing_Screener.py  # IV-error screener + RV strategy ideas
│
├── pipelines/                    # Headless CLI scripts
│   ├── run_pricing.py            # Build analytics table from chain snapshot
│   └── run_calibration.py        # Calibrate and dump JSON
│
├── config/
│   └── market_config.py          # Global constants (r, q, MC paths)
│
├── utils/
│   ├── discounting.py            # Discounting helpers
│   └── regression.py             # LSMC basis function utilities
│
├── results/
│   └── calibrations/             # Persisted calibration JSON files (auto-created)
│
├── vol surface app/
│   └── vol_surface.py            # Standalone volatility surface Streamlit app
│
├── nvda_vol.xlsx                 # Sample NVDA option chain snapshot
├── requirements.txt
└── CLAUDE.md
```

---

## 4. Setup

### Prerequisites

- Python 3.10 or newer
- Internet access for live Yahoo Finance data (not needed for sample mode)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd Heston-Options-Pricing-Engine

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Ubuntu / Debian** — if `python3 -m venv` fails:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv
```

### Dependencies

| Package | Role |
|---------|------|
| `numpy` | Numerical arrays, Monte Carlo paths |
| `scipy` | Fourier integration (`quad`), optimization (`minimize`), root-finding (`brentq`), interpolation (`griddata`) |
| `pandas` | DataFrame-based option chain representation |
| `yfinance` | Live option chain and historical price downloads |
| `streamlit` | Multi-page web app framework |
| `plotly` | Interactive 3-D volatility surface charts |
| `openpyxl` | Reading the sample NVDA Excel snapshot |

---

## 5. Data Layer

### Live Data — `data/market_data.py`

`get_all_options(ticker)` fetches the full option chain for a single ticker via `yfinance`:

1. Downloads the latest closing price as the spot.
2. Iterates over every available expiry in `tk.options`.
3. Concatenates calls and puts, tags them with `type`, `maturity`, `spot`, `ticker`, `ExerciseStyle = "american"`.
4. Computes `T` (year fraction from now to expiry), `mid_price = (bid + ask) / 2`, `rel_spread`, and `moneyness`.
5. Drops expired contracts (`T <= 0`) and replaces infinities with NaN.

`get_multiple_tickers(tickers)` loops over a list of tickers and concatenates results.

### Sample Data

`nvda_vol.xlsx` is a pre-downloaded NVDA option chain snapshot. It is loaded by the services layer via `openpyxl` and passed through `ensure_option_frame()` before any analytics are run.

### Filtering — `data/option_filters.py`

`apply_filters()` executes eleven sequential filters and tracks how many contracts each step removes:

| Step | Filter | Default |
|------|--------|---------|
| 1 | Expired contracts (`T <= 0`) | Always on |
| 2 | Near-zero mid price (`<= 1e-3`) | Always on |
| 3 | Bid-ask relative spread too wide | `spread_limit = 5%` |
| 4 | Moneyness outside band `[0.8, 1.2]` | Keeps near-ATM only |
| 5 | No-arbitrage lower bound violation | `lower = max(0, forward - disc_K)` |
| 6 | Ticker not in selection | Optional |
| 7 | Option type excluded | Optional |
| 8 | Volume below minimum | Optional |
| 9 | Open interest below minimum | Optional |
| 10 | Maturity beyond cap | Optional (`max_maturity = 2y`) |
| 11 | Hard contract cap (applied after sort) | Optional |

Returns `(filtered_df, stats)` where `stats` maps each reason string to the count of dropped contracts.

---

## 6. Schema Normalization

Every DataFrame entering the analytics pipeline must first pass through `ensure_option_frame()` in [analytics/schema.py](analytics/schema.py).

### What it does

1. **Numeric coercion** — converts `spot`, `strike`, `bid`, `ask`, `lastPrice`, `mid_price`, `volume`, `openInterest`, `T` to float (invalid strings become NaN).
2. **Type column** — lower-cases `"call"` / `"put"`.
3. **Ticker** — upper-cases and defaults to `"UNKNOWN"`.
4. **ExerciseStyle** — defaults to `"american"` if missing.
5. **T computation** — if `T` is absent but `maturity` exists, computes `T` as year fraction from `pd.Timestamp.now()`.
6. **mid_price** — fills from `(bid + ask) / 2`, falls back to `lastPrice`.
7. **rel_spread** — `(ask - bid) / mid_price`.
8. **moneyness** — `strike / spot`, plus `spot_over_strike` and `atm_distance = |log(moneyness)|`.
9. **contractSymbol** — synthesized from `ticker_type_maturity_strike` if absent.
10. **contract_id** — string version of `contractSymbol`.
11. Replaces all `+/-inf` with NaN.

This gate is idempotent — calling it twice does no harm. Every service and analytics function calls it first.

### Output Schema

After normalization, every DataFrame carries:

```
Market columns:
  ticker, type, maturity, strike, spot, bid, ask, mid_price,
  volume, openInterest, rel_spread, T, moneyness, atm_distance,
  spot_over_strike, ExerciseStyle, contract_id

Analytics columns (added by AnalyticsService):
  market_iv, market_delta, market_gamma, market_vega, market_theta,
  market_rho, market_abs_delta, liquidity_score, intrinsic_value, time_value

Model columns (when Heston params provided):
  model_price, model_iv, model_delta, model_gamma, model_vega,
  model_theta, model_rho, model_abs_delta, price_error, iv_error,
  relative_price_error, abs_iv_error, mispricing_score, mispricing_bias
```

---

## 7. Pricing Methods

### 7.1 European Options — Fourier Inversion

**File:** [models/heston_european.py](models/heston_european.py)

The Heston European price uses the Gil-Pelaez inversion of the characteristic function:

```
Call = S0 * P1 - K * exp(-r*T) * P2
Put  = K * exp(-r*T) * (1 - P2) - S0 * (1 - P1)

P_j = 0.5 + (1/pi) * integral_0^100 [ Re( exp(-i*u*ln(K)) * phi_j(u) / (i*u) ) ] du
```

The integral is evaluated with `scipy.integrate.quad` over `[0, 100]`.

This is the primary engine for all fast/European pricing. It is exact up to numerical integration tolerance with no discretization error.

### 7.2 American Call (No Dividends)

By the early exercise premium result, an American call on a non-dividend-paying asset equals a European call in value. [pricing/american.py](pricing/american.py) routes such contracts directly to the Fourier inversion pricer.

### 7.3 American Options — Explicit Finite-Difference PDE

**File:** [pricing/heston_pde_american.py](pricing/heston_pde_american.py)

For American options with dividends (or puts), the two-factor Heston PDE is solved on a `(S, v)` grid via explicit forward Euler time-stepping:

```
dV/dt = (r-q)*S * dV/dS
      + kappa*(theta-v) * dV/dv
      + 0.5*v*S^2 * d2V/dS2
      + 0.5*sigma^2*v * d2V/dv2
      + rho*sigma*v*S * d2V/(dS dv)
      - r*V
```

Grid setup:
- `S` in `[0, 3*S0]` with `Ns` steps; `v` in `[0, 1.0]` with `Nv` steps; `Nt` time steps backwards from expiry.
- Central differences for first- and second-order spatial derivatives.
- Mixed cross-derivative via four-corner stencil: `(V[i+1,j+1] - V[i+1,j-1] - V[i-1,j+1] + V[i-1,j-1]) / (4*dS*dv)`.
- American constraint applied at each time step: `V[i,:] = max(V[i,:], payoff(S_i))`.
- Final value interpolated at `(S0, v0)` from the grid.

Default grid: `Ns = 40, Nv = 20, Nt = 40`.

### 7.4 American Options — LSMC (Longstaff-Schwartz)

**Files:** [simulation/heston_path.py](simulation/heston_path.py), [simulation/lsmc.py](simulation/lsmc.py)

For American options requiring Monte Carlo, the engine:

**Step 1 — Path simulation** (`heston_path.py`):

Generates `N` paths of `(S_t, v_t)` over `M` time steps using correlated Brownian increments:

```
Correlated increments:
  W1 = sqrt(dt) * Z1
  W2 = sqrt(dt) * (rho * Z1 + sqrt(1 - rho^2) * Z2)
  where Z1, Z2 ~ N(0,1) independently

Variance (full-truncation Euler):
  v[t+dt] = max(v[t] + kappa*(theta - v[t])*dt + sigma*sqrt(max(v[t],0))*W2, 0)

Stock (log-Euler for positivity):
  S[t+dt] = S[t] * exp( (r - q - 0.5*v[t])*dt + sqrt(max(v[t],0))*W1 )
```

Full truncation prevents negative variance by flooring at 0 before the square root; log-Euler ensures stock prices remain positive.

Three variants handle: American put without dividends, American put with dividends (`q > 0`), American call with dividends.

**Step 2 — Longstaff-Schwartz regression** (`lsmc.py`):

Backward induction from expiry:

```
1. Terminal cashflow = max(K - S[:,-1], 0)  [for put]
2. For each time step t = M-1 down to 1:
   a. Identify in-the-money paths: ITM = S[:,t] < K
   b. Skip if no ITM paths
   c. Discount continuation values: Y = cashflow * exp(-r*dt)
   d. Fit polynomial C(S) = polyfit(S[ITM], Y[ITM], degree=2)
   e. Exercise if: payoff(S[ITM]) > C(S[ITM])
   f. Update cashflows for exercised/held paths
3. Final price = mean(cashflow * exp(-r*dt))
```

Defaults: `N = 10,000` paths, `M = 100` steps.

### 7.5 Pricing Dispatch Logic

`price_option_row()` in [services/pricing_service.py](services/pricing_service.py) routes each contract:

```
ExerciseStyle == "european"
  -> Fourier inversion (call or put)

ExerciseStyle == "american", call, q ~ 0
  -> American call = European call (no early exercise premium without dividends)

ExerciseStyle == "american", method="lsmc"
  -> LSMC (call with dividends, or put with/without dividends)

ExerciseStyle == "american" (default)
  -> PDE solver (Ns x Nv x Nt grid)
```

In **fast calibration mode**, all contracts are treated as European regardless of exercise style, enabling much faster calibration inner loops.

---

## 8. Implied Volatility Inversion

**File:** [calibration/implied_vol.py](calibration/implied_vol.py)

Since Black-Scholes has no closed-form inverse for sigma, market IV is computed numerically. `implied_volatility(price, S, K, r, T, type, q)`:

1. Validates inputs (positive spot, strike, time, price).
2. Checks no-arbitrage bounds: price must lie in `[max(0, F*exp(-q*T) - K*exp(-r*T)), F*exp(-q*T)]` for calls.
3. Sets up the root-finding objective: `f(sigma) = BS(S, K, r, T, sigma, type, q) - price`.
4. Brackets the root in `[1e-8, 5.0]`, expanding the upper bound up to `200` if needed.
5. Solves with `scipy.optimize.brentq` (guaranteed convergence for bracketed roots).
6. Returns `np.nan` gracefully on any failure.

This function is used for computing **market IV** from observed mid-prices and for computing **model IV** from Heston model prices — enabling direct IV-space comparison and the IV objective in calibration.

---

## 9. Calibration Pipeline

### 9.1 Overview

The calibration fits `(v0, kappa, theta, sigma, rho)` by minimizing a loss function over observed option prices or IVs using the L-BFGS-B optimizer from `scipy.optimize.minimize`.

### 9.2 Loss Function — `calibration/heston_loss_function.py`

`heston_loss(params, r, q, options_df, Ns, Nv, Nt, objective, pricing_mode)`:

1. Guards against invalid parameters: negative variances or `|rho| >= 1` immediately return `1e10`.
2. For each contract in the calibration universe:
   - Computes the model price via `_model_price_from_row()` (routing to European proxy, American proxy, or PDE).
   - **Price objective** (`objective = "price"`): `loss_i = ((P_model - P_market) / max(P_market, 1))^2`
   - **IV objective** (`objective = "iv"`): `loss_i = (IV_model - IV_market)^2` where both IVs are computed via BS inversion.
3. Returns mean squared error across all valid contracts. Returns `1e10` if no valid errors.

### 9.3 Calibration Modes

| Mode | Universe | Pricing | Objective | Use case |
|------|----------|---------|-----------|----------|
| **Fast** (recommended) | ATM calls only, European proxy | Fourier inversion | Price MSE | Real-time app; seconds |
| **Full** | All calls + puts, all expiries | Correct exercise style (American PDE) | IV MSE | Overnight batch; accurate |

In fast mode, `select_calibration_universe()` picks the `contracts_per_expiry` nearest-to-ATM contracts from up to `max_expiries` expiries (default 4 expiries × 4 contracts = 16 contracts), then overrides `ExerciseStyle = "european"` for all of them.

Contract ranking within each expiry is by: ATM distance (ascending), rel_spread (ascending), volume (descending).

### 9.4 Initial Guess and Bounds

Three strategies provide the starting point for the optimizer:

**Static defaults** (fallback when data is insufficient):
```
v0 = 0.04, kappa = 2.0, theta = 0.04, sigma = 0.5, rho = -0.7
Bounds: v0/theta/sigma in [1e-4, 2], kappa in [1e-4, 10], rho in [-0.999, 0.999]
```

**Data-driven bounds** — `calibration/data_driven_bounds.py`:

Reads the shape of the observed IV surface to estimate each parameter:

| Parameter | Estimator | Rationale |
|-----------|-----------|-----------|
| v0 | `sigma_ATM(T_short)^2` | ATM IV squared at nearest liquid expiry = spot variance |
| theta | `sigma_ATM(T_long)^2` | ATM IV squared at furthest liquid expiry = long-run variance |
| sigma | `sqrt(8 * c)` from smile curvature | Heston short-time approximation: curvature c ≈ sigma^2/8 |
| rho | `2 * b / sigma` from smile slope | Heston short-time approximation: slope b ≈ rho*sigma/2 |
| kappa | `-2 * sigma_mid * slope_T / (v0 - theta)` | From variance term structure derivative |

The smile is fit as `IV_imp ≈ a + b*k + c*k^2` where `k = log(K/S)` using least-squares.

Bounds are set as ±20–50% around each estimate, clipped to physically valid ranges. At least 2 liquid maturities (each with 5+ valid-IV contracts) are required; otherwise static defaults are used.

**Historical bounds** — `calibration/historical_bounds.py`:

Estimates parameters under the physical measure P from 2 years of daily closing prices via `yfinance`:

| Parameter | Physical Estimator |
|-----------|-------------------|
| v0 | Most recent 63-day rolling annualized realized variance |
| theta | Long-run mean of the full rolling RV series |
| kappa | AR(1) on RV series: `v[t+1] = a + b*v[t] + e`; `kappa = (1-b)/dt` |
| sigma | Vol-of-vol: `sigma = std(residuals / sqrt(v[t] * dt))` |
| rho | Pearson correlation between daily log-returns and daily changes in rolling RV |

Note: These are P-measure estimates. Use as sanity checks and starting ranges; exact values will differ from calibrated Q-measure parameters.

The historical module also reports Feller condition status, AR(1) coefficients, and RV distribution statistics in the `diagnostics` dict.

### 9.5 Calibration Service and Caching

`calibrate_option_chain()` in [services/calibration_service.py](services/calibration_service.py):

1. Calls `select_calibration_universe()` to pick the calibration subset.
2. Computes market IVs for the calibration subset upfront.
3. Runs `calibrate_heston()` (L-BFGS-B) and measures wall-clock runtime.
4. Returns a `CalibrationResult` frozen dataclass:

```python
CalibrationResult(
    params=HestonParameters(v0, kappa, theta, sigma, rho),
    loss=float,
    contract_count=int,
    objective=str,
    pricing_mode=str,
    calibration_style=str,
    runtime_seconds=float,
)
```

After optimization, results are persisted as JSON to:

```
results/calibrations/{scope_id}.json
```

The `scope_id` is derived from `(source, tickers, r, q, calibration_style)`. On the next run with identical settings, `load_saved_calibration()` returns the cached result immediately — no re-optimization. Different market conditions, rates, or tickers produce a new scope ID and a new file.

---

## 10. Analytics Engine

**File:** [analytics/chain_metrics.py](analytics/chain_metrics.py)

`enrich_option_chain(options_df, r, q, heston_params=None, compute_model_prices=False, ...)` enriches a filtered DataFrame.

### Market-Side Analytics (always computed)

| Column | Computation |
|--------|-------------|
| `intrinsic_value` | `max(S - K, 0)` for calls; `max(K - S, 0)` for puts |
| `time_value` | `mid_price - intrinsic_value` |
| `market_iv` | Brent-method BS IV inversion from `mid_price` |
| `market_delta` | BS delta at `market_iv` with continuous dividend yield |
| `market_gamma` | BS gamma at `market_iv` |
| `market_vega` | BS vega at `market_iv` |
| `market_theta` | BS theta at `market_iv` |
| `market_rho` | BS rho at `market_iv` |
| `market_abs_delta` | `abs(market_delta)` |
| `liquidity_score` | Composite 0-100 score (see below) |

### Liquidity Score Formula

```
score = 100 * (
    0.35 * tanh(volume / 100)
  + 0.30 * tanh(open_interest / 500)
  + 0.20 * (1 - clip(rel_spread, 0, 1))
  + 0.15 * tanh(mid_price / 10)
)
```

Weights: volume (35%), open interest (30%), tight spread (20%), non-trivial price (15%). All components use `tanh` for diminishing returns.

### Model-Side Analytics (when Heston params provided)

| Column | Derivation |
|--------|-----------|
| `model_price` | Heston price from `price_option_frame()` |
| `model_iv` | Brent-method BS IV from `model_price` |
| `model_delta/gamma/vega/theta/rho` | BS Greeks computed at `model_iv` |
| `price_error` | `model_price - mid_price` |
| `iv_error` | `model_iv - market_iv` |
| `relative_price_error` | `price_error / mid_price` |
| `abs_iv_error` | `abs(iv_error)` |
| `mispricing_score` | `abs_iv_error * (1 + liquidity_score / 100)` |
| `mispricing_bias` | `"buy"` if `iv_error > 0` (option cheap), `"sell"` if `iv_error < 0`, else `"hold"` |

The `mispricing_score` weights liquidity into the IV error: a 5-vol-point error in a highly liquid contract scores higher than the same error in a thinly traded deep OTM option.

### BS Greeks — `analytics/greeks.py`

`black_scholes_greeks(S, K, r, T, sigma, option_type, q)` implements continuous-dividend-yield Black-Scholes Greeks:

```
d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)

delta_call = exp(-q*T) * N(d1)
delta_put  = exp(-q*T) * (N(d1) - 1)

gamma = exp(-q*T) * phi(d1) / (S * sigma * sqrt(T))

vega = S * exp(-q*T) * phi(d1) * sqrt(T)

theta_call = -S*exp(-q*T)*phi(d1)*sigma / (2*sqrt(T))
             - r*K*exp(-r*T)*N(d2)
             + q*S*exp(-q*T)*N(d1)

rho_call = K * T * exp(-r*T) * N(d2)
```

Invalid inputs (expired contracts, zero sigma, non-positive inputs) return `NaN` dictionaries rather than raising exceptions.

---

## 11. Volatility Surfaces

**File:** [analytics/surfaces.py](analytics/surfaces.py)

`build_surface_grid(analytics_df, x_col, y_col, z_col, x_points=50, y_points=50)` constructs a smooth 2-D surface from scattered option data:

1. Drops rows with NaN in any of the three columns.
2. Requires at least 8 valid points and at least 2 unique values on each axis.
3. Creates a uniform 50×50 meshgrid over `[min_x, max_x] x [min_y, max_y]`.
4. Interpolates with `scipy.interpolate.griddata`:
   - **Cubic** interpolation if 16+ points (smooth, C2 continuity)
   - **Linear** interpolation for smaller datasets
5. Fills any remaining NaN holes with nearest-neighbour values (prevents blank regions in the surface chart).
6. Returns a frozen `SurfaceGrid` dataclass: `x_grid, y_grid, z_grid, x_col, y_col, z_col, point_count`.

The app renders these as interactive Plotly 3-D surfaces (`go.Surface`) with the Viridis colorscale at 750px height. Surfaces available in the app:

- **Market IV surface** over `(T, moneyness)` — observed volatility smile/skew surface
- **Model IV surface** over `(T, moneyness)` — Heston model's implied surface
- **IV error surface** — where the model under/over-fits the market
- **Delta surface**, **Gamma surface**, **Vega surface** — Greek landscapes

---

## 12. Strategies

### OptionLeg — `strategies/contracts.py`

The atomic unit of a strategy. Frozen dataclass:

```python
@dataclass(frozen=True)
class OptionLeg:
    contract_id: str
    ticker: str
    option_type: str      # "call" or "put"
    maturity: str
    strike: float
    premium: float        # mid_price at entry
    quantity: int         # positive = long, negative = short
    multiplier: int = 100 # standard equity options multiplier
    delta: float = np.nan
    gamma: float = np.nan
    vega: float = np.nan
    theta: float = np.nan
    rho: float = np.nan
    implied_vol: float = np.nan
```

### Payoff — `strategies/payoff.py`

| Function | Description |
|----------|-------------|
| `price_grid(spot)` | 200 uniformly spaced underlying prices from 50% to 150% of current spot |
| `intrinsic_value(prices, strike, type)` | Vectorized payoff at expiry |
| `leg_payoff(leg, prices)` | `quantity * multiplier * (intrinsic - premium)` |
| `strategy_payoff(legs, prices)` | Sum of all leg payoffs across the price grid |
| `estimate_break_even_points(prices, pnl)` | Linear interpolation of sign changes in the P&L curve |

### Portfolio Summary — `strategies/portfolio.py`

`summarize_strategy(legs, spot)` returns:

| Key | Value |
|-----|-------|
| `price_grid` | 200-point spot array |
| `payoff` | Total P&L at each spot price |
| `break_evens` | List of interpolated break-even prices |
| `max_profit_on_grid` | Peak P&L across the grid |
| `max_loss_on_grid` | Worst P&L across the grid (usually negative for premium-paying strategies) |
| `delta` | Net portfolio delta (`sum over legs: quantity * multiplier * delta`) |
| `gamma` | Net portfolio gamma |
| `vega` | Net portfolio vega |
| `theta` | Net portfolio theta |
| `rho` | Net portfolio rho |
| `contract_count` | Total contracts (`sum of abs(quantity)`) |
| `net_premium_paid` | Total cash paid (`quantity * premium * multiplier`) |
| `entry_cashflow` | `-net_premium_paid` |

### Mispricing Screener — `strategies/screener.py`

`rank_mispriced_contracts(analytics_df, min_abs_iv_error=0.02, top_n=15)`:
- Filters to contracts where `abs_iv_error >= min_abs_iv_error`.
- Sorts by `mispricing_score` descending (highest liquidity-weighted IV error first).
- Maps `mispricing_bias` to trade actions: `"Buy option: model IV > market IV"` / `"Sell option: model IV < market IV"`.

`build_relative_value_strategies(analytics_df, min_abs_iv_error=0.02, top_n=10)`:
- Groups by `(ticker, maturity, type)`.
- Within each group pairs the most underpriced contract (highest positive `iv_error`) with the most overpriced (most negative `iv_error`).
- Names the strategy based on strike ordering:
  - Long lower strike + short higher strike call → `"Bull call spread proxy"`
  - Long higher strike + short lower strike put → `"Bear put spread proxy"`
  - Otherwise → `"Relative value pair"`
- Scores pairs by `long_score + short_score`.

---

## 13. Risk Engine

**Files:** [risk/engine.py](risk/engine.py), [risk/limits.py](risk/limits.py), [risk/scenarios.py](risk/scenarios.py), [risk/exposure.py](risk/exposure.py)

### RiskLimits

Default thresholds for the strategy risk dashboard:

```python
@dataclass(frozen=True)
class RiskLimits:
    max_abs_delta:    float = 1500.0   # Net portfolio delta
    max_abs_gamma:    float = 250.0    # Net portfolio gamma
    max_abs_vega:     float = 4000.0   # Net portfolio vega
    max_premium_paid: float = 15000.0  # Total premium outflow
    max_contracts:    float = 20.0     # Total contract count
    max_loss_on_grid: float = 20000.0  # Worst-case P&L from payoff grid
```

### Limit Evaluation

`evaluate_limits(strategy_summary, limits)` checks each metric:

```
abs(value) > limit        -> "reject"
abs(value) > 0.8 * limit  -> "warn"   (80% utilization threshold)
otherwise                 -> "pass"
```

Returns a DataFrame with columns: `metric, value, limit, status`.

### Top-Level Risk Function

`evaluate_strategy_risk(strategy_summary, spot, limits)` returns:

```python
{
    "overall_status": "pass" | "warn" | "reject",  # worst status across all metrics
    "limits":    pd.DataFrame,   # per-metric limit table
    "scenarios": pd.DataFrame,   # 63-row spot x IV x time P&L table
}
```

### Scenario Analysis

`scenario_table(strategy_summary, spot)` produces a cross-product of:
- **Spot shocks**: −15%, −10%, −5%, 0%, +5%, +10%, +15%
- **IV shocks**: −10%, 0%, +10%
- **Day shifts**: 0, 7, 30 days

P&L is approximated using first and second-order Greeks:

```
P&L ≈ delta * dS + 0.5 * gamma * dS^2 + vega * d_sigma + theta * (days/365)
where dS = spot * spot_shock
```

This yields `7 * 3 * 3 = 63` scenario rows per strategy evaluation.

---

## 14. Services Layer

The services layer is the single entry point for all orchestration. Application code (pages, pipelines) should call services — never the lower-level analytics or models directly.

### `market_service.py`

| Function | Description |
|----------|-------------|
| `load_live_chain(tickers)` | Fetch from yfinance, normalize via `ensure_option_frame()` |
| `filter_chain_with_stats(raw_df, ...)` | Full 11-step filter; returns `(df, stats)` |
| `parse_tickers(raw)` | Split comma-separated string into uppercase list; defaults to `["NVDA"]` |

### `analytics_service.py`

| Function | Description |
|----------|-------------|
| `build_chain_analytics(options_df, r, q, heston_params, ...)` | Wraps `enrich_option_chain()` |
| `calibrate_and_build_analytics(options_df, r, q, ...)` | Calibrates first, then enriches with fitted params |

### `calibration_service.py`

| Function | Description |
|----------|-------------|
| `select_calibration_universe(options_df, ...)` | Pick near-ATM contracts per expiry for calibration |
| `calibrate_option_chain(options_df, r, q, ...)` | Full calibration; returns `(CalibrationResult, calibration_df)` |
| `save_calibration_result(scope_id, meta)` | Persist JSON to `results/calibrations/` |
| `load_saved_calibration(scope_id)` | Load cached result if it exists |
| `calibration_scope_id(source, tickers, r, q, style)` | Derive deterministic cache key |

### `pricing_service.py`

| Function | Description |
|----------|-------------|
| `HestonParameters` | Frozen dataclass: `v0, kappa, theta, sigma, rho` with `from_iterable()` and `as_tuple()` |
| `price_option_row(row, r, q, heston_params, ...)` | Price a single contract row with routing logic |
| `price_option_frame(options_df, r, q, heston_params, pricing_limit, ...)` | Price all contracts in a DataFrame; respects `pricing_limit` |
| `prioritize_contracts(options_df, max_contracts)` | Sort by `(T, atm_distance, rel_spread, volume)` for partial repricing |

The `pricing_limit` parameter caps how many contracts are repriced with the model; the highest-priority contracts (near-ATM, tight spread, high volume, short expiry) are selected first.

---

## 15. Web App

Launch with:

```bash
streamlit run app/Home.py
```

### Architecture

Every page loads data through `load_app_data(page_key)` in [app/shared.py](app/shared.py), which is the single source of truth for:
- Rendering sidebar controls
- `@st.cache_data`-wrapped data loading, filtering, analytics building, and calibration
- Calibration result storage (session state + JSON file)
- Scope ID computation for cache keying

### Caching Architecture

| Cache function | Keyed by | What it caches |
|----------------|----------|----------------|
| `cached_load_chain` | `tickers_text` | Raw option chain from yfinance |
| `cached_filter_chain` | `raw_df + all filter params` | Filtered chain + filter stats |
| `cached_build_analytics` | `filtered_df + model params + r, q, limits` | Enriched analytics DataFrame |
| `cached_calibrate_chain` | `filtered_df + r, q, grid sizes, expiry/contract counts, style` | `(calibration_meta, calibration_df)` |

Clicking **Refresh options chain** calls `clear_data_caches()` to invalidate all four caches and re-fetch live data.

### Model Modes

The sidebar **Model mode** selector determines how model prices are computed:

| Mode | Behavior |
|------|----------|
| `Use stored / calibrated Heston params` | Loads last calibration from session state or JSON cache; reprices with those params |
| `Use existing/precomputed model prices` | Uses `calibrated_heston_price` column if present in raw data |
| `Use manual Heston params` | User enters `v0,kappa,theta,sigma,rho` as text; reprices immediately |
| `Market metrics only` | No model pricing; only market IV and Greeks computed |

### Pages

#### Home (`app/Home.py`)

Landing page showing:
- **Summary metrics**: raw count, filtered count, expiry count, ticker count, IV point counts, median delta, top mispricing score.
- **Calibration panel**: JSON dump of calibrated params + the specific contracts used in calibration.
- **Two-tab explorer**: Raw Contracts table | Filtered Contracts table with per-filter drop counts.

#### Option Chain (`app/pages/1_Option_Chain.py`)

Full per-contract enriched table. Displays strike, spot, mid_price, market_iv, model_iv, all market and model Greeks, liquidity_score, mispricing_score, mispricing_bias.

#### Volatility Surfaces (`app/pages/2_Volatility_Surfaces.py`)

Interactive Plotly 3-D surfaces for:
- Market IV surface (observed smile/skew)
- Model IV surface (Heston fit)
- IV error surface (model minus market)
- Greek surfaces (delta, gamma, vega)

All surfaces plotted over `(maturity, moneyness)` axes.

#### Strategy Lab (`app/pages/3_Strategy_Lab.py`)

Multi-leg option strategy builder:
- Select up to N legs from the filtered chain (call/put, strike, maturity, quantity, long/short)
- Build `OptionLeg` objects from selected rows
- Plot P&L payoff diagram at expiry from 50%–150% of spot
- Display break-even points, max profit/loss, and net Greeks

#### Risk Dashboard (`app/pages/4_Risk_Dashboard.py`)

For the strategy from Strategy Lab:
- Runs `evaluate_strategy_risk()` against `RiskLimits`
- Color-coded limit table: green (pass), yellow (warn at 80%), red (reject)
- 63-row scenario P&L table across spot shocks × IV shocks × time decay

#### Mispricing Screener (`app/pages/5_Mispricing_Screener.py`)

Requires calibrated model prices. Features:
- **Min |IV error| slider** — filter threshold for mispricing signal
- **Top N contracts** slider
- **Ranked single-leg opportunities** sorted by `mispricing_score` with trade action column
- **Suggested relative value strategies**: matched long/short pairs (bull spread, bear spread, or generic pairs)

---

## 16. CLI Pipelines

### Analytics Pipeline — `pipelines/run_pricing.py`

Builds a fully enriched analytics table:

```bash
# Sample mode (uses nvda_vol.xlsx)
python pipelines/run_pricing.py \
  --source sample \
  --tickers NVDA \
  --max-contracts 100

# With Heston model pricing (pass calibrated params)
python pipelines/run_pricing.py \
  --source sample \
  --tickers NVDA \
  --max-contracts 100 \
  --heston-params 0.04,2.0,0.04,0.5,-0.7 \
  --pricing-limit 50 \
  --output analytics.csv

# Live data
python pipelines/run_pricing.py \
  --source live \
  --tickers NVDA \
  --spread-limit 0.05 \
  --max-contracts 200 \
  --output live_analytics.csv
```

Without `--output`, prints a summary table showing: contract_id, type, maturity, strike, mid_price, market_iv, market_delta, market_gamma, market_vega, model_iv, iv_error.

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `sample` | `sample` or `live` |
| `--tickers` | `NVDA` | Comma-separated ticker list |
| `--spread-limit` | `0.05` | Maximum relative bid-ask spread |
| `--risk-free-rate` | `0.05` | Continuous risk-free rate |
| `--dividend-yield` | `0.0` | Continuous dividend yield |
| `--min-volume` | `1` | Minimum daily volume |
| `--min-open-interest` | `0` | Minimum open interest |
| `--max-maturity` | `2.0` | Maximum time to expiry in years |
| `--max-contracts` | `400` | Hard contract cap |
| `--heston-params` | None | `v0,kappa,theta,sigma,rho` for model pricing |
| `--pricing-limit` | `150` | Max contracts repriced with Heston model |
| `--Ns` | `40` | PDE stock grid steps |
| `--Nv` | `20` | PDE variance grid steps |
| `--Nt` | `40` | PDE time steps |
| `--output` | None | CSV output path |

### Calibration Pipeline — `pipelines/run_calibration.py`

Calibrates Heston parameters and saves the result as JSON:

```bash
# Fast calibration on sample data
python pipelines/run_calibration.py \
  --source sample \
  --tickers NVDA \
  --calibration-style fast \
  --output results.json

# Full IV calibration on live data
python pipelines/run_calibration.py \
  --source live \
  --tickers NVDA \
  --calibration-style full \
  --max-expiries 6 \
  --contracts-per-expiry 6 \
  --output calibration_full.json
```

Output JSON structure:
```json
{
  "v0": 0.0412,
  "kappa": 1.987,
  "theta": 0.0621,
  "sigma": 0.4823,
  "rho": -0.6931,
  "loss": 0.000342,
  "contract_count": 16,
  "objective": "price",
  "pricing_mode": "european_proxy",
  "calibration_style": "fast",
  "runtime_seconds": 4.21,
  "calibration_contracts": ["NVDA_call_2024-01-19_500.0", "..."]
}
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `sample` | `sample` or `live` |
| `--tickers` | `NVDA` | Comma-separated ticker list |
| `--spread-limit` | `0.05` | Maximum relative bid-ask spread |
| `--risk-free-rate` | `0.05` | Risk-free rate |
| `--dividend-yield` | `0.0` | Dividend yield |
| `--calibration-style` | `fast` | `fast` or `full` |
| `--max-expiries` | `6` | Max expiry dates in calibration universe |
| `--contracts-per-expiry` | `6` | Contracts selected per expiry |
| `--initial-guess` | None | Optional `v0,kappa,theta,sigma,rho` override |
| `--Ns` | `40` | PDE stock grid steps |
| `--Nv` | `20` | PDE variance grid steps |
| `--Nt` | `40` | PDE time steps |
| `--output` | None | JSON output path |

---

## 17. Configuration

### Global Constants — `config/market_config.py`

```python
RISK_FREE_RATE    = 0.05    # Default r used in pricing and Greeks
DIVIDEND_YIELD    = 0.0     # Default q

MC_PATHS          = 10000   # Number of Monte Carlo paths (LSMC)
MC_STEPS          = 100     # Number of time steps per path

LSMC_BASIS_DEGREE = 2       # Polynomial degree for regression basis functions
```

### PDE Grid Size Trade-offs

| Setting | Ns | Nv | Nt | Speed | Accuracy |
|---------|----|----|----|----|---|
| Fast | 20 | 10 | 20 | Very fast | Lower |
| Default | 40 | 20 | 40 | Moderate | Good |
| Accurate | 80 | 40 | 80 | Slow | High |

The app exposes `Ns`, `Nv`, `Nt` as sidebar sliders for each page. CLI flags `--Ns`, `--Nv`, `--Nt` allow pipeline override.

---

## 18. Full End-to-End Workflow

### Using the Web App

```
1. Launch the app:
   streamlit run app/Home.py

2. Open the URL shown in the terminal (default: http://localhost:8501).

3. In the sidebar:
   a. Enter tickers (e.g., "NVDA") and select data source.
   b. Set risk-free rate and dividend yield.
   c. Click "Refresh options chain" to load and normalize the chain.

4. The Home page shows:
   - Summary metrics (contract counts, expiry count, IV points).
   - Filter breakdown: which filter dropped how many contracts.
   - Two-tab explorer: Raw Contracts | Filtered Contracts.

5. Click "Calibrate Heston" in the sidebar.
   - Fast proxy calibration runs in seconds.
   - Results are stored in results/calibrations/ and session state.
   - Loss, contract count, and runtime are shown in a success banner.

6. All pages now show model-calibrated values:
   - Option Chain: model_iv, model Greeks, iv_error, mispricing_score.
   - Volatility Surfaces: market/model/error surfaces in 3-D.
   - Mispricing Screener: ranked single-leg trade ideas + relative value pairs.

7. In Strategy Lab, select option legs from the filtered chain.
   - Each leg: option type, strike, maturity, quantity, long/short.
   - View the payoff diagram (P&L vs. spot at expiry).
   - Break-even points and net Greeks are displayed.

8. In Risk Dashboard, check the strategy against RiskLimits.
   - Color-coded limit table (pass/warn/reject).
   - 63-row scenario P&L table across spot x IV x time shocks.
```

### Using the CLI

```bash
# Step 1: Calibrate on sample data
python pipelines/run_calibration.py \
  --source sample --calibration-style fast \
  --output calibration.json

# Step 2: Read calibrated params and build enriched analytics
python pipelines/run_pricing.py \
  --source sample \
  --heston-params "$(python -c "
import json
d = json.load(open('calibration.json'))
print(f\"{d['v0']},{d['kappa']},{d['theta']},{d['sigma']},{d['rho']}\")
")" \
  --output analytics.csv
```

---

## 19. Key Design Decisions

### Why L-BFGS-B?

L-BFGS-B handles box constraints natively (essential for Heston parameter bounds), approximates the Hessian from gradients (avoids expensive second-derivative computations through Fourier integrals), and converges reliably from good starting points in the 5-dimensional parameter space. It is the standard choice for smooth, bounded, low-dimensional financial calibration problems.

### Why Two Calibration Modes?

Full IV calibration with American PDE pricing is accurate but slow — each function evaluation prices dozens of contracts via a 2-D finite-difference grid. For real-time use in the web app, fast mode replaces American pricing with European Fourier inversion (valid for calls without dividends by no-early-exercise-premium arguments) and uses price MSE instead of IV MSE, cutting calibration time from minutes to seconds.

### Why Fourier Inversion for European Options?

The characteristic function approach is exact up to numerical integration tolerance for European options — no grid discretization error, no Monte Carlo variance. `scipy.integrate.quad` is accurate and fast for smooth integrands, making it the default for any European or fast-proxy pricing.

### Why PDE for American Options (Default)?

LSMC requires thousands of paths and introduces Monte Carlo noise (results vary across runs). The explicit finite-difference PDE solver is fully deterministic, fast for the grid sizes used, and produces a clean interpolated result at `(S0, v0)`. LSMC is retained as an alternative for validation or for cases where Monte Carlo is preferred.

### Why `ensure_option_frame()` as a Gate?

Without a normalization gate, each analytics function would need to handle column aliasing, type coercion, and missing field logic independently — leading to fragile and duplicated code. `ensure_option_frame()` establishes a contract: everything downstream can assume the schema exists, types are correct, and computed columns are present.

### Why JSON-Based Calibration Cache?

JSON is human-readable and easily debuggable. The scope ID encodes all parameters that affect the calibration result, so different market conditions, rates, or tickers never collide. Re-running calibration with the same inputs is instant; re-running with different inputs produces a new file without disturbing old results.

### Why Liquidity-Weighted Mispricing Score?

A raw IV error does not distinguish between a heavily traded ATM option and a deep OTM contract with a 50% spread. Multiplying by `(1 + liquidity_score / 100)` up-weights mispricing signals in liquid, tight-spread options where the signal is most exploitable and least likely to be a data artifact or wide-spread noise.

### Why Data-Driven Bounds?

Static parameter bounds force the optimizer to search broadly even when the market already reveals the approximate answer. Estimating v0, theta, sigma, rho, and kappa directly from the observed IV surface — using known short-time Heston approximations — produces tight, data-consistent bounds that dramatically reduce the search space and improve convergence speed and solution quality.
