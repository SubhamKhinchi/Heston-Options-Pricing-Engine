# Heston Options Pricing Engine

A full-stack quantitative finance platform for pricing, calibrating, and analyzing options under the Heston stochastic volatility model. The system covers everything from raw market data ingestion to real-time volatility surface visualization, strategy payoff construction, and risk limit evaluation — all accessible through a multi-page Streamlit web app or headless CLI pipelines.

> **Architecture note (current state).** The calibration and pricing core have been
> rewritten since this document was first authored. The engine is now a
> **European-equivalent** pipeline:
> - Data is **live-only** (Yahoo Finance); the old `nvda_vol.xlsx` sample mode and the
>   `--source` CLI flag have been removed.
> - The implied **forward curve** `F(T)` (from put-call parity) is carried instead of a
>   dividend yield; the risk-free rate comes from a **SOFR/OIS curve** (`config/market_config.py`).
> - American quotes are **de-Americanized** once, up front, into European-equivalent prices
>   (`calibration/de_americanize.py`).
> - Calibration uses **Levenberg-Marquardt with an analytic Jacobian** (Cui et al. 2016) over a
>   **64-node Gauss-Legendre** characteristic-function pricer (`pricing/european_gl.py`,
>   `models/heston_cf_cui.py`) — **not** the L-BFGS-B / IV-MSE approach described in some
>   sections below.
> - The Streamlit app is an **8-page, self-contained, step-by-step pipeline** using
>   `st.session_state`; the centralized sidebar/cache layer was retired.
>
> Sections 7, 9, 14–17 and 19 reflecting the older design are being migrated. For the
> authoritative current overview see `DOCS.md` and `CLAUDE.md`.

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
│  Levenberg-Marquardt │        │                              │
│  + analytic Jacobian │        │  All contracts priced as     │
└──────────┬───────────┘        │  European-equivalent:        │
           │                    │  GL quad (Cui CF), carry q   │
           │ fitted params      │  (American = European-equiv; │
           │                    │   no PDE/LSMC)               │
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
          │   Fetch -> Filter -> Calibrate -> Price ->       │
          │   Vol Surface -> Strategy Lab -> Risk ->         │
          │   Mispricing Screener   (8 self-contained pages) │
          └──────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
Heston-Options-Pricing-Engine/
│
├── models/                       # Pure mathematical models
│   ├── heston_cf_cui.py          # Cui (2016) CF + analytic gradient (active)
│   ├── Heston_cf.py              # Legacy classic Heston characteristic function
│   ├── heston_european.py        # Legacy Fourier-inversion (scipy quad) European pricing
│   └── black_scholes.py          # BS price with continuous dividend yield
│
├── pricing/                      # European-equivalent pricing engine
│   ├── european_gl.py            # The engine: GL-quadrature European + analytic gradient
│   └── european.py               # Thin wrappers (delegate to european_gl)
│                                 # (American PDE/LSMC/C++ pricers removed → _graveyard.py)
│
├── calibration/                  # Parameter estimation
│   ├── calibrate_heston.py       # Levenberg-Marquardt (trf) entry point
│   ├── heston_loss_function.py   # Residuals + analytic Jacobian + vega weighting
│   ├── de_americanize.py         # CRR-tree de-Americanization (euro_mid, deam_iv)
│   ├── implied_vol.py            # Brent-method BS IV inversion
│   ├── data_driven_bounds.py     # IV-surface-based guess/bounds (tool, not default)
│   └── historical_bounds.py      # AR(1)-realized-variance P-measure estimates (tool)
│
├── analytics/                    # Option chain enrichment
│   ├── schema.py                 # ensure_option_frame() normalization gate
│   ├── greeks.py                 # Closed-form BS Greeks (delta, gamma, vega, theta, rho)
│   ├── chain_metrics.py          # enrich_option_chain() — full enrichment pipeline
│   └── surfaces.py               # SurfaceGrid: scipy griddata interpolation
│
├── data/                         # Market data ingestion and filtering
│   ├── market_data.py            # yfinance option chain fetcher
│   ├── forward_curve.py          # Implied forward F(T) from put-call parity
│   ├── instrument_classifier.py  # EQUITY / ETF / INDEX → exercise style
│   └── option_filters.py         # 11-step filter pipeline
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
│       ├── 01_Fetch_Data.py          # Step 1: live chain + SOFR/OIS curve
│       ├── 02_Filter_Options.py      # Step 2: filter + de-Americanize
│       ├── 03_Calibrate_Heston.py    # Step 3: vega-weighted LM calibration
│       ├── 04_Price_Contracts.py     # Step 4: model pricing + analytics table
│       ├── 05_Volatility_Surface.py  # Step 5: 3-D IV and Greek surfaces
│       ├── 06_Strategy_Lab.py        # Step 6: multi-leg builder + payoff chart
│       ├── 4_Risk_Dashboard.py       # Step 7: limit checks + scenario table
│       └── 07_Mispricing_Screener.py # Step 8: IV-error screener + RV ideas
│
├── pipelines/                    # Headless CLI scripts
│   ├── run_pricing.py            # Build analytics table from chain snapshot
│   └── run_calibration.py        # Calibrate and dump JSON
│
├── config/
│   └── market_config.py          # SOFR/OIS discount curve (FRED + Treasury)
│
├── utils/                        # Jupyter research notebooks (calibration experiments)
│
├── results/
│   └── calibrations/             # Persisted calibration JSON files (auto-created)
│
├── requirements.txt
├── DOCS.md                       # Concise current-state overview
└── CLAUDE.md                     # Developer/architecture guide
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
| `numpy` | Numerical arrays, Gauss-Legendre quadrature, vectorized math |
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

> **Live-only.** There is no bundled sample snapshot. The previous `nvda_vol.xlsx` sample mode
> and the `--source` CLI flag have been removed; all entry points fetch live chains.

### Implied Forward Curve — `data/forward_curve.py`

For each expiry, the implied forward `F(T)` is recovered from European put-call parity by
regressing `C − P` on `K` across the near-ATM strikes: slope = `−e^{-rT}`, intercept = `e^{-rT}·F`.
The forward — not a dividend yield — is the object option prices actually depend on, and it is
exercise-style-independent. The near-ATM window keeps the (American) early-exercise premium
negligible, a sanity clamp rejects economically impossible fits (falling back to the no-dividend
forward `S·e^{rT}`), and the implied `q = r − ln(F/S)/T` is exposed only as a diagnostic. This
breaks the chicken-and-egg with the downstream forward-based filter.

### Instrument Classification — `data/instrument_classifier.py`

Classifies a ticker EQUITY / ETF / INDEX (override maps → `^` prefix → yfinance `quoteType` →
default EQUITY). Cash-settled indices (SPX, NDX, …) are European; ETFs and single names are
American. Every instrument is priced with a *continuous* dividend yield (the implied forward
bakes discrete cash dividends into one effective rate), so there is no discrete-cashflow path.

### Discount Rates — `config/market_config.py`

A **SOFR/OIS** discount curve (FRED SOFR compound averages short end, US Treasury on-the-run
yields long end) is built and cached hourly; `interpolate_rate(curve, T)` interpolates log-linearly
in discount factors. Per-row `r` is stamped by maturity; a flat 4.5% fallback applies if all
network calls fail.

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

### 7.1 European Options — Gauss-Legendre Quadrature (Cui CF)

**Primary file:** [pricing/european_gl.py](pricing/european_gl.py) using [models/heston_cf_cui.py](models/heston_cf_cui.py)

The current European engine uses the **Cui et al. (2016)** characteristic function — numerically
continuous for long maturities (no branch-switching discontinuities) and analytically
differentiable — evaluated by **64-node Gauss-Legendre quadrature** truncated at `ū = 200`:

```
Call = ½·(S0·e^{-qT} − K·e^{-rT}) + e^{-rT}/π · (I1 − K·I2)

I1 = ∫_0^ū Re[ K^{-iu}/(iu) · φ(u − i) ] du      I2 = ∫_0^ū Re[ K^{-iu}/(iu) · φ(u) ] du
Put = Call − S0·e^{-qT} + K·e^{-rT}              (put-call parity)
```

This achieves ~1e-8 accuracy for all standard maturities and — crucially for calibration —
`heston_call_price_and_gradient()` returns the price **and** all five analytic gradient
components in a single vectorized pass over the quadrature nodes (Cui Theorem 1 / Eq. 22). The
continuous carry `q` (the implied-forward yield) enters only the price level, not the gradient.

> The legacy classic-Heston CF (`models/Heston_cf.py`) with `scipy.integrate.quad` inversion
> (`models/heston_european.py`) is retained but is no longer on the calibration path.

### 7.0 De-Americanization (pre-pricing market normalization)

**File:** [calibration/de_americanize.py](calibration/de_americanize.py)

Because the fast CF pricer produces *European* prices, American market quotes are converted to
European-equivalents once, after filtering: a CRR binomial tree backs out the constant vol σ*
that reproduces the American price (the tree handles early exercise explicitly, so the premium
does not leak into σ*), then a European BS price is taken at the same σ*. This adds `euro_mid`
(European-equivalent mid) and `deam_iv` (σ*, the de-Americanized implied vol) — the single
source of European-equivalent market vol used by both calibration and analytics.

### 7.2 American Options — Priced as European-Equivalent

There is no dedicated American pricer. Because American quotes are de-Americanized once
up front (§7.0) and the model is calibrated in European-equivalent space, every contract —
European or American — is priced with the European GL engine of §7.1, carrying the
continuous implied-forward yield `q`. `price_option_row()` in
[services/pricing_service.py](services/pricing_service.py) therefore dispatches only by
option type (call/put), not exercise style.

> **Removed (2026-06-27).** The earlier American pricers — the explicit finite-difference
> PDE solver, the LSMC (Longstaff-Schwartz) Monte Carlo path, and the C++ MC kernel — were
> removed. They were already unused in the live flow (analytics priced everything as
> European; calibration always used the European proxy). The code is preserved in the
> gitignored `_graveyard.py` at the repo root for future reference; a dedicated American
> pricer is a planned later addition.

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

The calibration fits `(v0, theta, sigma, rho)` by **nonlinear least squares** on price
residuals, following Cui et al. (2016). `kappa` is **fixed, not optimised**: the option
surface does not identify it (kappa and sigma trade off along a near-flat valley, so a
free kappa drifts to whatever bound the search box imposes). Instead kappa0 is estimated
from the chain's own ATM average-variance term structure,
`w(T)/T = theta + (v0 - theta)(1 - e^(-kappa*T))/(kappa*T)` — a Q-measure estimate with no
historical data — clipped to [0.5, 12] and held fixed (fallback kappa0 = 2.0 only when
fewer than 4 expiries are usable; kappa's standard error is reported as a diagnostic, not
used as a gate).
v0/theta search bounds are dynamic guard rails scaled to the chain's observed
de-Americanized IV range, so the box adapts per ticker and should never bind:

```
r_i(θ) = w_i · (C_model(θ; K_i, T_i) − C*_i)      f(θ) = ½ ‖r(θ)‖²
```

solved with `scipy.optimize.least_squares(method='trf')` — the bounded equivalent of
Levenberg-Marquardt. The decisive feature is an **analytic Jacobian**: the Cui CF gives
`∂φ/∂θ_j = φ · h_j(u)`, so the same 64-node Gauss-Legendre quadrature that prices the option
also returns all five gradient components in one pass (`pricing/european_gl.heston_call_price_and_gradient`).
This is 10–16× faster than finite differences and far more stable than the previous
L-BFGS-B / scalar-IV-MSE optimizer. Stopping tolerances mirror the paper (`ftol = gtol = xtol = 1e-10`).

> The `objective` argument is retained for backward compatibility but is ignored — LM always
> minimizes (weighted) price residuals.

### 9.2 Residuals, Jacobian, and Weighting — `calibration/heston_loss_function.py`

- `heston_residuals(params, r, q, options_df, Ns, Nv, Nt, pricing_mode, weights)` builds the
  residual vector `w_i · (price_i − market_i)`, plus one trailing **soft Feller penalty** term
  `FELLER_WEIGHT · max(0, σ² − 2κθ)`.
- `heston_jacobian(...)` returns the matching analytic Jacobian (Cui Theorem 1). Every
  contract is priced as European-equivalent, so the gradient is always analytic.
- `compute_residual_weights(...)` produces per-contract weights, normalized to mean 1:
  - `"vega"` (default) — `1/vega` at market IV, making the price-space fit behave like an
    IV-space fit so near-intrinsic high-price contracts don't dominate the skew-bearing wings.
  - `"none"` — plain price residuals.
  - `"inv_spread"` — `1/rel_spread`, trust tight markets more.

  Weights are constant in the parameters, so the analytic Jacobian stays exact (each row is just scaled).

**Feller penalty is OFF by default** (`FELLER_WEIGHT = 0.0`). High vol-of-vol relative to mean
reversion is the empirical norm for single names (NVDA especially); enforcing `2κθ ≥ σ²` pins
the fit to that boundary, forcing κ high and distorting σ/ρ. A synthetic recovery test (chain
priced from known params with σ = 1.5 > 2κθ) recovered the truth to machine precision at
weight 0 but failed at weight 50.

### 9.3 Calibration Universe — "calibrate tight, price broad"

`select_calibration_universe()` builds a deliberately tight fitting set off the
already-europeanized chain:

1. Maturity window + forward-moneyness band (`K/F`, the economically correct ATM measure).
2. Optional open-interest floor.
3. **OTM-only leg per strike** relative to the implied forward `F` (industry practice): the put
   when `K < F`, the call when `K > F`. This drops the wide-spread, near-intrinsic ITM mirror,
   removes early-exercise noise, and avoids double-counting the same IV from a call+put at one strike.
4. Per-expiry near-ATM cap: rank by ATM distance (asc), rel_spread (asc), volume (desc), keep
   `contracts_per_expiry` per expiry up to `max_expiries`.

The calibration target is the **European-equivalent** price: `euro_mid` is used as `mid_price`
and `deam_iv` (σ*) as `market_iv` (the raw American quote is preserved in `mid_price_market`).
`ExerciseStyle` is forced to `"european"` so the analytic Jacobian is always available — no
American pricer in the optimizer loop.

The broad pricing/analytics layer, by contrast, runs over the full filtered chain.

### 9.4 Initial Guess and Bounds

**Default (current path):** fixed search box and starting point following Cui et al. (2016),
which deliberately calibrates "without any presumption on the values of the parameters" and
shows the analytic-gradient LM converges from broad fixed ranges:

```
Initial guess: v0 = 0.20, kappa = 1.20, theta = 0.20, sigma = 0.30, rho = -0.60
Bounds: v0 ∈ [0.05, 0.95], kappa ∈ [0.50, 10.0], theta ∈ [0.05, 0.95],
        sigma ∈ [0.05, 3.0], rho ∈ [-0.90, -0.10]
```

The σ ceiling is raised to 3.0 (vs. the paper's 0.95) because high vol-of-vol single names
need room; the Cui CF stays numerically stable well past 1.0. The initial guess is clamped
into the box before the solve. Callers can override `bounds` / `initial_guess` per run.

The two estimators below are available as **starting-point and sanity tools** but are **not**
wired into the default calibration path:

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

1. Calls `select_calibration_universe()` to pick the OTM, near-ATM calibration subset (already
   carrying European-equivalent `euro_mid` / `deam_iv`).
2. Stamps per-row `r` from the OIS curve when a `rate_curve` is supplied.
3. Runs `calibrate_heston()` (Levenberg-Marquardt, analytic Jacobian) and measures runtime.
4. Returns a `CalibrationResult` frozen dataclass:

```python
CalibrationResult(
    params=HestonParameters(v0, kappa, theta, sigma, rho),
    loss=float,              # ½‖r(θ*)‖² at the optimum
    contract_count=int,
    objective=str,           # always "price"
    pricing_mode=str,        # "european_proxy" (the only supported mode)
    calibration_style=str,
    runtime_seconds=float,
    weight_scheme=str,       # "vega" (default), "none", or "inv_spread"
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

The app is a **step-by-step, self-contained pipeline**. Each page under `app/pages/` loads its
own data, runs its own step, and hands results to the next page via `st.session_state` (the raw
chain and OIS curve are stashed by Step 1, the filtered chain by Step 2, the calibration by
Step 3, and so on). `app/shared.py` is now minimal — just `configure_page()` and `sys.path`
setup. The earlier centralized sidebar + `@st.cache_data` layer and the three-slot calibration
panel were retired; that dead code lives in the gitignored `_graveyard.py`.

### Page Flow

`Home` → `01_Fetch_Data` → `02_Filter_Options` → `03_Calibrate_Heston` → `04_Price_Contracts`
→ `05_Volatility_Surface` → `06_Strategy_Lab` → `4_Risk_Dashboard` → `07_Mispricing_Screener`.

| Page | Step | Does |
|------|------|------|
| `Home.py` | — | Landing page; explains the pipeline and links the page flow |
| `01_Fetch_Data.py` | 1 | Load a live chain + the SOFR/OIS curve; stash both in session state |
| `02_Filter_Options.py` | 2 | Apply the 11-step filter and de-Americanize survivors (`euro_mid`, `deam_iv`) |
| `03_Calibrate_Heston.py` | 3 | Select OTM near-ATM universe; vega-weighted LM over the CF pricer |
| `04_Price_Contracts.py` | 4 | Price the chain under the calibrated params; build the analytics table |
| `05_Volatility_Surface.py` | 5 | Market / model / error IV surfaces and Greek surfaces (Plotly 3-D) |
| `06_Strategy_Lab.py` | 6 | Multi-leg strategy builder + payoff diagram + net Greeks |
| `4_Risk_Dashboard.py` | 7 | Limit checks + 63-row spot×vol×time scenario table |
| `07_Mispricing_Screener.py` | 8 | Model-vs-market dislocations, RV pairs, parity/bias/VRP lenses |

### Pages (detail)

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
# Market metrics only (live chain)
python pipelines/run_pricing.py \
  --tickers NVDA \
  --max-contracts 100

# With Heston model pricing (pass calibrated params)
python pipelines/run_pricing.py \
  --tickers NVDA \
  --max-contracts 100 \
  --heston-params 0.04,2.0,0.04,0.5,-0.7 \
  --pricing-limit 50 \
  --output analytics.csv
```

Without `--output`, prints a summary table showing: contract_id, type, maturity, strike, mid_price, market_iv, market_delta, market_gamma, market_vega, model_iv, iv_error.

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | `NVDA` | Comma-separated ticker list |
| `--spread-limit` | `0.05` | Maximum relative bid-ask spread |
| `--risk-free-rate` | `0.05` | Continuous risk-free rate (fallback if no curve) |
| `--dividend-yield` | `0.0` | Continuous dividend yield |
| `--min-volume` | `1` | Minimum daily volume |
| `--min-open-interest` | `0` | Minimum open interest |
| `--max-maturity` | `2.0` | Maximum time to expiry in years |
| `--max-contracts` | `400` | Hard contract cap |
| `--option-types` | `call,put` | Option types to keep |
| `--heston-params` | None | `v0,kappa,theta,sigma,rho` for model pricing |
| `--pricing-limit` | `150` | Max contracts repriced with Heston model |
| `--output` | None | CSV output path |

### Calibration Pipeline — `pipelines/run_calibration.py`

Calibrates Heston parameters and saves the result as JSON:

```bash
# Calibrate on a live chain (Levenberg-Marquardt, european-proxy)
python pipelines/run_calibration.py \
  --tickers NVDA \
  --max-expiries 6 \
  --contracts-per-expiry 6 \
  --output results.json
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
  "calibration_style": "european_proxy",
  "runtime_seconds": 4.21,
  "weight_scheme": "vega",
  "calibration_contracts": ["NVDA_call_2024-01-19_500.0", "..."]
}
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | `NVDA` | Comma-separated ticker list |
| `--spread-limit` | `0.05` | Maximum relative bid-ask spread |
| `--risk-free-rate` | `0.05` | Risk-free rate (fallback if no curve) |
| `--dividend-yield` | `0.0` | Dividend yield |
| `--min-volume` | `1` | Minimum daily volume |
| `--min-open-interest` | `0` | Minimum open interest |
| `--max-maturity` | `2.0` | Maximum time to expiry in years |
| `--max-contracts` | `400` | Hard contract cap on the loaded chain |
| `--max-expiries` | `6` | Max expiry dates in calibration universe |
| `--contracts-per-expiry` | `6` | Contracts selected per expiry |
| `--initial-guess` | None | Optional `v0,kappa,theta,sigma,rho` override |
| `--output` | None | JSON output path |

Calibration always uses Levenberg-Marquardt over the European-proxy CF pricer with vega-weighted
residuals; there is no `--calibration-style` flag.

---

## 17. Configuration

### SOFR/OIS Discount Curve — `config/market_config.py`

This module is no longer a bag of static constants — it builds the live discount curve:

```python
from config.market_config import get_ois_curve, interpolate_rate, fetch_sofr_rate

r_3m   = fetch_sofr_rate(T=0.25)          # single rate for most uses
curve  = get_ois_curve()                  # full {T_years: rate} dict, cached ~1h
r_at_T = interpolate_rate(curve, T=1.5)   # log-linear in discount factors
```

- **Short end (≤ 6M):** SOFR compound averages from FRED (`SOFR`, `SOFR30/90/180DAYAVG`).
- **Long end (≥ 3M):** US Treasury on-the-run yields from yfinance (`^IRX`, `^FVX`, `^TNX`).
- **Fallback:** flat `0.045` if every network call fails.
- Cached for `_CACHE_TTL = 3600s`, so it is safe to call on every Streamlit rerender.

---

## 18. Full End-to-End Workflow

### Using the Web App

```
1. Launch the app:
   streamlit run app/Home.py
   Open the URL shown in the terminal (default: http://localhost:8501).

2. Walk the page flow top to bottom — each page consumes the previous step's
   result from session state, so run them in order:

   Step 1  Fetch Data        Enter tickers; load the live chain + SOFR/OIS curve.
   Step 2  Filter Options    Apply the 11-step filter; survivors are de-Americanized
                             (euro_mid, deam_iv) automatically.
   Step 3  Calibrate Heston  Vega-weighted Levenberg-Marquardt over the CF pricer;
                             loss / contract count / runtime shown, result cached.
   Step 4  Price Contracts   Reprice the chain under the calibrated params; build the
                             enriched analytics table (model_iv, Greeks, iv_error).
   Step 5  Volatility Surface  Market / model / error IV surfaces + Greek surfaces (3-D).
   Step 6  Strategy Lab      Select legs; view payoff, break-evens, net Greeks.
   Step 7  Risk Dashboard    Limit checks (pass/warn/reject) + 63-row scenario table.
   Step 8  Mispricing Screener  Ranked dislocations + relative-value pairs.
```

### Using the CLI

```bash
# Step 1: Calibrate on a live chain
python pipelines/run_calibration.py \
  --tickers NVDA \
  --output calibration.json

# Step 2: Read calibrated params and build enriched analytics
python pipelines/run_pricing.py \
  --tickers NVDA \
  --heston-params "$(python -c "
import json
d = json.load(open('calibration.json'))
print(f\"{d['v0']},{d['kappa']},{d['theta']},{d['sigma']},{d['rho']}\")
")" \
  --output analytics.csv
```

---

## 19. Key Design Decisions

### Why Levenberg-Marquardt with an Analytic Jacobian?

Heston calibration is a nonlinear least-squares problem, and LM (Gauss-Newton with damping) is
its natural solver. The decisive advantage is the **Cui et al. (2016) analytic gradient**: it
delivers an exact Jacobian for the cost of the pricing integrals themselves (one vectorized GL
pass returns price + all five derivatives), so the optimizer never pays for finite differences
and never suffers their noise. This replaced the earlier L-BFGS-B / scalar-IV-MSE setup, which
was both slower (FD gradients) and less stable. `method='trf'` keeps the box constraints LM
needs for Heston bounds.

### Why De-Americanize Instead of an American Pricer in the Loop?

Calibrating to raw American quotes would force a PDE/LSMC pricer inside every optimizer
iteration — slow and numerically fragile. De-Americanizing once (CRR-tree implied vol → European
re-price) strips the early-exercise premium in a model-consistent way, so the fast CF pricer and
its analytic Jacobian can run the whole calibration. The only residual approximation is that the
premium is computed under constant-vol GBM rather than Heston — a small, smooth correction.

### Why a Characteristic-Function Engine for All Pricing?

The CF approach is exact up to integration tolerance — no grid discretization error, no
Monte Carlo variance — and the same Gauss-Legendre quadrature returns the analytic gradient
used by calibration. Pricing and calibration therefore share one engine (same CF, same
carry `q`), so model prices and the calibration objective are always mutually consistent.

### Why No American Pricer?

The engine is European-equivalent by construction: de-Americanization (§7.0) maps American
quotes to European-equivalent prices, and calibration fits that surface. Pricing American
contracts as their European equivalent is then the consistent choice — a separate
American pricer would reintroduce a second valuation basis. The previous PDE and LSMC
pricers were already off the live path, so they were removed (archived in `_graveyard.py`).
A dedicated American pricer can be reintroduced later if early-exercise P&L is needed
explicitly.

### Why `ensure_option_frame()` as a Gate?

Without a normalization gate, each analytics function would need to handle column aliasing, type coercion, and missing field logic independently — leading to fragile and duplicated code. `ensure_option_frame()` establishes a contract: everything downstream can assume the schema exists, types are correct, and computed columns are present.

### Why JSON-Based Calibration Cache?

JSON is human-readable and easily debuggable. The scope ID encodes all parameters that affect the calibration result, so different market conditions, rates, or tickers never collide. Re-running calibration with the same inputs is instant; re-running with different inputs produces a new file without disturbing old results.

### Why Liquidity-Weighted Mispricing Score?

A raw IV error does not distinguish between a heavily traded ATM option and a deep OTM contract with a 50% spread. Multiplying by `(1 + liquidity_score / 100)` up-weights mispricing signals in liquid, tight-spread options where the signal is most exploitable and least likely to be a data artifact or wide-spread noise.

### Why Data-Driven Bounds?

Static parameter bounds force the optimizer to search broadly even when the market already reveals the approximate answer. Estimating v0, theta, sigma, rho, and kappa directly from the observed IV surface — using known short-time Heston approximations — produces tight, data-consistent bounds that dramatically reduce the search space and improve convergence speed and solution quality.
