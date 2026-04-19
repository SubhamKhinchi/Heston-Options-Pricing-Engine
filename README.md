# Heston Options Pricing Engine

This repository now includes:

- reusable analytics and service layers for option-chain metrics, greeks, and surfaces
- pipeline scripts for analytics generation and Heston calibration
- a self-contained multi-page Streamlit app for option chains, surfaces, strategy payoff, and risk checks

## Requirements

- Python 3.10 or newer
- Internet access if you want live Yahoo Finance option-chain pulls

## Setup On Another System

Move the repository to the target machine, then create a virtual environment and install dependencies from the repo root.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3 -m venv` fails on Ubuntu or Debian, install the system package first:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

### Windows Setup

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The App

From the repo root:

```bash
.venv/bin/streamlit run app/Home.py
```

On Windows:

```powershell
python -m streamlit run app/Home.py
```

If port `8501` is already in use:

```bash
.venv/bin/streamlit run app/Home.py --server.port 8502
```

There is also a standalone compatibility surface view:

```bash
.venv/bin/streamlit run "vol surface app/vol_surface.py"
```

## How To Use The App

1. Start the app and open the Streamlit URL shown in the terminal.
2. In the sidebar, choose `Sample snapshot` to use `nvda_vol.xlsx`, or `Yahoo Finance` for live market data.
3. Click `Refresh options chain` to pull fresh data.
4. Leave `Calibration style` on `Fast proxy calibration (Recommended)` unless you specifically want the slower full calibration path.
5. Click `Calibrate Heston`.
6. Once calibration finishes, the app stores the calibrated parameter set and rerenders pricing, model IV, surfaces, and mispricing views using those updated parameters.
7. Use the pages as follows:
   - `Option Chain`: per-contract metrics and greeks
   - `Volatility Surfaces`: market IV, model IV, error, and greek surfaces
   - `Mispricing Screener`: model-vs-market opportunities and trade ideas
   - `Strategy Lab`: manual multi-leg construction and payoff view
   - `Risk Dashboard`: exposure and limit checks for the latest strategy

Stored calibration files are written to:

```bash
results/calibrations/
```

## Run Pipelines

Build an analytics table from the sample NVDA snapshot:

```bash
.venv/bin/python pipelines/run_pricing.py --source sample --max-contracts 100
```

Run a calibration job on the sample snapshot:

```bash
.venv/bin/python pipelines/run_calibration.py --source sample --calibration-style fast
```

Live Yahoo Finance example:

```bash
.venv/bin/python pipelines/run_pricing.py --source live --tickers NVDA --max-contracts 50
```

## Notes

- `Yahoo Finance` mode requires network access.
- Sample mode requires `nvda_vol.xlsx` to be present in the repo root.
- Calibration speed depends heavily on CPU and data size.
- The default fast proxy calibration mode is intended for normal app use because it is much faster than the full calibration path.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.venv/bin/streamlit run app/Home.py
```
