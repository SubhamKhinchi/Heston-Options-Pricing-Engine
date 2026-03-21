# Heston Options Pricing Engine

This repository now includes:

- reusable analytics and service layers for option-chain metrics, greeks, and surfaces
- pipeline scripts for analytics generation and Heston calibration
- a self-contained multi-page Streamlit app for option chains, surfaces, strategy payoff, and risk checks

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app/Home.py
```

There is also a standalone compatibility surface view:

```bash
streamlit run "vol surface app/vol_surface.py"
```

## Run Pipelines

Build an analytics table from the sample NVDA snapshot:

```bash
python3 pipelines/run_pricing.py --source sample --max-contracts 100
```

Run a small calibration job on the sample snapshot:

```bash
python3 pipelines/run_calibration.py --source sample --max-contracts 100
```
