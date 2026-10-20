---
title: Saaf Hawa v2 - Smog Forecasting & Ground Truth
emoji: 🌬️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: streamlit_app.py
pinned: false
license: apache-2.0
---

# Saaf Hawa v2: Ground-Truth Reconstruction & Uncertainty-Calibrated PM2.5 Forecasting

**Track:** Air Quality | **HackInsight '26** | **Zero-Cost Open Source Stack**

Live Streamlit deployment for Punjab District Education and Health Authorities following the March 2025 shutdown of U.S. Diplomatic Post AirNow monitors in Pakistan.

## Key Features
- **Reconstructed Ground Truth:** Low-cost community sensor network calibrated against pre-halt reference BAMs using Ridge regression with humidity interaction terms.
- **24-Hour Multi-Horizon Forecasts:** 15 LightGBM quantile regression models across $+6\text{h}, +12\text{h}, +24\text{h}$ horizons.
- **Calibrated Prediction Intervals:** Split-Conformal Prediction (CQR) providing honest 80% coverage intervals.
- **Responsible Insight Agent:** Strict SQL grounding with Number Guard and official WHO / PEQS health citations.
- **Zero-Dependency State Bundle:** App runs instantaneously on Hugging Face Spaces and Streamlit Cloud via lightweight, pre-computed state in `state/`.

## Running Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
