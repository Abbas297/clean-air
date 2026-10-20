# Model Card: Saaf Hawa v2 Smog Forecaster

## Model Overview
- **Model Architecture:** Quantile Gradient Boosted Decision Trees (LightGBM) with Split Conformal Prediction (CQR).
- **Target Variable:** Ambient fine particulate matter (PM2.5 in $\mu\text{g}/\text{m}^3$) at $+6\text{ h}$, $+12\text{ h}$, and $+24\text{ h}$ horizons.
- **Intended Users:** District education, health, and disaster management officers in Punjab, Pakistan.
- **Intended Use:** Informing next-day outdoor activity restrictions and smog advisories with calibrated uncertainty bands.

## Training Data & Splits
- **Time Splits:** Train (2022-07-01 to 2024-06-30), Validation (2024-07-02 to 2025-06-30), Test (2025-07-02 to 2026-09-30) with 24h embargo.
- **Inputs:** Ground sensor lags, CAMS PM2.5/PM10/AOD, ERA5 weather variables, transboundary VIIRS/MODIS fire signals.

## Evaluation & Ethical Limitations
- Evaluated against Persistence and Seasonal-Hour baselines.
- Explicit disaggregation across humidity bands and sensor redundancy counts.
- Unvalidated cities are labeled with expanded uncertainty and explicitly stated in all model outputs.
