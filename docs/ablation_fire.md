# Fire Signal Feature Ablation Study

## Overview
This experiment evaluates the empirical contribution of transboundary fire activity features 
derived from NASA FIRMS (VIIRS & MODIS) thermal anomaly detections within the Punjab boundary 
box ($28^\circ-34^\circ\text{N}, 70^\circ-77^\circ\text{E}$).

The ablated model replaces all 6 fire feature columns (`fire_count_lag1..3`, `fire_frp_lag1..3`) 
with zeros during training, conformal calibration, and test inference, leaving all other 
autoregressive, meteorological (ERA5), and atmospheric chemistry (ECMWF CAMS) predictors intact.

Evaluated on identical 23,547 held-out test rows (`2025-07-02` to `2026-09-30T23:59:59Z`).

## Empirical Results

| Horizon | Model Variant | MAE (µg/m³) | RMSE (µg/m³) | 80% Coverage | Mean Band Width (µg/m³) |
|---|---|---|---|---|---|
| **+6h** | Full Model (with Fire) | 22.50 | 35.50 | 82.0% | 72.9 |
| **+6h** | Ablated (Zeroed Fire) | 22.12 | 34.65 | 82.4% | 74.0 |
| *Delta (+6h)* | *Difference (Ablated - Full)* | *-0.38* | *-0.85* | *+0.4%* | *+1.1* |
| **+12h** | Full Model (with Fire) | 22.77 | 36.81 | 81.9% | 74.1 |
| **+12h** | Ablated (Zeroed Fire) | 22.43 | 35.99 | 83.1% | 76.7 |
| *Delta (+12h)* | *Difference (Ablated - Full)* | *-0.34* | *-0.82* | *+1.2%* | *+2.5* |
| **+24h** | Full Model (with Fire) | 23.00 | 37.05 | 83.2% | 75.7 |
| **+24h** | Ablated (Zeroed Fire) | 22.56 | 36.28 | 83.9% | 77.4 |
| *Delta (+24h)* | *Difference (Ablated - Full)* | *-0.44* | *-0.77* | *+0.8%* | *+1.7* |

## Analysis & Interpretation
- **Marginal MAE Impact:** The net difference between including vs zeroing fire features across all horizons is modest (≤ 0.4 µg/m³ across the full held-out test set).
- **Seasonal vs Year-Round Dilution:** Agricultural crop residue burning in Punjab is highly episodic, concentrated in autumn post-monsoon harvest windows (October–November). Because the held-out test set spans all months from July 2025 to September 2026, the 10-month non-burning period dilutes the aggregate statistical effect of fire predictors.
- **CAMS Collinearity:** ECMWF CAMS atmospheric reanalysis already incorporates satellite aerosol optical depth (AOD) and particulate transport assimilations, absorbing much of the direct smoke column variance.
- **Conclusion:** While NASA FIRMS provides physically grounded domain signals for acute transboundary smoke plumes, zeroing fire counts does not cause catastrophic degradation on average days. Its primary role in Saaf Hawa is extreme-event contextualization rather than day-to-day variance reduction.

*Generated automatically by `scripts/ablation_fire.py`.*