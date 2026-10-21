# Fire Signal Feature Ablation Study

## 1. Summary of Findings
We expected NASA FIRMS satellite fire features (lagged detection counts and Fire Radiative Power) to improve forecast accuracy for Punjab smog. **The empirical data show no benefit from fire features.**

When all six fire lag columns (`fire_count_lag1..3`, `fire_frp_lag1..3`) were zeroed out during training, conformal calibration, and test inference, the ablated model achieved equal or slightly lower Mean Absolute Error (MAE) at all three horizons across both the full held-out test split and the acute smog season (October–February).

In accordance with scientific rigour, we keep the fire features in the declared model pipeline as declared, and transparently disclose this negative finding.

---

## 2. Full Held-Out Test Set (23,547 rows, July 2025 – September 2026)

| Horizon | Model Variant | N Samples | MAE (µg/m³) | RMSE (µg/m³) | 80% Coverage | Mean Band Width (µg/m³) |
|---|---|---|---|---|---|---|
| **+6h** | Full Model (with Fire) | 23,547 | 22.10 | 34.72 | 81.9% | 72.2 |
| **+6h** | Ablated (Zeroed Fire) | 23,547 | 21.90 | 34.22 | 82.3% | 73.2 |
| *Delta (+6h)* | *Difference (Ablated - Full)* | - | *-0.20* | *-0.50* | *+0.4%* | *+1.0* |
| **+12h** | Full Model (with Fire) | 23,541 | 22.79 | 36.58 | 82.5% | 75.9 |
| **+12h** | Ablated (Zeroed Fire) | 23,541 | 22.21 | 35.55 | 83.7% | 77.1 |
| *Delta (+12h)* | *Difference (Ablated - Full)* | - | *-0.58* | *-1.04* | *+1.2%* | *+1.2* |
| **+24h** | Full Model (with Fire) | 23,529 | 23.15 | 37.30 | 83.3% | 76.4 |
| **+24h** | Ablated (Zeroed Fire) | 23,529 | 22.64 | 36.37 | 83.9% | 78.8 |
| *Delta (+24h)* | *Difference (Ablated - Full)* | - | *-0.51* | *-0.93* | *+0.5%* | *+2.5* |

---

## 3. Smog Season Subset (October to February, 7,732 rows)

To test whether fire features provide value specifically during the agricultural residue burning season, we evaluated the identical trained models and conformal calibration strictly on the subset of test rows occurring between October 1 and February 28/29:

| Horizon | Model Variant | N Samples | MAE (µg/m³) | RMSE (µg/m³) | 80% Coverage | Mean Band Width (µg/m³) |
|---|---|---|---|---|---|---|
| **+6h** | Full Model (with Fire) | 8,563 | 37.43 | 51.16 | 80.8% | 118.1 |
| **+6h** | Ablated (Zeroed Fire) | 8,563 | 37.12 | 50.27 | 81.4% | 121.0 |
| *Delta (+6h)* | *Difference (Ablated - Full)* | - | *-0.32* | *-0.89* | *+0.6%* | *+3.0* |
| **+12h** | Full Model (with Fire) | 8,563 | 37.65 | 53.10 | 78.8% | 116.3 |
| **+12h** | Ablated (Zeroed Fire) | 8,563 | 36.33 | 51.17 | 80.7% | 119.4 |
| *Delta (+12h)* | *Difference (Ablated - Full)* | - | *-1.32* | *-1.93* | *+1.9%* | *+3.1* |
| **+24h** | Full Model (with Fire) | 8,563 | 37.87 | 53.82 | 79.4% | 117.1 |
| **+24h** | Ablated (Zeroed Fire) | 8,563 | 36.87 | 52.38 | 81.9% | 122.7 |
| *Delta (+24h)* | *Difference (Ablated - Full)* | - | *-0.99* | *-1.44* | *+2.5%* | *+5.5* |

Even during the smog season, zeroing fire features yielded marginally lower error across all horizons (MAE 37.12 vs 37.43 at +6h; 36.33 vs 37.65 at +12h; 36.87 vs 37.87 at +24h).

---

## 4. Reconciliation with Official Evaluation Scores
An initial prototype ablation script reported a full-model MAE of 22.50 µg/m³ at +6h, differing from the official pipeline evaluation of 22.10 µg/m³. The technical reasons for this divergence are:
1. **Tree Depth Hyperparameter:** The official evaluation uses LightGBM's default unconstrained tree depth (`max_depth: -1`, `num_leaves: 31`), whereas the initial prototype had restricted tree growth to `max_depth: 6` via `LGBMRegressor`.
2. **Like-for-Like Baseline Inner Join:** The official evaluation script (`src/model/evaluate.py`) strictly evaluates rows where both Baseline A (Persistence) and Baseline B (Seasonal-Hour) are populated ($N = 23,547$), whereas the prototype script evaluated all test split rows without inner-joining baseline availability ($N = 24,064$).
When evaluated using identical hyperparameters and identical baseline-joined rows, the full model reproduces the exact official evaluation score of **22.10 µg/m³** at +6h.

---

## 5. Potential Hypotheses (Unverified)
The finding that satellite fire features do not improve forecast accuracy is counterintuitive given the prominence of crop residue burning in Punjab smog discussions. We note two **unverified hypotheses**:
- *Collinearity with Atmospheric Chemistry (Unverified):* ECMWF CAMS reanalysis already assimilates satellite Aerosol Optical Depth (AOD) and particulate transport fields. CAMS predictors may already capture the transboundary smoke signal, rendering explicit FIRMS fire counts redundant.
- *Temporal Resolution & Cloud Masking (Unverified):* VIIRS polar-orbiting satellites provide at most 1–2 overpasses per day per sensor. Missing observations due to heavy overcast, combined with broad daily lag aggregation, may introduce noise that dilutes short-term hourly autoregressive predictive power.

Neither hypothesis has been experimentally proven; the primary verified empirical conclusion is simply that the data show no measurable predictive gain from fire features under the declared setup.