# Saaf Hawa: 24-hour smog forecasting for Punjab after the loss of embassy monitors

## 1. Problem & Approach
Public reference-grade air quality data in Pakistani cities became unavailable after the U.S. embassy network stopped publishing to AirNow in 2025. Saaf Hawa rebuilds ground truth from community sensors (calibrated to the embassy record where the two overlap), adds ECMWF CAMS, weather and NASA FIRMS fires, and forecasts hourly PM2.5 at +6, +12 and +24 h with calibrated 80% intervals for district education and health officers in Punjab.

## 2. Data & Quality
Independent publishers: OpenAQ community sensors, U.S. AirNow embassy archive (not on the track list), ECMWF/Copernicus via Open-Meteo, NASA FIRMS. Gold holds 56,373 city-hourly PM2.5 rows and 133,536 feature rows. Every raw file has a SHA-256 row in manifest.csv. Gate 1 (schema, sentinels, stuck sensors, extreme-value peer check) and Gate 2 (freshness, drift) write each fix/fill/flag decision to quality_log.

## 3. Results vs declared baselines (held-out test, 24 h embargo)

| Horizon | A: Persistence MAE | B: Seasonal-hour MAE | LightGBM+CQR MAE | 80% coverage | Beats A and B |
|---|---|---|---|---|---|
| +6h | 30.35 | 25.08 | 22.10 | 81.9% | yes |
| +12h | 33.85 | 25.07 | 22.79 | 82.6% | yes |
| +24h | 24.88 | 25.08 | 23.15 | 83.3% | yes |

*MAE in µg/m³ on identical test rows. Declared target for the 80% interval: 75-85% (observed 81.9-83.3%). Seasonal-hour is the same at every horizon because it uses the previous 7 days of the target hour. Numbers are read from state/eval_summary.csv.*

## 4. Limitations (read these first)
• Fire signal: FIRMS series in Silver covers 1347 days (Jan 2023 - Oct 2026). Offline ablation found no predictive gain from fire features (MAE 21.90 vs 22.10 at +6h); kept as declared.
• Uncalibrated labels (no reference-monitor overlap in the training window): karachi, peshawar.
• Labels in the test period come from different sensors than in training (sensor change in May 2025), so test error includes a label-source shift.
• Weakest group: Smog (Oct-Feb) at +24h (MAE 37.87, 80% coverage 79.4%).
• Subgroups trailing baselines: Night (00-06 PKT) +6h; Afternoon (12-18 PKT) +12h; Very Unhealthy (>150) +24h (Night +6h trails persistence 25.49 vs 25.20; Afternoon +12h trails seasonal 25.18 vs 24.90).
• Coverage drops in critical slices: Lahore achieves 74.7-76.0% (target 80%); Very Unhealthy tier (>150 µg/m³) coverage drops to 72.2% (MAE 53.83).
• Peshawar transfer: Peshawar has 0 training hours and represents 45.5% of test rows (10,710 of 23,547); the result is largely an out-of-distribution transfer to an unseen city.
• Partial circularity: Islamabad calibration uses series s_1343270 which matches the embassy monitor in ~75% of overlap hours; treat its calibration R² as not independent evidence.
• Advisory rule: Criterion 4 met at +6h and +12h, but not at +24h (recall 75.0% vs 82.4% baseline; p50 chosen to avoid 65% false alarms); Lahore base rate is 72.3%.
• Run history: 11 runs (1 not Completed) from 2026-10-21 02:35 AM PKT to 2026-10-21 09:59 AM PKT; longest gap 4.66 h between unyielding-beluga and rough-snake; 42 missing fire dates treated as 0 fires (cause not established).

## 5. Responsible AI
The Insight Agent takes numbers only from SQL on Gold tables, stores row IDs with each answer, answers "I don't know" when retrieval or data is missing, points to Rescue 1122 / Health helpline 1033 for emergencies, and labels unvalidated cities. Outputs are information, not professional advice.
