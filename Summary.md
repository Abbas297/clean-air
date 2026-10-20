# Saaf Hawa: 24-hour smog forecasting for Punjab after the loss of embassy monitors

## 1. Problem & Approach
Public reference-grade air quality data in Pakistani cities became unavailable after the U.S. embassy network stopped publishing to AirNow in 2025. Saaf Hawa rebuilds ground truth from community sensors (calibrated to the embassy record where the two overlap), adds ECMWF CAMS, weather and NASA FIRMS fires, and forecasts hourly PM2.5 at +6, +12 and +24 h with calibrated 80% intervals for district education and health officers in Punjab.

## 2. Data & Quality
Independent publishers: OpenAQ community sensors, U.S. AirNow embassy archive (not on the track list), ECMWF/Copernicus via Open-Meteo, NASA FIRMS. Gold holds 56,369 city-hourly PM2.5 rows and 133,440 feature rows. Every raw file has a SHA-256 row in manifest.csv. Gate 1 (schema, sentinels, stuck sensors, extreme-value peer check) and Gate 2 (freshness, drift) write each fix/fill/flag decision to quality_log.

## 3. Results vs declared baselines (held-out test, 24 h embargo)

| Horizon | A: Persistence MAE | B: Seasonal-hour MAE | LightGBM+CQR MAE | 80% coverage | Beats A and B |
|---|---|---|---|---|---|
| +6h | 28.50 | 23.61 | 20.83 | 83.0% | yes |
| +12h | 31.77 | 23.62 | 21.14 | 84.5% | yes |
| +24h | 23.48 | 23.64 | 21.51 | 84.5% | yes |

*MAE in µg/m³ on identical test rows. Declared target for the 80% interval: 75-85% (observed 83.0-84.5%). Seasonal-hour is the same at every horizon because it uses the previous 7 days of the target hour. Numbers are read from state/eval_summary.csv.*

## 4. Limitations (read these first)
• Fire signal: the FIRMS series in Silver covers 8 day(s); where history is missing the lagged fire features are 0, so any fire effect is not yet evidenced.
• Labels in the test period come from different sensors than in training (sensor change in May 2025), so test error includes a label-source shift.
• Weakest group: lahore at +24h (MAE 35.57, 80% coverage 77.4%).
• Groups where the model does not beat both baselines: peshawar +6h; Night (00-06 PKT) +6h.

## 5. Responsible AI
The Insight Agent takes numbers only from SQL on Gold tables, stores row IDs with each answer, answers "I don't know" when retrieval or data is missing, points to Rescue 1122 / Health helpline 1033 for emergencies, and labels unvalidated cities. Outputs are information, not professional advice.
