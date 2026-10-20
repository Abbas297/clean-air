# Declaration Addendum (written Wed 21 Oct 2026, after results were first seen)

`DECLARATION.md` is unchanged. This file records, in the open, where what we actually did differs from what we declared. Nothing here changes a baseline or a success criterion.

1. **Effective training window.** Declared train start 2022-07-01. Open-Meteo/CAMS history and the sensor pulls start 2023-01-01, so training effectively starts 2023-01-01. Per-city training hours: Islamabad 11,168, Lahore 4,899, Peshawar 0, Karachi 0.
2. **Peshawar and Karachi.** Peshawar has no training rows (its sensor starts 2025-05-29) and is scored as a city the model never trained on. Karachi has 101 usable sensor hours in total, so it is excluded from results and marked inactive.
3. **Calibration.** Cities with fewer than 30 sensor/reference pairs in the training window (Peshawar, Karachi) now use the sensor values as published (identity) and are labelled uncalibrated. An earlier version applied fixed coefficients that were not fitted to data; that was removed.
4. **Label source changes between train and test.** Training labels for Islamabad come from the series `s_1343270` (AirNow Community); test labels come from the F-7 AirGradient sensor. Lahore moves from Clarity to Clarity + Barki AirGradient. Test error therefore includes a label-source shift.
5. **Reference independence.** `s_1343270` matches the embassy reference exactly in about 75% of the 8,201 overlapping hours (correlation 0.89), so for Islamabad the calibration check is partly circular. Treat the Islamabad calibration R² as not independent evidence.
6. **Fire features.** FIRMS history was not backfilled in the first pipeline (only the last 7 days), so lagged fire features were 0 for almost the whole history. Counts were also inflated by repeated downloads of the same detections; fixed by de-duplication. `scripts/backfill_firms.py` loads real history. Models must be retrained after the backfill; the retrain is a bug fix of a declared feature, not a new feature.
7. **Declared but not evaluated.** Success criterion 4 (advisory recall/precision vs the "current PM2.5 ≥ 55.5" rule) and the humidity-band and sensor-count group tables are not produced by the current evaluation code. They are reported as not done, not as passed.
8. **Test set use.** The evaluation code was corrected three times after test scores were first seen (commits at 02:48, 03:11, 03:26 PKT on 21 Oct) and the diurnal groups were moved from UTC to PKT afterwards. Final numbers should come from one last run, with no further tuning.
9. **Diurnal groups** are now in PKT (UTC+5). Earlier tables used UTC hours labelled as time of day.
10. **Test split upper bound.** Declared test end is 2026-09-30. `evaluate.py` initially evaluated all rows with `ts_utc >= test_start` without bounding at `test_end`, which inadvertently allowed new hourly ingestion rows to drift into the test evaluation. This was corrected to enforce strict `(ts_utc >= test_start) & (ts_utc <= test_end)`.
11. **Subgroup performance & weak spots (disclosed plainly):**
   - **Night at +6h (00–06 PKT):** The model (MAE 25.49) is slightly worse than persistence (25.20) and seasonal (25.19).
   - **Afternoon at +12h (12–18 PKT):** The model (MAE 25.18) is slightly worse than the seasonal baseline (24.90).
   - **Lahore interval coverage:** 76.0% at +6h and 74.7% at +12h, falling short of the nominal 80% target.
   - **Very Unhealthy tier (>150 µg/m³):** 80% interval coverage drops to 72.2% at +6h, with an MAE of 53.83 µg/m³. Operational decision-makers (DEAs) must be aware that interval uncertainty is under-covered during acute smog peaks.
