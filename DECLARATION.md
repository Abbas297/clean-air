# Saaf Hawa v2: Modeling & Evaluation Declaration

Committed before any model training or parameter selection.

---

## 1. Problem & Target
- **Target:** City-level ground-truth PM2.5 ($\mu\text{g}/\text{m}^3$) at $+6\text{ h}$, $+12\text{ h}$, $+24\text{ h}$ for monitored Pakistani cities (Lahore, Islamabad, Karachi, Peshawar).
- **Ground Truth Source:** Median of QC-passed, calibrated community low-cost sensors (PAQI and OpenAQ).
- **Calibration Reference:** Historical pre-halt US State Department / EPA AirNow embassy monitor record (pre-March 2025). Fitted strictly on training split.

---

## 2. Baselines (Declared & Frozen)
- **Baseline A (Persistence):** Observed ground-truth PM2.5 at issue time $t$.
- **Baseline B (Seasonal-Hour):** Mean observed PM2.5 for the target hour-of-day over the preceding 7 days ($t-168\text{ h}$ to $t$), known at time $t$.
- **Reference Benchmark (Not a baseline):** Raw CAMS hindcast at target horizon (from Open-Meteo), labeled as *"CAMS hindcast (optimistic, not a true forecast)"*.

---

## 3. Data Splits & Embargo Protocol
Strict chronological split with a **24-hour embargo** between contiguous partitions to eliminate label leakage from autoregressive lags:

- **Train:** 2022-07-01 00:00:00 UTC to 2024-06-30 23:59:59 UTC
- **Embargo 1:** 2024-07-01 00:00:00 UTC to 2024-07-01 23:59:59 UTC
- **Validation:** 2024-07-02 00:00:00 UTC to 2025-06-30 23:59:59 UTC (contains 2024–25 smog season)
- **Embargo 2:** 2025-07-01 00:00:00 UTC to 2025-07-01 23:59:59 UTC
- **Test:** 2025-07-02 00:00:00 UTC to 2026-09-30 23:59:59 UTC (contains 2025–26 smog season; touched exactly once)

*Note:* All preprocessing, sensor calibrations, scalers, and imputers are fit exclusively on the Train set. Random seed is locked to `42`.

---

## 4. Feature Space (Known at Issue Time $t$ Only)
- **Autoregressive Lags:** Ground PM2.5 at $t, t-1, t-3, t-6, t-12, t-24, t-48\text{ h}$; rolling 6h, 12h, 24h means and standard deviations.
- **Atmospheric Model:** CAMS PM2.5, PM10, Aerosol Optical Depth (AOD) at time $t$.
- **Meteorology (ERA5 / Open-Meteo):** 2m temperature, relative humidity, $u/v$ wind components, precipitation at time $t$.
- **Fire Signal (NASA FIRMS):** Fire detection counts and total Fire Radiative Power (FRP) aggregated over the Punjab transboundary bounding box ($28^\circ\text{N}-34^\circ\text{N}, 70^\circ\text{E}-77^\circ\text{E}$) lagged 1, 2, and 3 days.
- **Calendar & Spatial:** Hour of day, day of week, month, city categorical encoding.

---

## 5. Model Architecture & Uncertainty
- **Model:** LightGBM Quantile Regression estimating $p10, p25, p50, p75, p90$ independently for each horizon ($+6\text{ h}, +12\text{ h}, +24\text{ h}$) = 15 total estimators.
- **Conformal Calibration:** Conformalized Quantile Regression (CQR) adjustment fitted on Validation block to ensure the nominal 80% prediction interval ($[p10, p90]$) achieves verified empirical coverage.

---

## 6. Evaluation Metrics & Success Criteria
- **Metrics:** MAE, RMSE on $p50$, Pinball loss, Empirical 80% coverage rate, Mean interval width.
- **Decision Metric:** School / health advisory triggers when $P(\text{PM2.5} \ge 55.5\,\mu\text{g}/\text{m}^3) \ge \tau$, where $\tau$ is calibrated on validation to achieve recall $\ge 0.80$. Compared against baseline threshold rule (current $\text{PM2.5} \ge 55.5\,\mu\text{g}/\text{m}^3$).

### Declared Success Criteria:
1. $p50$ MAE strictly lower than Baseline A (Persistence) and Baseline B (Seasonal-Hour) across all 3 horizons on pooled test set.
2. Honest reporting of any individual city where the model does not beat baselines.
3. Pooled 80% interval coverage lies between $75\%$ and $85\%$.
4. At $+24\text{ h}$, advisory recall $\ge$ baseline recall, with precision no worse.

---

## 7. Group Disaggregation & Honesty
Performance is evaluated across groups:
- City (Lahore, Islamabad, Karachi, Peshawar)
- Diurnal cycle (Day vs Night)
- Season (Smog season: Oct–Jan vs Non-smog: Feb–Sep)
- Humidity band ($<60\%$, $60-80\%$, $>80\%$)
- Sensor redundancy ($n_{\text{sensors}} = 1$ vs $n_{\text{sensors}} \ge 2$)
