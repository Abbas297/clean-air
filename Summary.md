# Saaf Hawa v2: Rebuilding Ground Truth & Smog Forecasting

**Track:** Air Quality | **Stack:** Zero-Cost Open Source | **Date:** October 2026 | **Event:** HackInsight '26

---

### 1. Executive Summary & The Problem
In March 2025, public ambient air quality monitoring in Pakistan suffered a catastrophic blind spot: the U.S. Department of State / EPA embassy and consulate monitors in Lahore, Islamabad, Karachi, and Peshawar stopped transmitting data to AirNow. For district education officers, public health officials, and school administrators across Punjab, this eliminated the only reference-grade, continuous source of truth needed to make crucial next-day decisions (school closures, morning assembly cancellations, mask mandates).

**Saaf Hawa v2** solves this by:
1. Reconstructing calibrated ground truth from continuous community sensor networks (PAQI, Clarity, AirGradient on OpenAQ v3) calibrated against the pre-halt embassy baseline using Ridge regression with humidity-interaction terms.
2. Training 15 LightGBM quantile regression models across $+6\text{ h}$, $+12\text{ h}$, and $+24\text{ h}$ forecasting horizons using autoregressive lags, transboundary fire signals from NASA FIRMS, and ECMWF CAMS atmospheric reanalysis.
3. Providing uncertainty-calibrated prediction intervals via Split-Conformal Prediction (CQR) achieving 83.0% to 84.8% empirical coverage for nominal 80% prediction intervals.
4. Deploying a grounded Insight Agent with a Number Guard and SQL grounding, providing responsible, cited decision support in English and Urdu.

---

### 2. Multi-Source Architecture & Data Rigour
- **Multi-Publisher Stack:** 4 independent original publishers: OpenAQ (PAQI community network), U.S. State Dept / EPA AirNow historical archive, ECMWF/Copernicus (Open-Meteo ERA5 & CAMS), and NASA Earthdata (FIRMS VIIRS active fires). All fully recorded with live SHA-256 and open licences in `data/manifest.csv`.
- **Dual Data Quality Gates:** Automated Pandera schema enforcement, sentinel cleaning (`-999` $\to$ `NULL`), stuck-sensor detection ($\ge 6\text{h}$ identical), and peer-corroboration adjudication during extreme smog episodes ($500-1000\,\mu\text{g}/\text{m}^3$).
- **Volume:** Over 56,388 city-hourly PM2.5 ground-truth observations and 133,440 atmospheric feature rows in DuckDB (`data/gold/saaf_hawa.duckdb`), sourced from 28,652 real AirNow historical rows, 60,078 real OpenAQ community observations, and 133,344 Open-Meteo meteorological/CAMS records.

---

### 3. Model Benchmark & Evaluation Against Declared Baselines

All evaluations were conducted strictly on held-out test splits (`ts >= 2025-07-02`) with a mandatory **24-hour embargo** between train, validation, and test sets. All models and baselines were evaluated on strictly identical, like-for-like joined test rows ($N \approx 24,350$ per horizon):

| Horizon | Baseline A (Persistence) MAE | Baseline B (Seasonal-Hour) MAE | Saaf Hawa LightGBM + CQR MAE | Conformal 80% Coverage | Baselines Beaten? |
|---|---|---|---|---|---|
| **+6 Hours** | 28.48 µg/m³ | 23.62 µg/m³ | **20.52 µg/m³** | **82.8%** | **YES** |
| **+12 Hours** | 31.77 µg/m³ | 23.63 µg/m³ | **21.03 µg/m³** | **84.6%** | **YES** |
| **+24 Hours** | 23.50 µg/m³ | 23.65 µg/m³ | **21.49 µg/m³** | **84.9%** | **YES** |

> **Baseline Invariance Note:** Baseline B (Seasonal-Hour) yields an MAE of ~23.6 µg/m³ across all three horizons. Because $h \le 24\text{ h}$, the previous 7 daily historical occurrences for the target hour $(t+h)$ are identical regardless of horizon, producing an invariant empirical baseline.

---

### 4. Primary Operational Limitations & Scientific Honesty
1. **Calibrated Estimate Post-March 2025:** Following the shutdown of reference BAM monitors at U.S. diplomatic posts, ground-truth labels in the test period are reconstructed via humidity-aware Ridge calibration of community sensors (PAQI) rather than direct BAM instruments. Post-March 2025 ground truth is an operational estimate.
2. **Unmonitored Districts:** In districts without physical monitors (e.g., Multan, Faisalabad), automated health advisories are strictly suspended per Hackathon Guidelines. Regional CAMS bounds are provided with explicit unmonitored warnings.

---

### 5. Grounded Insight Agent & Responsible AI Safeguards
- **Zero Hallucinated Numbers:** Output Number Guard ensures every digit in the agent's response maps to retrieved SQL rows or official health guideline citations.
- **Traceable Row IDs:** Every forecast answer explicitly cites its database row ID (e.g., `LHE-20261020T1600Z`).
- **Emergency Health Pointers:** Medical emergency instructions directing severe cases to Punjab Emergency Service (**Rescue 1122**) and National Health Helpline (**1033**) are appended to health responses.
- **Unvalidated City Safeguards:** Unmonitored cities (Multan, Faisalabad) are explicitly labeled with widened uncertainty bounds and suspended advisories.
- **Bilingual Accessibility:** Native Urdu advisory templates for provincial education authorities.
