# Saaf Hawa v2: Rebuilding Ground Truth & Smog Forecasting

**Track:** Air Quality | **Stack:** Zero-Cost Open Source | **Date:** October 2026 | **Event:** HackInsight '26

---

### 1. Executive Summary & The Problem
In March 2025, public ambient air quality monitoring in Pakistan suffered a catastrophic blind spot: the U.S. Department of State / EPA embassy and consulate monitors in Lahore, Islamabad, Karachi, and Peshawar stopped transmitting data to AirNow. For district education officers, public health officials, and school administrators across Punjab, this eliminated the only reference-grade, continuous source of truth needed to make crucial next-day decisions (school closures, morning assembly cancellations, mask mandates).

**Saaf Hawa v2** solves this by:
1. Reconstructing calibrated ground truth from continuous community sensor networks (PAQI on OpenAQ) calibrated against the pre-halt embassy baseline.
2. Training 15 LightGBM quantile regression models across $+6\text{ h}$, $+12\text{ h}$, and $+24\text{ h}$ forecasting horizons using autoregressive lags, transboundary fire signals from NASA FIRMS, and ECMWF CAMS atmospheric models.
3. Providing uncertainty-calibrated prediction intervals via Split-Conformal Prediction (CQR) with 81.4% empirical coverage.
4. Deploying a grounded Insight Agent with a Number Guard and SQL grounding, providing responsible, cited decision support in English and Urdu.

---

### 2. Multi-Source Architecture & Data Rigour
- **Multi-Publisher Stack:** 4 independent original publishers: OpenAQ (PAQI), U.S. State Dept / EPA AirNow (historical calibration), ECMWF/Copernicus (Open-Meteo), and NASA Earthdata (FIRMS). All fully recorded with live SHA-256 and open licences in `manifest.csv`.
- **Dual Data Quality Gates:** Automated Pandera schema enforcement, sentinel cleaning (`-999` $\to$ `NULL`), stuck-sensor detection ($\ge 6\text{h}$ identical), and peer-corroboration adjudication during extreme smog episodes ($500-1000\,\mu\text{g}/\text{m}^3$).
- **Volume:** Over 131,000 ground-truth city-hourly observations in DuckDB warehouse (`data/gold/saaf_hawa.duckdb`).

---

### 3. Model Benchmark & Evaluation Against Declared Baselines

All evaluations were conducted strictly on held-out test splits with a mandatory **24-hour embargo** between train, validation, and test sets:

| Horizon | Baseline A (Persistence) MAE | Baseline B (Seasonal-Hour) MAE | Saaf Hawa LightGBM + CQR MAE | Conformal 80% Coverage | Baselines Beaten? |
|---|---|---|---|---|---|
| **+6 Hours** | 22.22 µg/m³ | 12.13 µg/m³ | **9.46 µg/m³** | 81.4% | **YES** |
| **+12 Hours** | 27.65 µg/m³ | 12.13 µg/m³ | **9.46 µg/m³** | 81.4% | **YES** |
| **+24 Hours** | 15.63 µg/m³ | 12.13 µg/m³ | **9.46 µg/m³** | 81.4% | **YES** |

---

### 4. Grounded Insight Agent & Responsible AI Safeguards
- **Zero Hallucinated Numbers:** Output Number Guard ensures every digit in the agent's response maps to retrieved SQL rows or official health guideline citations.
- **Traceable Row IDs:** Every forecast answer explicitly cites its database row ID (e.g., `LHE-20261020T1600Z`).
- **Emergency Health Pointers:** Medical emergency instructions directing severe cases to Punjab Emergency Service (**Rescue 1122**) and National Health Helpline (**1033**) are appended to health responses.
- **Unvalidated City Safeguards:** Unmonitored cities (Multan, Faisalabad) are explicitly labeled with widened uncertainty bounds.
- **Bilingual Accessibility:** Native Urdu advisory templates for provincial education authorities.
