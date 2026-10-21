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

**Track:** Air Quality | **HackInsight '26** | **Zero-Cost Open-Source Stack**

Following the March 2025 shutdown of U.S. Diplomatic Post AirNow monitors in Pakistan, public health and education authorities lost access to continuous reference-grade particulate measurements. **Saaf Hawa v2** reconstructs ground truth by calibrating community sensor networks against the historical reference record and delivers uncertainty-bounded 24-hour PM2.5 forecasts (+6h, +12h, +24h) for Punjab District Education and Health Authorities.

---

## 🌐 Live Deployments

- **Streamlit Application:** [https://clean-air.streamlit.app](https://clean-air.streamlit.app)  
  *(Alternative link: [Streamlit Share](https://share.streamlit.io/kazmi4700937-hub/clean-air/main/streamlit_app.py))*
- **Hugging Face Static Space:** [https://huggingface.co/spaces/abbaskazmi07/saaf-hawa](https://huggingface.co/spaces/abbaskazmi07/saaf-hawa)

---

## 🚀 How to Reproduce

### 1. Environment Configuration
The pipeline requires two free API keys for data ingestion. Only environment variable names are required (never commit secret values):
- `OPENAQ_API_KEY`: Free API key from [OpenAQ](https://openaq.org/) for fetching community sensor feeds.
- `FIRMS_MAP_KEY`: Free MAP key from [NASA LANCE FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key/) for near-real-time satellite thermal anomaly detections.

Set them in your shell or `.env` file (which is git-ignored):
```bash
export OPENAQ_API_KEY="your_openaq_key_here"
export FIRMS_MAP_KEY="your_firms_key_here"
```

### 2. Ingestion & Feature Engineering (`make data`)
To run ingestion across all upstream publishers, compute cryptographic SHA-256 hashes for data integrity, execute Quality Gate 1 & Gate 2, and build the Gold warehouse:
```bash
make data
# Equivalent to: python flows/hourly.py
```

### 3. Continuous Scheduler
To run the automated hourly ingestion and forecasting pipeline:
```bash
python scripts/serve_scheduler.py
```
The scheduler executes once per hour at minute :00, pulls live observations, refreshes forecasts, exports the state bundle to `state/`, and records pipeline execution metadata.

### 4. Running Tests
To verify test assertions (leakage-free 24-hour embargos, strict deterministic row ID schemas, data quality gate rules, and grounded SQL evaluation):
```bash
pytest
# Or: make test
```

### 5. Exporting Flow Run History
To inspect and export genuine Prefect pipeline execution logs with optional time filtering without altering existing records:
```bash
# Export all unedited runs to prefect_runs.csv
python scripts/export_runs.py

# Select specific UTC time windows (e.g., Tue 4 PM to Wed 4 PM PKT):
python scripts/export_runs.py --since 2026-10-20T11:00:00Z --until 2026-10-21T11:00:00Z
```

---

## 📊 Data Sources and Licences

Every raw feed is registered in `data/manifest.csv` with a SHA-256 hash, fetch timestamp, query URL, and publisher attribution:

| Data Source | Publisher / Provider | Licence / Usage Terms | Usage in Pipeline |
|---|---|---|---|
| **Community Sensors** | Pakistan Air Quality Initiative (PAQI) / Clarity / AirGradient via OpenAQ | CC0 1.0 / CC BY 4.0 / Open Government Data | Live target ground truth for Islamabad, Lahore, Peshawar, Karachi |
| **Historical Embassy Reference** | U.S. Diplomatic Post AirNow monitors via OpenAQ v3 API *(not on track list)* | U.S. Public Domain / Open Government Data | Reference-grade BAM data used to calibrate community sensors |
| **Atmospheric & Weather** | Copernicus ECMWF ERA5 & CAMS via Open-Meteo | CC BY 4.0 (Non-commercial attribution) | Meteorological predictors (wind, temperature, boundary layer height, aerosol optical depth) |
| **Active Fire Detections** | NASA Earthdata / LANCE FIRMS (VIIRS & MODIS) | NASA Open Data Policy (Public Domain) | Crop residue burning & transboundary agricultural fire counts and FRP lags |
| **Policy & Health Guidelines** | WHO, NDMA Pakistan, Punjab EPA, US EPA | CC BY-NC-SA 3.0 IGO / Official Public Documents | Grounded context and advisory thresholds for the Insight Agent |

---

## 📈 Evaluation vs Declared Baselines

Evaluated on 23,547 held-out test rows under a strict 24-hour temporal embargo (test window ending 2026-09-30T23:59:59Z):

| Horizon | Baseline A: Persistence MAE | Baseline B: Seasonal-Hour MAE | LightGBM + CQR MAE | 80% Coverage | Beats Both Baselines |
|---|---|---|---|---|---|
| **+6h** | 30.35 | 25.08 | **22.10** | 81.9% | Yes |
| **+12h** | 33.85 | 25.07 | **22.79** | 82.6% | Yes |
| **+24h** | 24.88 | 25.08 | **23.15** | 83.3% | Yes |

*All MAE values in µg/m³. Declared nominal coverage target for 80% prediction intervals is 75–85%. Numbers are synchronously produced by `src/model/evaluate.py`.*

---

## ⚠️ Transparent Limitations & Disclosures

From [Summary.md](Summary.md):
- **Fire signal:** The FIRMS series in Silver covers 1,347 days (Jan 2023 – Oct 2026), backfilling multi-year transboundary fire counts and FRP lags. Exactly 42 dates across 2023–2026 are missing in upstream records (cause not established; treated as zero fires).
- **Uncalibrated labels:** Karachi and Peshawar had fewer than 30 reference-monitor overlap hours in the training window. Raw sensor values are used as published (identity) and clearly marked uncalibrated.
- **Label source shift:** Test period labels come from different sensors than training (e.g. Islamabad training on `s_1343270` and testing on F-7 AirGradient; Lahore adding Barki AirGradient in May 2025). Test error reflects this real-world sensor transition.
- **Weakest overall group:** Smog season (Oct–Feb) at +24h (MAE 37.87 µg/m³, 80% coverage 79.4%).
- **Subgroups trailing baselines:** Night (00–06 PKT) at +6h trails persistence (25.49 vs 25.20 µg/m³); Afternoon (12–18 PKT) at +12h trails seasonal (25.18 vs 24.90 µg/m³).
- **Coverage drops during acute peaks:** Lahore achieves 74.7–76.0% coverage across horizons (target 80%); Very Unhealthy tier (>150 µg/m³) coverage drops to 72.2% at +6h and 67.3% at +12h (MAE 53.83 µg/m³). Interval uncertainty is under-covered in severe smog peaks.
- **Peshawar out-of-distribution transfer:** Peshawar has 0 training hours and accounts for 45.5% of test rows (10,710 of 23,547). Scored as pure zero-shot spatial transfer.
- **Partial circularity in calibration check:** Islamabad training series `s_1343270` matches the embassy monitor in ~75% of overlap hours; its calibration R² should not be treated as independent evidence.
- **Run history:** Pipeline execution records start at 02:35 AM PKT due to late scheduler activation.
