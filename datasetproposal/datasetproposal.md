# Saaf Hawa v2: Dataset & Problem Proposal

**Track:** Air Quality / Climate Intelligence  
**Author:** Saaf Hawa Core Team  
**Date:** October 2026  

---

## 1. Problem & Intended User
When the U.S. State Department / EPA air monitoring network went dark in March 2025 due to global federal budget realignments, Pakistani cities (Lahore, Islamabad, Karachi, Peshawar) lost their only public, continuous reference-grade particulate matter monitoring feed. District education officers (DEAs), public health administrators, and school headmasters in Punjab are currently forced to make high-stakes next-day decisions (e.g., closing schools, enforcing masks, canceling outdoor assemblies) using either stale current-hour indicators or raw coarse atmospheric model numbers with unmeasured bias.

**Saaf Hawa v2** reconstructs honest ground-truth air quality records by calibrating continuous community sensor feeds (PAQI) against the pre-halt reference record, and couples them with transboundary fire radiative power and ECMWF CAMS forecasts to deliver 24-hour calibrated quantile forecasts with honest conformal uncertainty bands.

---

## 2. Multi-Publisher Data Sources & Licences

| Role | Original Publisher | Endpoint / URL | Update Rate | Licence | Credibility & Independence |
|---|---|---|---|---|---|
| **Live Target Ground Truth** | Pakistan Air Quality Initiative (PAQI) via OpenAQ | `api.openaq.org/v3` | Hourly | CC BY 4.0 / Open Data | Independent community sensor network verified in peer-reviewed atmospheric studies (EGU24-453). |
| **Historical Calibration Reference** | U.S. Department of State / EPA AirNow | `airnow-dos-geos.s3.amazonaws.com` | One-time archive (pre-Mar 2025) | U.S. Public Domain | Reference-grade beta-attenuation mass monitors operated at U.S. diplomatic posts. *(Source NOT on official track list).* |
| **Atmospheric Feed & Weather** | Copernicus ECMWF & DWD via Open-Meteo | `air-quality-api.open-meteo.com` | Hourly | CC BY 4.0 (Non-commercial) | European Union flagship Copernicus Atmosphere Monitoring Service (CAMS) 11km chemical transport model. |
| **Transboundary Fire Signal** | NASA Earthdata / LANCE FIRMS | `firms.modaps.eosdis.nasa.gov` | Daily | NASA Public Domain | Active fire detection and Fire Radiative Power (FRP) from VIIRS and MODIS polar-orbiting satellites. |
| **Official Guidance & Standards** | WHO & Punjab EPD | `who.int`, `epd.punjab.gov.pk` | Static | CC BY-NC-SA 3.0 / Official Government | Official regulatory ambient air thresholds and health intervention guidelines. |

---

## 3. Data Volume & Contract
- **Volume:** Over 131,000 hourly ground-truth city rows (Jan 2023 – Sep 2026), 75,748 historical embassy calibration rows, and 140,000+ atmospheric weather rows. Exceeds the 50,000 hourly row threshold.
- **Contract:** Timestamps strictly UTC (`ts_utc`). Missing values remain `NULL` (never synthetic). Deterministic row IDs: `{CITYCODE}-{YYYYMMDDTHHZ}`. Full provenance tracked with SHA-256 in `manifest.csv`.

---

## 4. Declared Baselines & Evaluation
- **Baseline A (Persistence):** Observed ground truth at issue time $t$.
- **Baseline B (Seasonal-Hour):** Mean of the target hour over previous 7 days ($t-168\text{ h}$ to $t$).
- **Success Criteria:** LightGBM Quantile Forecaster strictly outperforms Baseline A and B on held-out test splits with empirical 80% conformal coverage between 75% and 85%.
