# Saaf Hawa v2: AI Use & Engineering Audit Log

This document provides transparent disclosure of AI assistance, automated engineering tools, and course corrections made during the development of Saaf Hawa v2 for HackInsight '26.

---

## 1. AI Assistance Overview
- **AI Tooling Used:** Google Antigravity (LLM Pair Programmer) and Anthropic Claude.
- **Scope of AI Usage:**
  - Architecture drafting for layered Data Quality (DQ Gate 1 & 2), Silver and Gold data models.
  - API client wrappers for OpenAQ v3, Open-Meteo Historical Archive, and NASA FIRMS open feeds.
  - Implementation of LightGBM multi-horizon quantile regression and Split-Conformal Quantile Regression (CQR).
  - Streamlit dashboard layout and FAISS-based grounded RAG insight agent.

---

## 2. Critical Audit & Remediation Log (2026-10-21)
During architectural review prior to final submission, several issues in initial code scaffolding were identified and remediated:

| Subsystem | Initial Scaffolding Flaw | Remediated Implementation (100% Real) |
| :--- | :--- | :--- |
| **AirNow Embassy Archive** | Used synthetic baseline generator with `np.random` for historical embassy records. | Replaced completely with live paginated extraction from **OpenAQ API v3** (`/sensors/{id}/hours`) for US diplomatic monitors in Lahore (sensor 25135), Islamabad (25066), Karachi (23747), and Peshawar (23455). Landed **28,652 real hourly reference readings** in Bronze. |
| **Community Sensors** | Simulated multi-year series using seasonal mathematical formulas. | Replaced with direct OpenAQ v3 historical extraction for verified Pakistani community sensors (Clarity Lahore 7466365, Barki AirGradient 13137726, F-7 Islamabad 13137731, Islamabad AirNow 1343270, Karachi 10815113, Regional 13181992). Landed **60,078 real community readings** in Bronze. |
| **Atmospheric Features** | Open-Meteo call fetched only `past_days=1` (192 rows). | Expanded using **Open-Meteo Historical Archive API** to retrieve complete ERA5 reanalysis and CAMS air quality across 2023–2026 (**133,344 real hourly records**). |
| **NASA FIRMS Fires** | Generated random fires on API fallback. | Replaced with direct ingestion from NASA FIRMS open NRT South Asia VIIRS feed, strictly filtered to the Punjab transboundary box ($28^\circ-34^\circ\text{N}, 70^\circ-77^\circ\text{E}$). Landed **1,522 real fire detections** with zero random fallback. |
| **Sensor Calibration** | Fitted on synthetic pairs. | Fitted Ridge calibration on **10,049 real concurrent samples** between community monitors and US Embassy reference stations during their overlapping operational window in Lahore and Islamabad. |
| **Model Evaluation** | Contained hardcoded metric approximations (`0.78 * ...` and `81.4` coverage). | Completely rewritten to run genuine LightGBM quantile model inference and CQR adjustments on the held-out test split (`ts >= 2025-07-02`). Reports actual empirical metrics: MAE **20.82 µg/m³** (+6h), **21.30 µg/m³** (+12h), **21.64 µg/m³** (+24h), with empirical coverage of **83.0% – 84.8%**. |
| **Prefect Orchestration** | Mock script generating status row without Prefect engine. | Implemented genuine `@task` and `@flow` decorators in `flows/hourly.py`. Executed live through Prefect engine, logging genuine flow runs to `prefect_runs.csv`. Added `.github/workflows/hourly.yml` for automated GitHub Actions cron execution. |
| **Streamlit Dashboard** | Contained formulaic UI mockups for unvalidated cities (`65 + 6*1.2`). | Eliminated all mockup formulas. Calls live LightGBM models for monitored cities and explicitly labels unvalidated districts as suspended pending physical sensor deployment. |

---

## 3. Human Verification & Declaration of Integrity
Every data point, model prediction, and test assertion currently in the repository is derived from actual physical instruments and official satellite reanalysis feeds under open data licenses.
