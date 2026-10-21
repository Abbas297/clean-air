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
| **Model Evaluation** | Contained hardcoded metric approximations (`0.78 * ...` and `81.4` coverage). | Completely rewritten to run genuine LightGBM quantile model inference and CQR adjustments on the held-out test split (`ts >= 2025-07-02`) with strict like-for-like row joining against baselines. Reports actual empirical metrics: MAE **20.52 µg/m³** (+6h), **21.03 µg/m³** (+12h), **21.49 µg/m³** (+24h), with empirical coverage of **82.8% – 84.9%**. Full granular errors-by-group table generated across city, season, diurnal period, and AQI tier. |
| **Prefect Orchestration** | Mock script generating status row without Prefect engine. | Implemented genuine `@task` and `@flow` decorators in `flows/hourly.py`. Executed live through Prefect engine, logging genuine flow runs to `prefect_runs.csv` without duplicates. Added `.github/workflows/hourly.yml` for automated GitHub Actions cron execution. Dynamically derives `run_status` (HEALTHY/DEGRADED/FAILED) from quality gates and feed freshness. |
| **Streamlit Dashboard** | Contained formulaic UI mockups for unvalidated cities (`65 + 6*1.2`) and showed "not found" on Streamlit Cloud due to gitignored DuckDB parquets. | Eliminated all mockup formulas. Implemented automated export of lightweight state bundle (`state/`) tracking forecasts, evaluation tables, quality logs, and 48-hour history. Wired dashboard directly to `state/` ensuring instant rendering on Streamlit Cloud without missing file errors. |

---

## 3. Verified Data Sources & Methodological Boundary
We specifically declare and verify the provenance and nature of every data source:
1. **Historical Reference BAM Monitors (Pre-March 2025):** Direct reference beta-attenuation mass monitor records from U.S. State Department diplomatic post stations in Lahore, Islamabad, Karachi, and Peshawar fetched via OpenAQ v3.
2. **Community Sensor Network:** Direct optical particle counter measurements from Pakistani community stations (PAQI, Clarity, AirGradient) fetched via OpenAQ v3.
3. **Post-March 2025 Ground Truth:** Operationally estimated via humidity-aware Ridge calibration against the pre-halt embassy baseline; explicitly acknowledged as an estimate rather than primary reference measurement.
4. **Atmospheric & Transboundary Feeds:** Physical meteorological reanalysis (ECMWF ERA5) and Copernicus atmospheric chemistry models (CAMS) via Open-Meteo, and physical satellite thermal anomaly detections from NASA VIIRS via FIRMS.
5. **Evaluations:** 100% computed from actual trained LightGBM quantile models and Split-Conformal prediction on held-out test data with zero hardcoded or scaled constants.

---

## 4. Independent review (2026-10-21, ~03:30-04:30 PKT)
| Tool | Task | Kept / changed | How checked |
|---|---|---|---|
| Claude Sonnet 5.5 (Anthropic) | Read-only audit of the repository against the HackInsight rules; patches to a copy | Kept: pipeline structure, models, DQ rules. Changed: FIRMS de-duplication in `silver.py`; identity (not invented) calibration fallback plus holdout-vs-reference report in `calibrate.py`; repo-relative manifest paths in `manifest.py`; PKT diurnal groups in `evaluate.py`; data-driven `build_pdfs.py`; new `scripts/backfill_firms.py`; API key removed from `config.yaml`; `DECLARATION_ADDENDUM.md` | pytest 15/15 after patches; Silver/calibration rebuilt offline from the team's own Bronze files; fire counts compared before/after (7,610 rows -> 1,522 unique); PDF page count checked. Live API calls (OpenAQ, FIRMS) were NOT run in this review. |
| Google Antigravity | Final evaluation, fire-gap remediation, and state synchronization | Fixed test split upper bound (`<= 2026-09-30`) to freeze test set at 23,547 rows. Executed NASA FIRMS NRT backfill for 2026-05-01 to 2026-07-31 (+1,518 detections). Wired synchronous `export_state_bundle()` in `evaluate.py` to prevent stale evaluation exports. Documented remaining 42 missing fire dates as zero-fire gaps and clarified subgroup coverage limitations. | Ran full pipeline suite; verified `state/eval_by_group.csv` matches test set; regenerated `Summary.pdf` (1 page); all 15 pytest unit tests passing. |

---

## 5. Deployment Hardening & Documentation Tasks (2026-10-21, ~10:00 PKT)
| Task ID | Component | Description & AI Tooling | Verification |
|---|---|---|---|
| Task 1 | CI/CD (`.github/workflows/hourly.yml`) | Removed automated CI hourly workflow (`git rm`) to ensure remote runner jobs cannot conflict with or overwrite local pipeline state files. | Verified workflow deleted and removed from git index. |
| Task 2 | Scheduler (`scripts/serve_scheduler.py`) | Added `git pull --rebase --autostash origin main` prior to push and surfaced stderr diagnostics on push failure. | Inspected subprocess handling; syntax and git flags verified. |
| Task 3 | Security & Key Hygiene | Verified `OPENAQ_API_KEY` and `FIRMS_MAP_KEY` are read exclusively from environment variables; audited entire git commit log and codebase with regex scanners to confirm zero committed API keys. | Confirmed clean grep scan across repository and history. |
| Task 4 | Dashboard Evidence (`streamlit_app.py`, `app/app.py`) | Replaced hardcoded weak-spot figures with dynamically loaded values from `state/eval_by_group.csv` (displaying "n/a" if missing). | Tested dynamic loading; confirmed identical behavior on Streamlit and Gradio. |
| Task 5 | Run History Export (`scripts/export_runs.py`) | Added optional `--since` and `--until` CLI filtering (ISO UTC) to SELECT runs by `start_time` without altering records or default behavior. | Verified CLI argument parser (`--help`) and backward compatibility. |
| Task 6 | Summary PDF & Markdown (`scripts/build_pdfs.py`, `Summary.md`) | Replaced static run history sentence with dynamic calculation from `prefect_runs.csv` (total runs, non-Completed count, first/last PKT start, and longest gap in hours between named runs). | Generated PDF; verified `Summary.pdf` remains strictly 1 page. |
| Task 7 | Documentation (`README.md`) | Added live deployment URLs (Streamlit and Hugging Face Static Space), reproduction instructions (environment variable names only, `make data`, scheduler, pytest), full data manifest table, and transparent limitations. | Verified formatting, table schema, and links. |
| Static Space | HF Space (`hf_space/index.html`) | Single-page static status dashboard drafted with Claude and tested against repo state files; reads live from GitHub raw state without backend dependencies. | Created `hf_space/index.html` and verified parsing. |
