# Saaf Hawa v2: MVP Spec for HackInsight '26

Track: Air quality. Zero-cost stack. Written Tue 20 Oct 2026, 8:30 PM PKT, with about 27 hours left before the event closes (Wed 11:59 PM; form closes Thu 12:00 AM).

---

## 0. Verdict on the previous MVP

**Keep about 60% of it, replace the ground-truth layer, and cut scope.** The skeleton is good: Bronze/Silver/Gold, two quality gates, LightGBM quantile models, FAISS RAG with numbers from SQL, Prefect, Gradio. It maps cleanly to the scoring rubric. It would not win as written, for these reasons:

| # | Problem | Evidence / reasoning | Fix in v2 |
|---|---|---|---|
| 1 | **The ground truth is dead.** The plan's target is the U.S. embassy monitors on AirNow. The State Department stopped transmitting embassy data to AirNow in March 2025 (AP/CNN, CREA, PNAS 2026). An aqicn mirror of the Islamabad embassy shows "Not Available" as of Feb 2026. | A live hourly pipeline cannot fetch from a feed that no longer exists, and the 2025-26 smog season has no embassy data to test on. | Use embassy data only as a **historical calibration reference (pre-halt)**. Use community sensors (Pakistan Air Quality Initiative on OpenAQ) as live ground truth. See section 3. |
| 2 | "Raw CAMS forecast at +24h" compared with our model uses archived Open-Meteo values, which are best estimates, not the forecast actually issued 24 h earlier. | Comparing against it flatters the raw model, so beating it is a conservative test. Presenting it as a true forecast would be a misrepresentation. | Label it **"CAMS hindcast (optimistic)"** everywhere. Feed only CAMS at time t as a feature. |
| 3 | The "primary baseline" (same hour yesterday) equals persistence at t for the +24h horizon, and is weak for +6h. | Two names for one baseline at one horizon. | Declare two baselines (section 8). |
| 4 | Business model: paid subscription on top of Open-Meteo's free tier, which is non-commercial. | Free use is CC BY 4.0, non-commercial; commercial plans are separate. | Position as a public-interest service (public budget/NGO grant). State the commercial path (Copernicus ADS direct, or Open-Meteo commercial plan). |
| 5 | Scope was written for 72 h with a heartbeat by 4 PM. Now it is 8:30 PM with about 27 h left. | Cutting late is how teams ship nothing. | Tiered scope with kill criteria (sections 4 and 13). |
| 6 | Volume claim (4 stations x 2 years = 70k rows) assumed live embassy data. | Only valid pre-Mar 2025. | Count rows from the actual sources after Gate 0 (section 3). |

The new angle is stronger and more Pakistani than a plain AQI forecast:

> **When the U.S. embassy monitors went dark in March 2025, Pakistani cities lost their only reference-grade, public air quality data. We rebuild trustworthy live ground truth from community sensors, calibrated against the pre-halt embassy record, and use it to produce a 24-hour smog forecast with honest uncertainty for Punjab school and health officers.**

---

## 1. Pitch and the three questions we answer

For a district education or health officer in Punjab, who must decide the evening before whether to issue a smog advisory or move school activity indoors, Saaf Hawa forecasts hourly PM2.5 for the next 24 hours with a calibrated uncertainty band, unlike today's single current-AQI number that is already stale when acted on.

**Research questions (these are what the jury and a journal reviewer will test):**

1. **RQ1: Ground truth.** Can calibrated community-sensor PM2.5 stand in for the lost reference monitors? Measured against pre-halt embassy data: bias, R², error by humidity band.
2. **RQ2: Forecast skill.** Does a model using calibrated ground-truth lags, CAMS at time t, weather, and upwind fires beat persistence and a seasonal-hour baseline at +6, +12, +24 h on held-out smog weeks?
3. **RQ3: Decision value.** Does the forecast trigger a "limit outdoor activity" advisory more accurately than the baseline rule, and with known confidence?

We report the answer whichever way it falls. Honest negative results score under "honesty" and "rigour".

---

## 2. Cost: everything is free

| Need | Tool | Cost note |
|---|---|---|
| Ground truth + embassy history | OpenAQ API v3 (free key), AirNow historical archive | free key; rate limits apply (verify current limits) |
| Model feed + weather | Open-Meteo Air Quality + Archive APIs (CAMS, ERA5) | free for non-commercial, CC BY 4.0, attribution required |
| Fires | NASA FIRMS Area API | free MAP_KEY (email only) |
| Warehouse / model | Python, DuckDB, Parquet, pandera, LightGBM | free |
| Orchestration | Prefect (Cloud free workspace for run history, verify plan limits) + GitHub Actions cron | free on a public repo |
| RAG | `intfloat/multilingual-e5-small` (CPU), FAISS | free |
| Text generation | Templates (default). Optional free-tier LLM only for paraphrase, behind a number guard | no paid API anywhere |
| Dashboard | Gradio on a Hugging Face Space (free CPU) | free; Space may sleep, see risks |
| Repo | GitHub public | free |

---

## 3. Data: sources, licences, and Gate 0

### Verified this session
- AirNow embassy data transmission stopped in 2025 for budget reasons; 44 countries affected (CREA, CNN/AP, PNAS).
- Open-Meteo free tier: non-commercial, CC BY 4.0, attribution required. CAMS grid is about 11 km. Archived air-quality values are revised with each model run, so treat them as best estimates. Start of air-quality history is reported inconsistently (2022-01-01 vs 2022-06-01): **check it.**
- NASA FIRMS Area API needs a MAP_KEY; archive sources include `VIIRS_SNPP_SP`, `VIIRS_NOAA20_SP`, `MODIS_SP`; `/api/data_availability/` reports coverage.
- OpenAQ free key exists; Pakistan Air Quality Initiative (PAQI) presented seven years of low-cost sensor data for Karachi, Lahore, Islamabad and Peshawar, with comparison to reference data (EGU24-453, OpenAQ co-author).
- Punjab EPD publishes daily "Monitoring of Air Quality of Lahore" PDFs with PM2.5 per location and the Punjab standard row (PM2.5 24-h = 35 µg/m³ in the 2018 samples).

### NOT verified (Gate 0 must settle these in the first hour)
- How many Pakistani PM2.5 sensors on OpenAQ are active in 2025-26 with at least about 2 years of hourly history, and under what licence (check each location's `licenses` field).
- Whether AirNow's pre-halt embassy archive for Lahore, Islamabad, Karachi and Peshawar is still downloadable, and in what format. Fallback: embassy locations may also appear in OpenAQ.
- Sensor type and any humidity bias on the community sensors (metadata).
- Whether recent Punjab EPD daily PDFs still exist and their terms of use.

### Source stack

| Role | Source (original publisher) | Independent? | On track list? |
|---|---|---|---|
| Live ground truth (target) | Community low-cost sensors via OpenAQ (PAQI and others) | yes | OpenAQ is on the list |
| Calibration reference (history only) | U.S. Dept. of State / EPA AirNow embassy monitors | yes | **not on list** (satisfies "1 source not on list") |
| Model feed + weather | ECMWF/Copernicus CAMS and ERA5 via Open-Meteo | yes | Open-Meteo is on the list |
| Fire signal | NASA FIRMS (VIIRS/MODIS) | yes | not on list |
| Local validation + RAG library | Punjab EPD ambient air PDFs, WHO 2021 AQG, U.S. EPA AQI guidance, official Punjab/PMD/NDMA advisories | n/a (documents) | n/a |

Kaggle uploads are **discovery only** (the rules ban re-uploads as a sole source). Use them to sanity-check our own pulls, never as a citation.

### Gate 0 decision tree (decide by **9:30 PM**)
1. OpenAQ returns at least 3 Pakistani PM2.5 sensors in at least 2 cities with at least 1.5 years of mostly continuous history, **and** the licence permits use → plan stands.
2. If embassy history is unreachable, ground truth stays "community sensor, uncalibrated"; RQ1 is reduced to cross-sensor consistency plus Punjab EPD spot checks, and labelled as such.
3. If OpenAQ Pakistan history is too thin → **Plan B**: Climate and heat-health track (Open-Meteo/ERA5 + NASA POWER + Meteostat station temperature, cases defined from PMD heat advisories). Switching costs about 2 hours; do not wait past 9:30 PM.

### Volume
Count from real pulls after Gate 0. Target: more than 50,000 hourly rows per main source (for example 4 or more sensors x 1.5 years ≈ 52,000+, CAMS for the same cities over 4 years ≈ 140,000, FIRMS detections in the bounding box over 4 years ≈ tens of thousands). If the count falls short, add sensors or cities, never synthetic data.

---

## 4. Scope tiers

**MUST (cutting any of these loses points or risks disqualification)**
- Hourly Prefect flow running by **10:00 PM tonight** (heartbeat) and staying up until Wed 4 PM.
- `make data` from an empty folder, `manifest.csv` with SHA-256 written by the fetch code.
- Declaration file committed **before any modelling**.
- Two DQ gates, `quality_log`, quarantine, at least one real flag-instead-of-fill case.
- Baselines + LightGBM quantile model + time split + errors by group.
- RAG with four passing tests; every number tied to row IDs.
- Space with freshness, quality status, last successful run.
- All submission files.

**SHOULD**
- Conformal calibration of intervals (CQR) with coverage by group.
- Decision metric for the advisory rule.
- Calibration of community sensors against embassy history (RQ1).
- Urdu advisory templates.
- SHAP-style "why is it high" via LightGBM contributions.

**STRETCH (cut first, in this order)**
Drift-triggered retraining, human-in-the-loop approval button, wind-cone fire feature, Data Enrichment Agent, 2-page data paper, extra unvalidated cities.

---

## 5. Architecture

```
OpenAQ (community PM2.5)   Open-Meteo (CAMS + ERA5)   NASA FIRMS   [AirNow embassy history: one-time]
        \                        |                       /                    |
         +-------- Prefect flow, cron hourly (run by GitHub Actions) ---------+
                                 |
        Bronze: raw Parquet + manifest.csv (source, query, fetch_time, sha256, licence)
                                 |
        DQ Gate 1 (pandera + evidence tools): schema, units, dupes, stuck, range, UTC check
             fail -> quarantine + log + alert; Gold keeps last good version
                                 |
        Silver: typed, UTC, de-duplicated, sensor-level
                                 |
        Calibration (sensor -> reference scale, fitted on TRAIN period only)
                                 |
        DQ Gate 2: city-hour aggregation, completeness, freshness, drift (PSI)
                                 |
        Gold (DuckDB): pm25_city_hourly, features_hourly, forecasts, eval_by_group, advisories
            |                 |                     |
      LightGBM quantiles   RAG Insight Agent     status.json + quality_log.parquet
      + conformal          (SQL numbers +                 |
      vs. baselines        cited documents)        Hugging Face Space (Gradio)
```

---

## 6. Data contract

- All timestamps stored in **UTC** (`ts_utc`); PKT only for display.
- Missing stays missing (NULL). Never fill the target.
- **City ground truth** = median of QC-passed calibrated sensors in the city with at least 2 sensors; if only one, mark `n_sensors = 1` and widen the interval downstream.
- Stable row ID: `{CITYCODE}-{YYYYMMDDTHHZ}` (for example `LHE-20251104T0300Z`). Forecast rows add `-H{horizon}`.

| Gold table | Key columns |
|---|---|
| `pm25_city_hourly` | row_id, city, ts_utc, pm25_cal, n_sensors, qc_flag, is_filled |
| `features_hourly` | row_id, cams_pm25, cams_pm10, aod, rh, temp, wind_u, wind_v, precip, fire_count_lag1..3, fire_frp_lag1..3 (all known at issue time t) |
| `forecasts` | row_id, city, issued_at_utc, horizon, p10, p25, p50, p75, p90, model_version, interval_method |
| `eval_by_group` | model, horizon, group_type, group, mae, pinball, coverage_80, n |
| `advisories` | row_id, city, window_utc, p_exceed, advisory_flag, status (`DRAFT`/`APPROVED`) |
| `quality_log` | ts, run_id, gate, check, rule, action (fix/fill/flag), reason, evidence_json, table, rows_affected |
| `quarantine` | original row + failed rule + run_id |
| `manifest.csv` | file, source, publisher, url_or_doi, query, fetch_time_utc, sha256, licence |

---

## 7. Data Quality Agent (the part that must really "decide")

Not a script of if-statements: **tools → evidence → policy → logged decision.**

**Tools the agent calls:** pandera validator, stuck-value detector, gap profiler, cross-sensor comparer, CAMS-agreement check, humidity check, fire-proximity check, UTC-offset detector.

**Evidence-based adjudication (the novel bit):** an extreme reading is not auto-dropped. The agent asks neighbours.

| Situation | Evidence gathered | Action | Notes |
|---|---|---|---|
| Sentinel (for example -999) or wrong unit | none needed | **FIX** → NULL | certain and reversible |
| Value above 500 µg/m³ | other sensors in city, CAMS ratio, fires | **keep + tag `extreme_confirmed`** if at least 2 other sensors exceed 300 in the same hour; otherwise **FLAG + quarantine** | lets real smog peaks survive |
| Value above 1000 | none | **FLAG + quarantine** | physically implausible for these sensors |
| Identical value for 6 or more consecutive hours | n/a | **FLAG** sensor-hours | stuck sensor |
| Gap of 1 to 3 h in a feature series | neighbours | **FILL** (linear), `is_filled = true` | features only |
| Gap longer than 3 h, or any target gap | n/a | **FLAG**, keep NULL | never guess |
| Humidity above 90% and sensor/CAMS ratio above 3 | RH | **FLAG** `possible_hygroscopic_bias` | feeds RQ1 error-by-humidity |
| Hour offset suspected | cross-correlation of sensor diurnal curve with CAMS | **FLAG**, halt merge | catches silent UTC/PKT shifts |
| Source stale (OpenAQ over 3 h, CAMS over 6 h, FIRMS over 12 h) | last timestamp | **FLAG** + show on dashboard | |
| Feature distribution shift (PSI above 0.25 vs train) | PSI | **FLAG** drift | stretch: retrain trigger |

Every decision writes one `quality_log` row with rule, evidence, action, reason, and row count. The agent's explanation text is template-generated from the log row (an LLM is optional and gets no say in the decision).

**Supervisor behaviour (bonus):** a small controller in the Prefect flow reads the gate results and chooses `continue`, `halt-and-keep-last-good`, or `continue-with-warning`. Log that choice too.

---

## 8. Model and evaluation protocol (commit as `DECLARATION.md` before modelling)

**Target.** City PM2.5 at +6 h, +12 h, +24 h, per monitored city.

**Features (known at issue time t only).** Lagged ground PM2.5 (1, 3, 6, 12, 24, 48 h; rolling means), CAMS PM2.5/PM10/AOD at t, weather at t, fire counts and FRP lagged 1 to 3 days from a bounding box covering Punjab on both sides of the border (config file), hour of day, day of week, month, city.

**Models.** LightGBM quantile regression for p10, p25, p50, p75, p90 per horizon (15 models; city as a categorical). Then **split-conformal (CQR)** adjustment fitted on the validation block so the 80% band has measured coverage.

**Baselines (declared now, never changed).**
- **A. Persistence:** value at t.
- **B. Seasonal-hour:** mean of the same target hour over the previous 7 days (known at t).
- **Reference (not a baseline):** CAMS hindcast, labelled "optimistic, not a true forecast".

**Splits (embargo 24 h between blocks).**
- Train: 2022-07-01 to 2024-06-30
- Validation: 2024-07-01 to 2025-06-30 (contains the 2024-25 smog season)
- Test: 2025-07-01 to 2026-09-30 (contains the 2025-26 smog season). Touch once.
- Calibration of community sensors against embassy data is fitted on the **train** window only, and checked on the embassy overlap in validation.
- If CAMS history starts later than July 2022, move the train start only.
- Scalers and imputers fit on train only; seeds fixed.
- **Forward check:** the live hourly pipeline accumulates new out-of-sample hours. Report them as a sanity check, not as scored results.

**Metrics.** MAE and RMSE (p50), pinball loss, empirical 80% coverage, mean band width. Decision metric: advisory fires when P(PM2.5 ≥ 55.5 µg/m³) ≥ τ (τ chosen on validation to reach recall of at least 0.8); compare recall and precision with baseline rule "advise if current PM2.5 ≥ 55.5". Also show Punjab's 35 µg/m³ 24-h standard for context. Threshold values to be double-checked against the WHO and EPA documents we cite.

**Declared success criteria.**
1. p50 MAE below both baselines at all three horizons on the pooled test set.
2. Report every city where it is not; do not hide it.
3. Pooled 80% coverage within 75 to 85%.
4. At +24 h, advisory recall at least the baseline's, at precision no worse.

**Group errors.** City, hour of day, AQI tier, season (smog Oct to Jan vs other), humidity band, and `n_sensors` (1 vs 2 or more). Name the weakest group and write what an officer should do about it.

**Unvalidated cities (for example Faisalabad, Multan).** Show CAMS-only estimate, widen the band using raw-CAMS residual quantiles from monitored cities, label "unvalidated", and make the RAG say so.

---

## 9. Insight Agent (RAG)

- **Library (about 8 to 12 docs, saved manually from official sites with URL, retrieval date and SHA-256 in the manifest):** WHO 2021 Air Quality Guidelines, Punjab EPD standards/ambient reports, U.S. EPA AQI health guidance, official Punjab/PMD/NDMA smog advisories. Respect each site's terms and robots.txt; no scraping of aggregators.
- **Index:** sections chunked with ID `docid#section`; `multilingual-e5-small` embeddings in FAISS plus keyword (BM25) hybrid.
- **Intents (small closed set):** forecast lookup, advisory recommendation, "why is it high", data status. Each intent maps to a **SQL template** on Gold tables. The language layer never produces digits: numbers are inserted by code from query results, and **row IDs are stored with every answer**.
- **Number guard:** after generation, every number in the answer must exist in the retrieved rows or the answer is rejected.
- **"I don't know" rules:** no intent match, retrieval score below a threshold tuned on 20 answerable and 20 unanswerable hand-written questions, no Gold row for the requested city/time, or the city is unvalidated and the question needs validation. It says what is missing.
- **Language:** English plus about 30 hand-written Urdu template strings (reviewed by a native reader); the agent never machine-translates health advice.
- **Tests that must pass before the viva:** (1) answerable question cites the right section, (2) unanswerable question returns "I don't know", (3) every number matches stored row IDs, (4) a misleading question ("confirm tomorrow will be under 20") does not invent a fact. Plus an emergency pointer to official services in every health answer.
- Outputs are information, not professional advice. Say so in the UI.

---

## 10. Pipeline and free hosting

**Hourly flow:** fetch (incremental) → land Bronze + manifest → Gate 1 → Silver → calibrate → Gate 2 → Gold → forecast → advisories → write `status.json` → push to a `state` branch. A failed gate stops the run and the Space keeps showing the last good version.

**Scheduler choice (pick one tonight):**
- **Primary: GitHub Actions cron** (public repo, free) runs the flow; **Prefect Cloud free workspace** records run history for `prefect_runs.csv`. Use an off-peak minute such as `7 * * * *`. Actions schedules can drift by many minutes and are not guaranteed, so do not promise exact hourly in the demo; show the real run history.
- **Alternative: a teammate's always-on laptop** running `flow.serve(cron=...)` against a local Prefect server (sleep disabled, plugged in, stable network). Do not run both at once.
- State lives on the `state` branch (Parquet and JSON, a few MB). Secrets (OpenAQ key, FIRMS MAP_KEY, Prefect key) go in repo secrets, never in code.
- `scripts/export_runs.py` pulls the run history through the Prefect API and writes `prefect_runs.csv` untouched. Export after **Wed 4:00 PM**.

---

## 11. Dashboard (Hugging Face Space, Gradio)

Tabs: **Status** (freshness per source with late badge, latest DQ summary of fix/fill/flag counts, last successful run, run duration), **Forecast** (city picker, 24 h with p10 to p90 band, baselines and CAMS hindcast overlaid, advisory badge DRAFT/APPROVED), **Evidence** (baseline vs model chart, per-group table, coverage), **Quality log** (filterable), **Ask** (RAG, English/Urdu). Read live from raw GitHub URLs on the `state` branch with short caching. Never hard-code a number. Free Spaces can sleep: open it before demo recording, and keep the status data in the repo so a cold start still works.

---

## 12. Repo layout and commands

```
saaf-hawa/
  DECLARATION.md          # baselines, splits, metrics (committed before modelling)
  datasetproposal/        # source for datasetproposal.pdf
  config/                 # cities, bbox, thresholds, source URLs
  src/ingest/             # openaq.py, openmeteo.py, firms.py, airnow_archive.py, manifest.py
  src/dq/                 # tools.py, policy.py, agent.py, log.py
  src/warehouse/          # silver.py, calibrate.py, gold.py (DuckDB)
  src/model/              # baselines.py, lgbm.py, conformal.py, evaluate.py
  src/rag/                # library/, index.py, intents.py, sql_templates.py, guard.py
  flows/hourly.py
  app/                    # Gradio Space
  tests/                  # pandera, row-id, RAG four tests, no-leakage test
  docs/                   # model_card.md, AI_Use_Log.md, optimization_log.md
  Makefile                # data, quality, model, eval, rag, app, test
```

`make data` = start from empty folder → fetch or reload raw → verify SHA-256 → Silver → calibrate → Gold → run gates → print summary → exit 0 with no manual step. Add a **no-leakage test** (assert max train timestamp is below min test timestamp minus embargo) and a **row-ID test** (every number in a sample answer maps to a Gold row).

---

## 13. Timeline (owners are suggestions)

Roles: **A** data engineering, **B** DQ agent + orchestration, **C** model + evaluation, **D** RAG + dashboard + documents.

| When (PKT) | Do | Hard gate |
|---|---|---|
| Tue 8:30 to 9:30 PM | **Gate 0.** A: OpenAQ key, list Pakistani PM2.5 sensors with first/last dates and licences, check AirNow archive. B: repo, Prefect Cloud, Actions skeleton. C: write and commit `DECLARATION.md`. D: FIRMS MAP_KEY, download library PDFs, start datasetproposal. | **Go / no-go at 9:30** |
| Tue 9:30 to 10:00 PM | **Heartbeat:** minimal hourly flow fetching OpenAQ + Open-Meteo into Bronze with manifest rows. | Running by **10 PM** |
| Tue 10 PM to Wed 2 AM | Backfill, Silver, Gold v0, `make data` v0. B builds Gate 1. C computes baselines on validation. D indexes docs. | `make data` works once from scratch |
| Wed 2 to 8 AM | **Sleep in two shifts** (two people at a time). The pipeline runs unattended and is the evidence. | Check overnight runs at 8 AM |
| Wed 8 to 11 AM | LightGBM v0, conformal, group errors on validation; Gate 2; calibration vs embassy. | Features frozen at 11 AM |
| Wed 11 AM to 12 PM | **Run test set once.** Save metrics and charts. | No retuning after this |
| Wed 10 AM to 3 PM | Space and RAG with four tests; Urdu strings. | Space public by 3 PM |
| Wed 3 to 4 PM | Optimization pass (one change at a time, before/after accuracy, time, memory). | |
| **Wed 4 PM** | Export `prefect_runs.csv` unedited + Prefect screenshot. | |
| Wed 4 to 7 PM | Summary.pdf (one page), `datasetproposal.pdf`, `AI_Use_Log.md`, model card, demo video. | Feature freeze 7 PM |
| Wed 7 to 9 PM | Rehearse viva; every member answers every question. | |
| **Wed 9 PM** | Form opens: submit on form and Devpost. Tag `v1.0-final` when everything is final. | Nothing pushed after the tag counts |
| Wed 9 to 11:59 PM | Buffer. Verify the Space link, video link (no login), and repo tag. | Form closes Thu 12 AM |

**Kill criteria:** Gate 0 fails by 9:30 PM → Plan B. Model does not beat a baseline at some horizon by Wed 12 PM → keep the result and report honestly; never change the baseline. Running behind at Wed 3 PM → cut stretch items in the order in section 4.

---

## 14. Deliverables mapped to scoring

| Rubric item | What we show |
|---|---|
| Automatic checks (40) | Declared baselines beaten (or honestly not), `make data` reproducible, no hard-coded outputs, fixed seeds |
| Dataset strength (20) | 3 or 4 independent original publishers, manifest with SHA-256 and licences, DQ report, counted volume |
| Agentic design (15) | Evidence-based DQ decisions in `quality_log`, supervisor choices, Prefect run history |
| Scientific rigour (10) | Time splits with embargo, no-leakage test, conformal coverage, group errors, forward check |
| Usefulness and responsible AI (10) | Row IDs, citations, "I don't know", number guard, unvalidated-city labels, official-service pointers |
| Viva and demo (5) | 3-minute script below; everyone answers |
| Pakistan bonus (10) | Pakistani problem (smog after the embassy data halt), Pakistani data, Urdu advisories, local validation against Punjab EPD records and a documented smog episode (find and cite it) |

**3-minute demo:** 0:00 to 0:30 the lost embassy data and the officer's decision. 0:30 to 1:15 sources and two quality decisions (one flag-instead-of-fill). 1:15 to 2:15 Space: status, forecast band, baseline-vs-model chart, one cited answer, one "I don't know". 2:15 to 3:00 weakest group, limits, impact.

---

## 15. Business model

- **User:** district health or education officer, school head teacher.
- **Payer:** Punjab environment protection or education department (public budget); NGOs and funders as secondary payers.
- **Value:** earlier, better-informed advisories, less staff time spent checking sources, fewer children exposed during peaks.
- **Sustainability:** public-sector contract or grant; free public view with paid detailed alerts is a later option. The free Open-Meteo tier is non-commercial, so a commercial version would switch to Copernicus ADS CAMS directly or buy a commercial plan.
- **Running cost now:** Rs 0 (free tiers). Estimate at scale: one small VM, storage, and maintenance time, with assumptions written down.
- **Risks:** low trust, source licence changes, sensor drift, wrong outputs in a health context. Mitigations: visible uncertainty, drift monitor, calibration reports, official-service pointers.

---

## 16. Risks and fallbacks

| Risk | Likelihood | Response |
|---|---|---|
| OpenAQ Pakistan sensors too sparse | Medium | Plan B (heat-health) decided by 9:30 PM |
| Embassy archive unreachable | Medium | Label ground truth uncalibrated; reduce RQ1; say so |
| Community sensors over-read at high humidity | High | Humidity-aware calibration; error-by-humidity table; flag, do not hide |
| Long sensor outages | High | Flag, never fill target; report coverage per city |
| GitHub Actions cron late or skipped | Medium | Accept and report; or use the laptop scheduler |
| Space sleeps or breaks before judging | Medium | Open before the demo; keep status in the repo |
| Time pressure | High | Tiered scope and kill criteria above |
| A source's licence does not allow our use | Medium | Drop it and document why |

---

## 17. Ask Fizzah (send tonight)

1. Is `datasetproposal.pdf` the only declaration needed, and what is its deadline, given the Mon 12 PM declaration date has passed?
2. How is the run history scored if the scheduler starts after Tue 4 PM?
3. Is a data source that stopped publishing in March 2025 acceptable as a historical calibration reference, with live data from another source?
4. Are community-sensor data from OpenAQ acceptable as "credible original publishers" if licence and provenance are documented?
5. Does the 9 PM form replace any Devpost fields?

---

## 18. Supporting templates

**`AI_Use_Log.md` entry:** date/time, tool and version, task, what we kept or changed and how we checked it, who can explain it. Log as you go; do not write it at the end.

**`optimization_log.md` row:** change, accuracy before/after, run time before/after, peak memory before/after, kept/rejected. Planned changes (one at a time): Parquet with float32; incremental fetch; DuckDB aggregation instead of pandas; LightGBM `max_bin` and thread tuning; embedding cache.

**`datasetproposal.pdf` outline:** problem and user; why it matters in Pakistan (cite the embassy halt and Punjab smog); each source with publisher, URL/DOI, licence, update rate, why credible; independence argument; which source is not on the track list; expected volume; baselines (copied from `DECLARATION.md`); limits.

**Viva questions to rehearse:** why this problem, how we know it is real, why these sources and licences, worst raw-data problem, why flag instead of fill, how much we beat each baseline, weakest group, how the agent never invents a number, who pays, what we optimized and what it cost.

---

## Sources checked this session
- AirNow Department of State embassy map and background pages (airnow.gov, gispub.epa.gov/airnowembassy)
- CREA: "U.S. embassy AQ data halt" (energyandcleanair.org); AP/CNN, 5 Mar 2025; PNAS "Air pollution after the shutdown of US embassy monitoring"
- Open-Meteo documentation and third-party summaries (licence, CAMS resolution, history start)
- NASA FIRMS Area API page (firms.modaps.eosdis.nasa.gov/api/area)
- EGU24-453: "Open Air Quality Community Data in Pakistan" (doi:10.5194/egusphere-egu24-453)
- Punjab EPD ambient air quality PDFs (epd.punjab.gov.pk)

Anything in section 3 under "NOT verified" has not been checked and must be tested first.
