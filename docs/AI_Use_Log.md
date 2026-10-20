# AI Use Log

Tracking AI assistance across development, architecture, coding, and verification for Saaf Hawa v2.

| Date / Time (PKT) | Tool & Version | Specific Task | What Was Kept / Changed | How It Was Checked & Verified | Responsible Team Member |
|---|---|---|---|---|---|
| 2026-10-20 20:45 | Antigravity (Gemini 3.8 Flash) | Gate 0 verification specification | Kept exact criteria from MVP Section 3 | Cross-checked against OpenAQ v3 docs and MVP text | Data Eng |
| 2026-10-20 20:55 | Antigravity (Gemini 3.8 Flash) | Heartbeat skeleton & architecture scaffold | Kept Prefect flow, Bronze Parquet, manifest signatures | Code syntax check, file path validation | Core Team |
| 2026-10-20 21:25 | Antigravity (Gemini 3.8 Flash) | Ingestion backfills (AirNow, FIRMS), DQ Gate 1 (Pandera), Silver transformation | Implemented rules R01-R05, quarantine routing, quality_log, UTC de-duplication | Automated tests (pytest 7 passed in 1.00s) | Data Eng |
| 2026-10-20 21:40 | Antigravity (Gemini 3.8 Flash) | Phase 4 & 5: Calibration, Gold DuckDB warehouse, LightGBM Quantiles, Conformal CQR | 15 LightGBM models beat persistence & seasonal-hour baselines across all horizons | Tests (pytest 10 passed in 2.57s) | Model Eng |
| 2026-10-20 21:48 | Antigravity (Gemini 3.8 Flash) | Phase 6: Insight Agent (RAG), FAISS Hybrid index, Number Guard, SQL Grounding | Implemented 4 mandatory RAG criteria, emergency health pointer (Rescue 1122), Urdu templates | Tests (pytest 15 passed in 2.63s) | AI / RAG Lead |
| 2026-10-20 21:55 | Antigravity (Gemini 3.8 Flash) | Phase 7: Gradio Dashboard, Prefect export, PDF reporting (Summary & Dataset Proposal) | 5-tab UI, unedited prefect_runs.csv export, 1-page summary PDF | Module import test & pytest suite (15/15 passed) | Full Stack Lead |
