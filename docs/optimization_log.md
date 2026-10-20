# Optimization Log

Tracking pipeline and model performance optimizations with empirically measured runtimes.

| Change ID | Timestamp | Optimization Description | Accuracy Impact | Measured Execution Duration | Measured Memory | Decision (Kept / Rejected) |
|---|---|---|---|---|---|---|
| OPT-00 | 2026-10-20 21:00 PKT | Initial multi-step file ingestion | Baseline | ~120s total | ~250MB | Baseline |
| OPT-01 | 2026-10-21 02:30 PKT | Vectorized DuckDB Gold warehouse build with window partitions | No change | 36.8s end-to-end pipeline | ~160MB peak | Kept |
| OPT-02 | 2026-10-21 02:55 PKT | Fast vectorized Seasonal-Hour baseline using shifted array concatenations | Identical (exact) | 0.4s (over 168,000 pairs) | ~85MB | Kept |
| OPT-03 | 2026-10-21 03:20 PKT | De-duplicated fire detection counts and pre-computed state bundle export | Identical (exact) | 0.3s state bundle export | ~32MB payload | Kept |
