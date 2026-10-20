# Optimization Log

Tracking pipeline and model performance optimizations one change at a time.

| Change ID | Timestamp | Optimization Description | Accuracy Before / After | Runtime Before / After | Peak Memory Before / After | Decision (Kept / Rejected) |
|---|---|---|---|---|---|---|
| OPT-00 | 2026-10-20 21:00 PKT | Baseline pipeline configuration | N/A | N/A | N/A | Baseline |
| OPT-01 | 2026-10-20 21:35 PKT | Vectorized stuck-sensor grouping & peer-corroboration hash lookup in DQ Gate 1 | N/A | 65s -> 0.8s | 280MB -> 110MB | Kept |
| OPT-02 | 2026-10-20 21:40 PKT | Vectorized Seasonal-Hour baseline calculation via lag-shift matrix | N/A | 240s -> 0.4s | 320MB -> 85MB | Kept |
