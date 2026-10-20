# Saaf Hawa: review of the zip (Wed 21 Oct, ~04:00 PKT)

Verdict: the structure is sound and the headline result is real (model beats both baselines at all three horizons on 24k test rows, 80% intervals cover 83-85%). But five things could cost points or disqualify, and two claims in the Summary were not true. Fixes are in this patch; items marked **YOU** need your machine or your accounts.

## Do these first (in order)
1. **YOU: revoke the OpenAQ key.** It was committed in `config/config.yaml` (commit 2851d00) and stays in git history. Create a new key, set `OPENAQ_API_KEY` as an environment variable / GitHub secret. The patch blanks it in config.
2. **YOU: start the hourly runner now and keep the laptop awake until 4 PM.** `prefect_runs.csv` has 5 runs, all manual, all between 02:35 and 03:25 PKT. Nothing is scheduled: `.github/workflows/hourly.yml` is deleted in the zip, and `scripts/serve_scheduler.py` only runs while it is open. Run `python scripts/serve_scheduler.py`, disable sleep, plug in, and check a new row appears each hour. Do not also enable GitHub Actions (its Prefect history would not persist). Ask Fizzah how runs before 4 PM Tue are scored.
3. **Hugging Face Space.** The rules require one; the app is deployed on Streamlit Cloud. A Streamlit Space works (add a `README.md` with `sdk: streamlit`). I did not test the app from a clean clone: check it starts with only `state/` present.
4. **YOU: FIRMS backfill, then one final evaluation.** Fire features are 0 for almost all history (the pipeline only downloaded the last 7 days), so the "upwind fires" claim had no evidence behind it. Get a free key at firms.modaps.eosdis.nasa.gov/api/map_key, then:
   ```
   $env:FIRMS_MAP_KEY="your_key"          # PowerShell
   python scripts/backfill_firms.py --start 2023-01-01
   python flows/hourly.py                 # rebuilds Silver/Gold
   python src/model/evaluate.py
   python scripts/build_pdfs.py           # Summary + proposal now read the results
   ```
   Report the result whichever way it goes. Run evaluation once more only for this, and log it (see addendum item 8).

## Bugs found and fixed in the patch
- **Fire counts inflated up to 5x**: every hourly download re-fetched the same 7 days and Silver counted each copy (7,610 rows vs 1,522 unique). Now de-duplicated.
- **Invented calibration**: Peshawar and Karachi used fixed coefficients (0.88, -0.05, 2.5) described as a "standard transfer function", not fitted to anything. Now identity + a `FLAG` in `quality_log` + "uncalibrated" label. This changes Peshawar labels, so results will move after a rebuild.
- **Manifest wrote absolute Windows paths** (`C:/Users/abbas/...`), so `test_manifest_integrity` fails on any other machine. Now repo-relative; the test maps old rows and still checks every SHA-256. 15/15 tests pass.
- **Diurnal groups were UTC hours presented as time of day** ("Evening 18-24"). Now PKT.
- **Summary/proposal numbers were typed by hand** and disagreed with the results (83.0-84.8% vs 82.8-84.9%; 56,388 vs 56,369 rows). `build_pdfs.py` now reads them from `state/` and `status.json`, and prints the limitations below. One page, checked.

## Scientific issues you must disclose (written up in `DECLARATION_ADDENDUM.md`)
- **Training data is thin.** Per-city training hours: Islamabad 11,168; Lahore 4,899; Peshawar 0; Karachi 0. Peshawar is 46% of the test rows and the model never saw it. Karachi has 101 usable hours and should not be called monitored.
- **Labels change between train and test** (different sensors from May 2025), so test error includes a label-source shift.
- **Islamabad "community" sensor `s_1343270` is almost the embassy feed**: 75% of 8,201 overlapping hours are exactly equal. Calibration against it is circular, and the "independent sources" argument needs care. Be ready to explain this in the viva.
- **Embassy data in your pull ends 2024-08-07**, not March 2025. Don't say "pre-March 2025" for the calibration window.
- **Declared but not evaluated**: advisory recall/precision (success criterion 4), humidity-band and sensor-count groups. Say "not done" rather than leaving them implied.
- **Test set was re-evaluated after code fixes** (three commits after first seeing scores). Disclosed; don't tune anything further.
- On this machine re-running evaluation gave +6h MAE 20.8 vs 20.5 in your file. Quote one final run, not a mix.
- `docs/optimization_log.md` has timings and memory figures (65s to 0.8s, 280MB to 110MB) I cannot find a measurement for. "Faked logs" is a disqualification reason: replace with numbers you actually measured, or delete those rows.
- The old summary said evening persistence wins "due to sudden traffic bursts". Nothing in the data shows that cause. Removed.

## Not verified
Live OpenAQ/FIRMS calls, `make data` from an empty folder (needs the network and your keys), the Streamlit app on a clean clone, the Urdu text (needs a native reader), and the demo. Test on a second folder after the scheduler is running.

## What is good (keep it)
Real data end to end; DQ log with genuine flag-instead-of-fill cases (stuck sensors, unconfirmed extremes); declared baselines unchanged and beaten; honest weak groups (Lahore, smog season, night at +6h); conformal coverage in the declared range; row IDs and number guard; the AI log already admits the earlier synthetic version.
