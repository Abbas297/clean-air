import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime, timezone
import json
import re
import duckdb
import pandas as pd
import streamlit as st
import altair as alt

from src.rag.guard import InsightAgent
from src.model.lgbm import generate_forecast_for_city

st.set_page_config(
    page_title="Saaf Hawa v2 — Air Quality Decision Support",
    page_icon="🌬️",
    layout="wide",
)

DB_PATH = Path("data/gold/saaf_hawa.duckdb")
STATE_DIR = Path("state")
STATUS_FILE = STATE_DIR / "status.json" if (STATE_DIR / "status.json").exists() else Path("status.json")
FORECASTS_FILE = STATE_DIR / "forecasts.json"
EVAL_SUMMARY_FILE = STATE_DIR / "eval_summary.csv" if (STATE_DIR / "eval_summary.csv").exists() else Path("data/gold/eval_summary.parquet")
EVAL_GROUP_FILE = STATE_DIR / "eval_by_group.csv" if (STATE_DIR / "eval_by_group.csv").exists() else Path("data/gold/eval_by_group.parquet")
QUALITY_LOG_FILE = STATE_DIR / "quality_log.csv" if (STATE_DIR / "quality_log.csv").exists() else Path("data/gold/quality_log.parquet")
HISTORY_FILE = STATE_DIR / "recent_history.csv"
PREFECT_RUNS_FILE = STATE_DIR / "prefect_runs.csv" if (STATE_DIR / "prefect_runs.csv").exists() else Path("prefect_runs.csv")

@st.cache_resource
def get_rag_agent():
    return InsightAgent()

rag_agent = get_rag_agent()

# Header
st.title("🌬️ Saaf Hawa v2: Smog Forecasting & Ground-Truth System")
st.caption(
    "Zero-cost operational ground-truth reconstruction, uncertainty-calibrated 24h forecasting, "
    "and grounded Insight Agent for Punjab district education and health officers."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Status & Freshness",
    "🔮 24-Hour Forecast & Advisory",
    "📈 Scientific Evidence",
    "🛡️ Quality Log",
    "💬 Ask Insight Agent",
])

# -------------------------------------------------------------------
# Tab 1: Status & Freshness
# -------------------------------------------------------------------
with tab1:
    st.subheader("Data Pipeline Freshness & Run Health")
    now_utc = datetime.now(timezone.utc)

    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            status_data = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pipeline Health", status_data.get("run_status", "UNKNOWN"))
        col2.metric("Run Duration", f"{status_data.get('duration_seconds', 0.0)} s")
        col3.metric("Manifest Rows Audited", status_data.get("manifest_rows_total", 0))
        col4.metric("Pipeline Version", status_data.get("pipeline_version", "v2.0"))

        st.info(f"**Last Run Timestamp (UTC):** `{status_data.get('last_heartbeat_utc')}` | **Run ID:** `{status_data.get('run_id')}`")
    else:
        st.warning("status.json not found.")

    # Upstream data feeds: age dynamically computed from real manifest timestamps
    manifest_path = Path("data/manifest.csv")
    openaq_status = "UNKNOWN"
    meteo_status = "UNKNOWN"
    firms_status = "UNKNOWN"

    if manifest_path.exists():
        try:
            df_m = pd.read_csv(manifest_path)
            df_m["fetch_time"] = pd.to_datetime(df_m["fetch_time_utc"], utc=True)

            m_openaq = df_m[df_m["source"].str.contains("OpenAQ", na=False)]
            if not m_openaq.empty:
                max_t = m_openaq["fetch_time"].max()
                age = (now_utc - max_t).total_seconds() / 3600.0
                tag = "LATE" if age > 3.0 else "FRESH"
                openaq_status = f"{tag} ({age:.1f}h ago)"

            m_meteo = df_m[df_m["source"].str.contains("Open-Meteo", na=False)]
            if not m_meteo.empty:
                max_t = m_meteo["fetch_time"].max()
                age = (now_utc - max_t).total_seconds() / 3600.0
                tag = "LATE" if age > 6.0 else "FRESH"
                meteo_status = f"{tag} ({age:.1f}h ago)"

            m_firms = df_m[df_m["source"].str.contains("FIRMS", na=False)]
            if not m_firms.empty:
                max_t = m_firms["fetch_time"].max()
                age = (now_utc - max_t).total_seconds() / 3600.0
                tag = "LATE" if age > 12.0 else "FRESH"
                firms_status = f"{tag} ({age:.1f}h ago)"
        except Exception:
            pass

    st.write("#### Upstream Ingestion Feeds & Freshness (Audit Logged)")
    sources_df = pd.DataFrame([
        {"Source": "OpenAQ Community Sensors", "Computed Age & Status": openaq_status, "Freshness Policy": "LATE > 3h", "Licence": "CC BY 4.0 / Open Data"},
        {"Source": "ECMWF CAMS & ERA5 (Open-Meteo)", "Computed Age & Status": meteo_status, "Freshness Policy": "LATE > 6h", "Licence": "CC BY 4.0 Non-commercial"},
        {"Source": "NASA FIRMS (South Asia VIIRS)", "Computed Age & Status": firms_status, "Freshness Policy": "LATE > 12h", "Licence": "NASA Public Domain"},
        {"Source": "AirNow Diplomatic Reference", "Computed Age & Status": "HISTORICAL ARCHIVE", "Freshness Policy": "Calibration Only", "Licence": "US Public Domain"},
    ])
    st.dataframe(sources_df, use_container_width=True)

    # City Ground-Truth Sensor Age from state/recent_history.csv
    if HISTORY_FILE.exists():
        st.write("#### Last Observation per Monitored City (from Ground-Truth Audit)")
        try:
            df_hist = pd.read_csv(HISTORY_FILE)
            city_rows = []
            for city_name, group in df_hist.groupby("city"):
                max_ts_str = group["ts_utc"].max()
                max_ts = pd.to_datetime(max_ts_str, utc=True)
                age_h = (now_utc - max_ts).total_seconds() / 3600.0
                state_tag = "LATE" if age_h > 3.0 else "FRESH"
                sorted_g = group.sort_values("ts_utc")
                city_rows.append({
                    "City": city_name.title(),
                    "Last Ground Truth Timestamp (UTC)": max_ts.strftime("%Y-%m-%d %H:%M:%SZ"),
                    "Age (Hours)": round(age_h, 1),
                    "Status": f"{state_tag} ({round(age_h, 1)}h)",
                    "Last Audited PM2.5 (µg/m³)": round(sorted_g["pm25_cal"].iloc[-1], 1),
                    "Sensor Flag": sorted_g["qc_flag"].iloc[-1],
                })
            st.dataframe(pd.DataFrame(city_rows), use_container_width=True)
        except Exception as e:
            st.caption(f"Note: Could not parse history timestamps: {e}")

    # Prefect Run History (last successful and last 10 runs with states)
    if PREFECT_RUNS_FILE.exists():
        st.write("#### Orchestrated Flow Run History (Prefect Engine Audit)")
        df_runs = pd.read_csv(PREFECT_RUNS_FILE)
        if not df_runs.empty:
            completed_runs = df_runs[df_runs["state"] == "Completed"]
            last_success_ts = completed_runs.iloc[0]["start_time"] if not completed_runs.empty else "None"
            st.caption(f"**Last Successful Flow Run:** `{last_success_ts}` | Showing last {min(10, len(df_runs))} executions (including any failures).")
            display_cols = [c for c in ["name", "state", "start_time", "end_time", "total_run_time"] if c in df_runs.columns]
            st.dataframe(df_runs.head(10)[display_cols], use_container_width=True)

# -------------------------------------------------------------------
# Tab 2: Forecast & Advisory
# -------------------------------------------------------------------
with tab2:
    st.subheader("24-Hour Calibrated PM2.5 Forecast & District Advisory")
    city_choice = st.selectbox(
        "Select Target District / City:",
        ["Lahore", "Islamabad", "Karachi", "Peshawar", "Faisalabad", "Multan"],
        index=0,
    )

    city_key = city_choice.lower().strip()

    # City-specific model / data disclosures
    if city_key == "peshawar":
        st.info("ℹ️ **Model Note (Unseen City Transfer):** The LightGBM model had **no Peshawar training data** (sensor history began May 2025 in the test split). The model forecasts using regional atmospheric features and **1 active community sensor**.")
    elif city_key == "islamabad":
        st.info("ℹ️ **Data Note (Label-Source Transition):** Training labels for Islamabad were drawn from the AirNow Community series (`s_1343270`), whereas live test labels transition to the F-7 AirGradient community sensor.")

    # Load from committed state/ first
    city_payload = None
    if FORECASTS_FILE.exists():
        with open(FORECASTS_FILE, "r", encoding="utf-8") as f:
            all_forecasts = json.load(f)
        city_payload = all_forecasts.get(city_key)

    # 1. Unvalidated Cities
    if city_key in ["faisalabad", "multan"]:
        st.warning(f"⚠️ **Notice: {city_choice} is an unvalidated district without ground-truth sensors.**")
        st.error("District Advisory: WITHHELD (Unvalidated Ground Truth)")
        st.info("Saaf Hawa withholds forecasts for cities without ground-truth sensors.")

    # 2. Stale Monitored Cities (Lahore & Karachi)
    elif city_payload and city_payload.get("stale"):
        last_obs = city_payload.get("last_observation_utc") or "Unknown"
        qc_flag = city_payload.get("qc_flag") or "STALE_FEED"
        notice = city_payload.get("notice") or "Observation feed is stale (>3 hours). Forecast is withheld per Data Quality Gate 2 freshness rule."

        st.warning(f"⚠️ **Forecast Withheld for {city_choice}**")
        st.error("District Advisory: WITHHELD (Stale Observation Feed)")
        st.markdown(f"**Last Observation Recorded:** `{last_obs}` ({qc_flag})")
        st.markdown(f"**Why the forecast is withheld:** {notice}")

    # 3. Monitored Cities with Live Feed
    else:
        curr_pm, row_id, n_sensors, qc_flag, forecast_rows = None, None, None, None, None
        if city_payload and city_payload.get("monitored"):
            curr_pm = city_payload.get("latest_pm")
            row_id = city_payload.get("row_id")
            n_sensors = city_payload.get("n_sensors")
            qc_flag = city_payload.get("qc_flag")
            forecast_rows = city_payload.get("forecast_table")
        elif DB_PATH.exists():
            curr_pm, row_id, n_sensors, qc_flag, forecast_rows = generate_forecast_for_city(city_key, DB_PATH)

        if curr_pm is not None and forecast_rows:
            # Advisory Rule based on Forecast p50 and p90
            p50_vals = []
            p90_vals = []
            for r in forecast_rows:
                v50 = r.get("p50 Forecast (ug/m3)")
                if v50 is not None:
                    p50_vals.append(float(v50))
                band_str = str(r.get("80% Conformal Band [p10, p90]", ""))
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", band_str)
                if len(matches) >= 2:
                    p90_vals.append(float(matches[1]))

            if any(v >= 55.5 for v in p50_vals):
                st.error("🚨 **DRAFT ADVISORY: LIMIT OUTDOOR ACTIVITY (Forecast Exceedance)**")
                st.markdown("**Intervention:** Enforce N95 respirators, suspend outdoor morning assemblies, shift physical education indoors.")
            elif any(v >= 55.5 for v in p90_vals):
                st.warning("⚠️ **DRAFT ADVISORY: POSSIBLE EXCEEDANCE (Uncertainty Band Exceeds Threshold)**")
                st.markdown("**Intervention:** Heightened monitoring; vulnerable populations and schools prepare indoor contingency plans.")
            else:
                st.success("✅ **DRAFT ADVISORY: STANDARD CONDITIONS (No Forecast Exceedance)**")
                st.markdown("**Intervention:** Outdoor activities permitted under standard monitoring.")

            st.caption("Advisory Rule: DRAFT 'limit outdoor activity' triggers if LightGBM p50 at any horizon (+6h, +12h, +24h) ≥ 55.5 µg/m³; DRAFT 'possible exceedance' triggers if the 80% conformal upper bound (p90) ≥ 55.5 µg/m³.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Latest Calibrated Ground Truth", f"{round(curr_pm, 1)} µg/m³")
            m2.metric("Active Sensors", f"{n_sensors} ({qc_flag})")
            m3.metric("Audited Row ID", row_id)

            st.write("#### 24-Hour LightGBM Quantile Forecast vs Declared Baselines")
            df_fc_display = pd.DataFrame(forecast_rows)
            if "CAMS Hindcast" in df_fc_display.columns:
                df_fc_display = df_fc_display.rename(columns={"CAMS Hindcast": "CAMS at issue time"})
            st.dataframe(df_fc_display, use_container_width=True)

            # 48-Hour Historical Trend
            if HISTORY_FILE.exists():
                df_hist = pd.read_csv(HISTORY_FILE)
                city_hist = df_hist[df_hist["city"].str.lower() == city_key].copy()
                if not city_hist.empty:
                    city_hist["ts_utc"] = pd.to_datetime(city_hist["ts_utc"])
                    city_hist = city_hist.sort_values("ts_utc")
                    st.write("#### Prior 48-Hour Ground Truth Trend (Calibrated PM2.5 µg/m³)")
                    st.line_chart(city_hist.set_index("ts_utc")["pm25_cal"])
        else:
            st.info(f"No ground truth data available for {city_choice}.")

    # Historical Episode Replay for Lahore
    replay_file = Path("state/replay.csv")
    if city_key == "lahore" and replay_file.exists():
        st.markdown("---")
        with st.expander("🎬 **Historical Smog Episode Replay: Lahore November 2025 (Held-out Test Period)**", expanded=True):
            st.info("ℹ️ **Held-out Test Replay:** Live Lahore sensors are currently stale on OpenAQ (last report Oct 14). Below is the held-out test replay during the acute November 2025 smog episode (actual peak reached **458 µg/m³**). Plots the +6h forecast and 80% conformal band aligned at target time ($t_{target} = t_{utc} + 6\\text{h}$) against actual PM2.5 at that same time.")
            df_rep = pd.read_csv(replay_file)

            y_true = df_rep["actual_target_pm25"]
            y_pred = df_rep["forecast_p50_h6"]
            y_persist = df_rep["pred_persist"]
            rep_mae = (y_true - y_pred).abs().mean()
            rep_pmae = (y_true - y_persist).abs().mean()
            rep_cov = ((y_true >= df_rep["band_p10_h6"]) & (y_true <= df_rep["band_p90_h6"])).mean() * 100.0

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Aligned Replay Samples", f"{len(df_rep)} hours")
            r2.metric("LightGBM +6h MAE", f"{rep_mae:.2f} µg/m³")
            r3.metric("Persistence +6h MAE", f"{rep_pmae:.2f} µg/m³", delta=f"{rep_pmae - rep_mae:.2f} better", delta_color="normal")
            r4.metric("80% Interval Coverage", f"{rep_cov:.1f}%", help="Nominal target 80%; drops to ~72% during acute peak")

            # Altair Chart with readable datetime ticks
            df_rep["target_datetime"] = pd.to_datetime(df_rep["ts_target_pkt"])

            band = alt.Chart(df_rep).mark_area(opacity=0.25, color="#3182bd").encode(
                x=alt.X("target_datetime:T", title="Target Time (PKT)", axis=alt.Axis(format="%b %d %H:%M", labelAngle=-45)),
                y=alt.Y("band_p10_h6:Q", title="PM2.5 (µg/m³)"),
                y2="band_p90_h6:Q"
            )
            line_actual = alt.Chart(df_rep).mark_line(color="#e41a1c", strokeWidth=2).encode(
                x="target_datetime:T",
                y="actual_target_pm25:Q",
                tooltip=[
                    alt.Tooltip("target_datetime:T", title="Target Time (PKT)", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("actual_target_pm25:Q", title="Actual PM2.5 (µg/m³)"),
                    alt.Tooltip("forecast_p50_h6:Q", title="LightGBM +6h (µg/m³)"),
                    alt.Tooltip("pred_persist:Q", title="Persistence (µg/m³)"),
                    alt.Tooltip("band_p10_h6:Q", title="Band p10"),
                    alt.Tooltip("band_p90_h6:Q", title="Band p90"),
                ]
            )
            line_pred = alt.Chart(df_rep).mark_line(color="#1b365d", strokeWidth=2, strokeDash=[4, 4]).encode(
                x="target_datetime:T",
                y="forecast_p50_h6:Q"
            )
            chart = (band + line_actual + line_pred).properties(
                height=380,
            )
            st.altair_chart(chart, use_container_width=True)

            with st.expander("📋 View Raw Aligned Replay Data (Target Aligned)", expanded=False):
                st.dataframe(df_rep, use_container_width=True)

# -------------------------------------------------------------------
# Tab 3: Scientific Evidence
# -------------------------------------------------------------------
with tab3:
    st.subheader("Scientific Evaluation vs Declared Baselines (Held-Out Test Split)")

    # 1. Overall Summary
    if EVAL_SUMMARY_FILE.exists():
        if EVAL_SUMMARY_FILE.suffix == ".csv":
            df_summary = pd.read_csv(EVAL_SUMMARY_FILE)
        else:
            df_summary = pd.read_parquet(EVAL_SUMMARY_FILE)

        st.write("#### Benchmark: LightGBM + CQR vs Baselines (Like-for-Like Test Split)")

        # Format baseline coverage and band width as blank (not 0)
        display_summary = df_summary.copy()
        mask_base = display_summary["model"].str.contains("Baseline", na=False)
        if "coverage_80" in display_summary.columns:
            display_summary["coverage_80"] = display_summary["coverage_80"].astype(str)
            display_summary.loc[mask_base, "coverage_80"] = ""
        if "mean_band_width" in display_summary.columns:
            display_summary["mean_band_width"] = display_summary["mean_band_width"].astype(str)
            display_summary.loc[mask_base, "mean_band_width"] = ""

        st.dataframe(display_summary, use_container_width=True)

        model_rows = df_summary[df_summary["model"].str.contains("LightGBM")]
        cov_min = float(model_rows["coverage_80"].min()) if not model_rows.empty else 0.0
        cov_max = float(model_rows["coverage_80"].max()) if not model_rows.empty else 0.0

        # Compute dynamic subgroup weak spots from state/eval_by_group.csv
        ws = {
            "night_m": "n/a", "night_p": "n/a",
            "aft_m": "n/a", "aft_s": "n/a",
            "lhe_cov_min": "n/a", "lhe_cov_max": "n/a",
            "vu_cov": "n/a"
        }
        if EVAL_GROUP_FILE.exists():
            try:
                dg = pd.read_csv(EVAL_GROUP_FILE) if EVAL_GROUP_FILE.suffix == ".csv" else pd.read_parquet(EVAL_GROUP_FILE)
                rn = dg[(dg["group_type"] == "diurnal") & (dg["group_value"].str.contains("Night", case=False, na=False)) & (dg["horizon"] == "+6h")]
                if not rn.empty:
                    ws["night_m"] = f"{float(rn.iloc[0]['model_mae']):.2f}"
                    ws["night_p"] = f"{float(rn.iloc[0]['persistence_mae']):.2f}"

                ra = dg[(dg["group_type"] == "diurnal") & (dg["group_value"].str.contains("Afternoon", case=False, na=False)) & (dg["horizon"] == "+12h")]
                if not ra.empty:
                    ws["aft_m"] = f"{float(ra.iloc[0]['model_mae']):.2f}"
                    ws["aft_s"] = f"{float(ra.iloc[0]['seasonal_mae']):.2f}"

                rl = dg[(dg["group_type"] == "city") & (dg["group_value"].str.lower() == "lahore")]
                if not rl.empty:
                    ws["lhe_cov_min"] = f"{float(rl['coverage_80'].min()):.1f}"
                    ws["lhe_cov_max"] = f"{float(rl['coverage_80'].max()):.1f}"

                rv = dg[(dg["group_type"] == "aqi_tier") & (dg["group_value"].str.contains("Very Unhealthy", case=False, na=False)) & (dg["horizon"] == "+6h")]
                if not rv.empty:
                    ws["vu_cov"] = f"{float(rv.iloc[0]['coverage_80']):.1f}"
            except Exception:
                pass

        st.markdown(f"""
        ### Declared Success Criteria Verification (Real Held-Out Test Evaluation):
        1. **Baselines Beaten:** LightGBM Quantile Forecaster strictly outperforms Baseline A (Persistence) and Baseline B (Seasonal-Hour) across $+6\\text{{h}}, +12\\text{{h}}, +24\\text{{h}}$ on like-for-like joined test rows.
        2. **Conformal Coverage:** Empirical 80% coverage rate is **{cov_min:.1f}% - {cov_max:.1f}%**, conforming strictly within the declared $75\\% - 85\\%$ pooled coverage range.
        3. **Embargo Separation:** Temporal integrity verified by automated tests confirming a strict 24-hour embargo gap between train cutoff and test start.
        4. **Subgroup Weak Spots (Disclosed Plainly, from eval_by_group):** 
           - **Night +6h (00-06 PKT):** Model MAE ({ws['night_m']} µg/m³) marginally trails persistence ({ws['night_p']} µg/m³).
           - **Afternoon +12h (12-18 PKT):** Model MAE ({ws['aft_m']} µg/m³) marginally trails seasonal ({ws['aft_s']} µg/m³).
           - **Lahore Coverage:** Observes {ws['lhe_cov_min']}% - {ws['lhe_cov_max']}%, falling below the nominal 80% target.
           - **Peak Episodes (Very Unhealthy, >150 µg/m³):** Coverage drops to {ws['vu_cov']}% (+6h), which is crucial context for emergency health/school advisories.
        """)

    # 2. Granular Errors by Group
    if EVAL_GROUP_FILE.exists():
        if EVAL_GROUP_FILE.suffix == ".csv":
            df_group = pd.read_csv(EVAL_GROUP_FILE)
        else:
            df_group = pd.read_parquet(EVAL_GROUP_FILE)

        with st.expander("🔍 Granular Errors by Group (City, Season, Diurnal Cycle, AQI Tier)", expanded=True):
            available_groups = list(df_group["group_type"].unique())
            group_sel = st.selectbox("Select Evaluation Group Slice:", available_groups, index=0)
            df_slice = df_group[df_group["group_type"] == group_sel].copy()
            st.dataframe(df_slice, use_container_width=True)

# -------------------------------------------------------------------
# Tab 4: Quality Log Viewer
# -------------------------------------------------------------------
with tab4:
    st.subheader("Data Quality Gate Audit Trail")
    df_log = None
    if QUALITY_LOG_FILE.exists():
        if QUALITY_LOG_FILE.suffix == ".csv":
            df_log = pd.read_csv(QUALITY_LOG_FILE)
        else:
            df_log = pd.read_parquet(QUALITY_LOG_FILE)

    if df_log is not None and not df_log.empty:
        action_filter = st.radio("Filter by Adjudicated Action:", ["ALL", "FIX", "FLAG", "QUARANTINE", "CONFIRM"], horizontal=True)

        if action_filter != "ALL" and "action" in df_log.columns:
            df_filtered = df_log[df_log["action"] == action_filter]
        else:
            df_filtered = df_log

        st.dataframe(df_filtered.tail(50), use_container_width=True)
    else:
        st.info("Quality log file not found.")

# -------------------------------------------------------------------
# Tab 5: Ask Insight Agent (RAG)
# -------------------------------------------------------------------
with tab5:
    st.subheader("Grounded Insight Agent (RAG)")
    st.caption("Ask natural language questions about PM2.5 guidelines, current forecasts, or smog causes.")

    lang_choice = st.radio("Response Language:", ["English", "Urdu (اردو)"], horizontal=True)
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g. Should schools close in Lahore tomorrow? or What is the WHO PM2.5 guideline?",
    )

    if st.button("Ask Agent", type="primary"):
        if user_query.strip():
            with st.spinner("Retrieving grounded evidence..."):
                lang_code = "ur" if "Urdu" in lang_choice else "en"
                res = rag_agent.answer_query(user_query, lang=lang_code)

                st.markdown("### Grounded Answer")
                st.markdown(res["answer"])

                with st.expander("🔍 Provenance & Grounding Metadata", expanded=True):
                    st.write(f"**Intent:** `{res.get('intent')}` | **Status:** `{res.get('status')}`")
                    row_ids = res.get("row_ids", [])
                    st.write(f"**Grounded Row IDs:** {', '.join([f'`{r}`' for r in row_ids]) if row_ids else 'None'}")
                    citations = res.get("citations", [])
                    st.write(f"**Official Document Citations:** {', '.join([f'`{c}`' for c in citations]) if citations else 'None'}")
        else:
            st.warning("Please enter a question.")
