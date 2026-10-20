from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

def generate_summary_pdf(output_path: Path = Path("Summary.pdf")):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1b365d"))
    heading_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#2b547e"))
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=9, leading=12)
    bold_body = ParagraphStyle("BoldBody", parent=styles["Normal"], fontSize=9, leading=12, fontName="Helvetica-Bold")

    elements = []

    # Header
    elements.append(Paragraph("Saaf Hawa v2: Rebuilding Ground Truth & Smog Forecasting", title_style))
    elements.append(Paragraph("<b>Track:</b> Air Quality | <b>Event:</b> HackInsight '26 | <b>Zero-Cost Open Source Stack</b>", body_style))
    elements.append(Spacer(1, 10))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary & Problem Statement", heading_style))
    elements.append(Paragraph(
        "In March 2025, public ambient air quality monitoring in Pakistan suffered a severe blind spot when U.S. State Department / EPA "
        "embassy monitors in Lahore, Islamabad, Karachi, and Peshawar stopped transmitting data to AirNow. For district education officers, "
        "health officials, and school heads across Punjab, this eliminated the only public reference-grade source of truth for high-stakes decisions "
        "(school closures, morning assembly cancellations, mask mandates). "
        "<b>Saaf Hawa v2</b> reconstructs ground truth from community sensor networks (PAQI) calibrated against the pre-halt embassy record, "
        "and couples it with NASA FIRMS transboundary fires and ECMWF CAMS to provide 24h quantile forecasts with calibrated conformal uncertainty.",
        body_style
    ))
    elements.append(Spacer(1, 8))

    # Architecture & Data Rigour
    elements.append(Paragraph("2. Multi-Publisher Data Stack & Volume", heading_style))
    elements.append(Paragraph(
        "• <b>Four Independent Publishers:</b> OpenAQ (PAQI), AirNow DoS archive (source not on track list), ECMWF/Copernicus (Open-Meteo), and NASA FIRMS.<br/>"
        "• <b>Data Volume:</b> 56,388 city-hourly PM2.5 rows, 133,440 atmospheric features in DuckDB, backed by 28,652 AirNow historical calibration rows and 60,078 community sensor rows.<br/>"
        "• <b>Data Contract:</b> Timestamps in UTC, deterministic row IDs ({CITY}-{YYYYMMDDTHHZ}), full SHA-256 provenance in manifest.csv.<br/>"
        "• <b>Dual Quality Gates:</b> Pandera schema validation, sentinel conversion (-999 -> NULL), stuck-sensor detection (6h), and peer corroboration during extreme smog spikes (500-1000 ug/m3).",
        body_style
    ))
    elements.append(Spacer(1, 8))

    # Model Evaluation Table
    elements.append(Paragraph("3. Scientific Evaluation vs Declared Baselines (Like-for-Like Held-out Test Split, 24h Embargo)", heading_style))
    table_data = [
        ["Forecast Horizon", "Baseline A (Persistence) MAE", "Baseline B (Seasonal-Hour) MAE", "Saaf Hawa LightGBM+CQR MAE", "80% Conformal Coverage"],
        ["+6 Hours", "28.48 ug/m3", "23.62 ug/m3", "20.52 ug/m3", "82.8% (PASS)"],
        ["+12 Hours", "31.77 ug/m3", "23.63 ug/m3", "21.03 ug/m3", "84.6% (PASS)"],
        ["+24 Hours", "23.50 ug/m3", "23.65 ug/m3", "21.49 ug/m3", "84.9% (PASS)"],
    ]
    t = Table(table_data, colWidths=[90, 115, 120, 120, 95])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#e8f4f8")),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "<i>*Baseline Invariance & Weakest Groups: Baseline B (Seasonal-Hour) MAE is ~23.6 ug/m3 across horizons because previous 7-day occurrences for target hour (t+h) are identical for h<=24h. Weakest subgroup performance: Lahore (MAE 30.78, coverage 77.8%) and Smog Season (MAE 33.27). Post-March 2025 ground truth is an operational calibrated estimate. Karachi feed stopped transmitting in March 2025 and live forecasts are suppressed per data freshness policy.</i>",
        body_style
    ))
    elements.append(Spacer(1, 8))

    # Grounded RAG & Responsible AI
    elements.append(Paragraph("4. Responsible AI & Insight Agent", heading_style))
    elements.append(Paragraph(
        "• <b>Number Guard:</b> Every numeric value in agent outputs is strictly verified against DuckDB Gold query results or official guidelines, preventing hallucinated numbers.<br/>"
        "• <b>Data Provenance:</b> All forecast answers attach real database row IDs (e.g. LHE-20261020T1600Z).<br/>"
        "• <b>Emergency Health Routing:</b> Directs acute symptoms to Punjab Emergency Rescue (1122) and Health Helpline (1033).<br/>"
        "• <b>Bilingual Support:</b> Native Urdu templates for provincial school education administrators.<br/>"
        "• <b>Unvalidated City Guard:</b> Multan & Faisalabad outputs are explicitly labeled with widened uncertainty bounds.",
        body_style
    ))

    doc.build(elements)
    print(f"Generated {output_path}")

def generate_dataset_proposal_pdf(output_path: Path = Path("datasetproposal/datasetproposal.pdf")):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1b365d"))
    heading_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#2b547e"))
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=9, leading=13)

    elements = []
    elements.append(Paragraph("Dataset & Problem Proposal: Saaf Hawa v2", title_style))
    elements.append(Paragraph("<b>Air Quality Track | Zero-Cost Infrastructure</b>", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("1. Problem Statement & Operational User", heading_style))
    elements.append(Paragraph(
        "Following the March 2025 shutdown of U.S. Diplomatic Post AirNow monitors in Pakistan, public health and education authorities "
        "lost all continuous reference particulate data. District Education Authorities (DEAs) in Punjab must make critical operational decisions "
        "without trustworthy next-day forecasts. Saaf Hawa v2 reconstructs ground truth by calibrating community sensor networks against the pre-halt "
        "reference record and produces 24h uncertainty-bounded quantile forecasts.",
        body_style
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("2. Original Publishers & Multi-Source Independence", heading_style))
    elements.append(Paragraph(
        "1. <b>Live Target Ground Truth:</b> Pakistan Air Quality Initiative (PAQI) via OpenAQ (CC BY 4.0 / Open Data).<br/>"
        "2. <b>Historical Calibration Reference:</b> U.S. Diplomatic Post AirNow monitors via OpenAQ v3 (Public Domain). <i>[Source not on track list]</i>.<br/>"
        "3. <b>Model Atmospheric Feed:</b> Copernicus ECMWF CAMS and ERA5 via Open-Meteo (CC BY 4.0 Non-commercial).<br/>"
        "4. <b>Transboundary Fire Signal:</b> NASA Earthdata / LANCE FIRMS VIIRS & MODIS (Public Domain).<br/>"
        "5. <b>Standards & Guidance:</b> WHO Global Air Quality Guidelines 2021 & Punjab Environmental Quality Standards (PEQS).",
        body_style
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("3. Storage Contract, Provenance & Baselines", heading_style))
    elements.append(Paragraph(
        "• <b>Contract:</b> Timestamps stored in UTC. Missing target values remain NULL. Deterministic row IDs: {CITY}-{YYYYMMDDTHHZ}.<br/>"
        "• <b>Volume:</b> 56,388 ground-truth city-hourly records, 28,652 embassy calibration rows, and 133,440 atmospheric feature rows, exceeding the 50,000 threshold.<br/>"
        "• <b>Baselines:</b> Baseline A (Persistence at t) and Baseline B (Seasonal-Hour mean of target hour over past 7 days).<br/>"
        "• <b>Primary Limitation:</b> Post-March 2025 ground truth is an operational estimate calibrated from community sensors rather than direct BAM reference measurements.<br/>"
        "• <b>Integrity:</b> SHA-256 computed on ingest and audited in manifest.csv. 24-hour embargo enforced between splits.",
        body_style
    ))

    doc.build(elements)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_summary_pdf()
    generate_dataset_proposal_pdf()
