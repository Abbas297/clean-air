import re
from typing import Any, Dict, List, Optional, Tuple, Set

from src.rag.intents import UserIntent, classify_intent
from src.rag.sql_templates import execute_intent_query
from src.rag.index import HybridRAGIndex

EMERGENCY_HEALTH_POINTER = (
    "\n\n[Medical Emergency Pointer]: For acute respiratory distress or severe symptoms during smog peaks, "
    "immediately contact Punjab Emergency Rescue Service (Rescue 1122) or consult the National Health Helpline at 1033."
)

DISCLAIMER = "\n[Notice]: Informational decision support only. Official decisions must comply with Punjab EPD / PDMA directives."

URDU_ADVISORY_TEMPLATES = {
    "high_smog": "سموگ کی شدت زیادہ ہے۔ اسکولوں میں کھلی جگہ پر اسمبلی اور کھیل کے پیریڈز معطل رکھیں۔ طلبا اور اساتذہ کے لیے ماسک (N95) کا استعمال لازمی بنائیں۔",
    "moderate_smog": "ہوا کا معیار درمیانے درجے پر ہے۔ حساس طلبا اور بچوں کو کھلی فضا میں طویل مشقت سے گریز کی ہدایت کریں۔",
    "unvalidated_city": "یہ شہر غیر تصدیق شدہ ہے۔ براہ کرم سیٹلائٹ ماڈل کے وسیع تر غیر یقینی بینڈ کو مدنظر رکھیں۔",
}

def extract_numbers_from_text(text: str) -> Set[float]:
    """Extracts all floating point or integer numbers appearing in text, ignoring PM2.5 and PM10 tokens."""
    # Strip pollutant names like PM2.5, PM10
    cleaned = re.sub(r"pm\s*2\.5", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"pm\s*10", "", cleaned, flags=re.IGNORECASE)
    tokens = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    nums = set()
    for t in tokens:
        try:
            nums.add(float(t))
        except ValueError:
            pass
    return nums

class InsightAgent:
    def __init__(self):
        self.index = HybridRAGIndex()

    def answer_query(self, query: str, lang: str = "en") -> Dict[str, Any]:
        """
        Executes grounded RAG pipeline:
        1. Classifies intent & entities
        2. Retrieves official document citations
        3. Executes SQL template against Gold warehouse
        4. Validates answers via Number Guard & 'I don't know' rules
        5. Injects row IDs and official emergency pointers
        """
        q_lower = query.lower()

        # Check for misleading questions attempting to force false confirmations
        if any(phrase in q_lower for phrase in ["confirm tomorrow will be under", "guarantee pm2.5 is low", "promise clean air"]):
            return {
                "answer": (
                    "I cannot confirm or guarantee that pollution levels will remain low. "
                    "Atmospheric forecasts are governed by empirical uncertainty and calibrated prediction intervals. "
                    "Current data indicates persistence of seasonal smog."
                ) + EMERGENCY_HEALTH_POINTER + DISCLAIMER,
                "intent": "MISLEADING_QUERY_REJECTED",
                "citations": ["docid#who_pm25_24h"],
                "row_ids": [],
                "status": "REFUSAL_TO_INVENT_FACTS",
            }

        # 1. Intent Classification
        intent, entities = classify_intent(query)
        city = entities.get("city")

        # Unvalidated City Check (e.g., Multan, Faisalabad)
        if city in ["faisalabad", "multan"]:
            urdu_extra = f"\n\nاردو رہنمائی: {URDU_ADVISORY_TEMPLATES['unvalidated_city']}" if lang == "ur" else ""
            return {
                "answer": (
                    f"Notice: {city.capitalize()} is currently an unvalidated city without ground-truth reference monitors. "
                    "Estimates for this city are derived from raw CAMS satellite hindcasts with wide uncertainty intervals. "
                    "Ground validation is pending deployment."
                ) + urdu_extra + DISCLAIMER,
                "intent": "UNVALIDATED_CITY",
                "citations": ["docid#punjab_epd_pm25_standard"],
                "row_ids": [],
                "status": "UNVALIDATED_CITY_LABEL",
            }

        # 2. Document Search
        doc_matches = self.index.search(query, top_k=2, score_threshold=0.10)
        citations = [d["chunk_id"] for d in doc_matches]

        # 3. 'I don't know' Fallback for Unmatched Intents or low-relevance
        if intent == UserIntent.UNKNOWN and not doc_matches:
            return {
                "answer": "I don't know. The requested question does not match any official air quality standard, available city forecast, or pipeline status record.",
                "intent": UserIntent.UNKNOWN.value,
                "citations": [],
                "row_ids": [],
                "status": "I_DONT_KNOW_FALLBACK",
            }

        # If document query only (no SQL intent needed)
        if intent == UserIntent.UNKNOWN and doc_matches:
            best_chunk = doc_matches[0]
            ans_text = f"According to {best_chunk['title']} ({best_chunk['chunk_id']}):\n{best_chunk['text']}"
            return {
                "answer": ans_text + DISCLAIMER,
                "intent": "DOCUMENT_LOOKUP",
                "citations": citations,
                "row_ids": [],
                "status": "GROUNDED_DOCUMENT_ANSWER",
            }

        # 4. SQL Execution
        sql_data, row_ids = execute_intent_query(intent, entities)
        if not sql_data:
            return {
                "answer": f"I don't know. No data records are currently available for {city.capitalize() if city else 'the requested location'}.",
                "intent": intent.value,
                "citations": citations,
                "row_ids": [],
                "status": "MISSING_DATA",
            }

        # 5. Answer Assembly via Grounded Template
        allowed_numbers: Set[float] = set()
        for v in sql_data.values():
            if isinstance(v, (int, float)):
                allowed_numbers.add(float(v))

        # Build Answer Text
        if intent == UserIntent.FORECAST_LOOKUP:
            ans = (
                f"For {sql_data['city']}, current observed PM2.5 is {sql_data['current_pm25']} ug/m3. "
                f"The calibrated {sql_data['forecast_horizon']} forecast projects a median of {sql_data['p50_forecast']} ug/m3, "
                f"with an 80% conformal prediction band of [{sql_data['p10_band']} to {sql_data['p90_band']}] ug/m3 "
                f"(derived from {sql_data['n_sensors']} active sensor(s), QC flag: {sql_data['qc_flag']})."
            )
            citations.append("docid#epa_pm25_breakpoints")

        elif intent == UserIntent.ADVISORY_RECOMMENDATION:
            is_active = sql_data["advisory_triggered"]
            ans = (
                f"Advisory Assessment for {sql_data['city']}: "
                f"Current PM2.5 is {sql_data['current_pm25']} ug/m3 (US EPA Unhealthy threshold: {sql_data['epa_threshold']} ug/m3, "
                f"Punjab Standard: {sql_data['punjab_standard']} ug/m3). "
                f"Status: {'ADVISORY ACTIVE' if is_active else 'STANDARD'}. "
                f"Recommended action: {sql_data['action_required']}"
            )
            citations.append("docid#punjab_epd_school_closures")
            citations.append("docid#epa_action_threshold_55")
            ans += EMERGENCY_HEALTH_POINTER

        elif intent == UserIntent.WHY_IS_IT_HIGH:
            ans = (
                f"Atmospheric Diagnostics for {sql_data['city']}: "
                f"CAMS estimated PM2.5 at issue time was {sql_data['cams_pm25']} ug/m3 with "
                f"ambient relative humidity of {sql_data['relative_humidity']} % and temperature of {sql_data['temperature']} C. "
                f"Transboundary agricultural fire detections over Punjab (lagged 1 day) recorded {sql_data['transboundary_fires_lag1']} "
                f"active fire counts with total Fire Radiative Power (FRP) of {sql_data['fire_frp_lag1']} MW."
            )
            citations.append("docid#ndma_emergency_advisory")

        elif intent == UserIntent.DATA_STATUS:
            ans = (
                f"Data Pipeline Status: Operational. Total warehouse records: {sql_data['total_hourly_records']}. "
                f"Latest recorded UTC timestamp: {sql_data['latest_timestamp_utc']}."
            )

        # Append Urdu if requested
        if lang == "ur" and intent == UserIntent.ADVISORY_RECOMMENDATION:
            ans += f"\n\nاردو رہنمائی: {URDU_ADVISORY_TEMPLATES['high_smog'] if sql_data.get('advisory_triggered') else URDU_ADVISORY_TEMPLATES['moderate_smog']}"

        ans += DISCLAIMER

        # 6. Number Guard Validation
        # Verify that all numbers in the template string exist in allowed_numbers or standard constants
        text_numbers = extract_numbers_from_text(ans)
        exempt_numbers = {
            2.5, 10.0, 2021.0, 2023.0, 2024.0, 2026.0, 1122.0, 1033.0,
            80.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 12.0, 24.0, 48.0, 35.0, 55.5, 150.0, 200.0, 300.0
        }
        unauthorized_numbers = text_numbers - allowed_numbers - exempt_numbers

        if unauthorized_numbers:
            return {
                "answer": "Number Guard Violation: Answer contained ungrounded numeric values.",
                "intent": intent.value,
                "citations": citations,
                "row_ids": row_ids,
                "status": "REJECTED_BY_NUMBER_GUARD",
            }

        return {
            "answer": ans,
            "intent": intent.value,
            "citations": list(set(citations)),
            "row_ids": row_ids,
            "status": "VERIFIED_GROUNDED_ANSWER",
        }
