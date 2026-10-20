from enum import Enum
import re
from typing import Any, Dict, Optional, Tuple

class UserIntent(str, Enum):
    FORECAST_LOOKUP = "FORECAST_LOOKUP"
    ADVISORY_RECOMMENDATION = "ADVISORY_RECOMMENDATION"
    WHY_IS_IT_HIGH = "WHY_IS_IT_HIGH"
    DATA_STATUS = "DATA_STATUS"
    UNKNOWN = "UNKNOWN"

CITY_SYNONYMS = {
    "lahore": "lahore",
    "lhe": "lahore",
    "islamabad": "islamabad",
    "isb": "islamabad",
    "karachi": "karachi",
    "khi": "karachi",
    "peshawar": "peshawar",
    "pew": "peshawar",
    "faisalabad": "faisalabad",
    "fsd": "faisalabad",
    "multan": "multan",
    "mux": "multan",
}

def classify_intent(query: str) -> Tuple[UserIntent, Dict[str, Any]]:
    """
    Classifies user question into closed-set intents and extracts entities (city, horizon).
    """
    q = query.lower()
    entities: Dict[str, Any] = {"city": None, "horizon": 24}

    # Check for Document/Guideline Standards lookup
    if any(k in q for k in ["who", "guideline", "standard", "peqs", "regulation", "breakpoint", "rule", "target"]):
        # Unless asking explicitly for a city forecast
        if not any(k in q for k in ["tomorrow", "predict", "forecast"]):
            return UserIntent.UNKNOWN, entities

    # Extract City
    for syn, canonical in CITY_SYNONYMS.items():
        if re.search(rf"\b{syn}\b", q):
            entities["city"] = canonical
            break

    # Extract Horizon
    if "6" in q or "6h" in q or "six" in q:
        entities["horizon"] = 6
    elif "12" in q or "12h" in q or "twelve" in q:
        entities["horizon"] = 12
    elif "24" in q or "24h" in q or "tomorrow" in q or "next day" in q:
        entities["horizon"] = 24

    # Classify Intent
    if any(k in q for k in ["status", "freshness", "last run", "sensor health", "pipeline"]):
        return UserIntent.DATA_STATUS, entities

    if any(k in q for k in ["why is it high", "why high", "reasons", "stubble", "fire", "smoke", "causes", "wind"]):
        return UserIntent.WHY_IS_IT_HIGH, entities

    if any(k in q for k in ["advisory", "close school", "mask", "outdoor", "assembly", "action", "guidance", "safe"]):
        return UserIntent.ADVISORY_RECOMMENDATION, entities

    if any(k in q for k in ["forecast", "pm2.5", "pm25", "predict", "tomorrow", "expected", "level", "aqi"]):
        return UserIntent.FORECAST_LOOKUP, entities

    return UserIntent.UNKNOWN, entities
