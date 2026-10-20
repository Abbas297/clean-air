from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

class DQAction(str, Enum):
    FIX = "FIX"
    FLAG = "FLAG"
    QUARANTINE = "QUARANTINE"
    CONFIRM = "CONFIRM"

@dataclass
class PolicyRule:
    rule_id: str
    description: str
    action: DQAction
    reason_template: str

RULES = {
    "R01_SENTINEL": PolicyRule(
        rule_id="R01_SENTINEL",
        description="Negative or -999 sentinel sensor reading",
        action=DQAction.FIX,
        reason_template="Replaced sentinel reading ({val}) with NULL.",
    ),
    "R02_PHYSICALLY_IMPLAUSIBLE": PolicyRule(
        rule_id="R02_PHYSICALLY_IMPLAUSIBLE",
        description="PM2.5 value exceeding 1000 ug/m3",
        action=DQAction.QUARANTINE,
        reason_template="Reading ({val} ug/m3) exceeded physical ceiling of 1000 ug/m3; routed to quarantine.",
    ),
    "R03_STUCK_SENSOR": PolicyRule(
        rule_id="R03_STUCK_SENSOR",
        description="Sensor output identical for 6 or more consecutive hours",
        action=DQAction.FLAG,
        reason_template="Sensor {sensor} output stuck at {val} for >= 6 consecutive hours; flagged as stuck_sensor.",
    ),
    "R04_EXTREME_ADJUDICATION": PolicyRule(
        rule_id="R04_EXTREME_ADJUDICATION",
        description="PM2.5 value between 500 and 1000 ug/m3",
        action=DQAction.FLAG,
        reason_template="Extreme value ({val} ug/m3) without peer sensor confirmation; flagged & quarantined.",
    ),
    "R05_EXTREME_CONFIRMED": PolicyRule(
        rule_id="R05_EXTREME_CONFIRMED",
        description="PM2.5 value between 500 and 1000 ug/m3 confirmed by >=2 peer sensors > 300 ug/m3",
        action=DQAction.CONFIRM,
        reason_template="Extreme value ({val} ug/m3) corroborated by neighbouring sensors during severe smog episode.",
    ),
}
