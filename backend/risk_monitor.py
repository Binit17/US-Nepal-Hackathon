"""Risk monitoring module for crisis detection in conversation."""
from __future__ import annotations

from models import RiskLevel

# Keywords and phrases that indicate potential crisis
CRISIS_KEYWORDS_HIGH = [
    "kill myself", "want to die", "end my life", "suicide",
    "self-harm", "cutting myself", "hurt myself",
    "no reason to live", "better off dead", "can't go on",
    "end it all", "not worth living",
]

CRISIS_KEYWORDS_MODERATE = [
    "hopeless", "worthless", "nobody cares",
    "can't take it anymore", "give up", "no point",
    "trapped", "burden to everyone", "alone forever",
    "overwhelming pain", "constant suffering",
]

CRISIS_KEYWORDS_LOW = [
    "depressed", "anxious", "panic", "scared",
    "can't sleep", "nightmares", "crying",
    "exhausted", "empty", "numb",
]


def assess_risk(text: str) -> tuple[RiskLevel, str | None]:
    """Assess risk level based on conversation text.
    
    Returns:
        Tuple of (risk_level, alert_message or None)
    """
    text_lower = text.lower()

    for keyword in CRISIS_KEYWORDS_HIGH:
        if keyword in text_lower:
            return (
                RiskLevel.HIGH,
                "⚠️ HIGH RISK DETECTED: The patient has expressed thoughts "
                "that may indicate immediate danger. Crisis resources have "
                "been made available. Please contact emergency services if "
                "in immediate danger."
            )

    for keyword in CRISIS_KEYWORDS_MODERATE:
        if keyword in text_lower:
            return (
                RiskLevel.MODERATE,
                "⚡ Moderate risk indicators detected. The patient may "
                "benefit from additional support. Consider reaching out "
                "to a mental health professional."
            )

    for keyword in CRISIS_KEYWORDS_LOW:
        if keyword in text_lower:
            return (RiskLevel.LOW, None)

    return (RiskLevel.NONE, None)
