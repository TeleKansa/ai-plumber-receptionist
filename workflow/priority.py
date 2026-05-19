import json
import re
from typing import Optional


DEFAULT_EMERGENCY_KEYWORDS = [
    "active leak",
    "burst pipe",
    "cannot shut",
    "can't shut",
    "cant shut",
    "flooding",
    "sewage",
    "sewer backup",
    "water is still coming out",
    "water still coming out",
    "water still running",
]

URGENT_KEYWORDS = [
    "today",
    "urgent",
    "soon",
    "no hot water",
    "leaking",
]


def _coerce_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = re.split(r"[\n,]+", value)
        return _coerce_list(parsed)
    return []


def _combined_text(args: dict) -> str:
    parts = [
        args.get("issue"),
        args.get("urgency"),
    ]
    extra_fields = args.get("extra_fields") or {}
    if isinstance(extra_fields, dict):
        parts.extend(str(value) for value in extra_fields.values())
    return " ".join(str(part or "") for part in parts).lower()


def classify_lead_priority(args: dict, policy: Optional[dict] = None) -> dict:
    text = _combined_text(args)
    keywords = _coerce_list((policy or {}).get("emergency_keywords_json")) or DEFAULT_EMERGENCY_KEYWORDS
    matched = [keyword for keyword in keywords if keyword.lower() in text]
    if matched:
        return {
            "priority": "emergency",
            "priority_reason": f"Matched emergency keyword: {matched[0]}",
            "classification": {"matched_keywords": matched, "source": "rule_based"},
        }

    urgent_matches = [keyword for keyword in URGENT_KEYWORDS if keyword in text]
    if urgent_matches:
        return {
            "priority": "urgent",
            "priority_reason": f"Matched urgent keyword: {urgent_matches[0]}",
            "classification": {"matched_keywords": urgent_matches, "source": "rule_based"},
        }

    return {
        "priority": "normal",
        "priority_reason": "No emergency or urgent keywords matched.",
        "classification": {"matched_keywords": [], "source": "rule_based"},
    }
