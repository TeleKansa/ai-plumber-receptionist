import json
import re
from typing import Optional


SUPPORTED_CONDITIONS = {"always", "urgency_contains", "issue_contains"}


def default_intake_policy() -> dict:
    return {
        "enabled": True,
        "extra_questions_json": "[]",
        "conditional_questions_json": "[]",
        "sms_include_extra_fields_json": "[]",
        "admin_display_fields_json": "[]",
        "notes": "",
    }


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_json_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _coerce_keywords(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = re.split(r"[\n,]+", text)
        return _coerce_keywords(parsed)
    return []


def _normalize_key(value) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def normalize_extra_question(question: dict) -> Optional[dict]:
    if not isinstance(question, dict):
        return None
    key = _normalize_key(question.get("key"))
    if not key:
        return None
    label = str(question.get("label") or key.replace("_", " ").title()).strip()
    question_text = str(question.get("question_text") or "").strip()
    return {
        "key": key,
        "label": label,
        "question_text": question_text,
        "required": _as_bool(question.get("required"), False),
        "include_in_sms": _as_bool(question.get("include_in_sms"), False),
        "include_in_admin": _as_bool(question.get("include_in_admin"), True),
        "active": _as_bool(question.get("active"), True),
    }


def normalize_conditional_question(question: dict) -> Optional[dict]:
    normalized = normalize_extra_question(question)
    if normalized is None:
        return None
    condition_type = str(question.get("condition_type") or "always").strip().lower()
    if condition_type not in SUPPORTED_CONDITIONS:
        condition_type = "always"
    normalized.update(
        {
            "condition_type": condition_type,
            "condition_keywords": _coerce_keywords(question.get("condition_keywords")),
        }
    )
    return normalized


def extra_questions(policy: Optional[dict], include_inactive: bool = False) -> list[dict]:
    if not policy:
        return []
    if not policy.get("enabled", True) and not include_inactive:
        return []
    questions = []
    for raw_question in _coerce_json_list(policy.get("extra_questions_json")):
        question = normalize_extra_question(raw_question)
        if question and (include_inactive or question["active"]):
            questions.append(question)
    return questions


def conditional_questions(policy: Optional[dict], include_inactive: bool = False) -> list[dict]:
    if not policy:
        return []
    if not policy.get("enabled", True) and not include_inactive:
        return []
    questions = []
    for raw_question in _coerce_json_list(policy.get("conditional_questions_json")):
        question = normalize_conditional_question(raw_question)
        if question and (include_inactive or question["active"]):
            questions.append(question)
    return questions


def _contains_any(text: object, keywords: list[str]) -> bool:
    haystack = str(text or "").lower()
    return any(keyword.lower() in haystack for keyword in keywords if keyword)


def conditional_question_applies(question: dict, args: Optional[dict] = None) -> bool:
    args = args or {}
    condition_type = question.get("condition_type") or "always"
    keywords = question.get("condition_keywords") or []
    if condition_type == "always":
        return True
    if condition_type == "urgency_contains":
        return _contains_any(args.get("urgency"), keywords)
    if condition_type == "issue_contains":
        return _contains_any(args.get("issue"), keywords)
    return False


def applicable_questions(policy: Optional[dict], args: Optional[dict] = None) -> list[dict]:
    questions = list(extra_questions(policy))
    questions.extend(
        question
        for question in conditional_questions(policy)
        if conditional_question_applies(question, args)
    )
    return questions


def _extra_fields(args: Optional[dict]) -> dict:
    if not args:
        return {}
    value = args.get("extra_fields") or {}
    return value if isinstance(value, dict) else {}


def validate_required_extra_fields(args: dict, policy: Optional[dict]) -> dict[str, str]:
    errors = {}
    fields = _extra_fields(args)
    for question in applicable_questions(policy, args):
        if not question.get("required"):
            continue
        value = fields.get(question["key"])
        if str(value or "").strip():
            continue
        errors[question["key"]] = f"{question['label']} is required by this tenant's intake policy."
    return errors


def _policy_key_list(policy: Optional[dict], field: str) -> set[str]:
    if not policy:
        return set()
    return {_normalize_key(item) for item in _coerce_json_list(policy.get(field)) if _normalize_key(item)}


def sms_extra_field_rows(policy: Optional[dict], args: Optional[dict]) -> list[tuple[str, str]]:
    fields = _extra_fields(args)
    include_keys = _policy_key_list(policy, "sms_include_extra_fields_json")
    rows = []
    seen = set()
    for question in applicable_questions(policy, args):
        key = question["key"]
        if key in seen:
            continue
        if not question.get("include_in_sms") and key not in include_keys:
            continue
        value = str(fields.get(key) or "").strip()
        if not value:
            continue
        rows.append((question["label"], value))
        seen.add(key)
    return rows


def admin_extra_field_rows(policy: Optional[dict], args: Optional[dict]) -> list[tuple[str, str]]:
    fields = _extra_fields(args)
    include_keys = _policy_key_list(policy, "admin_display_fields_json")
    rows = []
    seen = set()
    for question in applicable_questions(policy, args):
        key = question["key"]
        if key in seen:
            continue
        if not question.get("include_in_admin") and key not in include_keys:
            continue
        value = str(fields.get(key) or "").strip()
        if not value:
            continue
        rows.append((question["label"], value))
        seen.add(key)
    return rows


def policy_to_json(policy: Optional[dict], field: str) -> str:
    value = _coerce_json_list((policy or {}).get(field))
    return json.dumps(value, indent=2)
