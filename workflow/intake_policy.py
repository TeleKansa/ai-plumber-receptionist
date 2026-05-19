import json
import re
from typing import Optional


SUPPORTED_CONDITIONS = {"always", "urgency_contains", "issue_contains"}
COLLECTION_MODES = {"required", "ask_once", "passive"}
ADDITIONAL_NOTES_KEY = "additional_notes"
DEFAULT_ADDITIONAL_NOTES_QUESTION = {
    "key": ADDITIONAL_NOTES_KEY,
    "label": "Additional notes",
    "question_text": "Anything else the plumber should know before I send this over?",
    "collection_mode": "ask_once",
    "required": False,
    "include_in_sms": True,
    "include_in_admin": True,
    "active": True,
}
DECLINED_OR_UNKNOWN_VALUES = {
    "all set",
    "that's all",
    "that is all",
    "declined",
    "do not know",
    "dont know",
    "don't know",
    "i do not know",
    "i dont know",
    "i don't know",
    "n a",
    "na",
    "no",
    "no thanks",
    "none",
    "nope",
    "nothing",
    "nothing else",
    "no idea",
    "not provided",
    "not sure",
    "refused",
    "unknown",
    "unsure",
}


def default_intake_policy() -> dict:
    return {
        "enabled": True,
        "extra_questions_json": json.dumps([DEFAULT_ADDITIONAL_NOTES_QUESTION]),
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


def _normalize_words(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_is_declined_or_unknown(value) -> bool:
    return _normalize_words(value) in DECLINED_OR_UNKNOWN_VALUES


def _collection_mode(question: dict) -> str:
    mode = str(question.get("collection_mode") or "").strip().lower()
    if mode in COLLECTION_MODES:
        return mode
    return "required" if _as_bool(question.get("required"), False) else "ask_once"


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
        "collection_mode": _collection_mode(question),
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
    regular_questions = [question for question in questions if question["key"] != ADDITIONAL_NOTES_KEY]
    final_questions = [question for question in questions if question["key"] == ADDITIONAL_NOTES_KEY]
    return regular_questions + final_questions


def _extra_fields(args: Optional[dict]) -> dict:
    if not args:
        return {}
    value = args.get("extra_fields") or {}
    return value if isinstance(value, dict) else {}


def validate_required_extra_fields(args: dict, policy: Optional[dict]) -> dict[str, str]:
    errors = {}
    fields = _extra_fields(args)
    for question in applicable_questions(policy, args):
        if question.get("collection_mode") != "required":
            continue
        value = fields.get(question["key"])
        if str(value or "").strip() and not answer_is_declined_or_unknown(value):
            continue
        errors[question["key"]] = f"{question['label']} is required by this tenant's intake policy."
    return errors


def _pending_question(intake_state: Optional[dict], key: str) -> dict:
    if not intake_state:
        return {}
    pending = intake_state.get("pending_extra_question") or {}
    return pending if pending.get("key") == key else {}


def caller_responded_after_pending_question(intake_state: Optional[dict], key: str) -> bool:
    pending = _pending_question(intake_state, key)
    return bool(pending.get("caller_response_after_pending") or pending.get("caller_response_text"))


def missing_policy_extra_fields(args: dict, policy: Optional[dict], intake_state: Optional[dict] = None) -> list[dict]:
    missing = []
    fields = _extra_fields(args)
    for question in applicable_questions(policy, args):
        mode = question.get("collection_mode") or "ask_once"
        if mode == "passive":
            continue
        value = fields.get(question["key"])
        has_value = bool(str(value or "").strip())
        declined_or_unknown = answer_is_declined_or_unknown(value)
        if mode == "ask_once" and has_value:
            if declined_or_unknown and not caller_responded_after_pending_question(intake_state, question["key"]):
                missing.append(
                    {
                        "key": question["key"],
                        "label": question["label"],
                        "question_text": question.get("question_text") or f"Please ask: {question['label']}",
                        "collection_mode": mode,
                        "reason": "intake_policy_unanswered_extra_field",
                        "error": "The AI cannot use declined or unknown until the caller has responded after this question.",
                    }
                )
                continue
            continue
        if mode == "required" and has_value and not declined_or_unknown:
            continue
        missing.append(
            {
                "key": question["key"],
                "label": question["label"],
                "question_text": question.get("question_text") or f"Please ask: {question['label']}",
                "collection_mode": mode,
                "reason": "intake_policy_missing_extra_fields",
                "error": (
                    "A useful answer is required."
                    if mode == "required" and declined_or_unknown
                    else "This question must be asked before submit."
                ),
            }
        )
    return missing


def missing_extra_guidance(missing_fields: list[dict]) -> str:
    if not missing_fields:
        return ""
    question = missing_fields[0]
    question_text = question.get("question_text") or question.get("label") or question.get("key")
    key = question.get("key")
    if question.get("collection_mode") == "required":
        return (
            f"Ask exactly this one question and wait for the caller's answer: {question_text} "
            f"This tenant marked it required, so submit extra_fields.{key} only after the caller gives a useful answer."
        )
    return (
        f"Ask exactly this one question and wait for the caller's answer: {question_text} "
        f"If they decline or do not know, record 'declined' or 'unknown' in extra_fields.{key} after they say so."
    )


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
    if field == "extra_questions_json":
        value = [question for question in (normalize_extra_question(item) for item in value) if question]
    elif field == "conditional_questions_json":
        value = [question for question in (normalize_conditional_question(item) for item in value) if question]
    return json.dumps(value, indent=2)
