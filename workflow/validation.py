import re
from collections.abc import Mapping


INVALID_ADDRESS_VALUES = {
    "",
    "address on file",
    "at my house",
    "caller house",
    "callers house",
    "caller s house",
    "caller\u2019s house",
    "home",
    "my home",
    "my house",
    "same place",
    "the address on file",
    "the same place",
    "their house",
    "unknown",
}

INVALID_NAME_VALUES = {
    "",
    "caller",
    "customer",
    "unknown",
}

STREET_INDICATORS = {
    "ave",
    "avenue",
    "blvd",
    "boulevard",
    "cir",
    "circle",
    "court",
    "ct",
    "dr",
    "drive",
    "hwy",
    "highway",
    "ln",
    "lane",
    "parkway",
    "pkwy",
    "pl",
    "place",
    "rd",
    "road",
    "st",
    "street",
    "ter",
    "terrace",
    "trail",
    "trl",
    "way",
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_words(value: object) -> str:
    text = _clean(value).lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_phone(value: object) -> bool:
    text = _clean(value)
    digits = re.sub(r"\D", "", text)
    if text.startswith("+"):
        return len(digits) in (11, 12)
    return len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))


def looks_like_address(value: object) -> bool:
    text = _clean(value)
    normalized = _normalize_words(text)
    if normalized in INVALID_ADDRESS_VALUES:
        return False
    if "address on file" in normalized:
        return False
    if "house" in normalized and not re.search(r"\d", normalized):
        return False
    if len(normalized) < 8:
        return False
    if not re.search(r"\d", normalized):
        return False
    if not re.search(r"[a-z]", normalized):
        return False
    tokens = set(normalized.split())
    return bool(tokens & STREET_INDICATORS)


def looks_like_name(value: object) -> bool:
    text = _clean(value)
    normalized = _normalize_words(text)
    if normalized in INVALID_NAME_VALUES:
        return False
    return bool(text)


def name_supported_by_caller_text(name: object, caller_text: object) -> bool:
    normalized_name = _normalize_words(name)
    normalized_caller_text = _normalize_words(caller_text)
    if not normalized_name:
        return False
    name_tokens = [token for token in normalized_name.split() if len(token) > 1]
    if not normalized_caller_text:
        return bool(name_tokens)
    if not name_tokens:
        return False
    return all(re.search(rf"\b{re.escape(token)}\b", normalized_caller_text) for token in name_tokens)


def validate_service_request_args(args: Mapping[str, object], caller_text: object = "") -> dict[str, str]:
    errors: dict[str, str] = {}

    if not _clean(args.get("issue")):
        errors["issue"] = "Issue is required."

    if not _clean(args.get("urgency")):
        errors["urgency"] = "Urgency is required."

    if not looks_like_address(args.get("address")):
        errors["address"] = "Address is required and cannot be a placeholder."

    if not looks_like_phone(args.get("callback")):
        errors["callback"] = "Callback must look like a phone number."

    if not looks_like_name(args.get("name")):
        errors["name"] = "Name is required and cannot be a placeholder."
    elif not name_supported_by_caller_text(args.get("name"), caller_text):
        errors["name"] = "Name must be supported by caller-provided transcript."

    return errors
