import re
from collections.abc import Mapping


INVALID_ADDRESS_VALUES = {
    "",
    "home",
    "my home",
    "my house",
    "same place",
    "unknown",
}

INVALID_NAME_VALUES = {
    "",
    "caller",
    "customer",
    "unknown",
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def looks_like_phone(value: object) -> bool:
    text = _clean(value)
    digits = re.sub(r"\D", "", text)
    if text.startswith("+"):
        return len(digits) in (11, 12)
    return len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))


def looks_like_address(value: object) -> bool:
    text = _clean(value)
    normalized = re.sub(r"\s+", " ", text.lower())
    if normalized in INVALID_ADDRESS_VALUES:
        return False
    return bool(text)


def looks_like_name(value: object) -> bool:
    text = _clean(value)
    normalized = re.sub(r"\s+", " ", text.lower())
    if normalized in INVALID_NAME_VALUES:
        return False
    return bool(text)


def validate_service_request_args(args: Mapping[str, object]) -> dict[str, str]:
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

    return errors
