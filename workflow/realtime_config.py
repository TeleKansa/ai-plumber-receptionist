from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DEFAULT_REALTIME_MODEL = "gpt-realtime-1.5"
REALTIME_2_MODEL = "gpt-realtime-2"
REALTIME_2_REASONING_EFFORT = "low"
SUPPORTED_REALTIME_MODELS = (DEFAULT_REALTIME_MODEL, REALTIME_2_MODEL)
DEFAULT_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={DEFAULT_REALTIME_MODEL}"


def normalize_realtime_model(model: Optional[str]) -> str:
    candidate = (model or "").strip()
    if candidate in SUPPORTED_REALTIME_MODELS:
        return candidate
    return DEFAULT_REALTIME_MODEL


def effective_realtime_model(prompt_profile: Optional[dict] = None, settings=None) -> str:
    profile_model = (prompt_profile or {}).get("realtime_model")
    settings_model = getattr(settings, "openai_realtime_model", None)
    return normalize_realtime_model(profile_model or settings_model)


def realtime_reasoning_effort(model: Optional[str]) -> Optional[str]:
    if normalize_realtime_model(model) == REALTIME_2_MODEL:
        return REALTIME_2_REASONING_EFFORT
    return None


def realtime_session_overrides(model: Optional[str]) -> dict:
    effort = realtime_reasoning_effort(model)
    if not effort:
        return {}
    return {"reasoning": {"effort": effort}}


def build_realtime_url(model: Optional[str], configured_url: Optional[str] = None) -> str:
    realtime_model = normalize_realtime_model(model)
    base_url = (configured_url or DEFAULT_REALTIME_URL).strip() or DEFAULT_REALTIME_URL
    parsed = urlsplit(base_url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "model"
    ]
    query_items.append(("model", realtime_model))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )
