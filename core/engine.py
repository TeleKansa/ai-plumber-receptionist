"""Industry-agnostic prompt/session engine.

Given a vertical config (defaults + instructions_template + tool schema) plus the
per-call tenant / prompt profile / intake policy, this renders the OpenAI Realtime
`instructions` string and the function tool list. All industry wording lives in the
vertical config; this module contains only generic assembly logic.
"""

from __future__ import annotations

import copy
import json
from string import Template
from typing import Optional

from workflow.intake_policy import ADDITIONAL_NOTES_KEY, conditional_questions, extra_questions


def coerce_list(value) -> list[str]:
    """Normalize a list / JSON string / delimited string into a clean list[str]."""
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
            parsed = [part.strip() for part in text.replace("\r", "\n").replace(",", "\n").split("\n")]
        return coerce_list(parsed)
    return []


def render_intake_policy(policy: Optional[dict], submit_tool_name: str) -> str:
    """Render the tenant-specific extra/conditional intake questions block.

    Generic: the qualification-tool name is supplied by the vertical so this text
    references the correct tool for any industry.
    """
    if not policy or not policy.get("enabled", True):
        return "No tenant-specific extra questions are active."

    extra_lines = []
    for question in extra_questions(policy):
        mode = question.get("collection_mode") or "ask_once"
        extra_lines.append(
            f"- {question['label']} ({question['key']}, {mode}): {question.get('question_text') or 'No wording configured.'}"
        )

    conditional_lines = []
    for question in conditional_questions(policy):
        mode = question.get("collection_mode") or "ask_once"
        condition_type = question.get("condition_type") or "always"
        keywords = ", ".join(question.get("condition_keywords") or [])
        condition_text = "always" if condition_type == "always" else f"{condition_type}: {keywords}"
        conditional_lines.append(
            f"- {question['label']} ({question['key']}, {mode}, if {condition_text}): "
            f"{question.get('question_text') or 'No wording configured.'}"
        )

    if not extra_lines and not conditional_lines:
        return "No tenant-specific extra questions are active."

    return "\n".join(
        [
            "Tenant-specific extra questions:",
            *(extra_lines or ["- None."]),
            "",
            "Tenant-specific conditional questions:",
            *(conditional_lines or ["- None."]),
            "",
            f"Submit answers for tenant-specific questions in {submit_tool_name}.extra_fields as an object keyed by the field key.",
            f"Before calling {submit_tool_name}, ask every active required and ask_once tenant-specific question that applies.",
            f"If {ADDITIONAL_NOTES_KEY} is active, ask it last as the final pre-submit question.",
            "Do not silently skip ask_once questions. Ask once before submit.",
            "If the caller answers an ask_once question, put the answer in extra_fields.",
            "Do not set extra_fields values to \"unknown\" unless the caller actually said they do not know, declined, or gave no answer after you asked.",
            "If the caller declines, does not know, or is rushed on an ask_once question, put \"declined\" or \"unknown\" in extra_fields only after they say so, then continue.",
            "Do not infer homeowner/renter or any other tenant-specific answer.",
            "Required tenant-specific questions must have a useful answer; declined, unknown, or not provided does not satisfy required questions.",
            "Passive questions can be captured if naturally provided, but they do not have to be asked before submit.",
            "Do not let passive questions delay emergency lead capture. Core required fields still matter more than style or extra questions.",
            "Avoid pushy wording like \"I need to know.\" Give the caller a moment to answer.",
        ]
    )


def submit_tool_name(vertical: dict) -> str:
    return (vertical.get("tool") or {}).get("name", "")


def profile_defaults(vertical: dict, tenant: Optional[dict] = None) -> dict:
    """Default prompt profile for a tenant, sourced from the vertical's defaults."""
    defaults = vertical["defaults"]
    tenant = tenant or {}
    business_name = tenant.get("business_name") or tenant.get("name") or defaults["business_name"]
    greeting = tenant.get("greeting") or defaults["greeting"]
    return {
        "label": "Default prompt",
        "business_name": business_name,
        "greeting": greeting,
        "tone": defaults["tone"],
        "verbosity": defaults["verbosity"],
        "closing_line": defaults["closing_line"],
        "avoid_phrases": list(defaults["avoid_phrases"]),
        "preferred_terms": list(defaults["preferred_terms"]),
        "extra_instructions_text": "",
        "realtime_model": "",
    }


def build_instructions(
    vertical: dict,
    caller_number: str,
    tenant: Optional[dict] = None,
    profile: Optional[dict] = None,
    intake_policy: Optional[dict] = None,
) -> str:
    """Render the OpenAI Realtime `instructions` string for one call."""
    defaults = vertical["defaults"]
    tenant = tenant or {}
    profile = profile or profile_defaults(vertical, tenant)

    business_name = (
        profile.get("business_name")
        or tenant.get("business_name")
        or tenant.get("name")
        or defaults["business_name_inline_fallback"]
    )
    greeting = profile.get("greeting") or tenant.get("greeting") or defaults["greeting"]
    tone = profile.get("tone") or defaults["tone"]
    verbosity = profile.get("verbosity") or defaults["verbosity"]
    closing_line = profile.get("closing_line") or defaults["closing_line"]
    avoid_phrases = coerce_list(profile.get("avoid_phrases_json") or profile.get("avoid_phrases")) or defaults["avoid_phrases"]
    preferred_terms = coerce_list(profile.get("preferred_terms_json") or profile.get("preferred_terms")) or defaults["preferred_terms"]
    extra_instructions = (profile.get("extra_instructions_text") or "").strip()

    return Template(vertical["instructions_template"]).substitute(
        business_name=business_name,
        tone=tone,
        caller_number=caller_number,
        greeting=greeting,
        verbosity=verbosity,
        closing_line=closing_line,
        preferred_text="\n".join(f"- {term}" for term in preferred_terms),
        avoid_text="\n".join(f'- "{phrase}"' for phrase in avoid_phrases),
        extra_text=extra_instructions or "None.",
        intake_policy_text=render_intake_policy(intake_policy, submit_tool_name(vertical)),
    )


def build_tools(vertical: dict) -> list:
    """The OpenAI function tool list for this vertical."""
    return [copy.deepcopy(vertical["tool"])]
