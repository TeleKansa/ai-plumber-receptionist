import json
import re
from html import escape
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from admin.auth import check_admin_credentials
from config.settings import Settings
from storage import repository
from workflow.intake_policy import conditional_questions, extra_questions, policy_to_json
from workflow.notifications import build_sms_body
from workflow.prompt_builder import PromptBuilder
from workflow.realtime_config import SUPPORTED_REALTIME_MODELS, effective_realtime_model, realtime_reasoning_effort


security = HTTPBasic(auto_error=False)


def _require_admin(settings: Settings):
    def dependency(credentials: HTTPBasicCredentials = Depends(security)):
        if not settings.admin_password:
            raise HTTPException(status_code=503, detail="Admin disabled. Set ADMIN_PASSWORD to enable it.")
        if credentials is None or not check_admin_credentials(
            settings.admin_password,
            credentials.username,
            credentials.password,
        ):
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Basic"},
            )
        return True

    return dependency


def _render_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "<p>No records yet.</p>"
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(row.get(column, '') or ''))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _tenant_actions_table(tenants: list[dict]) -> str:
    if not tenants:
        return "<p>No tenants yet.</p>"
    rows = []
    for tenant in tenants:
        tenant_id = tenant["id"]
        detail_href = f"/admin/tenants/{tenant_id}"
        prompt_href = f"/admin/tenants/{tenant_id}/prompt"
        intake_href = f"/admin/tenants/{tenant_id}/intake-policy"
        notification_href = f"/admin/tenants/{tenant_id}/notification-policy"
        rows.append(
            "<tr>"
            f'<td><a href="{detail_href}">{escape(tenant.get("name") or "")}</a></td>'
            f"<td>{escape(str(tenant_id))}</td>"
            f"<td>{escape(tenant.get('slug') or '')}</td>"
            f"<td>{escape(tenant.get('status') or '')}</td>"
            f"<td>{escape(tenant.get('business_name') or '')}</td>"
            f"<td>{escape(tenant.get('notification_sms_number') or '')}</td>"
            f'<td><a href="{detail_href}">Details</a></td>'
            f'<td><a href="{prompt_href}">Prompt/persona settings</a></td>'
            f'<td><a href="{intake_href}">Intake policy</a></td>'
            f'<td><a href="{notification_href}">Notification policy</a></td>'
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>tenant</th><th>id</th><th>slug</th><th>status</th><th>business_name</th>"
        "<th>notification_sms_number</th><th>details</th><th>prompt</th><th>intake</th><th>notifications</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "tenant"


def _input(name: str, value: str = "", placeholder: str = "") -> str:
    return (
        f'<label>{escape(name)}<br>'
        f'<input name="{escape(name)}" value="{escape(value or "")}" placeholder="{escape(placeholder)}" style="width: 420px; max-width: 100%;"></label>'
    )


def _checkbox(name: str, checked: bool = False) -> str:
    checked_attr = " checked" if checked else ""
    return f'<label><input type="checkbox" name="{escape(name)}" value="1"{checked_attr}> {escape(name)}</label>'


def _select(name: str, options: list[tuple[str, str]], selected: str = "") -> str:
    option_html = []
    for value, label in options:
        selected_attr = " selected" if value == selected else ""
        option_html.append(f'<option value="{escape(value)}"{selected_attr}>{escape(label)}</option>')
    return f'<label>{escape(name)}<br><select name="{escape(name)}">{"".join(option_html)}</select></label>'


def _realtime_model_select(selected: str = "", fallback_model: str = "gpt-realtime-1.5") -> str:
    options = [
        ("", "env/default"),
        ("gpt-realtime-1.5", "gpt-realtime-1.5"),
        ("gpt-realtime-2", "gpt-realtime-2"),
    ]
    option_html = []
    for value, label in options:
        selected_attr = " selected" if value == (selected or "") else ""
        option_html.append(f'<option value="{escape(value)}"{selected_attr}>{escape(label)}</option>')
    return (
        '<div style="border: 1px solid #ddd; padding: 12px; margin: 12px 0; background: #fafafa;">'
        '<label><strong>Realtime Model:</strong><br>'
        f'<select name="realtime_model" style="min-width: 260px;">{"".join(option_html)}</select>'
        "</label>"
        f"<p><code>env/default</code> uses Railway <code>OPENAI_REALTIME_MODEL</code> or the app default "
        f"<code>{escape(fallback_model)}</code>.</p>"
        "<p><code>gpt-realtime-2</code> uses <code>reasoning.effort=low</code>.</p>"
        "</div>"
    )


def _textarea(name: str, value: str = "", rows: int = 2) -> str:
    return (
        f'<label>{escape(name)}<br>'
        f'<textarea name="{escape(name)}" rows="{rows}" style="width: 720px; max-width: 100%;">{escape(value or "")}</textarea></label>'
    )


def _lines_from_json(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, list):
        return "\n".join(str(item) for item in parsed)
    return value


def _lines_to_list(value: str) -> list[str]:
    return [line.strip() for line in value.replace("\r", "\n").split("\n") if line.strip()]


def _phone_live_table(tenant_id: int, phones: list[dict]) -> str:
    if not phones:
        return "<p>No phone numbers yet.</p>"
    rows = []
    for phone in phones:
        checked = " checked" if phone.get("accepts_live_calls") else ""
        rows.append(
            "<tr>"
            f"<td>{escape(str(phone.get('id') or ''))}</td>"
            f"<td>{escape(phone.get('twilio_number') or '')}</td>"
            f"<td>{escape(phone.get('label') or '')}</td>"
            f"<td>{escape(phone.get('purpose') or '')}</td>"
            f"<td>{escape('yes' if phone.get('active') else 'no')}</td>"
            "<td>"
            f'<form method="post" action="/admin/tenants/{tenant_id}/phones/{phone["id"]}/live">'
            f'<label><input type="checkbox" name="accepts_live_calls" value="1"{checked}> accepts live calls</label> '
            '<button type="submit">Save</button>'
            "</form>"
            "</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>id</th><th>twilio_number</th><th>label</th><th>purpose</th><th>active</th><th>live switch</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
    )


def _prompt_history_table(tenant_id: int, profiles: list[dict]) -> str:
    if not profiles:
        return "<p>No prompt versions yet.</p>"
    rows = []
    for profile in profiles:
        activate = "active"
        if not profile.get("is_active"):
            activate = (
                f'<form method="post" action="/admin/tenants/{tenant_id}/prompt/{profile["id"]}/activate">'
                '<button type="submit">Activate</button>'
                "</form>"
            )
        rows.append(
            "<tr>"
            f"<td>{escape(str(profile.get('id') or ''))}</td>"
            f"<td>{escape(str(profile.get('version') or ''))}</td>"
            f"<td>{escape(profile.get('label') or '')}</td>"
            f"<td>{escape(profile.get('business_name') or '')}</td>"
            f"<td>{escape(profile.get('greeting') or '')}</td>"
            f"<td>{escape(profile.get('realtime_model') or 'env/default')}</td>"
            f"<td>{escape('yes' if profile.get('is_active') else 'no')}</td>"
            f"<td>{activate}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>id</th><th>version</th><th>label</th><th>business_name</th><th>greeting</th><th>model</th><th>active</th><th>action</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
    )


def _question_table(questions: list[dict], conditional: bool = False) -> str:
    if not questions:
        return "<p>No questions configured.</p>"
    columns = ["key", "label", "question_text", "collection_mode", "include_in_sms", "include_in_admin", "active"]
    if conditional:
        columns = ["key", "label", "condition_type", "condition_keywords", "question_text", "collection_mode", "include_in_sms", "active"]
    rows = []
    for question in questions:
        cells = []
        for column in columns:
            value = question.get(column)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            elif isinstance(value, bool):
                value = "yes" if value else "no"
            cells.append(f"<td>{escape(str(value or ''))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _notification_policy_preview(policy: dict, intake_policy: Optional[dict]) -> str:
    sample = {
        "name": "Sam",
        "callback": "732-789-0675",
        "issue": "Kitchen sink leaking underneath",
        "urgency": "Water still coming out but shut off",
        "address": "6100 West 120th Street",
        "priority": "normal",
        "priority_reason": "No emergency or urgent keywords matched.",
        "extra_fields": {"property_role": "homeowner", "additional_notes": "gate code 1234"},
    }
    emergency_sample = dict(sample)
    emergency_sample["urgency"] = "Water is still coming out and caller cannot shut it off"
    emergency_sample["priority"] = "emergency"
    emergency_sample["priority_reason"] = "Matched emergency keyword: water is still coming out"
    return "\n".join(
        [
            "<h3>SMS Preview</h3>",
            "<h4>Normal</h4>",
            f"<pre>{escape(build_sms_body(sample, '732-789-0675', intake_policy, policy))}</pre>",
            "<h4>Emergency</h4>",
            f"<pre>{escape(build_sms_body(emergency_sample, '732-789-0675', intake_policy, policy))}</pre>",
        ]
    )


def _json_list_from_text(value: str) -> Optional[list]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.4; }}
    table {{ border-collapse: collapse; margin-bottom: 32px; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    code, pre {{ background: #f5f5f5; padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; }}
    nav a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <nav><a href="/admin/review">Review queue</a><a href="/admin/metrics">Metrics</a><a href="/admin/leads">Dashboard</a><a href="/admin/tenants">Tenants</a></nav>
  <h1>{escape(title)}</h1>
  {body}
</body>
</html>
"""
    return HTMLResponse(html, status_code=status_code)


def _admin_not_found(title: str, message: str) -> HTMLResponse:
    body = "\n".join(
        [
            f"<p>{escape(message)}</p>",
            '<p><a href="/admin/tenants">Back to tenants</a></p>',
        ]
    )
    return _page(title, body, status_code=404)


def _parse_tenant_id(value: str) -> Optional[int]:
    try:
        tenant_id = int(value)
    except (TypeError, ValueError):
        return None
    return tenant_id if tenant_id > 0 else None


def _parse_optional_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_form_bool(value) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _mask_phone(value: str) -> str:
    text = str(value or "")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 4:
        return text
    return f"...{digits[-4:]}"


def _review_status_options(selected: str = "unreviewed") -> str:
    options = [
        ("unreviewed", "Unreviewed"),
        ("good", "Good"),
        ("needs_review", "Needs review"),
        ("bad", "Bad"),
        ("follow_up_needed", "Follow-up needed"),
    ]
    return _select("review_status", options, selected or "unreviewed")


def _review_tag_checkboxes(selected: list[str]) -> str:
    selected_set = set(selected or [])
    rows = []
    for tag in sorted(repository.REVIEW_TAGS):
        checked = " checked" if tag in selected_set else ""
        rows.append(f'<label style="display:inline-block; min-width: 220px;"><input type="checkbox" name="review_tags" value="{escape(tag)}"{checked}> {escape(tag)}</label>')
    return "<div>" + "".join(rows) + "</div>"


def _review_queue_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No calls match this review queue.</p>"
    rendered = []
    for row in rows:
        call_sid = row.get("call_sid") or ""
        rendered.append(
            "<tr>"
            f"<td>{escape(str(row.get('started_at') or ''))}</td>"
            f"<td>{escape(row.get('tenant_name') or '')}</td>"
            f'<td><a href="/admin/calls/{escape(call_sid)}">{escape(call_sid)}</a></td>'
            f"<td>{escape(_mask_phone(row.get('from_number') or ''))}</td>"
            f"<td>{escape(row.get('status') or '')}</td>"
            f"<td>{escape(row.get('review_status') or '')}</td>"
            f"<td>{escape(row.get('priority') or '')}</td>"
            f"<td>{escape(row.get('lead_created') or '')}</td>"
            f"<td>{escape(row.get('notification_status') or '')}</td>"
            f"<td>{escape(row.get('realtime_model') or '')}</td>"
            f"<td>{escape(str(row.get('prompt_version_id') or ''))}</td>"
            f"<td>{escape(row.get('review_tags_text') or '')}</td>"
            f"<td>{escape(row.get('attention_reasons_text') or '')}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>time</th><th>tenant</th><th>call_sid</th><th>from</th><th>status</th>"
        "<th>review</th><th>priority</th><th>lead</th><th>notification</th>"
        "<th>model</th><th>prompt</th><th>tags</th><th>why queued</th>"
        "</tr></thead><tbody>"
        f"{''.join(rendered)}"
        "</tbody></table>"
    )


def _call_review_form(call_sid: str, review: dict) -> str:
    return "\n".join(
        [
            f'<form method="post" action="/admin/calls/{escape(call_sid)}/review">',
            _review_status_options(review.get("review_status") or "unreviewed"),
            "<h4>Tags</h4>",
            _review_tag_checkboxes(review.get("review_tags") or []),
            "<br>",
            _textarea("internal_notes", review.get("internal_notes") or "", rows=5),
            "<br><br>",
            '<button type="submit">Save Review</button>',
            "</form>",
        ]
    )


def _feedback_form(call_sid: str) -> str:
    return "\n".join(
        [
            f'<form method="post" action="/admin/calls/{escape(call_sid)}/feedback">',
            _select("feedback_source", [("internal", "Internal"), ("plumber", "Plumber"), ("caller", "Caller")], "internal"),
            "<br><br>",
            _select(
                "action_needed",
                [
                    ("none", "None"),
                    ("prompt_update", "Prompt update"),
                    ("intake_policy_update", "Intake policy update"),
                    ("notification_policy_update", "Notification policy update"),
                    ("bug_fix", "Bug fix"),
                    ("conversation_polish", "Conversation polish"),
                    ("follow_up_with_customer", "Follow up with customer"),
                ],
                "none",
            ),
            "<br><br>",
            _checkbox("resolved", False),
            "<br><br>",
            _textarea("feedback_text", "", rows=4),
            "<br><br>",
            '<button type="submit">Add Feedback</button>',
            "</form>",
        ]
    )


def _feedback_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No feedback yet.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('created_at') or ''))}</td>"
            f"<td>{escape(row.get('feedback_source') or '')}</td>"
            f"<td>{escape(row.get('feedback_text') or '')}</td>"
            f"<td>{escape(row.get('action_needed') or '')}</td>"
            f"<td>{escape('yes' if row.get('resolved') else 'no')}</td>"
            f"<td>{escape(str(row.get('resolved_at') or ''))}</td>"
            f"<td>{row.get('action') or ''}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>created_at</th><th>source</th><th>feedback</th><th>action_needed</th>"
        "<th>resolved</th><th>resolved_at</th><th>action</th>"
        "</tr></thead><tbody>"
        f"{''.join(body)}"
        "</tbody></table>"
    )


def _important_events(events: list[dict]) -> list[dict]:
    important = {
        "validation_failed",
        "tool_args_parse_failed",
        "tool_call_incomplete",
        "submit_service_request_attempt",
        "lead_created",
        "priority_classified",
        "sms_sent",
        "sms_failed",
        "openai_realtime_error",
        "twilio_stream_stopped",
        "twilio_websocket_disconnected",
        "media_stream_done",
        "assistant_transcript",
        "caller_transcript",
    }
    return [event for event in events if event.get("event_type") in important]


def create_admin_router(settings: Settings) -> APIRouter:
    router = APIRouter(dependencies=[Depends(_require_admin(settings))])

    @router.get("/admin", response_class=HTMLResponse)
    async def admin_home():
        return await admin_leads()

    @router.get("/admin/leads", response_class=HTMLResponse)
    async def admin_leads():
        calls = repository.list_recent_calls()
        leads = repository.list_recent_leads()
        notifications = repository.list_recent_notifications()
        events = repository.list_recent_call_events()
        body = "\n".join(
            [
                "<h2>Recent Calls</h2>",
                _render_table(calls, ["tenant_id", "prompt_version_id", "realtime_model", "realtime_reasoning_effort", "call_sid", "from_number", "to_number", "stream_sid", "status", "started_at", "ended_at"]),
                "<h2>Recent Leads</h2>",
                _render_table(leads, ["id", "tenant_id", "call_sid", "name", "callback", "address", "issue", "urgency", "priority", "priority_reason", "extra_fields", "status", "created_at"]),
                "<h2>Recent Notifications</h2>",
                _render_table(notifications, ["id", "tenant_id", "lead_id", "channel", "to_number", "recipient_type", "status", "attempt_number", "provider_message_sid", "error", "created_at", "sent_at"]),
                "<h2>Recent Call Events</h2>",
                _render_table(events, ["id", "tenant_id", "call_sid", "event_type", "payload_json", "created_at"]),
            ]
        )
        return _page("Plumber Receptionist Admin", body)

    @router.get("/admin/review", response_class=HTMLResponse)
    async def admin_review_queue(
        tenant_id: str = "",
        review_status: str = "",
        tag: str = "",
        has_lead: str = "",
        notification_status: str = "",
        priority: str = "",
        realtime_model: str = "",
    ):
        tenants = repository.list_tenants()
        parsed_tenant_id = _parse_optional_int(tenant_id)
        rows = repository.list_call_review_queue(
            tenant_id=parsed_tenant_id,
            review_status=review_status,
            tag=tag,
            has_lead=has_lead,
            notification_status=notification_status,
            priority=priority,
            realtime_model=realtime_model,
        )
        tenant_options = [("", "all tenants")] + [(str(tenant["id"]), tenant["name"]) for tenant in tenants]
        tag_options = [("", "any tag")] + [(tag_value, tag_value) for tag_value in sorted(repository.REVIEW_TAGS)]
        body = "\n".join(
            [
                "<p>Use this queue to review real pilot calls, tag problems, and decide what to tune next.</p>",
                '<form method="get" action="/admin/review">',
                _select("tenant_id", tenant_options, str(parsed_tenant_id or "")),
                " ",
                _select("review_status", [("", "attention queue"), ("unreviewed", "unreviewed"), ("good", "good"), ("needs_review", "needs_review"), ("bad", "bad"), ("follow_up_needed", "follow_up_needed")], review_status),
                " ",
                _select("tag", tag_options, tag),
                " ",
                _select("has_lead", [("", "any lead status"), ("yes", "has lead"), ("no", "no lead")], has_lead),
                " ",
                _select("notification_status", [("", "any notification"), ("none", "none"), ("sent", "sent"), ("failed", "failed"), ("pending", "pending")], notification_status),
                " ",
                _select("priority", [("", "any priority"), ("normal", "normal"), ("urgent", "urgent"), ("emergency", "emergency")], priority),
                " ",
                _select("realtime_model", [("", "any model"), ("gpt-realtime-1.5", "gpt-realtime-1.5"), ("gpt-realtime-2", "gpt-realtime-2")], realtime_model),
                ' <button type="submit">Filter</button>',
                "</form>",
                "<h2>Calls Needing Attention</h2>",
                _review_queue_table(rows),
            ]
        )
        return _page("Pilot Review Queue", body)

    @router.get("/admin/metrics", response_class=HTMLResponse)
    async def admin_metrics(tenant_id: str = ""):
        parsed_tenant_id = _parse_optional_int(tenant_id)
        metrics = repository.pilot_metrics(tenant_id=parsed_tenant_id)
        tenants = repository.list_tenants()
        tenant_options = [("", "all tenants")] + [(str(tenant["id"]), tenant["name"]) for tenant in tenants]
        summary_row = {key: value for key, value in metrics.items() if key != "by_tenant"}
        body = "\n".join(
            [
                '<form method="get" action="/admin/metrics">',
                _select("tenant_id", tenant_options, str(parsed_tenant_id or "")),
                ' <button type="submit">Filter</button>',
                "</form>",
                "<h2>Pilot Metrics</h2>",
                _render_table([summary_row], list(summary_row.keys())),
                "<h2>By Tenant</h2>",
                _render_table(metrics.get("by_tenant") or [], ["tenant_id", "tenant", "calls", "leads"]),
            ]
        )
        return _page("Pilot Metrics", body)

    @router.get("/admin/tenants", response_class=HTMLResponse)
    async def admin_tenants():
        tenants = repository.list_tenants()
        phones = repository.list_tenant_phone_numbers()
        body = "\n".join(
            [
                "<h2>Tenants</h2>",
                _tenant_actions_table(tenants),
                "<h2>Tenant Phone Numbers</h2>",
                _render_table(phones, ["id", "tenant_id", "twilio_number", "label", "active", "created_at"]),
                "<h2>Create Tenant</h2>",
                '<form method="post" action="/admin/tenants">',
                _input("name", placeholder="Acme Plumbing"),
                "<br><br>",
                _input("slug", placeholder="acme-plumbing"),
                "<br><br>",
                _input("business_name", placeholder="Acme Plumbing"),
                "<br><br>",
                _textarea("greeting", "Plumbing office, what's going on?"),
                "<br><br>",
                _input("notification_sms_number", placeholder="+15551234567"),
                "<br><br>",
                _input("backup_notification_sms_number", placeholder="+15557654321"),
                "<br><br>",
                "<button type=\"submit\">Create Tenant</button>",
                "</form>",
            ]
        )
        return _page("Tenants", body)

    @router.post("/admin/tenants")
    async def admin_create_tenant(request: Request):
        form = await request.form()
        name = str(form.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="Tenant name is required")
        slug = str(form.get("slug", "")).strip() or _slugify(name)
        tenant = repository.create_tenant(
            name=name,
            slug=slug,
            business_name=str(form.get("business_name", "")).strip() or name,
            greeting=str(form.get("greeting", "")).strip() or "Plumbing office, what's going on?",
            notification_sms_number=str(form.get("notification_sms_number", "")).strip(),
            backup_notification_sms_number=str(form.get("backup_notification_sms_number", "")).strip(),
        )
        return RedirectResponse(f"/admin/tenants/{tenant['id']}", status_code=303)

    @router.get("/admin/tenants/{tenant_id}", response_class=HTMLResponse)
    async def admin_tenant_detail(tenant_id: str):
        parsed_tenant_id = _parse_tenant_id(tenant_id)
        if parsed_tenant_id is None:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        tenant_id = parsed_tenant_id
        tenant = repository.get_tenant(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        phones = repository.list_tenant_phone_numbers(tenant_id)
        telephony_profile = repository.get_telephony_profile(tenant_id)
        calls = repository.list_recent_calls(tenant_id=tenant_id)
        leads = repository.list_recent_leads(tenant_id=tenant_id)
        notifications = repository.list_recent_notifications(tenant_id=tenant_id)
        events = repository.list_recent_call_events(tenant_id=tenant_id)
        tenant_summary = [
            {
                "id": tenant["id"],
                "name": tenant["name"],
                "slug": tenant["slug"],
                "status": tenant["status"],
                "business_name": tenant.get("business_name") or "",
                "notification_sms_number": tenant.get("notification_sms_number") or "",
                "backup_notification_sms_number": tenant.get("backup_notification_sms_number") or "",
                "public_business_number": telephony_profile.get("public_business_number") if telephony_profile else "",
                "ai_ingress_twilio_number": telephony_profile.get("ai_ingress_twilio_number") if telephony_profile else "",
                "forwarding_setup_status": telephony_profile.get("forwarding_setup_status") if telephony_profile else "",
                "test_mode_enabled": telephony_profile.get("test_mode_enabled") if telephony_profile else False,
            }
        ]
        allowed_test_callers = _lines_from_json(telephony_profile.get("allowed_test_callers_json") if telephony_profile else "")
        body = "\n".join(
            [
                f"<h2>{escape(tenant['name'])}</h2>",
                '<p><a href="/admin/tenants">Back to tenants</a></p>',
                f'<p><strong><a href="/admin/tenants/{tenant_id}/prompt">Prompt/persona settings</a></strong> | <strong><a href="/admin/tenants/{tenant_id}/intake-policy">Intake policy</a></strong> | <strong><a href="/admin/tenants/{tenant_id}/notification-policy">Notification policy</a></strong></p>',
                "<h3>Tenant Summary</h3>",
                _render_table(
                    tenant_summary,
                    ["id", "name", "slug", "status", "business_name", "notification_sms_number", "backup_notification_sms_number"],
                ),
                "<h3>Onboarding / Telephony</h3>",
                "<p>Customer keeps their Google Maps number. They forward missed/after-hours calls to this AI forwarding number. Live calls are blocked until Go Live is enabled and the phone live switch is on.</p>",
                _render_table(
                    tenant_summary,
                    ["status", "public_business_number", "ai_ingress_twilio_number", "forwarding_setup_status", "test_mode_enabled"],
                ),
                f'<form method="post" action="/admin/tenants/{tenant_id}/telephony">',
                _input("status", tenant.get("status") or "onboarding", "draft/onboarding/testing/live/paused/archived"),
                "<br><br>",
                _input("public_business_number", telephony_profile.get("public_business_number") if telephony_profile else "", "+15551234567"),
                "<br><br>",
                _input("ai_ingress_twilio_number", telephony_profile.get("ai_ingress_twilio_number") if telephony_profile else "", "+15557654321"),
                "<br><br>",
                _input("forwarding_setup_status", telephony_profile.get("forwarding_setup_status") if telephony_profile else "not_started", "not_started/instructions_sent/customer_configured/verified"),
                "<br><br>",
                _checkbox("test_mode_enabled", bool(telephony_profile.get("test_mode_enabled") if telephony_profile else False)),
                "<br><br>",
                _textarea("allowed_test_callers", allowed_test_callers, rows=4),
                "<br><br>",
                _textarea("notes", telephony_profile.get("notes") if telephony_profile else "", rows=4),
                "<br><br>",
                "<button type=\"submit\">Save Onboarding / Telephony</button>",
                "</form>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/go-live" style="display:inline; margin-right: 12px;"><button type="submit">Go Live</button></form>',
                f'<form method="post" action="/admin/tenants/{tenant_id}/pause" style="display:inline;"><button type="submit">Pause</button></form>',
                "<h3>Settings</h3>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/settings">',
                _input("business_name", tenant.get("business_name") or tenant.get("name") or ""),
                "<br><br>",
                _textarea("greeting", tenant.get("greeting") or ""),
                "<br><br>",
                _input("notification_sms_number", tenant.get("notification_sms_number") or ""),
                "<br><br>",
                _input("backup_notification_sms_number", tenant.get("backup_notification_sms_number") or ""),
                "<br><br>",
                _input("status", tenant.get("status") or "active"),
                "<br><br>",
                "<button type=\"submit\">Save Settings</button>",
                "</form>",
                "<h3>Phone Numbers</h3>",
                _phone_live_table(tenant_id, phones),
                f'<form method="post" action="/admin/tenants/{tenant_id}/phones">',
                _input("twilio_number", placeholder="+15551234567"),
                "<br><br>",
                _input("label", placeholder="Main line"),
                "<br><br>",
                _input("purpose", "ai_forwarding", "ai_forwarding"),
                "<br><br>",
                "<button type=\"submit\">Add Phone Number</button>",
                "</form>",
                "<h3>Recent Calls</h3>",
                _render_table(calls, ["call_sid", "prompt_version_id", "realtime_model", "realtime_reasoning_effort", "from_number", "to_number", "status", "started_at", "ended_at"]),
                "<h3>Recent Leads</h3>",
                _render_table(leads, ["id", "call_sid", "name", "callback", "address", "issue", "urgency", "priority", "priority_reason", "extra_fields", "status", "created_at"]),
                "<h3>Recent Notifications</h3>",
                _render_table(notifications, ["id", "lead_id", "channel", "to_number", "recipient_type", "status", "attempt_number", "error", "created_at", "sent_at"]),
                "<h3>Recent Events</h3>",
                _render_table(events, ["id", "call_sid", "event_type", "payload_json", "created_at"]),
            ]
        )
        return _page(f"Tenant {tenant_id}", body)

    @router.get("/admin/tenants/{tenant_id}/intake-policy", response_class=HTMLResponse)
    async def admin_tenant_intake_policy(tenant_id: str):
        parsed_tenant_id = _parse_tenant_id(tenant_id)
        if parsed_tenant_id is None:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        tenant_id = parsed_tenant_id
        tenant = repository.get_tenant(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        policy = repository.get_intake_policy(tenant_id)
        active_profile = repository.get_active_prompt_profile(tenant_id)
        preview = PromptBuilder().build("913-555-0123", tenant=tenant, profile=active_profile, intake_policy=policy)
        body = "\n".join(
            [
                f"<h2>{escape(tenant['name'])} Intake Policy</h2>",
                f'<p><a href="/admin/tenants/{tenant_id}">Back to tenant detail</a> | <a href="/admin/tenants/{tenant_id}/prompt">Prompt/persona settings</a> | <a href="/admin/tenants">Back to tenants</a></p>',
                "<h3>Core workflow is locked</h3>",
                "<p>Required core fields stay: issue, urgency, address, callback, name. First name is enough; last name is not required. Intake policy can only add tenant-specific questions.</p>",
                "<p><strong>Collection modes:</strong> Required blocks submit until a useful answer is collected. Ask once means the AI will ask this question before submitting, but the caller can decline or say unknown. Passive means the AI may collect it only if it comes up naturally.</p>",
                "<h3>Current Extra Questions</h3>",
                _question_table(extra_questions(policy, include_inactive=True)),
                "<h3>Current Conditional Questions</h3>",
                _question_table(conditional_questions(policy, include_inactive=True), conditional=True),
                "<h3>Add Extra Question</h3>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/intake-policy/extra">',
                _input("key", placeholder="property_role"),
                "<br><br>",
                _input("label", placeholder="Homeowner or renter"),
                "<br><br>",
                _textarea("question_text", "Are you the homeowner or are you renting?", rows=2),
                "<br><br>",
                _select("collection_mode", [("ask_once", "Ask once"), ("required", "Required"), ("passive", "Passive")], "ask_once"),
                "<br><br>",
                _checkbox("include_in_sms", True),
                "<br><br>",
                _checkbox("include_in_admin", True),
                "<br><br>",
                _checkbox("active", True),
                "<br><br>",
                "<button type=\"submit\">Add Extra Question</button>",
                "</form>",
                "<h3>Add Conditional Question</h3>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/intake-policy/conditional">',
                _input("key", placeholder="can_shut_water_off"),
                "<br><br>",
                _input("label", placeholder="Can shut water off"),
                "<br><br>",
                _input("condition_type", "urgency_contains", "always/urgency_contains/issue_contains"),
                "<br><br>",
                _textarea("condition_keywords", "active leak\nwater is still running\nflooding", rows=4),
                "<br><br>",
                _textarea("question_text", "Can you shut the water off there?", rows=2),
                "<br><br>",
                _select("collection_mode", [("ask_once", "Ask once"), ("required", "Required"), ("passive", "Passive")], "ask_once"),
                "<br><br>",
                _checkbox("include_in_sms", True),
                "<br><br>",
                _checkbox("active", True),
                "<br><br>",
                "<button type=\"submit\">Add Conditional Question</button>",
                "</form>",
                "<h3>Edit Policy JSON</h3>",
                "<p>This is the fallback editor for changing, deactivating, or removing configured questions.</p>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/intake-policy">',
                _checkbox("enabled", bool(policy.get("enabled") if policy else True)),
                "<br><br>",
                _textarea("extra_questions_json", policy_to_json(policy, "extra_questions_json"), rows=10),
                "<br><br>",
                _textarea("conditional_questions_json", policy_to_json(policy, "conditional_questions_json"), rows=10),
                "<br><br>",
                _textarea("sms_include_extra_fields", _lines_from_json(policy.get("sms_include_extra_fields_json") if policy else ""), rows=4),
                "<br><br>",
                _textarea("admin_display_fields", _lines_from_json(policy.get("admin_display_fields_json") if policy else ""), rows=4),
                "<br><br>",
                _textarea("notes", policy.get("notes") if policy else "", rows=4),
                "<br><br>",
                "<button type=\"submit\">Save Intake Policy</button>",
                "</form>",
                "<h3>Generated Prompt Preview</h3>",
                f"<pre>{escape(preview)}</pre>",
            ]
        )
        return _page(f"Tenant {tenant_id} Intake Policy", body)

    @router.post("/admin/tenants/{tenant_id}/intake-policy")
    async def admin_update_intake_policy(tenant_id: int, request: Request):
        tenant = repository.get_tenant(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        form = await request.form()
        extra_question_values = _json_list_from_text(str(form.get("extra_questions_json", "[]")))
        conditional_question_values = _json_list_from_text(str(form.get("conditional_questions_json", "[]")))
        if extra_question_values is None or conditional_question_values is None:
            return _page(
                "Invalid Intake Policy JSON",
                f'<p>Extra questions and conditional questions must be valid JSON arrays.</p><p><a href="/admin/tenants/{tenant_id}/intake-policy">Back to intake policy</a></p>',
                status_code=400,
            )
        repository.update_intake_policy(
            tenant_id,
            enabled=_parse_form_bool(form.get("enabled")),
            extra_questions=extra_question_values,
            conditional_questions=conditional_question_values,
            sms_include_extra_fields=_lines_to_list(str(form.get("sms_include_extra_fields", ""))),
            admin_display_fields=_lines_to_list(str(form.get("admin_display_fields", ""))),
            notes=str(form.get("notes", "")).strip(),
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}/intake-policy", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/intake-policy/extra")
    async def admin_add_intake_extra_question(tenant_id: int, request: Request):
        policy = repository.get_intake_policy(tenant_id)
        if not policy:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        form = await request.form()
        extra_question_values = _json_list_from_text(policy.get("extra_questions_json") or "[]") or []
        conditional_question_values = _json_list_from_text(policy.get("conditional_questions_json") or "[]") or []
        extra_question_values.append(
            {
                "key": str(form.get("key", "")).strip(),
                "label": str(form.get("label", "")).strip(),
                "question_text": str(form.get("question_text", "")).strip(),
                "collection_mode": str(form.get("collection_mode", "ask_once")).strip() or "ask_once",
                "include_in_sms": _parse_form_bool(form.get("include_in_sms")),
                "include_in_admin": _parse_form_bool(form.get("include_in_admin")),
                "active": _parse_form_bool(form.get("active")),
            }
        )
        repository.update_intake_policy(
            tenant_id,
            enabled=bool(policy.get("enabled")),
            extra_questions=extra_question_values,
            conditional_questions=conditional_question_values,
            sms_include_extra_fields=_lines_to_list(_lines_from_json(policy.get("sms_include_extra_fields_json") or "")),
            admin_display_fields=_lines_to_list(_lines_from_json(policy.get("admin_display_fields_json") or "")),
            notes=policy.get("notes") or "",
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}/intake-policy", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/intake-policy/conditional")
    async def admin_add_intake_conditional_question(tenant_id: int, request: Request):
        policy = repository.get_intake_policy(tenant_id)
        if not policy:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        form = await request.form()
        extra_question_values = _json_list_from_text(policy.get("extra_questions_json") or "[]") or []
        conditional_question_values = _json_list_from_text(policy.get("conditional_questions_json") or "[]") or []
        conditional_question_values.append(
            {
                "key": str(form.get("key", "")).strip(),
                "label": str(form.get("label", "")).strip(),
                "condition_type": str(form.get("condition_type", "")).strip(),
                "condition_keywords": _lines_to_list(str(form.get("condition_keywords", ""))),
                "question_text": str(form.get("question_text", "")).strip(),
                "collection_mode": str(form.get("collection_mode", "ask_once")).strip() or "ask_once",
                "include_in_sms": _parse_form_bool(form.get("include_in_sms")),
                "include_in_admin": True,
                "active": _parse_form_bool(form.get("active")),
            }
        )
        repository.update_intake_policy(
            tenant_id,
            enabled=bool(policy.get("enabled")),
            extra_questions=extra_question_values,
            conditional_questions=conditional_question_values,
            sms_include_extra_fields=_lines_to_list(_lines_from_json(policy.get("sms_include_extra_fields_json") or "")),
            admin_display_fields=_lines_to_list(_lines_from_json(policy.get("admin_display_fields_json") or "")),
            notes=policy.get("notes") or "",
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}/intake-policy", status_code=303)

    @router.get("/admin/tenants/{tenant_id}/notification-policy", response_class=HTMLResponse)
    async def admin_tenant_notification_policy(tenant_id: str):
        parsed_tenant_id = _parse_tenant_id(tenant_id)
        if parsed_tenant_id is None:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        tenant_id = parsed_tenant_id
        tenant = repository.get_tenant(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        policy = repository.get_notification_policy(tenant_id)
        intake_policy = repository.get_intake_policy(tenant_id)
        notifications = repository.list_recent_notifications(tenant_id=tenant_id)
        failed_notifications = repository.list_failed_notifications(tenant_id=tenant_id)
        body = "\n".join(
            [
                f"<h2>{escape(tenant['name'])} Notification Policy</h2>",
                f'<p><a href="/admin/tenants/{tenant_id}">Back to tenant detail</a> | <a href="/admin/tenants">Back to tenants</a></p>',
                "<p>Emergency leads are saved first, then notifications are attempted. If no emergency recipient is configured, emergency leads fall back to normal recipients and are clearly marked emergency.</p>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/notification-policy">',
                "<h3>Recipients</h3>",
                _textarea("normal_sms_recipients", _lines_from_json(policy.get("normal_sms_recipients_json") if policy else ""), rows=4),
                "<br><br>",
                _textarea("emergency_sms_recipients", _lines_from_json(policy.get("emergency_sms_recipients_json") if policy else ""), rows=4),
                "<br><br>",
                _textarea("backup_sms_recipients", _lines_from_json(policy.get("backup_sms_recipients_json") if policy else ""), rows=4),
                "<br><br>",
                _checkbox("send_normal_leads", bool(policy.get("send_normal_leads") if policy else True)),
                "<br><br>",
                _checkbox("send_emergency_leads", bool(policy.get("send_emergency_leads") if policy else True)),
                "<br><br>",
                _checkbox("include_extra_fields", bool(policy.get("include_extra_fields") if policy else True)),
                "<br><br>",
                _checkbox("include_additional_notes", bool(policy.get("include_additional_notes") if policy else True)),
                "<h3>Emergency Classification</h3>",
                _textarea("emergency_keywords", _lines_from_json(policy.get("emergency_keywords_json") if policy else ""), rows=8),
                "<br><br>",
                _textarea("emergency_rules_json", policy.get("emergency_rules_json") if policy else "[]", rows=4),
                "<br><br>",
                _textarea("notes", policy.get("notes") if policy else "", rows=4),
                "<br><br>",
                "<button type=\"submit\">Save Notification Policy</button>",
                "</form>",
                _notification_policy_preview(policy or {}, intake_policy),
                "<h3>Recent Notifications</h3>",
                _render_table(notifications, ["id", "lead_id", "to_number", "recipient_type", "status", "attempt_number", "provider_message_sid", "error", "created_at", "sent_at"]),
                "<h3>Failed Notifications</h3>",
                _render_table(failed_notifications, ["id", "lead_id", "to_number", "recipient_type", "status", "attempt_number", "error", "created_at"]),
                "<p>Retry tool: run <code>python scripts/retry_failed_notifications.py --tenant-id "
                f"{tenant_id}</code> from the repo with production environment variables loaded.</p>",
            ]
        )
        return _page(f"Tenant {tenant_id} Notification Policy", body)

    @router.post("/admin/tenants/{tenant_id}/notification-policy")
    async def admin_update_notification_policy(tenant_id: int, request: Request):
        tenant = repository.get_tenant(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        form = await request.form()
        emergency_rules = _json_list_from_text(str(form.get("emergency_rules_json", "[]")))
        if emergency_rules is None:
            return _page(
                "Invalid Notification Policy JSON",
                f'<p>Emergency rules must be a valid JSON array.</p><p><a href="/admin/tenants/{tenant_id}/notification-policy">Back to notification policy</a></p>',
                status_code=400,
            )
        repository.update_notification_policy(
            tenant_id,
            normal_sms_recipients=_lines_to_list(str(form.get("normal_sms_recipients", ""))),
            emergency_sms_recipients=_lines_to_list(str(form.get("emergency_sms_recipients", ""))),
            backup_sms_recipients=_lines_to_list(str(form.get("backup_sms_recipients", ""))),
            send_normal_leads=_parse_form_bool(form.get("send_normal_leads")),
            send_emergency_leads=_parse_form_bool(form.get("send_emergency_leads")),
            include_extra_fields=_parse_form_bool(form.get("include_extra_fields")),
            include_additional_notes=_parse_form_bool(form.get("include_additional_notes")),
            emergency_keywords=_lines_to_list(str(form.get("emergency_keywords", ""))),
            emergency_rules=emergency_rules,
            notes=str(form.get("notes", "")).strip(),
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}/notification-policy", status_code=303)

    @router.get("/admin/tenants/{tenant_id}/prompt", response_class=HTMLResponse)
    async def admin_tenant_prompt(tenant_id: str):
        parsed_tenant_id = _parse_tenant_id(tenant_id)
        if parsed_tenant_id is None:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        tenant_id = parsed_tenant_id
        tenant = repository.get_tenant(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        active_profile = repository.get_active_prompt_profile(tenant_id)
        intake_policy = repository.get_intake_policy(tenant_id)
        profiles = repository.list_prompt_profiles(tenant_id)
        preview = PromptBuilder().build("913-555-0123", tenant=tenant, profile=active_profile, intake_policy=intake_policy) if active_profile else ""
        active_realtime_model = effective_realtime_model(active_profile, settings)
        active_reasoning_effort = realtime_reasoning_effort(active_realtime_model) or "not used"
        active_prompt_row = dict(active_profile) if active_profile else {}
        if active_prompt_row:
            active_prompt_row["realtime_model"] = active_profile.get("realtime_model") or f"env/default ({active_realtime_model})"
            active_prompt_row["realtime_reasoning_effort"] = active_reasoning_effort
            active_prompt_row["supported_model_options"] = ", ".join(SUPPORTED_REALTIME_MODELS)
        body = "\n".join(
            [
                f"<h2>{escape(tenant['name'])} Prompt</h2>",
                f'<p><a href="/admin/tenants/{tenant_id}">Back to tenant detail</a> | <a href="/admin/tenants/{tenant_id}/intake-policy">Intake policy</a> | <a href="/admin/tenants">Back to tenants</a></p>',
                "<h3>Core workflow is locked</h3>",
                "<p>Style/persona settings can change wording, but required fields cannot be changed here. Required fields stay: issue, urgency, address, callback, name. First name is enough; last name is not required.</p>",
                "<h3>Active Prompt Version</h3>",
                _render_table([active_prompt_row] if active_prompt_row else [], ["id", "version", "label", "business_name", "greeting", "tone", "verbosity", "closing_line", "realtime_model", "realtime_reasoning_effort", "is_active"]),
                "<h3>Realtime Model</h3>",
                f"<p>Current active model: <strong>{escape(active_realtime_model)}</strong>. Reasoning effort: <strong>{escape(active_reasoning_effort)}</strong>. Realtime 2 uses low reasoning effort for voice latency testing.</p>",
                "<h3>Create New Prompt Version</h3>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/prompt">',
                _input("label", active_profile.get("label") if active_profile else "Updated prompt"),
                "<br><br>",
                _realtime_model_select(active_profile.get("realtime_model") if active_profile else "", settings.openai_realtime_model),
                "<br>",
                _input("business_name", active_profile.get("business_name") if active_profile else tenant.get("business_name") or tenant.get("name") or ""),
                "<br><br>",
                _textarea("greeting", active_profile.get("greeting") if active_profile else tenant.get("greeting") or "", rows=2),
                "<br><br>",
                _textarea("tone", active_profile.get("tone") if active_profile else "", rows=2),
                "<br><br>",
                _textarea("verbosity", active_profile.get("verbosity") if active_profile else "", rows=2),
                "<br><br>",
                _textarea("closing_line", active_profile.get("closing_line") if active_profile else "", rows=2),
                "<br><br>",
                _textarea("avoid_phrases", _lines_from_json(active_profile.get("avoid_phrases_json") if active_profile else ""), rows=5),
                "<br><br>",
                _textarea("preferred_terms", _lines_from_json(active_profile.get("preferred_terms_json") if active_profile else ""), rows=5),
                "<br><br>",
                _textarea("extra_instructions_text", active_profile.get("extra_instructions_text") if active_profile else "", rows=6),
                "<br><br>",
                "<button type=\"submit\">Create and Activate New Version</button>",
                "</form>",
                "<h3>Prompt Version History</h3>",
                _prompt_history_table(tenant_id, profiles),
                "<h3>Generated Prompt Preview</h3>",
                f"<pre>{escape(preview)}</pre>",
            ]
        )
        return _page(f"Tenant {tenant_id} Prompt", body)

    @router.post("/admin/tenants/{tenant_id}/prompt")
    async def admin_create_prompt_version(tenant_id: int, request: Request):
        form = await request.form()
        profile = repository.create_prompt_profile(
            tenant_id=tenant_id,
            label=str(form.get("label", "")).strip(),
            business_name=str(form.get("business_name", "")).strip(),
            greeting=str(form.get("greeting", "")).strip(),
            tone=str(form.get("tone", "")).strip(),
            verbosity=str(form.get("verbosity", "")).strip(),
            closing_line=str(form.get("closing_line", "")).strip(),
            avoid_phrases=_lines_to_list(str(form.get("avoid_phrases", ""))),
            preferred_terms=_lines_to_list(str(form.get("preferred_terms", ""))),
            extra_instructions_text=str(form.get("extra_instructions_text", "")).strip(),
            realtime_model=str(form.get("realtime_model", "")).strip(),
            activate=True,
        )
        if not profile:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        return RedirectResponse(f"/admin/tenants/{tenant_id}/prompt", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/prompt/{profile_id}/activate")
    async def admin_activate_prompt_version(tenant_id: int, profile_id: int):
        profile = repository.activate_prompt_profile(tenant_id, profile_id)
        if not profile:
            return _admin_not_found("Prompt Version Not Found", f"Prompt version {profile_id} was not found for tenant {tenant_id}.")
        return RedirectResponse(f"/admin/tenants/{tenant_id}/prompt", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/telephony")
    async def admin_update_telephony(tenant_id: int, request: Request):
        form = await request.form()
        tenant = repository.set_tenant_status(tenant_id, str(form.get("status", "onboarding")).strip())
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        repository.update_telephony_profile(
            tenant_id,
            public_business_number=str(form.get("public_business_number", "")).strip(),
            ai_ingress_twilio_number=str(form.get("ai_ingress_twilio_number", "")).strip(),
            forwarding_setup_status=str(form.get("forwarding_setup_status", "not_started")).strip(),
            test_mode_enabled=_parse_form_bool(form.get("test_mode_enabled")),
            allowed_test_callers=_lines_to_list(str(form.get("allowed_test_callers", ""))),
            notes=str(form.get("notes", "")).strip(),
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/go-live")
    async def admin_go_live(tenant_id: int):
        tenant = repository.set_tenant_live(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/pause")
    async def admin_pause_tenant(tenant_id: int):
        tenant = repository.set_tenant_paused(tenant_id)
        if not tenant:
            return _admin_not_found("Tenant Not Found", f"Tenant {tenant_id} was not found.")
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/phones/{phone_id}/live")
    async def admin_set_phone_live(tenant_id: int, phone_id: int, request: Request):
        form = await request.form()
        phone = repository.set_tenant_phone_live(
            tenant_id,
            phone_id,
            accepts_live_calls=_parse_form_bool(form.get("accepts_live_calls")),
        )
        if not phone:
            return _admin_not_found("Phone Not Found", f"Phone {phone_id} was not found for tenant {tenant_id}.")
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/settings")
    async def admin_update_tenant_settings(tenant_id: int, request: Request):
        form = await request.form()
        tenant = repository.update_tenant_settings(
            tenant_id,
            business_name=str(form.get("business_name", "")).strip(),
            greeting=str(form.get("greeting", "")).strip(),
            notification_sms_number=str(form.get("notification_sms_number", "")).strip(),
            backup_notification_sms_number=str(form.get("backup_notification_sms_number", "")).strip(),
            status=str(form.get("status", "active")).strip() or "active",
        )
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.post("/admin/tenants/{tenant_id}/phones")
    async def admin_add_tenant_phone(tenant_id: int, request: Request):
        form = await request.form()
        twilio_number = str(form.get("twilio_number", "")).strip()
        if not twilio_number:
            raise HTTPException(status_code=400, detail="Twilio phone number is required")
        repository.add_tenant_phone_number(
            tenant_id,
            twilio_number=twilio_number,
            label=str(form.get("label", "")).strip(),
            active=True,
            accepts_live_calls=False,
            purpose=str(form.get("purpose", "")).strip(),
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.post("/admin/calls/{call_sid}/review")
    async def admin_save_call_review(call_sid: str, request: Request):
        form = await request.form()
        review = repository.save_call_review(
            call_sid,
            review_status=str(form.get("review_status", "unreviewed")).strip(),
            review_tags=[str(tag) for tag in form.getlist("review_tags")],
            internal_notes=str(form.get("internal_notes", "")).strip(),
            reviewed_by="admin",
        )
        if not review:
            return _admin_not_found("Call Not Found", f"Call {call_sid} was not found.")
        return RedirectResponse(f"/admin/calls/{call_sid}", status_code=303)

    @router.post("/admin/calls/{call_sid}/feedback")
    async def admin_add_call_feedback(call_sid: str, request: Request):
        form = await request.form()
        feedback = repository.add_call_feedback(
            call_sid,
            feedback_source=str(form.get("feedback_source", "internal")).strip(),
            feedback_text=str(form.get("feedback_text", "")).strip(),
            action_needed=str(form.get("action_needed", "none")).strip(),
            resolved=_parse_form_bool(form.get("resolved")),
        )
        if not feedback:
            return _admin_not_found("Call Not Found", f"Call {call_sid} was not found.")
        return RedirectResponse(f"/admin/calls/{call_sid}", status_code=303)

    @router.post("/admin/feedback/{feedback_id}/resolve")
    async def admin_resolve_feedback(feedback_id: int, request: Request):
        form = await request.form()
        call_sid = str(form.get("call_sid", "")).strip()
        feedback = repository.set_call_feedback_resolved(feedback_id, True)
        if not feedback:
            return _admin_not_found("Feedback Not Found", f"Feedback {feedback_id} was not found.")
        return RedirectResponse(f"/admin/calls/{call_sid or feedback['call_sid']}", status_code=303)

    @router.post("/admin/leads/{lead_id}/review")
    async def admin_save_lead_review(lead_id: int, request: Request):
        form = await request.form()
        call_sid = str(form.get("call_sid", "")).strip()
        lead = repository.update_lead_review(
            lead_id,
            lead_quality=str(form.get("lead_quality", "unknown")).strip(),
            lead_notes=str(form.get("lead_notes", "")).strip(),
        )
        if not lead:
            return _admin_not_found("Lead Not Found", f"Lead {lead_id} was not found.")
        return RedirectResponse(f"/admin/calls/{call_sid or lead['call_sid']}", status_code=303)

    @router.get("/admin/calls/{call_sid}", response_class=HTMLResponse)
    async def admin_call_detail(call_sid: str):
        detail = repository.get_call_detail(call_sid)
        call_summary = detail.get("call")
        if not call_summary:
            return _admin_not_found("Call Not Found", f"Call {call_sid} was not found.")
        lifecycle_event_names = {
            "media_stream_started",
            "response_create_sent",
            "twilio_stream_stopped",
            "media_stream_stopped",
            "twilio_websocket_disconnected",
            "openai_reader_error",
            "openai_websocket_closed",
            "openai_realtime_error",
            "openai_reader_cancelled",
            "media_stream_exception",
            "media_stream_done",
            "hangup_scheduled",
            "hangup_schedule_blocked",
            "call_ended",
            "barge_in_detected",
            "caller_audio_forwarded_during_assistant",
            "function_args_done",
            "tool_args_output_item_partial_ignored",
            "tool_args_parse_failed",
            "tool_call_incomplete",
            "tool_call_duplicate_ignored",
            "response_create_delayed",
            "response_create_delay_flushed",
            "response_cancel_not_active_reconciled",
        }
        events = detail.get("events") or []
        lifecycle_events = [
            event for event in events if event.get("event_type") in lifecycle_event_names
        ]
        latest_done = next(
            (event for event in lifecycle_events if event.get("event_type") == "media_stream_done"),
            None,
        )
        review = detail.get("review") or {}
        feedback = detail.get("feedback") or []
        leads = detail.get("leads") or []
        notifications = detail.get("notifications") or []
        important_events = _important_events(events)
        transcript_events = [
            event for event in events if event.get("event_type") in {"assistant_transcript", "caller_transcript"}
        ]
        lead_review_forms = []
        for lead in leads:
            lead_review_forms.append(
                "\n".join(
                    [
                        f'<form method="post" action="/admin/leads/{lead["id"]}/review">',
                        f'<input type="hidden" name="call_sid" value="{escape(call_sid)}">',
                        _select("lead_quality", [("unknown", "unknown"), ("good", "good"), ("incomplete", "incomplete"), ("wrong_info", "wrong_info"), ("duplicate", "duplicate"), ("test", "test")], lead.get("lead_quality") or "unknown"),
                        "<br><br>",
                        _textarea("lead_notes", lead.get("lead_notes") or "", rows=3),
                        "<br><br>",
                        '<button type="submit">Save Lead Review</button>',
                        "</form>",
                    ]
                )
            )
        feedback_rows = []
        for item in feedback:
            resolve = ""
            if not item.get("resolved"):
                resolve = (
                    f'<form method="post" action="/admin/feedback/{item["id"]}/resolve">'
                    f'<input type="hidden" name="call_sid" value="{escape(call_sid)}">'
                    '<button type="submit">Mark resolved</button>'
                    "</form>"
                )
            feedback_rows.append({**item, "action": resolve})
        body = "\n".join(
            [
                "<h2>Call Summary</h2>",
                _render_table(
                    [call_summary] if call_summary else [],
                    ["call_sid", "tenant_id", "prompt_version_id", "realtime_model", "realtime_reasoning_effort", "from_number", "to_number", "status", "started_at", "ended_at"],
                ),
                f'<p><a href="/admin/review">Back to review queue</a> | <a href="/admin/leads">Back to dashboard</a></p>',
                "<h2>Copy-Friendly PM Summary</h2>",
                f'<textarea readonly rows="14" style="width: 100%; max-width: 900px;">{escape(detail.get("summary_text") or "")}</textarea>',
                "<h2>Review</h2>",
                _render_table([review], ["review_status", "review_tags", "internal_notes", "reviewed_at", "reviewed_by"]),
                _call_review_form(call_sid, review),
                "<h2>Feedback</h2>",
                _feedback_table(feedback_rows),
                "<h3>Add Feedback</h3>",
                _feedback_form(call_sid),
                "<h2>Lead</h2>",
                _render_table(leads, ["id", "tenant_id", "call_sid", "name", "callback", "address", "issue", "urgency", "priority", "priority_reason", "extra_fields", "lead_quality", "lead_notes", "status", "created_at"]),
                "".join(lead_review_forms),
                "<h2>Notification Attempts</h2>",
                _render_table(notifications, ["id", "tenant_id", "lead_id", "channel", "to_number", "recipient_type", "status", "attempt_number", "provider_message_sid", "error", "created_at", "sent_at"]),
                "<h2>Important Events</h2>",
                _render_table(important_events, ["created_at", "event_type", "payload_json"]),
                "<h2>Transcript Snippets</h2>",
                _render_table(transcript_events, ["created_at", "event_type", "payload_json"]),
                "<h2>Lifecycle Debug</h2>",
                "<p>These events distinguish app hangup, Twilio stop, websocket disconnect, OpenAI reader failure, and unknown exits.</p>",
                "<h3>Latest Exit Snapshot</h3>",
                f"<pre>{escape(latest_done.get('payload_json') if latest_done else 'No media_stream_done event recorded yet.')}</pre>",
                "<h3>Lifecycle Events</h3>",
                _render_table(lifecycle_events, ["created_at", "event_type", "payload_json"]),
                "<h3>Full Call Detail</h3>",
                f"<pre>{escape(json.dumps(detail, default=str, indent=2))}</pre>",
            ]
        )
        return _page(f"Call {call_sid}", body)

    return router
