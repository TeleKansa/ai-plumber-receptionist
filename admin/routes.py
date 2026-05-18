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
from workflow.prompt_builder import PromptBuilder


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
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>tenant</th><th>id</th><th>slug</th><th>status</th><th>business_name</th>"
        "<th>notification_sms_number</th><th>details</th><th>prompt</th>"
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
            f"<td>{escape('yes' if profile.get('is_active') else 'no')}</td>"
            f"<td>{activate}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>id</th><th>version</th><th>label</th><th>business_name</th><th>greeting</th><th>active</th><th>action</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
    )


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
  <nav><a href="/admin/leads">Dashboard</a><a href="/admin/tenants">Tenants</a></nav>
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
                _render_table(calls, ["tenant_id", "prompt_version_id", "call_sid", "from_number", "to_number", "stream_sid", "status", "started_at", "ended_at"]),
                "<h2>Recent Leads</h2>",
                _render_table(leads, ["id", "tenant_id", "call_sid", "name", "callback", "address", "issue", "urgency", "status", "created_at"]),
                "<h2>Recent Notifications</h2>",
                _render_table(notifications, ["id", "tenant_id", "lead_id", "channel", "to_number", "status", "provider_message_sid", "error", "created_at", "sent_at"]),
                "<h2>Recent Call Events</h2>",
                _render_table(events, ["id", "tenant_id", "call_sid", "event_type", "payload_json", "created_at"]),
            ]
        )
        return _page("Plumber Receptionist Admin", body)

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
            }
        ]
        body = "\n".join(
            [
                f"<h2>{escape(tenant['name'])}</h2>",
                '<p><a href="/admin/tenants">Back to tenants</a></p>',
                f'<p><strong><a href="/admin/tenants/{tenant_id}/prompt">Prompt/persona settings</a></strong></p>',
                "<h3>Tenant Summary</h3>",
                _render_table(
                    tenant_summary,
                    ["id", "name", "slug", "status", "business_name", "notification_sms_number", "backup_notification_sms_number"],
                ),
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
                _render_table(phones, ["id", "tenant_id", "twilio_number", "label", "active", "created_at"]),
                f'<form method="post" action="/admin/tenants/{tenant_id}/phones">',
                _input("twilio_number", placeholder="+15551234567"),
                "<br><br>",
                _input("label", placeholder="Main line"),
                "<br><br>",
                "<button type=\"submit\">Add Phone Number</button>",
                "</form>",
                "<h3>Recent Calls</h3>",
                _render_table(calls, ["call_sid", "prompt_version_id", "from_number", "to_number", "status", "started_at", "ended_at"]),
                "<h3>Recent Leads</h3>",
                _render_table(leads, ["id", "call_sid", "name", "callback", "address", "issue", "urgency", "status", "created_at"]),
                "<h3>Recent Notifications</h3>",
                _render_table(notifications, ["id", "lead_id", "channel", "to_number", "status", "error", "created_at", "sent_at"]),
                "<h3>Recent Events</h3>",
                _render_table(events, ["id", "call_sid", "event_type", "payload_json", "created_at"]),
            ]
        )
        return _page(f"Tenant {tenant_id}", body)

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
        profiles = repository.list_prompt_profiles(tenant_id)
        preview = PromptBuilder().build("913-555-0123", tenant=tenant, profile=active_profile) if active_profile else ""
        body = "\n".join(
            [
                f"<h2>{escape(tenant['name'])} Prompt</h2>",
                f'<p><a href="/admin/tenants/{tenant_id}">Back to tenant detail</a> | <a href="/admin/tenants">Back to tenants</a></p>',
                "<h3>Core workflow is locked</h3>",
                "<p>Style/persona settings can change wording, but required fields cannot be changed here. Required fields stay: issue, urgency, address, callback, name. First name is enough; last name is not required.</p>",
                "<h3>Active Prompt Version</h3>",
                _render_table([active_profile] if active_profile else [], ["id", "version", "label", "business_name", "greeting", "tone", "verbosity", "closing_line", "is_active"]),
                "<h3>Create New Prompt Version</h3>",
                f'<form method="post" action="/admin/tenants/{tenant_id}/prompt">',
                _input("label", active_profile.get("label") if active_profile else "Updated prompt"),
                "<br><br>",
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
        )
        return RedirectResponse(f"/admin/tenants/{tenant_id}", status_code=303)

    @router.get("/admin/calls/{call_sid}", response_class=HTMLResponse)
    async def admin_call_detail(call_sid: str):
        detail = repository.get_call_detail(call_sid)
        body = f"<pre>{escape(json.dumps(detail, default=str, indent=2))}</pre>"
        return _page(f"Call {call_sid}", body)

    return router
