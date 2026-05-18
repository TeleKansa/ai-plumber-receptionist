import json
from html import escape

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from admin.auth import check_admin_credentials
from config.settings import Settings
from storage import repository


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


def _page(title: str, body: str) -> HTMLResponse:
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
  <nav><a href="/admin/leads">Dashboard</a></nav>
  <h1>{escape(title)}</h1>
  {body}
</body>
</html>
"""
    return HTMLResponse(html)


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
                _render_table(calls, ["call_sid", "from_number", "to_number", "stream_sid", "status", "started_at", "ended_at"]),
                "<h2>Recent Leads</h2>",
                _render_table(leads, ["id", "call_sid", "name", "callback", "address", "issue", "urgency", "status", "created_at"]),
                "<h2>Recent Notifications</h2>",
                _render_table(notifications, ["id", "lead_id", "channel", "to_number", "status", "provider_message_sid", "error", "created_at", "sent_at"]),
                "<h2>Recent Call Events</h2>",
                _render_table(events, ["id", "call_sid", "event_type", "payload_json", "created_at"]),
            ]
        )
        return _page("Plumber Receptionist Admin", body)

    @router.get("/admin/calls/{call_sid}", response_class=HTMLResponse)
    async def admin_call_detail(call_sid: str):
        detail = repository.get_call_detail(call_sid)
        body = f"<pre>{escape(json.dumps(detail, default=str, indent=2))}</pre>"
        return _page(f"Call {call_sid}", body)

    return router
