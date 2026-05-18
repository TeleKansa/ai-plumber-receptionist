from datetime import datetime, timezone

from sqlalchemy import inspect, text


TENANT_SCOPED_TABLES = ("calls", "leads", "notifications", "call_events")


def _normalize_phone(value: str) -> str:
    if not value:
        return ""
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 10:
        return f"1{digits}"
    return digits


def _columns_for(engine, table_name: str) -> set[str]:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def add_missing_tenant_columns(engine):
    with engine.begin() as conn:
        for table_name in TENANT_SCOPED_TABLES:
            columns = _columns_for(engine, table_name)
            if columns and "tenant_id" not in columns:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN tenant_id INTEGER"))


def _scalar(conn, sql: str, params: dict):
    return conn.execute(text(sql), params).scalar_one_or_none()


def ensure_default_tenant(engine, settings):
    now = datetime.now(timezone.utc)
    slug = settings.default_tenant_slug or "default"
    name = settings.default_tenant_name or "Default Plumbing"
    greeting = settings.default_tenant_greeting or "Plumbing office, what's going on?"

    with engine.begin() as conn:
        tenant_id = _scalar(conn, "SELECT id FROM tenants WHERE slug = :slug", {"slug": slug})
        if tenant_id is None:
            conn.execute(
                text(
                    "INSERT INTO tenants (name, slug, status, created_at, updated_at) "
                    "VALUES (:name, :slug, 'active', :created_at, :updated_at)"
                ),
                {"name": name, "slug": slug, "created_at": now, "updated_at": now},
            )
            tenant_id = _scalar(conn, "SELECT id FROM tenants WHERE slug = :slug", {"slug": slug})

        settings_id = _scalar(
            conn,
            "SELECT id FROM tenant_settings WHERE tenant_id = :tenant_id",
            {"tenant_id": tenant_id},
        )
        if settings_id is None:
            conn.execute(
                text(
                    "INSERT INTO tenant_settings "
                    "(tenant_id, business_name, greeting, notification_sms_number, "
                    "backup_notification_sms_number, voice, model, active) "
                    "VALUES (:tenant_id, :business_name, :greeting, :notification_sms_number, "
                    "NULL, NULL, NULL, :active)"
                ),
                {
                    "tenant_id": tenant_id,
                    "business_name": name,
                    "greeting": greeting,
                    "notification_sms_number": settings.plumber_phone_number,
                    "active": True,
                },
            )
        elif settings.plumber_phone_number:
            conn.execute(
                text(
                    "UPDATE tenant_settings "
                    "SET notification_sms_number = :notification_sms_number "
                    "WHERE tenant_id = :tenant_id AND "
                    "(notification_sms_number IS NULL OR notification_sms_number = '')"
                ),
                {"tenant_id": tenant_id, "notification_sms_number": settings.plumber_phone_number},
            )

        if settings.twilio_phone_number:
            normalized_default_phone = _normalize_phone(settings.twilio_phone_number)
            phone_id = _scalar(
                conn,
                "SELECT id FROM tenant_phone_numbers WHERE twilio_number = :twilio_number",
                {"twilio_number": settings.twilio_phone_number},
            )
            if phone_id is None and normalized_default_phone:
                rows = conn.execute(text("SELECT id, twilio_number FROM tenant_phone_numbers")).mappings().all()
                for row in rows:
                    if _normalize_phone(row["twilio_number"]) == normalized_default_phone:
                        phone_id = row["id"]
                        break
            if phone_id is None:
                conn.execute(
                    text(
                        "INSERT INTO tenant_phone_numbers "
                        "(tenant_id, twilio_number, label, active, created_at) "
                        "VALUES (:tenant_id, :twilio_number, :label, :active, :created_at)"
                    ),
                    {
                        "tenant_id": tenant_id,
                        "twilio_number": settings.twilio_phone_number,
                        "label": "Default Twilio number",
                        "active": True,
                        "created_at": now,
                    },
                )
            else:
                conn.execute(
                    text(
                        "UPDATE tenant_phone_numbers "
                        "SET tenant_id = :tenant_id, active = :active "
                        "WHERE id = :phone_id"
                    ),
                    {"tenant_id": tenant_id, "active": True, "phone_id": phone_id},
                )

        for table_name in TENANT_SCOPED_TABLES:
            if "tenant_id" in _columns_for(engine, table_name):
                conn.execute(
                    text(f"UPDATE {table_name} SET tenant_id = :tenant_id WHERE tenant_id IS NULL"),
                    {"tenant_id": tenant_id},
                )

    return tenant_id


def run_schema_migrations(engine, settings):
    add_missing_tenant_columns(engine)
    return ensure_default_tenant(engine, settings)
