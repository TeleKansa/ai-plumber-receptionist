import json
from typing import Optional

from sqlalchemy import desc

from storage.database import session_scope
from storage.models import Call, CallEvent, Lead, Notification, Tenant, TenantPhoneNumber, TenantSettings, utcnow


def normalize_phone_number(value: Optional[str]) -> str:
    if not value:
        return ""
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 10:
        return f"1{digits}"
    return digits


def _normalize_phone(value: Optional[str]) -> str:
    return normalize_phone_number(value)


def _tenant_summary(tenant: Tenant) -> dict:
    settings = tenant.settings
    return {
        "id": tenant.id,
        "name": tenant.name,
        "slug": tenant.slug,
        "status": tenant.status,
        "created_at": tenant.created_at,
        "updated_at": tenant.updated_at,
        "business_name": settings.business_name if settings else tenant.name,
        "greeting": settings.greeting if settings else "Plumbing office, what's going on?",
        "notification_sms_number": settings.notification_sms_number if settings else None,
        "backup_notification_sms_number": settings.backup_notification_sms_number if settings else None,
        "settings_active": settings.active if settings else False,
    }


def _tenant_phone_summary(phone: TenantPhoneNumber) -> dict:
    return {
        "id": phone.id,
        "tenant_id": phone.tenant_id,
        "twilio_number": phone.twilio_number,
        "label": phone.label,
        "active": phone.active,
        "created_at": phone.created_at,
    }


def _call_summary(call: Call) -> dict:
    return {
        "id": call.id,
        "tenant_id": call.tenant_id,
        "call_sid": call.call_sid,
        "stream_sid": call.stream_sid,
        "from_number": call.from_number,
        "to_number": call.to_number,
        "status": call.status,
        "started_at": call.started_at,
        "ended_at": call.ended_at,
    }


def _lead_summary(lead: Lead) -> dict:
    return {
        "id": lead.id,
        "tenant_id": lead.tenant_id,
        "call_id": lead.call_id,
        "call_sid": lead.call_sid,
        "name": lead.name,
        "callback": lead.callback,
        "address": lead.address,
        "issue": lead.issue,
        "urgency": lead.urgency,
        "raw_args_json": lead.raw_args_json,
        "status": lead.status,
        "created_at": lead.created_at,
    }


def _notification_summary(notification: Notification) -> dict:
    return {
        "id": notification.id,
        "tenant_id": notification.tenant_id,
        "lead_id": notification.lead_id,
        "channel": notification.channel,
        "to_number": notification.to_number,
        "status": notification.status,
        "provider_message_sid": notification.provider_message_sid,
        "error": notification.error,
        "created_at": notification.created_at,
        "sent_at": notification.sent_at,
    }


def _event_summary(event: CallEvent) -> dict:
    return {
        "id": event.id,
        "tenant_id": event.tenant_id,
        "call_id": event.call_id,
        "call_sid": event.call_sid,
        "event_type": event.event_type,
        "payload_json": event.payload_json,
        "created_at": event.created_at,
    }


def _get_call(db, call_sid: str) -> Optional[Call]:
    return db.query(Call).filter(Call.call_sid == call_sid).one_or_none()


def _default_tenant(db) -> Tenant:
    tenant = db.query(Tenant).filter(Tenant.slug == "default").one_or_none()
    if tenant is None:
        tenant = db.query(Tenant).order_by(Tenant.id).first()
    if tenant is None:
        raise RuntimeError("No tenant configured. Database bootstrap did not create a default tenant.")
    return tenant


def get_default_tenant() -> dict:
    with session_scope() as db:
        return _tenant_summary(_default_tenant(db))


def get_tenant(tenant_id: int) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        return _tenant_summary(tenant) if tenant else None


def list_tenants() -> list[dict]:
    with session_scope() as db:
        tenants = db.query(Tenant).order_by(Tenant.id).all()
        return [_tenant_summary(tenant) for tenant in tenants]


def list_tenant_phone_numbers(tenant_id: Optional[int] = None) -> list[dict]:
    with session_scope() as db:
        query = db.query(TenantPhoneNumber).order_by(TenantPhoneNumber.id)
        if tenant_id is not None:
            query = query.filter(TenantPhoneNumber.tenant_id == tenant_id)
        return [_tenant_phone_summary(phone) for phone in query.all()]


def find_tenant_by_phone_number(twilio_number: str) -> Optional[dict]:
    normalized = _normalize_phone(twilio_number)
    with session_scope() as db:
        phone = (
            db.query(TenantPhoneNumber)
            .join(Tenant)
            .filter(TenantPhoneNumber.twilio_number == twilio_number, TenantPhoneNumber.active.is_(True), Tenant.status == "active")
            .one_or_none()
        )
        if phone is None and normalized:
            phones = (
                db.query(TenantPhoneNumber)
                .join(Tenant)
                .filter(TenantPhoneNumber.active.is_(True), Tenant.status == "active")
                .all()
            )
            matches = [candidate for candidate in phones if _normalize_phone(candidate.twilio_number) == normalized]
            if len(matches) == 1:
                phone = matches[0]
        return _tenant_summary(phone.tenant) if phone else None


def resolve_tenant_for_phone(twilio_number: str) -> tuple[Optional[dict], bool]:
    tenant = find_tenant_by_phone_number(twilio_number)
    if tenant:
        return tenant, True
    return None, False


def get_call_tenant(call_sid: str) -> Optional[dict]:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call and call.tenant:
            return _tenant_summary(call.tenant)
        return None


def create_tenant(name: str, slug: str, business_name: str, greeting: str, notification_sms_number: str, backup_notification_sms_number: str = "", status: str = "active") -> dict:
    with session_scope() as db:
        tenant = Tenant(name=name.strip(), slug=slug.strip(), status=status or "active")
        db.add(tenant)
        db.flush()
        settings = TenantSettings(
            tenant_id=tenant.id,
            business_name=business_name.strip() or name.strip(),
            greeting=greeting.strip() or "Plumbing office, what's going on?",
            notification_sms_number=notification_sms_number.strip(),
            backup_notification_sms_number=backup_notification_sms_number.strip() or None,
            active=True,
        )
        db.add(settings)
        db.flush()
        return _tenant_summary(tenant)


def update_tenant_settings(tenant_id: int, business_name: str, greeting: str, notification_sms_number: str, backup_notification_sms_number: str = "", status: str = "active") -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        tenant.name = business_name.strip() or tenant.name
        tenant.status = status or tenant.status
        tenant.updated_at = utcnow()
        settings = tenant.settings
        if settings is None:
            settings = TenantSettings(tenant_id=tenant.id, business_name=tenant.name, greeting="Plumbing office, what's going on?", active=True)
            db.add(settings)
            db.flush()
        settings.business_name = business_name.strip() or settings.business_name
        settings.greeting = greeting.strip() or settings.greeting
        settings.notification_sms_number = notification_sms_number.strip()
        settings.backup_notification_sms_number = backup_notification_sms_number.strip() or None
        settings.active = status == "active"
        db.flush()
        return _tenant_summary(tenant)


def add_tenant_phone_number(tenant_id: int, twilio_number: str, label: str = "", active: bool = True) -> dict:
    with session_scope() as db:
        existing = db.query(TenantPhoneNumber).filter(TenantPhoneNumber.twilio_number == twilio_number).one_or_none()
        if existing is not None:
            existing.tenant_id = tenant_id
            existing.label = label.strip() or existing.label
            existing.active = active
            db.flush()
            return _tenant_phone_summary(existing)
        phone = TenantPhoneNumber(
            tenant_id=tenant_id,
            twilio_number=twilio_number.strip(),
            label=label.strip() or None,
            active=active,
        )
        db.add(phone)
        db.flush()
        return _tenant_phone_summary(phone)


def create_or_update_call(
    call_sid: str,
    from_number: str,
    to_number: str,
    tenant_id: Optional[int] = None,
    default_to_tenant: bool = True,
    status: str = "voice_received",
) -> dict:
    with session_scope() as db:
        resolved_tenant_id = tenant_id
        if resolved_tenant_id is None and default_to_tenant:
            resolved_tenant_id = _default_tenant(db).id
        call = _get_call(db, call_sid)
        if call is None:
            call = Call(
                tenant_id=resolved_tenant_id,
                call_sid=call_sid,
                from_number=from_number,
                to_number=to_number,
                status=status,
            )
            db.add(call)
            db.flush()
        else:
            if tenant_id is not None or not default_to_tenant:
                call.tenant_id = resolved_tenant_id
            elif call.tenant_id is None:
                call.tenant_id = resolved_tenant_id
            call.from_number = from_number or call.from_number
            call.to_number = to_number or call.to_number
            if status:
                call.status = status
        return _call_summary(call)


def update_call_stream_started(call_sid: str, stream_sid: str) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call is None:
            call = Call(tenant_id=_default_tenant(db).id, call_sid=call_sid, stream_sid=stream_sid, status="stream_started")
            db.add(call)
        else:
            call.stream_sid = stream_sid
            call.status = "stream_started"
        db.flush()
        return _call_summary(call)


def mark_call_ended(call_sid: str, status: str = "ended") -> None:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call is None:
            call = Call(tenant_id=_default_tenant(db).id, call_sid=call_sid, status=status, ended_at=utcnow())
            db.add(call)
        else:
            call.status = status
            call.ended_at = utcnow()


def record_call_event(
    call_sid: str,
    event_type: str,
    payload: dict,
    tenant_id: Optional[int] = None,
    default_to_tenant: bool = True,
) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if tenant_id is not None:
            event_tenant_id = tenant_id
        elif call and call.tenant_id:
            event_tenant_id = call.tenant_id
        elif default_to_tenant:
            event_tenant_id = _default_tenant(db).id
        else:
            event_tenant_id = None
        event = CallEvent(
            tenant_id=event_tenant_id,
            call_id=call.id if call else None,
            call_sid=call_sid,
            event_type=event_type,
            payload_json=json.dumps(payload, default=str),
        )
        db.add(event)
        db.flush()
        return _event_summary(event)


def get_lead_by_call_sid(call_sid: str) -> Optional[dict]:
    with session_scope() as db:
        lead = db.query(Lead).filter(Lead.call_sid == call_sid).one_or_none()
        return _lead_summary(lead) if lead else None


def get_notification_for_lead(lead_id: int, channel: str = "sms") -> Optional[dict]:
    with session_scope() as db:
        notification = (
            db.query(Notification)
            .filter(Notification.lead_id == lead_id, Notification.channel == channel)
            .order_by(desc(Notification.created_at))
            .first()
        )
        return _notification_summary(notification) if notification else None


def create_lead_with_pending_notification(call_sid: str, args: dict, to_number: str, tenant_id: Optional[int] = None) -> tuple[dict, dict]:
    raw_args_json = json.dumps(args, default=str)
    with session_scope() as db:
        call = _get_call(db, call_sid)
        lead_tenant_id = tenant_id or (call.tenant_id if call and call.tenant_id else _default_tenant(db).id)
        existing = db.query(Lead).filter(Lead.call_sid == call_sid).one_or_none()
        if existing is not None:
            notification = (
                db.query(Notification)
                .filter(
                    Notification.lead_id == existing.id,
                    Notification.channel == "sms",
                    Notification.to_number == to_number,
                )
                .one_or_none()
            )
            if notification is None:
                notification = Notification(
                    tenant_id=existing.tenant_id or lead_tenant_id,
                    lead_id=existing.id,
                    channel="sms",
                    to_number=to_number,
                    status="pending",
                )
                db.add(notification)
                db.flush()
            return _lead_summary(existing), _notification_summary(notification)

        lead = Lead(
            tenant_id=lead_tenant_id,
            call_id=call.id if call else None,
            call_sid=call_sid,
            name=str(args.get("name", "")).strip(),
            callback=str(args.get("callback", "")).strip(),
            address=str(args.get("address", "")).strip(),
            issue=str(args.get("issue", "")).strip(),
            urgency=str(args.get("urgency", "")).strip(),
            raw_args_json=raw_args_json,
            status="submitted",
        )
        db.add(lead)
        db.flush()

        notification = Notification(
            tenant_id=lead_tenant_id,
            lead_id=lead.id,
            channel="sms",
            to_number=to_number,
            status="pending",
        )
        db.add(notification)
        db.flush()
        return _lead_summary(lead), _notification_summary(notification)


def mark_notification_sent(notification_id: int, provider_message_sid: Optional[str]) -> dict:
    with session_scope() as db:
        notification = db.query(Notification).filter(Notification.id == notification_id).one()
        notification.status = "sent"
        notification.provider_message_sid = provider_message_sid
        notification.error = None
        notification.sent_at = utcnow()
        db.flush()
        return _notification_summary(notification)


def mark_notification_failed(notification_id: int, error: str) -> dict:
    with session_scope() as db:
        notification = db.query(Notification).filter(Notification.id == notification_id).one()
        notification.status = "failed"
        notification.error = error
        notification.sent_at = None
        db.flush()
        return _notification_summary(notification)


def list_recent_calls(limit: int = 25, tenant_id: Optional[int] = None) -> list[dict]:
    with session_scope() as db:
        query = db.query(Call)
        if tenant_id is not None:
            query = query.filter(Call.tenant_id == tenant_id)
        calls = query.order_by(desc(Call.started_at)).limit(limit).all()
        return [_call_summary(call) for call in calls]


def list_recent_leads(limit: int = 25, tenant_id: Optional[int] = None) -> list[dict]:
    with session_scope() as db:
        query = db.query(Lead)
        if tenant_id is not None:
            query = query.filter(Lead.tenant_id == tenant_id)
        leads = query.order_by(desc(Lead.created_at)).limit(limit).all()
        return [_lead_summary(lead) for lead in leads]


def list_recent_notifications(limit: int = 25, tenant_id: Optional[int] = None) -> list[dict]:
    with session_scope() as db:
        query = db.query(Notification)
        if tenant_id is not None:
            query = query.filter(Notification.tenant_id == tenant_id)
        notifications = query.order_by(desc(Notification.created_at)).limit(limit).all()
        return [_notification_summary(notification) for notification in notifications]


def list_recent_call_events(limit: int = 50, tenant_id: Optional[int] = None) -> list[dict]:
    with session_scope() as db:
        query = db.query(CallEvent)
        if tenant_id is not None:
            query = query.filter(CallEvent.tenant_id == tenant_id)
        events = query.order_by(desc(CallEvent.created_at)).limit(limit).all()
        return [_event_summary(event) for event in events]


def get_call_detail(call_sid: str) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        leads = db.query(Lead).filter(Lead.call_sid == call_sid).order_by(desc(Lead.created_at)).all()
        lead_ids = [lead.id for lead in leads]
        notifications = []
        if lead_ids:
            notifications = (
                db.query(Notification)
                .filter(Notification.lead_id.in_(lead_ids))
                .order_by(desc(Notification.created_at))
                .all()
            )
        events = db.query(CallEvent).filter(CallEvent.call_sid == call_sid).order_by(desc(CallEvent.created_at)).all()
        return {
            "call": _call_summary(call) if call else None,
            "tenant": _tenant_summary(call.tenant) if call and call.tenant else None,
            "leads": [_lead_summary(lead) for lead in leads],
            "notifications": [_notification_summary(notification) for notification in notifications],
            "events": [_event_summary(event) for event in events],
        }
