import json
from typing import Optional

from sqlalchemy import desc

from storage.database import session_scope
from storage.models import Call, CallEvent, Lead, Notification, utcnow


def _call_summary(call: Call) -> dict:
    return {
        "id": call.id,
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
        "call_id": event.call_id,
        "call_sid": event.call_sid,
        "event_type": event.event_type,
        "payload_json": event.payload_json,
        "created_at": event.created_at,
    }


def _get_call(db, call_sid: str) -> Optional[Call]:
    return db.query(Call).filter(Call.call_sid == call_sid).one_or_none()


def create_or_update_call(call_sid: str, from_number: str, to_number: str) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call is None:
            call = Call(
                call_sid=call_sid,
                from_number=from_number,
                to_number=to_number,
                status="voice_received",
            )
            db.add(call)
            db.flush()
        else:
            call.from_number = from_number or call.from_number
            call.to_number = to_number or call.to_number
            if call.status == "new":
                call.status = "voice_received"
        return _call_summary(call)


def update_call_stream_started(call_sid: str, stream_sid: str) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call is None:
            call = Call(call_sid=call_sid, stream_sid=stream_sid, status="stream_started")
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
            call = Call(call_sid=call_sid, status=status, ended_at=utcnow())
            db.add(call)
        else:
            call.status = status
            call.ended_at = utcnow()


def record_call_event(call_sid: str, event_type: str, payload: dict) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        event = CallEvent(
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


def create_lead_with_pending_notification(call_sid: str, args: dict, to_number: str) -> tuple[dict, dict]:
    raw_args_json = json.dumps(args, default=str)
    with session_scope() as db:
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
                    lead_id=existing.id,
                    channel="sms",
                    to_number=to_number,
                    status="pending",
                )
                db.add(notification)
                db.flush()
            return _lead_summary(existing), _notification_summary(notification)

        call = _get_call(db, call_sid)
        lead = Lead(
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


def list_recent_calls(limit: int = 25) -> list[dict]:
    with session_scope() as db:
        calls = db.query(Call).order_by(desc(Call.started_at)).limit(limit).all()
        return [_call_summary(call) for call in calls]


def list_recent_leads(limit: int = 25) -> list[dict]:
    with session_scope() as db:
        leads = db.query(Lead).order_by(desc(Lead.created_at)).limit(limit).all()
        return [_lead_summary(lead) for lead in leads]


def list_recent_notifications(limit: int = 25) -> list[dict]:
    with session_scope() as db:
        notifications = db.query(Notification).order_by(desc(Notification.created_at)).limit(limit).all()
        return [_notification_summary(notification) for notification in notifications]


def list_recent_call_events(limit: int = 50) -> list[dict]:
    with session_scope() as db:
        events = db.query(CallEvent).order_by(desc(CallEvent.created_at)).limit(limit).all()
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
            "leads": [_lead_summary(lead) for lead in leads],
            "notifications": [_notification_summary(notification) for notification in notifications],
            "events": [_event_summary(event) for event in events],
        }
