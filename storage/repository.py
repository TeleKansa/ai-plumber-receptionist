import json
from typing import Optional

from sqlalchemy import desc

from storage.database import session_scope
from storage.models import (
    Call,
    CallEvent,
    Lead,
    Notification,
    Tenant,
    TenantAIProfile,
    TenantIntakePolicy,
    TenantNotificationPolicy,
    TenantPhoneNumber,
    TenantSettings,
    TenantTelephonyProfile,
    utcnow,
)
from workflow.intake_policy import default_intake_policy
from workflow.prompt_builder import prompt_profile_defaults
from workflow.realtime_config import normalize_realtime_model


def normalize_phone_number(value: Optional[str]) -> str:
    if not value:
        return ""
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 10:
        return f"1{digits}"
    return digits


def _json_text_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


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
        "accepts_live_calls": phone.accepts_live_calls,
        "purpose": phone.purpose,
        "created_at": phone.created_at,
    }


def _telephony_profile_summary(profile: TenantTelephonyProfile) -> dict:
    return {
        "id": profile.id,
        "tenant_id": profile.tenant_id,
        "public_business_number": profile.public_business_number,
        "ai_ingress_twilio_number": profile.ai_ingress_twilio_number,
        "routing_mode": profile.routing_mode,
        "forwarding_setup_status": profile.forwarding_setup_status,
        "test_mode_enabled": profile.test_mode_enabled,
        "allowed_test_callers_json": profile.allowed_test_callers_json,
        "live_enabled_at": profile.live_enabled_at,
        "notes": profile.notes,
    }


def _intake_policy_summary(policy: TenantIntakePolicy) -> dict:
    return {
        "id": policy.id,
        "tenant_id": policy.tenant_id,
        "enabled": policy.enabled,
        "extra_questions_json": policy.extra_questions_json,
        "conditional_questions_json": policy.conditional_questions_json,
        "sms_include_extra_fields_json": policy.sms_include_extra_fields_json,
        "admin_display_fields_json": policy.admin_display_fields_json,
        "notes": policy.notes,
        "created_at": policy.created_at,
        "updated_at": policy.updated_at,
    }


def _notification_policy_summary(policy: TenantNotificationPolicy) -> dict:
    return {
        "id": policy.id,
        "tenant_id": policy.tenant_id,
        "normal_sms_recipients_json": policy.normal_sms_recipients_json,
        "emergency_sms_recipients_json": policy.emergency_sms_recipients_json,
        "backup_sms_recipients_json": policy.backup_sms_recipients_json,
        "send_normal_leads": policy.send_normal_leads,
        "send_emergency_leads": policy.send_emergency_leads,
        "include_extra_fields": policy.include_extra_fields,
        "include_additional_notes": policy.include_additional_notes,
        "emergency_keywords_json": policy.emergency_keywords_json,
        "emergency_rules_json": policy.emergency_rules_json,
        "notes": policy.notes,
        "created_at": policy.created_at,
        "updated_at": policy.updated_at,
    }


def _prompt_profile_summary(profile: TenantAIProfile) -> dict:
    return {
        "id": profile.id,
        "tenant_id": profile.tenant_id,
        "version": profile.version,
        "label": profile.label,
        "business_name": profile.business_name,
        "greeting": profile.greeting,
        "tone": profile.tone,
        "verbosity": profile.verbosity,
        "closing_line": profile.closing_line,
        "avoid_phrases_json": profile.avoid_phrases_json,
        "preferred_terms_json": profile.preferred_terms_json,
        "extra_instructions_text": profile.extra_instructions_text,
        "realtime_model": profile.realtime_model,
        "is_active": profile.is_active,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def _call_summary(call: Call) -> dict:
    return {
        "id": call.id,
        "tenant_id": call.tenant_id,
        "prompt_version_id": call.prompt_version_id,
        "realtime_model": call.realtime_model,
        "realtime_reasoning_effort": call.realtime_reasoning_effort,
        "call_sid": call.call_sid,
        "stream_sid": call.stream_sid,
        "from_number": call.from_number,
        "to_number": call.to_number,
        "status": call.status,
        "started_at": call.started_at,
        "ended_at": call.ended_at,
    }


def _lead_summary(lead: Lead) -> dict:
    try:
        extra_fields = json.loads(lead.extra_fields_json or "{}")
    except json.JSONDecodeError:
        extra_fields = {}
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
        "extra_fields": extra_fields if isinstance(extra_fields, dict) else {},
        "extra_fields_json": lead.extra_fields_json,
        "raw_args_json": lead.raw_args_json,
        "priority": lead.priority,
        "priority_reason": lead.priority_reason,
        "classification_json": lead.classification_json,
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
        "recipient_type": notification.recipient_type,
        "status": notification.status,
        "provider_message_sid": notification.provider_message_sid,
        "error": notification.error,
        "policy_snapshot_json": notification.policy_snapshot_json,
        "attempt_number": notification.attempt_number,
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


def _create_default_prompt_profile(db, tenant: Tenant, activate: bool = True) -> TenantAIProfile:
    tenant_summary = _tenant_summary(tenant)
    defaults = prompt_profile_defaults(tenant_summary)
    max_version = (
        db.query(TenantAIProfile.version)
        .filter(TenantAIProfile.tenant_id == tenant.id)
        .order_by(desc(TenantAIProfile.version))
        .first()
    )
    next_version = (max_version[0] if max_version else 0) + 1
    if activate:
        db.query(TenantAIProfile).filter(TenantAIProfile.tenant_id == tenant.id).update({"is_active": False})
    profile = TenantAIProfile(
        tenant_id=tenant.id,
        version=next_version,
        label=defaults["label"],
        business_name=defaults["business_name"],
        greeting=defaults["greeting"],
        tone=defaults["tone"],
        verbosity=defaults["verbosity"],
        closing_line=defaults["closing_line"],
        avoid_phrases_json=json.dumps(defaults["avoid_phrases"]),
        preferred_terms_json=json.dumps(defaults["preferred_terms"]),
        extra_instructions_text=defaults["extra_instructions_text"],
        realtime_model=defaults.get("realtime_model") or None,
        is_active=activate,
    )
    db.add(profile)
    db.flush()
    return profile


def _create_default_telephony_profile(db, tenant: Tenant) -> TenantTelephonyProfile:
    profile = TenantTelephonyProfile(
        tenant_id=tenant.id,
        public_business_number="",
        ai_ingress_twilio_number="",
        routing_mode="forwarded_google_maps_number",
        forwarding_setup_status="not_started",
        test_mode_enabled=False,
        allowed_test_callers_json="[]",
        notes="",
    )
    db.add(profile)
    db.flush()
    return profile


def _create_default_intake_policy(db, tenant: Tenant) -> TenantIntakePolicy:
    defaults = default_intake_policy()
    policy = TenantIntakePolicy(
        tenant_id=tenant.id,
        enabled=defaults["enabled"],
        extra_questions_json=defaults["extra_questions_json"],
        conditional_questions_json=defaults["conditional_questions_json"],
        sms_include_extra_fields_json=defaults["sms_include_extra_fields_json"],
        admin_display_fields_json=defaults["admin_display_fields_json"],
        notes=defaults["notes"],
    )
    db.add(policy)
    db.flush()
    return policy


def _create_default_notification_policy(db, tenant: Tenant) -> TenantNotificationPolicy:
    settings = tenant.settings
    normal = [settings.notification_sms_number] if settings and settings.notification_sms_number else []
    backup = [settings.backup_notification_sms_number] if settings and settings.backup_notification_sms_number else []
    policy = TenantNotificationPolicy(
        tenant_id=tenant.id,
        normal_sms_recipients_json=json.dumps(normal),
        emergency_sms_recipients_json="[]",
        backup_sms_recipients_json=json.dumps(backup),
        send_normal_leads=True,
        send_emergency_leads=True,
        include_extra_fields=True,
        include_additional_notes=True,
        emergency_keywords_json=json.dumps([
            "active leak",
            "burst pipe",
            "cannot shut",
            "can't shut",
            "cant shut",
            "flooding",
            "sewage",
            "sewer backup",
            "water is still coming out",
            "water still coming out",
            "water still running",
        ]),
        emergency_rules_json="[]",
        notes="",
    )
    db.add(policy)
    db.flush()
    return policy


def _get_or_create_telephony_profile(db, tenant: Tenant) -> TenantTelephonyProfile:
    if tenant.telephony_profile:
        return tenant.telephony_profile
    return _create_default_telephony_profile(db, tenant)


def _get_or_create_intake_policy(db, tenant: Tenant) -> TenantIntakePolicy:
    if tenant.intake_policy:
        return tenant.intake_policy
    return _create_default_intake_policy(db, tenant)


def _get_or_create_notification_policy(db, tenant: Tenant) -> TenantNotificationPolicy:
    if tenant.notification_policy:
        return tenant.notification_policy
    return _create_default_notification_policy(db, tenant)


def get_telephony_profile(tenant_id: int) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        return _telephony_profile_summary(_get_or_create_telephony_profile(db, tenant))


def get_intake_policy(tenant_id: int) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        return _intake_policy_summary(_get_or_create_intake_policy(db, tenant))


def get_notification_policy(tenant_id: int) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        return _notification_policy_summary(_get_or_create_notification_policy(db, tenant))


def update_notification_policy(
    tenant_id: int,
    normal_sms_recipients: Optional[list[str]] = None,
    emergency_sms_recipients: Optional[list[str]] = None,
    backup_sms_recipients: Optional[list[str]] = None,
    send_normal_leads: bool = True,
    send_emergency_leads: bool = True,
    include_extra_fields: bool = True,
    include_additional_notes: bool = True,
    emergency_keywords: Optional[list[str]] = None,
    emergency_rules: Optional[list] = None,
    notes: str = "",
) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        policy = _get_or_create_notification_policy(db, tenant)
        policy.normal_sms_recipients_json = json.dumps(normal_sms_recipients or [])
        policy.emergency_sms_recipients_json = json.dumps(emergency_sms_recipients or [])
        policy.backup_sms_recipients_json = json.dumps(backup_sms_recipients or [])
        policy.send_normal_leads = bool(send_normal_leads)
        policy.send_emergency_leads = bool(send_emergency_leads)
        policy.include_extra_fields = bool(include_extra_fields)
        policy.include_additional_notes = bool(include_additional_notes)
        policy.emergency_keywords_json = json.dumps(emergency_keywords or [])
        policy.emergency_rules_json = json.dumps(emergency_rules or [])
        policy.notes = notes.strip()
        policy.updated_at = utcnow()
        db.flush()
        return _notification_policy_summary(policy)


def update_intake_policy(
    tenant_id: int,
    enabled: bool = True,
    extra_questions: Optional[list[dict]] = None,
    conditional_questions: Optional[list[dict]] = None,
    sms_include_extra_fields: Optional[list[str]] = None,
    admin_display_fields: Optional[list[str]] = None,
    notes: str = "",
) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        policy = _get_or_create_intake_policy(db, tenant)
        policy.enabled = bool(enabled)
        policy.extra_questions_json = json.dumps(extra_questions or [])
        policy.conditional_questions_json = json.dumps(conditional_questions or [])
        policy.sms_include_extra_fields_json = json.dumps(sms_include_extra_fields or [])
        policy.admin_display_fields_json = json.dumps(admin_display_fields or [])
        policy.notes = notes.strip()
        policy.updated_at = utcnow()
        db.flush()
        return _intake_policy_summary(policy)


def update_telephony_profile(
    tenant_id: int,
    public_business_number: str = "",
    ai_ingress_twilio_number: str = "",
    forwarding_setup_status: str = "not_started",
    test_mode_enabled: bool = False,
    allowed_test_callers: Optional[list[str]] = None,
    notes: str = "",
) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        profile = _get_or_create_telephony_profile(db, tenant)
        profile.public_business_number = public_business_number.strip()
        profile.ai_ingress_twilio_number = ai_ingress_twilio_number.strip()
        profile.routing_mode = "forwarded_google_maps_number"
        profile.forwarding_setup_status = forwarding_setup_status.strip() or "not_started"
        profile.test_mode_enabled = bool(test_mode_enabled)
        profile.allowed_test_callers_json = json.dumps(allowed_test_callers or [])
        profile.notes = notes.strip()
        db.flush()
        return _telephony_profile_summary(profile)


def set_tenant_status(tenant_id: int, status: str) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        tenant.status = status.strip() or tenant.status
        tenant.updated_at = utcnow()
        db.flush()
        return _tenant_summary(tenant)


def set_tenant_live(tenant_id: int) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        tenant.status = "live"
        tenant.updated_at = utcnow()
        profile = _get_or_create_telephony_profile(db, tenant)
        profile.live_enabled_at = utcnow()
        if profile.forwarding_setup_status in ("not_started", "instructions_sent", "customer_configured"):
            profile.forwarding_setup_status = "verified"
        db.flush()
        return _tenant_summary(tenant)


def set_tenant_paused(tenant_id: int) -> Optional[dict]:
    return set_tenant_status(tenant_id, "paused")


def set_tenant_phone_live(tenant_id: int, phone_id: int, accepts_live_calls: bool) -> Optional[dict]:
    with session_scope() as db:
        phone = (
            db.query(TenantPhoneNumber)
            .filter(TenantPhoneNumber.tenant_id == tenant_id, TenantPhoneNumber.id == phone_id)
            .one_or_none()
        )
        if phone is None:
            return None
        phone.accepts_live_calls = bool(accepts_live_calls)
        db.flush()
        return _tenant_phone_summary(phone)


def list_prompt_profiles(tenant_id: int) -> list[dict]:
    with session_scope() as db:
        profiles = (
            db.query(TenantAIProfile)
            .filter(TenantAIProfile.tenant_id == tenant_id)
            .order_by(desc(TenantAIProfile.version))
            .all()
        )
        return [_prompt_profile_summary(profile) for profile in profiles]


def get_active_prompt_profile(tenant_id: int) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        profile = (
            db.query(TenantAIProfile)
            .filter(TenantAIProfile.tenant_id == tenant_id, TenantAIProfile.is_active.is_(True))
            .order_by(desc(TenantAIProfile.version))
            .first()
        )
        if profile is None:
            profile = _create_default_prompt_profile(db, tenant)
        return _prompt_profile_summary(profile)


def get_prompt_profile(tenant_id: int, profile_id: int) -> Optional[dict]:
    with session_scope() as db:
        profile = (
            db.query(TenantAIProfile)
            .filter(TenantAIProfile.tenant_id == tenant_id, TenantAIProfile.id == profile_id)
            .one_or_none()
        )
        return _prompt_profile_summary(profile) if profile else None


def get_call_prompt_profile(call_sid: str) -> Optional[dict]:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call and call.prompt_profile:
            return _prompt_profile_summary(call.prompt_profile)
        return None


def create_prompt_profile(
    tenant_id: int,
    label: str,
    business_name: str,
    greeting: str,
    tone: str,
    verbosity: str,
    closing_line: str,
    avoid_phrases: list[str],
    preferred_terms: list[str],
    extra_instructions_text: str = "",
    realtime_model: str = "",
    activate: bool = True,
) -> Optional[dict]:
    with session_scope() as db:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one_or_none()
        if tenant is None:
            return None
        max_version = (
            db.query(TenantAIProfile.version)
            .filter(TenantAIProfile.tenant_id == tenant_id)
            .order_by(desc(TenantAIProfile.version))
            .first()
        )
        next_version = (max_version[0] if max_version else 0) + 1
        if activate:
            db.query(TenantAIProfile).filter(TenantAIProfile.tenant_id == tenant_id).update({"is_active": False})
        defaults = prompt_profile_defaults(_tenant_summary(tenant))
        profile = TenantAIProfile(
            tenant_id=tenant_id,
            version=next_version,
            label=label.strip() or f"Prompt v{next_version}",
            business_name=business_name.strip() or defaults["business_name"],
            greeting=greeting.strip() or defaults["greeting"],
            tone=tone.strip() or defaults["tone"],
            verbosity=verbosity.strip() or defaults["verbosity"],
            closing_line=closing_line.strip() or defaults["closing_line"],
            avoid_phrases_json=json.dumps([phrase.strip() for phrase in avoid_phrases if phrase.strip()]),
            preferred_terms_json=json.dumps([term.strip() for term in preferred_terms if term.strip()]),
            extra_instructions_text=extra_instructions_text.strip(),
            realtime_model=normalize_realtime_model(realtime_model) if (realtime_model or "").strip() else None,
            is_active=activate,
        )
        db.add(profile)
        db.flush()
        return _prompt_profile_summary(profile)


def activate_prompt_profile(tenant_id: int, profile_id: int) -> Optional[dict]:
    with session_scope() as db:
        profile = (
            db.query(TenantAIProfile)
            .filter(TenantAIProfile.tenant_id == tenant_id, TenantAIProfile.id == profile_id)
            .one_or_none()
        )
        if profile is None:
            return None
        db.query(TenantAIProfile).filter(TenantAIProfile.tenant_id == tenant_id).update({"is_active": False})
        profile.is_active = True
        profile.updated_at = utcnow()
        db.flush()
        return _prompt_profile_summary(profile)


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


def find_tenant_phone_by_number(twilio_number: str) -> Optional[dict]:
    normalized = _normalize_phone(twilio_number)
    with session_scope() as db:
        phone = (
            db.query(TenantPhoneNumber)
            .join(Tenant)
            .filter(TenantPhoneNumber.twilio_number == twilio_number, TenantPhoneNumber.active.is_(True))
            .one_or_none()
        )
        if phone is None and normalized:
            phones = (
                db.query(TenantPhoneNumber)
                .join(Tenant)
                .filter(TenantPhoneNumber.active.is_(True))
                .all()
            )
            matches = [candidate for candidate in phones if _normalize_phone(candidate.twilio_number) == normalized]
            if len(matches) == 1:
                phone = matches[0]
        if not phone or not phone.tenant or phone.tenant.status == "archived":
            return None
        return _tenant_phone_summary(phone)


def find_tenant_by_phone_number(twilio_number: str) -> Optional[dict]:
    phone = find_tenant_phone_by_number(twilio_number)
    return get_tenant(phone["tenant_id"]) if phone else None


def resolve_tenant_for_phone(twilio_number: str) -> tuple[Optional[dict], bool]:
    tenant = find_tenant_by_phone_number(twilio_number)
    if tenant:
        return tenant, True
    return None, False


def resolve_tenant_phone_for_number(twilio_number: str) -> tuple[Optional[dict], Optional[dict], bool]:
    phone = find_tenant_phone_by_number(twilio_number)
    if not phone:
        return None, None, False
    tenant = get_tenant(phone["tenant_id"])
    if not tenant:
        return None, None, False
    return tenant, phone, True


def get_call_tenant(call_sid: str) -> Optional[dict]:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call and call.tenant:
            return _tenant_summary(call.tenant)
        return None


def create_tenant(name: str, slug: str, business_name: str, greeting: str, notification_sms_number: str, backup_notification_sms_number: str = "", status: str = "onboarding") -> dict:
    with session_scope() as db:
        tenant = Tenant(name=name.strip(), slug=slug.strip(), status=status or "onboarding")
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
        _create_default_telephony_profile(db, tenant)
        _create_default_intake_policy(db, tenant)
        _create_default_notification_policy(db, tenant)
        _create_default_prompt_profile(db, tenant)
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
        old_notification_number = settings.notification_sms_number or ""
        old_backup_number = settings.backup_notification_sms_number or ""
        settings.business_name = business_name.strip() or settings.business_name
        settings.greeting = greeting.strip() or settings.greeting
        settings.notification_sms_number = notification_sms_number.strip()
        settings.backup_notification_sms_number = backup_notification_sms_number.strip() or None
        settings.active = tenant.status not in {"paused", "archived"}
        policy = _get_or_create_notification_policy(db, tenant)
        current_normal_recipients = _json_text_list(policy.normal_sms_recipients_json)
        old_normal_recipients = [old_notification_number] if old_notification_number else []
        if not current_normal_recipients or current_normal_recipients == old_normal_recipients:
            policy.normal_sms_recipients_json = json.dumps([notification_sms_number.strip()] if notification_sms_number.strip() else [])
        current_backup_recipients = _json_text_list(policy.backup_sms_recipients_json)
        old_backup_recipients = [old_backup_number] if old_backup_number else []
        if not current_backup_recipients or current_backup_recipients == old_backup_recipients:
            policy.backup_sms_recipients_json = json.dumps([backup_notification_sms_number.strip()] if backup_notification_sms_number.strip() else [])
        policy.updated_at = utcnow()
        db.flush()
        return _tenant_summary(tenant)


def add_tenant_phone_number(
    tenant_id: int,
    twilio_number: str,
    label: str = "",
    active: bool = True,
    accepts_live_calls: bool = False,
    purpose: str = "",
) -> dict:
    with session_scope() as db:
        existing = db.query(TenantPhoneNumber).filter(TenantPhoneNumber.twilio_number == twilio_number).one_or_none()
        if existing is not None:
            existing.tenant_id = tenant_id
            existing.label = label.strip() or existing.label
            existing.active = active
            existing.accepts_live_calls = accepts_live_calls
            existing.purpose = purpose.strip() or existing.purpose
            db.flush()
            return _tenant_phone_summary(existing)
        phone = TenantPhoneNumber(
            tenant_id=tenant_id,
            twilio_number=twilio_number.strip(),
            label=label.strip() or None,
            active=active,
            accepts_live_calls=accepts_live_calls,
            purpose=purpose.strip() or None,
        )
        db.add(phone)
        db.flush()
        return _tenant_phone_summary(phone)


def create_or_update_call(
    call_sid: str,
    from_number: str,
    to_number: str,
    tenant_id: Optional[int] = None,
    prompt_version_id: Optional[int] = None,
    realtime_model: Optional[str] = None,
    realtime_reasoning_effort: Optional[str] = None,
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
                prompt_version_id=prompt_version_id,
                realtime_model=realtime_model,
                realtime_reasoning_effort=realtime_reasoning_effort,
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
            if prompt_version_id is not None:
                call.prompt_version_id = prompt_version_id
            if realtime_model is not None:
                call.realtime_model = realtime_model
            if realtime_reasoning_effort is not None:
                call.realtime_reasoning_effort = realtime_reasoning_effort
            call.from_number = from_number or call.from_number
            call.to_number = to_number or call.to_number
            if status:
                call.status = status
        return _call_summary(call)


def update_call_stream_started(
    call_sid: str,
    stream_sid: str,
    prompt_version_id: Optional[int] = None,
    realtime_model: Optional[str] = None,
    realtime_reasoning_effort: Optional[str] = None,
) -> dict:
    with session_scope() as db:
        call = _get_call(db, call_sid)
        if call is None:
            call = Call(
                tenant_id=_default_tenant(db).id,
                prompt_version_id=prompt_version_id,
                realtime_model=realtime_model,
                realtime_reasoning_effort=realtime_reasoning_effort,
                call_sid=call_sid,
                stream_sid=stream_sid,
                status="stream_started",
            )
            db.add(call)
        else:
            call.stream_sid = stream_sid
            if prompt_version_id is not None and call.prompt_version_id is None:
                call.prompt_version_id = prompt_version_id
            if realtime_model is not None:
                call.realtime_model = realtime_model
            if realtime_reasoning_effort is not None:
                call.realtime_reasoning_effort = realtime_reasoning_effort
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


def get_lead(lead_id: int) -> Optional[dict]:
    with session_scope() as db:
        lead = db.query(Lead).filter(Lead.id == lead_id).one_or_none()
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


def get_notification(notification_id: int) -> Optional[dict]:
    with session_scope() as db:
        notification = db.query(Notification).filter(Notification.id == notification_id).one_or_none()
        return _notification_summary(notification) if notification else None


def list_notifications_for_lead(lead_id: int, channel: str = "sms") -> list[dict]:
    with session_scope() as db:
        notifications = (
            db.query(Notification)
            .filter(Notification.lead_id == lead_id, Notification.channel == channel)
            .order_by(Notification.created_at)
            .all()
        )
        return [_notification_summary(notification) for notification in notifications]


def create_lead(
    call_sid: str,
    args: dict,
    tenant_id: Optional[int] = None,
    priority: str = "normal",
    priority_reason: str = "",
    classification: Optional[dict] = None,
) -> dict:
    raw_args_json = json.dumps(args, default=str)
    with session_scope() as db:
        call = _get_call(db, call_sid)
        lead_tenant_id = tenant_id or (call.tenant_id if call and call.tenant_id else _default_tenant(db).id)
        existing = db.query(Lead).filter(Lead.call_sid == call_sid).one_or_none()
        if existing is not None:
            return _lead_summary(existing)

        lead = Lead(
            tenant_id=lead_tenant_id,
            call_id=call.id if call else None,
            call_sid=call_sid,
            name=str(args.get("name", "")).strip(),
            callback=str(args.get("callback", "")).strip(),
            address=str(args.get("address", "")).strip(),
            issue=str(args.get("issue", "")).strip(),
            urgency=str(args.get("urgency", "")).strip(),
            extra_fields_json=json.dumps(args.get("extra_fields") or {}, default=str),
            raw_args_json=raw_args_json,
            priority=priority,
            priority_reason=priority_reason,
            classification_json=json.dumps(classification or {}, default=str),
            status="submitted",
        )
        db.add(lead)
        db.flush()
        return _lead_summary(lead)


def create_notification_attempt(
    lead_id: int,
    to_number: str,
    tenant_id: Optional[int] = None,
    recipient_type: str = "normal",
    policy_snapshot: Optional[dict] = None,
    status: str = "pending",
) -> dict:
    with session_scope() as db:
        lead = db.query(Lead).filter(Lead.id == lead_id).one()
        notification_tenant_id = tenant_id or lead.tenant_id
        existing = (
            db.query(Notification)
            .filter(
                Notification.lead_id == lead_id,
                Notification.channel == "sms",
                Notification.to_number == to_number,
            )
            .one_or_none()
        )
        if existing is not None:
            if existing.status == "sent":
                return _notification_summary(existing)
            existing.recipient_type = recipient_type or existing.recipient_type
            existing.policy_snapshot_json = json.dumps(policy_snapshot or {}, default=str)
            existing.status = status
            existing.error = None if status == "pending" else existing.error
            existing.attempt_number = (existing.attempt_number or 1) + 1
            db.flush()
            return _notification_summary(existing)
        notification = Notification(
            tenant_id=notification_tenant_id,
            lead_id=lead.id,
            channel="sms",
            to_number=to_number,
            recipient_type=recipient_type,
            status=status,
            policy_snapshot_json=json.dumps(policy_snapshot or {}, default=str),
            attempt_number=1,
        )
        db.add(notification)
        db.flush()
        return _notification_summary(notification)


def create_lead_with_pending_notification(call_sid: str, args: dict, to_number: str, tenant_id: Optional[int] = None) -> tuple[dict, dict]:
    lead = create_lead(call_sid, args, tenant_id=tenant_id)
    notification = create_notification_attempt(
        lead["id"],
        to_number,
        tenant_id=lead.get("tenant_id"),
        recipient_type="normal",
        policy_snapshot={},
    )
    return lead, notification


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


def list_failed_notifications(limit: int = 25, tenant_id: Optional[int] = None) -> list[dict]:
    with session_scope() as db:
        query = db.query(Notification).filter(Notification.status == "failed")
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
            "prompt_profile": _prompt_profile_summary(call.prompt_profile) if call and call.prompt_profile else None,
            "leads": [_lead_summary(lead) for lead in leads],
            "notifications": [_notification_summary(notification) for notification in notifications],
            "events": [_event_summary(event) for event in events],
        }
