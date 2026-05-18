from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True)
    prompt_version_id = Column(Integer, ForeignKey("tenant_ai_profiles.id"), nullable=True, index=True)
    call_sid = Column(String(128), unique=True, nullable=False, index=True)
    stream_sid = Column(String(128), nullable=True, index=True)
    from_number = Column(String(64), nullable=True)
    to_number = Column(String(64), nullable=True)
    status = Column(String(64), nullable=False, default="new")
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="calls")
    prompt_profile = relationship("TenantAIProfile")
    leads = relationship("Lead", back_populates="call")
    events = relationship("CallEvent", back_populates="call")


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=True, index=True)
    call_sid = Column(String(128), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    callback = Column(String(64), nullable=False)
    address = Column(Text, nullable=False)
    issue = Column(Text, nullable=False)
    urgency = Column(Text, nullable=False)
    extra_fields_json = Column(Text, nullable=True)
    raw_args_json = Column(Text, nullable=False)
    status = Column(String(64), nullable=False, default="submitted")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    tenant = relationship("Tenant", back_populates="leads")
    call = relationship("Call", back_populates="leads")
    notifications = relationship("Notification", back_populates="lead")


class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (UniqueConstraint("lead_id", "channel", "to_number", name="uq_notification_lead_channel_to"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False, index=True)
    channel = Column(String(32), nullable=False)
    to_number = Column(String(64), nullable=False)
    status = Column(String(64), nullable=False, default="pending")
    provider_message_sid = Column(String(128), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    sent_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="notifications")
    lead = relationship("Lead", back_populates="notifications")


class CallEvent(Base):
    __tablename__ = "call_events"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=True, index=True)
    call_sid = Column(String(128), nullable=False, index=True)
    event_type = Column(String(128), nullable=False, index=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    tenant = relationship("Tenant", back_populates="events")
    call = relationship("Call", back_populates="events")


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(128), unique=True, nullable=False, index=True)
    status = Column(String(64), nullable=False, default="onboarding")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    phone_numbers = relationship("TenantPhoneNumber", back_populates="tenant")
    settings = relationship("TenantSettings", back_populates="tenant", uselist=False)
    telephony_profile = relationship("TenantTelephonyProfile", back_populates="tenant", uselist=False)
    intake_policy = relationship("TenantIntakePolicy", back_populates="tenant", uselist=False)
    ai_profiles = relationship("TenantAIProfile", back_populates="tenant")
    calls = relationship("Call", back_populates="tenant")
    leads = relationship("Lead", back_populates="tenant")
    notifications = relationship("Notification", back_populates="tenant")
    events = relationship("CallEvent", back_populates="tenant")


class TenantPhoneNumber(Base):
    __tablename__ = "tenant_phone_numbers"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    twilio_number = Column(String(64), unique=True, nullable=False, index=True)
    label = Column(String(255), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    accepts_live_calls = Column(Boolean, nullable=False, default=False)
    purpose = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    tenant = relationship("Tenant", back_populates="phone_numbers")


class TenantTelephonyProfile(Base):
    __tablename__ = "tenant_telephony_profiles"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), unique=True, nullable=False, index=True)
    public_business_number = Column(String(64), nullable=True)
    ai_ingress_twilio_number = Column(String(64), nullable=True)
    routing_mode = Column(String(128), nullable=False, default="forwarded_google_maps_number")
    forwarding_setup_status = Column(String(64), nullable=False, default="not_started")
    test_mode_enabled = Column(Boolean, nullable=False, default=False)
    allowed_test_callers_json = Column(Text, nullable=False, default="[]")
    live_enabled_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=False, default="")

    tenant = relationship("Tenant", back_populates="telephony_profile")


class TenantIntakePolicy(Base):
    __tablename__ = "tenant_intake_policies"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), unique=True, nullable=False, index=True)
    enabled = Column(Boolean, nullable=False, default=True)
    extra_questions_json = Column(Text, nullable=False, default="[]")
    conditional_questions_json = Column(Text, nullable=False, default="[]")
    sms_include_extra_fields_json = Column(Text, nullable=False, default="[]")
    admin_display_fields_json = Column(Text, nullable=False, default="[]")
    notes = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    tenant = relationship("Tenant", back_populates="intake_policy")


class TenantSettings(Base):
    __tablename__ = "tenant_settings"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), unique=True, nullable=False, index=True)
    business_name = Column(String(255), nullable=False)
    greeting = Column(Text, nullable=False)
    notification_sms_number = Column(String(64), nullable=True)
    backup_notification_sms_number = Column(String(64), nullable=True)
    voice = Column(String(128), nullable=True)
    model = Column(String(128), nullable=True)
    active = Column(Boolean, nullable=False, default=True)

    tenant = relationship("Tenant", back_populates="settings")


class TenantAIProfile(Base):
    __tablename__ = "tenant_ai_profiles"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    label = Column(String(255), nullable=False)
    business_name = Column(String(255), nullable=False)
    greeting = Column(Text, nullable=False)
    tone = Column(Text, nullable=False)
    verbosity = Column(Text, nullable=False)
    closing_line = Column(Text, nullable=False)
    avoid_phrases_json = Column(Text, nullable=False)
    preferred_terms_json = Column(Text, nullable=False)
    extra_instructions_text = Column(Text, nullable=False, default="")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    tenant = relationship("Tenant", back_populates="ai_profiles")
