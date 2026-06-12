import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from admin.routes import create_admin_router
from config.settings import Settings
from main import build_sms_body
from storage import repository
from storage.database import configure_database, init_db
from workflow.notifications import SmsSendResult
from workflow.priority import classify_lead_priority
from workflow.service_request import process_service_request


BASE_ARGS = {
    "issue": "Slow drip under bathroom sink",
    "urgency": "Not active, can wait",
    "address": "6100 West 120th Street",
    "callback": "732-789-0675",
    "name": "Sam",
    "extra_fields": {"property_role": "homeowner", "additional_notes": "gate code 1234"},
}

CALLER_TEXT = "My name is Sam. The address is 6100 West 120th Street."


class RoutingSmsSender:
    def __init__(self, fail_numbers=None):
        self.fail_numbers = set(fail_numbers or [])
        self.calls = []

    async def __call__(self, call_sid, args, from_number, to_number=None):
        self.calls.append((call_sid, args, from_number, to_number))
        if to_number in self.fail_numbers:
            return SmsSendResult(success=False, error=f"forced failure for {to_number}")
        return SmsSendResult(success=True, provider_message_sid=f"SM_{len(self.calls)}")


class NotificationPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db(
            Settings(
                openai_api_key="",
                twilio_account_sid="",
                twilio_auth_token="",
                twilio_phone_number="+15551234567",
                plumber_phone_number="+15557654321",
                host="example.test",
                oai_url="wss://example.test/realtime",
                database_url=f"sqlite:///{self.tmpdir.name}/test.db",
                admin_password="secret",
                default_tenant_name="Default Plumbing",
                default_tenant_slug="default",
                default_tenant_greeting="Plumbing office, what's going on?",
            )
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_notification_policy_uses_current_default_recipient(self):
        tenant = repository.get_default_tenant()
        policy = repository.get_notification_policy(tenant["id"])

        self.assertIn("+15557654321", policy["normal_sms_recipients_json"])
        self.assertTrue(policy["send_normal_leads"])
        self.assertTrue(policy["send_emergency_leads"])

    def test_tenant_settings_edit_preserves_explicit_notification_policy_recipients(self):
        tenant = repository.get_default_tenant()
        repository.update_notification_policy(
            tenant["id"],
            normal_sms_recipients=["+15550000001", "+15550000002"],
            emergency_sms_recipients=["+15550000003"],
            backup_sms_recipients=["+15550000004"],
        )

        repository.update_tenant_settings(
            tenant["id"],
            business_name="Default Plumbing Updated",
            greeting="Plumbing office, what's going on?",
            notification_sms_number="+15559999999",
            backup_notification_sms_number="+15558888888",
            status="live",
        )

        policy = repository.get_notification_policy(tenant["id"])
        self.assertIn("+15550000001", policy["normal_sms_recipients_json"])
        self.assertIn("+15550000002", policy["normal_sms_recipients_json"])
        self.assertIn("+15550000004", policy["backup_sms_recipients_json"])
        self.assertNotIn("+15559999999", policy["normal_sms_recipients_json"])

    def test_rule_based_emergency_classification(self):
        emergency = classify_lead_priority({"issue": "Burst pipe", "urgency": "Water is still coming out"})
        sewer = classify_lead_priority({"issue": "Sewer backup in basement", "urgency": "Bad smell"})
        normal = classify_lead_priority({"issue": "Slow drip under sink", "urgency": "Not active"})
        cannot_shut = classify_lead_priority({"issue": "Leak", "urgency": "Cannot shut water off"})

        self.assertEqual(emergency["priority"], "emergency")
        self.assertEqual(sewer["priority"], "emergency")
        self.assertEqual(normal["priority"], "normal")
        self.assertEqual(cannot_shut["priority"], "emergency")

    async def test_normal_lead_sends_to_normal_recipient(self):
        tenant = repository.get_default_tenant()
        repository.create_or_update_call("CALL_NORMAL_NOTIFY", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = RoutingSmsSender()

        result = await process_service_request(
            "CALL_NORMAL_NOTIFY",
            dict(BASE_ARGS),
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(sender.calls[0][3], "+15557654321")
        lead = repository.get_lead_by_call_sid("CALL_NORMAL_NOTIFY")
        self.assertEqual(lead["priority"], "normal")

    async def test_emergency_lead_sends_to_emergency_recipient(self):
        tenant = repository.get_default_tenant()
        repository.update_notification_policy(
            tenant["id"],
            normal_sms_recipients=["+15550000001"],
            emergency_sms_recipients=["+15550000002"],
            backup_sms_recipients=[],
            emergency_keywords=["water is still coming out"],
        )
        args = dict(BASE_ARGS)
        args["issue"] = "Kitchen sink leak"
        args["urgency"] = "Water is still coming out"
        repository.create_or_update_call("CALL_EMERGENCY_NOTIFY", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = RoutingSmsSender()

        result = await process_service_request(
            "CALL_EMERGENCY_NOTIFY",
            args,
            "+19135550123",
            "+15550000001",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(sender.calls[0][3], "+15550000002")
        lead = repository.get_lead_by_call_sid("CALL_EMERGENCY_NOTIFY")
        self.assertEqual(lead["priority"], "emergency")

    async def test_emergency_without_emergency_recipient_falls_back_to_normal(self):
        tenant = repository.get_default_tenant()
        repository.update_notification_policy(
            tenant["id"],
            normal_sms_recipients=["+15550000001"],
            emergency_sms_recipients=[],
            emergency_keywords=["burst pipe"],
        )
        args = dict(BASE_ARGS)
        args["issue"] = "Burst pipe"
        args["urgency"] = "Flooding"
        repository.create_or_update_call("CALL_EMERGENCY_FALLBACK", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = RoutingSmsSender()

        result = await process_service_request(
            "CALL_EMERGENCY_FALLBACK",
            args,
            "+19135550123",
            "+15550000001",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(sender.calls[0][3], "+15550000001")
        events = [event for event in repository.list_recent_call_events() if event["call_sid"] == "CALL_EMERGENCY_FALLBACK"]
        self.assertIn("notification_policy_routing_note", {event["event_type"] for event in events})

    async def test_duplicate_recipient_only_sends_once(self):
        tenant = repository.get_default_tenant()
        repository.update_notification_policy(
            tenant["id"],
            normal_sms_recipients=["+15550000001"],
            emergency_sms_recipients=["+15550000001"],
            emergency_keywords=["flooding"],
        )
        args = dict(BASE_ARGS)
        args["urgency"] = "Flooding"
        repository.create_or_update_call("CALL_DUP_RECIPIENT", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = RoutingSmsSender()

        result = await process_service_request(
            "CALL_DUP_RECIPIENT",
            args,
            "+19135550123",
            "+15550000001",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)

    async def test_tenant_a_does_not_notify_tenant_b_recipient(self):
        tenant_a = repository.create_tenant("Tenant A", "tenant-a-notify", "Tenant A", "Tenant A, what's going on?", "+15550000001")
        tenant_b = repository.create_tenant("Tenant B", "tenant-b-notify", "Tenant B", "Tenant B, what's going on?", "+15550000002")
        repository.create_or_update_call("CALL_TENANT_NOTIFY", "+19135550123", "+15550000001", tenant_id=tenant_a["id"])
        sender = RoutingSmsSender()

        await process_service_request(
            "CALL_TENANT_NOTIFY",
            dict(BASE_ARGS),
            "+19135550123",
            tenant_a["notification_sms_number"],
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant_a["id"],
        )

        self.assertEqual(sender.calls[0][3], tenant_a["notification_sms_number"])
        self.assertNotEqual(sender.calls[0][3], tenant_b["notification_sms_number"])

    async def test_failed_primary_attempts_backup_and_saves_lead(self):
        tenant = repository.get_default_tenant()
        repository.update_notification_policy(
            tenant["id"],
            normal_sms_recipients=["+15550000001"],
            backup_sms_recipients=["+15550000002"],
        )
        repository.create_or_update_call("CALL_BACKUP_NOTIFY", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = RoutingSmsSender(fail_numbers={"+15550000001"})

        result = await process_service_request(
            "CALL_BACKUP_NOTIFY",
            dict(BASE_ARGS),
            "+19135550123",
            "+15550000001",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual([call[3] for call in sender.calls], ["+15550000001", "+15550000002"])
        lead = repository.get_lead_by_call_sid("CALL_BACKUP_NOTIFY")
        notifications = repository.list_notifications_for_lead(lead["id"])
        statuses = {notification["to_number"]: notification["status"] for notification in notifications}
        self.assertEqual(statuses["+15550000001"], "failed")
        self.assertEqual(statuses["+15550000002"], "sent")

    async def test_duplicate_submit_does_not_duplicate_notifications(self):
        tenant = repository.get_default_tenant()
        repository.create_or_update_call("CALL_DUP_NOTIFY", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = RoutingSmsSender()

        first = await process_service_request(
            "CALL_DUP_NOTIFY",
            dict(BASE_ARGS),
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )
        second = await process_service_request(
            "CALL_DUP_NOTIFY",
            dict(BASE_ARGS),
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(first.output["success"])
        self.assertEqual(second.output["reason"], "already_submitted")
        self.assertEqual(len(sender.calls), 1)

    def test_sms_body_includes_priority_and_policy_controlled_extra_fields(self):
        tenant = repository.get_default_tenant()
        intake_policy = repository.get_intake_policy(tenant["id"])
        policy = repository.update_notification_policy(
            tenant["id"],
            normal_sms_recipients=["+15550000001"],
            include_extra_fields=True,
            include_additional_notes=True,
        )
        emergency_args = dict(BASE_ARGS)
        emergency_args["priority"] = "emergency"
        emergency_args["priority_reason"] = "Matched emergency keyword: flooding"

        body = build_sms_body(emergency_args, "+19135550123", intake_policy, policy)

        self.assertTrue(body.startswith("EMERGENCY PLUMBING LEAD"))
        self.assertIn("Priority: EMERGENCY", body)
        self.assertIn("Additional notes: gate code 1234", body)
        self.assertNotIn("Homowner", body)

    def test_admin_notification_policy_page_renders(self):
        tenant = repository.get_default_tenant()
        app = FastAPI()
        app.include_router(
            create_admin_router(
                Settings(
                    openai_api_key="",
                    twilio_account_sid="",
                    twilio_auth_token="",
                    twilio_phone_number="+15551234567",
                    plumber_phone_number="+15557654321",
                    host="example.test",
                    oai_url="wss://example.test/realtime",
                    database_url=f"sqlite:///{self.tmpdir.name}/test.db",
                    admin_password="secret",
                    default_tenant_name="Default Plumbing",
                    default_tenant_slug="default",
                    default_tenant_greeting="Plumbing office, what's going on?",
                )
            )
        )
        client = TestClient(app)

        detail = client.get(f"/admin/tenants/{tenant['id']}", auth=("admin", "secret"))
        page = client.get(f"/admin/tenants/{tenant['id']}/notification-policy", auth=("admin", "secret"))

        self.assertEqual(page.status_code, 200)
        self.assertIn("Notification Policy", page.text)
        self.assertIn("emergency_sms_recipients", page.text)
        self.assertIn("EMERGENCY PLUMBING LEAD", page.text)
        self.assertIn(f'href="/admin/tenants/{tenant["id"]}/notification-policy"', detail.text)


if __name__ == "__main__":
    unittest.main()
