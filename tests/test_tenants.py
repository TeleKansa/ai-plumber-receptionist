import tempfile
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import text

import main
from config.settings import Settings
from storage import database, repository
from storage.database import configure_database, init_db


def tenant_settings(tmpdir, twilio_number="+15551234567", notification_number="+15557654321"):
    return Settings(
        openai_api_key="",
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_phone_number=twilio_number,
        plumber_phone_number=notification_number,
        host="example.test",
        oai_url="wss://example.test/realtime",
        database_url=f"sqlite:///{tmpdir}/test.db",
        admin_password="secret",
        default_tenant_name="Default Plumbing",
        default_tenant_slug="default",
        default_tenant_greeting="Plumbing office, what's going on?",
    )


class TenantTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        main.sessions.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_tenant_is_created_from_settings(self):
        init_db(tenant_settings(self.tmpdir.name))

        tenant = repository.get_default_tenant()
        phones = repository.list_tenant_phone_numbers(tenant["id"])
        active_prompt = repository.get_active_prompt_profile(tenant["id"])
        telephony_profile = repository.get_telephony_profile(tenant["id"])

        self.assertEqual(tenant["name"], "Default Plumbing")
        self.assertEqual(tenant["status"], "live")
        self.assertEqual(tenant["notification_sms_number"], "+15557654321")
        self.assertEqual(phones[0]["twilio_number"], "+15551234567")
        self.assertTrue(phones[0]["accepts_live_calls"])
        self.assertEqual(phones[0]["purpose"], "ai_forwarding")
        self.assertEqual(telephony_profile["ai_ingress_twilio_number"], "+15551234567")
        self.assertEqual(telephony_profile["forwarding_setup_status"], "verified")
        self.assertIsNotNone(active_prompt)
        self.assertEqual(active_prompt["version"], 1)
        self.assertTrue(active_prompt["is_active"])

    def test_new_tenant_defaults_to_onboarding_and_phone_not_live(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant Onboarding",
            "tenant-onboarding",
            "Tenant Onboarding Plumbing",
            "Tenant Onboarding plumbing, what's going on?",
            "+15550000001",
        )
        phone = repository.add_tenant_phone_number(tenant["id"], "+15551110001", "AI forwarding")
        telephony_profile = repository.get_telephony_profile(tenant["id"])

        self.assertEqual(tenant["status"], "onboarding")
        self.assertTrue(phone["active"])
        self.assertFalse(phone["accepts_live_calls"])
        self.assertEqual(telephony_profile["routing_mode"], "forwarded_google_maps_number")
        self.assertEqual(telephony_profile["forwarding_setup_status"], "not_started")
        self.assertFalse(telephony_profile["test_mode_enabled"])

    def test_phone_number_normalization_matches_common_formats(self):
        expected = "19135551234"

        self.assertEqual(repository.normalize_phone_number("+19135551234"), expected)
        self.assertEqual(repository.normalize_phone_number("19135551234"), expected)
        self.assertEqual(repository.normalize_phone_number("(913) 555-1234"), expected)
        self.assertEqual(repository.normalize_phone_number("913-555-1234"), expected)

    def test_startup_migration_backfills_existing_call_rows(self):
        with database.engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE calls ("
                    "id INTEGER PRIMARY KEY, "
                    "call_sid VARCHAR(128) NOT NULL UNIQUE, "
                    "stream_sid VARCHAR(128), "
                    "from_number VARCHAR(64), "
                    "to_number VARCHAR(64), "
                    "status VARCHAR(64) NOT NULL, "
                    "started_at DATETIME NOT NULL, "
                    "ended_at DATETIME)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO calls "
                    "(call_sid, from_number, to_number, status, started_at) "
                    "VALUES ('OLD_CALL', '+19135550123', '+15551234567', 'voice_received', '2026-01-01 00:00:00')"
                )
            )

        init_db(tenant_settings(self.tmpdir.name))

        detail = repository.get_call_detail("OLD_CALL")
        self.assertIsNotNone(detail["call"]["tenant_id"])
        self.assertEqual(detail["tenant"]["slug"], "default")

    def test_default_twilio_number_routes_to_default_tenant(self):
        init_db(tenant_settings(self.tmpdir.name, twilio_number="+15551234567"))
        default_tenant = repository.get_default_tenant()

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ROUTE_DEFAULT", "From": "+19135550123", "To": "(555) 123-4567"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        detail = repository.get_call_detail("CALL_ROUTE_DEFAULT")
        active_prompt = repository.get_active_prompt_profile(default_tenant["id"])
        self.assertEqual(detail["call"]["tenant_id"], default_tenant["id"])
        self.assertEqual(detail["call"]["prompt_version_id"], active_prompt["id"])
        self.assertEqual(detail["prompt_profile"]["id"], active_prompt["id"])
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertNotIn("tenant_lookup_failed", event_types)

    def test_live_tenant_with_live_phone_streams_to_tenant(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant A",
            "tenant-a-routing",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        phone = repository.add_tenant_phone_number(tenant["id"], "+15551110001", "Main")
        repository.set_tenant_live(tenant["id"])
        repository.set_tenant_phone_live(tenant["id"], phone["id"], True)

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ROUTE_A", "From": "+19135550123", "To": "(555) 111-0001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        detail = repository.get_call_detail("CALL_ROUTE_A")
        active_prompt = repository.get_active_prompt_profile(tenant["id"])
        self.assertEqual(detail["call"]["tenant_id"], tenant["id"])
        self.assertEqual(detail["call"]["prompt_version_id"], active_prompt["id"])
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertNotIn("tenant_lookup_failed", event_types)

    def test_known_tenant_b_phone_routes_to_tenant_b(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant_a = repository.create_tenant(
            "Tenant A",
            "tenant-a-routing",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        tenant_b = repository.create_tenant(
            "Tenant B",
            "tenant-b-routing",
            "Tenant B Plumbing",
            "Tenant B plumbing, what's going on?",
            "+15550000002",
        )
        phone_a = repository.add_tenant_phone_number(tenant_a["id"], "+15551110001", "Tenant A main")
        phone_b = repository.add_tenant_phone_number(tenant_b["id"], "555-222-0002", "Tenant B main")
        repository.set_tenant_live(tenant_a["id"])
        repository.set_tenant_live(tenant_b["id"])
        repository.set_tenant_phone_live(tenant_a["id"], phone_a["id"], True)
        repository.set_tenant_phone_live(tenant_b["id"], phone_b["id"], True)

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ROUTE_B", "From": "+19135550123", "To": "+15552220002"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        detail = repository.get_call_detail("CALL_ROUTE_B")
        self.assertEqual(detail["call"]["tenant_id"], tenant_b["id"])
        self.assertNotEqual(detail["call"]["tenant_id"], tenant_a["id"])

    def test_onboarding_tenant_blocks_before_stream(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant Onboarding Block",
            "tenant-onboarding-block",
            "Tenant Onboarding Block Plumbing",
            "Tenant Onboarding Block plumbing, what's going on?",
            "+15550000001",
        )
        repository.add_tenant_phone_number(tenant["id"], "+15551110001", "AI forwarding")

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ONBOARDING_BLOCK", "From": "+19135550123", "To": "+15551110001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sorry, this line is not active yet.", response.text)
        self.assertNotIn("<Stream", response.text)
        self.assertNotIn("CALL_ONBOARDING_BLOCK", main.sessions)
        detail = repository.get_call_detail("CALL_ONBOARDING_BLOCK")
        self.assertEqual(detail["call"]["tenant_id"], tenant["id"])
        self.assertEqual(detail["call"]["status"], "onboarding_blocked")
        self.assertIn("call_blocked", {event["event_type"] for event in detail["events"]})
        self.assertEqual(repository.list_recent_leads(), [])
        self.assertEqual(repository.list_recent_notifications(), [])

    def test_live_tenant_with_phone_live_switch_off_blocks_before_stream(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant Phone Switch",
            "tenant-phone-switch",
            "Tenant Phone Switch Plumbing",
            "Tenant Phone Switch plumbing, what's going on?",
            "+15550000001",
        )
        repository.add_tenant_phone_number(tenant["id"], "+15551110001", "AI forwarding")
        repository.set_tenant_live(tenant["id"])

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_PHONE_SWITCH_OFF", "From": "+19135550123", "To": "+15551110001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sorry, this line is not active yet.", response.text)
        self.assertNotIn("<Stream", response.text)
        detail = repository.get_call_detail("CALL_PHONE_SWITCH_OFF")
        self.assertEqual(detail["call"]["status"], "not_live")
        self.assertIn("call_blocked", {event["event_type"] for event in detail["events"]})

    def test_paused_tenant_blocks_before_stream(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant Paused",
            "tenant-paused",
            "Tenant Paused Plumbing",
            "Tenant Paused plumbing, what's going on?",
            "+15550000001",
        )
        phone = repository.add_tenant_phone_number(tenant["id"], "+15551110001", "AI forwarding")
        repository.set_tenant_live(tenant["id"])
        repository.set_tenant_phone_live(tenant["id"], phone["id"], True)
        repository.set_tenant_paused(tenant["id"])

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_PAUSED_BLOCK", "From": "+19135550123", "To": "+15551110001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sorry, this line is not active yet.", response.text)
        self.assertNotIn("<Stream", response.text)
        detail = repository.get_call_detail("CALL_PAUSED_BLOCK")
        self.assertEqual(detail["call"]["status"], "paused")
        self.assertIn("call_blocked", {event["event_type"] for event in detail["events"]})

    def test_testing_tenant_allows_only_allowed_callers(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant Testing",
            "tenant-testing",
            "Tenant Testing Plumbing",
            "Tenant Testing plumbing, what's going on?",
            "+15550000001",
        )
        repository.add_tenant_phone_number(tenant["id"], "+15551110001", "AI forwarding")
        repository.set_tenant_status(tenant["id"], "testing")
        repository.update_telephony_profile(
            tenant["id"],
            public_business_number="+19135550000",
            ai_ingress_twilio_number="+15551110001",
            forwarding_setup_status="customer_configured",
            test_mode_enabled=True,
            allowed_test_callers=["(913) 555-0123"],
            notes="testing",
        )

        client = TestClient(main.app)
        allowed_response = client.post(
            "/voice",
            data={"CallSid": "CALL_TEST_ALLOWED", "From": "+19135550123", "To": "+15551110001"},
        )
        blocked_response = client.post(
            "/voice",
            data={"CallSid": "CALL_TEST_BLOCKED", "From": "+19135559999", "To": "+15551110001"},
        )

        self.assertEqual(allowed_response.status_code, 200)
        self.assertIn("<Stream", allowed_response.text)
        self.assertEqual(blocked_response.status_code, 200)
        self.assertNotIn("<Stream", blocked_response.text)
        self.assertIn("Sorry, this line is not active yet.", blocked_response.text)
        blocked_detail = repository.get_call_detail("CALL_TEST_BLOCKED")
        self.assertEqual(blocked_detail["call"]["status"], "test_caller_not_allowed")
        self.assertIn("call_blocked", {event["event_type"] for event in blocked_detail["events"]})

    def test_unknown_to_number_rejects_without_default_fallback(self):
        init_db(tenant_settings(self.tmpdir.name))

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ROUTE_UNKNOWN", "From": "+19135550123", "To": "+15559999999"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sorry, this line is not configured yet.", response.text)
        self.assertIn("<Hangup", response.text)
        self.assertNotIn("<Stream", response.text)
        self.assertNotIn("CALL_ROUTE_UNKNOWN", main.sessions)

        detail = repository.get_call_detail("CALL_ROUTE_UNKNOWN")
        self.assertIsNone(detail["call"]["tenant_id"])
        self.assertIsNone(detail["tenant"])
        self.assertEqual(detail["call"]["status"], "tenant_lookup_failed")
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertIn("tenant_lookup_failed", event_types)
        self.assertEqual(repository.list_recent_leads(), [])
        self.assertEqual(repository.list_recent_notifications(), [])


if __name__ == "__main__":
    unittest.main()
