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

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_tenant_is_created_from_settings(self):
        init_db(tenant_settings(self.tmpdir.name))

        tenant = repository.get_default_tenant()
        phones = repository.list_tenant_phone_numbers(tenant["id"])

        self.assertEqual(tenant["name"], "Default Plumbing")
        self.assertEqual(tenant["notification_sms_number"], "+15557654321")
        self.assertEqual(phones[0]["twilio_number"], "+15551234567")

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

    def test_voice_maps_to_number_to_tenant(self):
        init_db(tenant_settings(self.tmpdir.name))
        tenant = repository.create_tenant(
            "Tenant A",
            "tenant-a-routing",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        repository.add_tenant_phone_number(tenant["id"], "+15551110001", "Main")

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ROUTE_A", "From": "+19135550123", "To": "+15551110001"},
        )

        self.assertEqual(response.status_code, 200)
        detail = repository.get_call_detail("CALL_ROUTE_A")
        self.assertEqual(detail["call"]["tenant_id"], tenant["id"])
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertNotIn("tenant_lookup_failed", event_types)

    def test_unknown_to_number_falls_back_to_default_and_records_event(self):
        init_db(tenant_settings(self.tmpdir.name))
        default_tenant = repository.get_default_tenant()

        client = TestClient(main.app)
        response = client.post(
            "/voice",
            data={"CallSid": "CALL_ROUTE_UNKNOWN", "From": "+19135550123", "To": "+15559999999"},
        )

        self.assertEqual(response.status_code, 200)
        detail = repository.get_call_detail("CALL_ROUTE_UNKNOWN")
        self.assertEqual(detail["call"]["tenant_id"], default_tenant["id"])
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertIn("tenant_lookup_failed", event_types)


if __name__ == "__main__":
    unittest.main()
