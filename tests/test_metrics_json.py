"""Phase C change #2: /admin/metrics.json read-only metrics endpoint (A-005).

Reuses repository.pilot_metrics and exposes the charter's derived rates as JSON
so weekly numbers have a citable source instead of manual log scraping.
"""

import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from admin.routes import create_admin_router
from config.settings import Settings
from storage import repository
from storage.database import configure_database, init_db


VALID_ARGS = {
    "issue": "Kitchen sink leaking underneath",
    "urgency": "Water still coming out but caller can shut it off",
    "address": "6100 West 120th Street",
    "callback": "732-789-0675",
    "name": "Sam",
    "extra_fields": {"property_role": "renting", "additional_notes": "dog in backyard"},
}


def _settings(tmpdir):
    return Settings(
        openai_api_key="",
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_phone_number="+15551234567",
        plumber_phone_number="+15557654321",
        host="example.test",
        oai_url="wss://example.test/realtime",
        database_url=f"sqlite:///{tmpdir}/test.db",
        admin_password="secret",
        default_tenant_name="Default Plumbing",
        default_tenant_slug="default",
        default_tenant_greeting="Plumbing office, what's going on?",
    )


class MetricsJsonTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db(_settings(self.tmpdir.name))
        app = FastAPI()
        app.include_router(create_admin_router(_settings(self.tmpdir.name)))
        self.client = TestClient(app)
        self.tenant = repository.get_default_tenant()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_call(self, call_sid, with_lead=False, sms_failed=False):
        repository.create_or_update_call(
            call_sid,
            "+19135550123",
            "+15551234567",
            tenant_id=self.tenant["id"],
            prompt_version_id=1,
            realtime_model="gpt-realtime-2",
            realtime_reasoning_effort="low",
        )
        if not with_lead:
            return
        lead = repository.create_lead(
            call_sid,
            dict(VALID_ARGS),
            tenant_id=self.tenant["id"],
            priority="normal",
            priority_reason="No emergency keywords matched.",
        )
        notification = repository.create_notification_attempt(
            lead["id"],
            "+15557654321",
            tenant_id=self.tenant["id"],
            recipient_type="normal",
        )
        if sms_failed:
            repository.mark_notification_failed(notification["id"], "forced failure")
            repository.record_call_event(call_sid, "sms_failed", {"notification_id": notification["id"]})
        else:
            repository.mark_notification_sent(notification["id"], "SM_TEST")
            repository.record_call_event(call_sid, "sms_sent", {"notification_id": notification["id"]})

    def test_requires_admin_auth(self):
        self.assertEqual(self.client.get("/admin/metrics.json").status_code, 401)

    def test_returns_metrics_and_derived_rates(self):
        # 4 calls answered: 3 produced leads (one SMS delivery failed), 1 produced none.
        self._create_call("CALL_A", with_lead=True)
        self._create_call("CALL_B", with_lead=True)
        self._create_call("CALL_C", with_lead=True, sms_failed=True)
        self._create_call("CALL_D", with_lead=False)

        response = self.client.get("/admin/metrics.json", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["calls_answered"], 4)
        self.assertEqual(data["calls_with_leads"], 3)
        self.assertEqual(data["calls_without_leads"], 1)
        self.assertAlmostEqual(data["qualification_completion_rate"], 0.75)
        self.assertEqual(data["sms_sent"], 2)
        self.assertEqual(data["sms_failed"], 1)
        self.assertAlmostEqual(data["sms_delivery_rate"], round(2 / 3, 4))
        self.assertIn("generated_at", data)
        self.assertTrue(
            any(row["tenant_id"] == self.tenant["id"] for row in data["by_tenant"]),
            "default tenant should appear in by_tenant",
        )

    def test_rates_handle_zero_denominator(self):
        response = self.client.get("/admin/metrics.json", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["calls_answered"], 0)
        self.assertIsNone(data["qualification_completion_rate"])
        self.assertIsNone(data["sms_delivery_rate"])


if __name__ == "__main__":
    unittest.main()
