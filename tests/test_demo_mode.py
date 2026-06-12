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
    "extra_fields": {
        "property_role": "renting",
        "additional_notes": "dog in backyard",
    },
}


def demo_settings(tmpdir):
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


class DemoModeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        self.settings = demo_settings(self.tmpdir.name)
        init_db(self.settings)
        app = FastAPI()
        app.include_router(create_admin_router(self.settings))
        self.client = TestClient(app)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_demo_lead(self, call_sid="CALL_DEMO_SUCCESS", emergency=False):
        tenant = repository.ensure_demo_tenant(
            notification_sms_number="+15557654321",
            ai_ingress_twilio_number="+15558889999",
            allowed_test_callers=["+17327890675"],
        )
        repository.create_or_update_call(
            call_sid,
            "+17327890675",
            "+15558889999",
            tenant_id=tenant["id"],
            realtime_model="gpt-realtime-1.5",
        )
        args = dict(VALID_ARGS)
        if emergency:
            args["urgency"] = "Water is still coming out and caller cannot shut it off"
        lead = repository.create_lead(
            call_sid,
            args,
            tenant_id=tenant["id"],
            priority="emergency" if emergency else "normal",
            priority_reason="Matched emergency keyword" if emergency else "No emergency keywords matched.",
        )
        notification = repository.create_notification_attempt(
            lead["id"],
            "+15557654321",
            tenant_id=tenant["id"],
            recipient_type="emergency" if emergency else "normal",
        )
        repository.mark_notification_sent(notification["id"], "SM_DEMO")
        return tenant, lead

    def test_demo_tenant_can_be_created_and_is_idempotent(self):
        default_tenant = repository.get_default_tenant()
        self.assertFalse(default_tenant["is_demo"])

        first = repository.ensure_demo_tenant(
            notification_sms_number="+15557654321",
            ai_ingress_twilio_number="+15558889999",
            allowed_test_callers=["+17327890675"],
        )
        second = repository.ensure_demo_tenant(
            notification_sms_number="+15557654321",
            ai_ingress_twilio_number="+15558889999",
            allowed_test_callers=["+17327890675"],
        )

        self.assertEqual(first["id"], second["id"])
        self.assertTrue(first["is_demo"])
        self.assertEqual(first["slug"], "demo-plumbing")
        self.assertEqual(first["status"], "testing")
        phones = repository.list_tenant_phone_numbers(first["id"])
        self.assertEqual(phones[0]["twilio_number"], "+15558889999")
        self.assertFalse(phones[0]["accepts_live_calls"])

    def test_demo_defaults_include_prompt_intake_and_notification_policy(self):
        tenant = repository.ensure_demo_tenant(notification_sms_number="+15557654321")

        profile = repository.get_active_prompt_profile(tenant["id"])
        intake_policy = repository.get_intake_policy(tenant["id"])
        notification_policy = repository.get_notification_policy(tenant["id"])

        self.assertIn("Sales demo", profile["label"])
        self.assertIn("Demo Plumbing", profile["business_name"])
        self.assertIn("property_role", intake_policy["extra_questions_json"])
        self.assertIn("Homeowner or renter", intake_policy["extra_questions_json"])
        self.assertIn("additional_notes", intake_policy["extra_questions_json"])
        self.assertIn("+15557654321", notification_policy["normal_sms_recipients_json"])
        self.assertTrue(notification_policy["send_emergency_leads"])

    def test_admin_demo_page_renders_scripts_and_checklist(self):
        repository.ensure_demo_tenant(notification_sms_number="+15557654321")

        response = self.client.get("/admin/demo", auth=("admin", "secret"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sales Demo Mode", response.text)
        self.assertIn("Demo Readiness Checklist", response.text)
        self.assertIn("Normal leak", response.text)
        self.assertIn("Emergency active leak", response.text)
        self.assertIn("One-shot info dump", response.text)
        self.assertIn("Create / Ensure Demo Tenant", response.text)

    def test_admin_create_demo_redirects_to_demo_tenant(self):
        response = self.client.post(
            "/admin/demo/ensure",
            data={
                "notification_sms_number": "+15557654321",
                "ai_ingress_twilio_number": "+15558889999",
                "allowed_test_callers": "+17327890675",
                "status": "testing",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        tenant = repository.get_demo_tenant()
        self.assertEqual(response.headers["location"], f"/admin/tenants/{tenant['id']}")

    def test_demo_filters_distinguish_demo_and_non_demo_records(self):
        demo_tenant, _lead = self._create_demo_lead("CALL_DEMO_FILTER")
        default_tenant = repository.get_default_tenant()
        repository.create_or_update_call(
            "CALL_REAL_FILTER",
            "+19135550123",
            "+15551234567",
            tenant_id=default_tenant["id"],
        )

        demo_rows = repository.list_call_review_queue(demo_filter="demo")
        real_rows = repository.list_call_review_queue(demo_filter="hide_demo")
        demo_metrics = repository.pilot_metrics(demo_filter="demo")
        real_metrics = repository.pilot_metrics(demo_filter="hide_demo")

        self.assertIn("CALL_DEMO_FILTER", {row["call_sid"] for row in demo_rows})
        self.assertNotIn("CALL_REAL_FILTER", {row["call_sid"] for row in demo_rows})
        self.assertIn("CALL_REAL_FILTER", {row["call_sid"] for row in real_rows})
        self.assertEqual(demo_metrics["total_calls"], 1)
        self.assertEqual(real_metrics["total_calls"], 1)
        self.assertTrue(repository.get_tenant(demo_tenant["id"])["is_demo"])

    def test_admin_leads_review_and_metrics_have_demo_filter(self):
        self._create_demo_lead("CALL_DEMO_ADMIN_FILTER")

        leads = self.client.get("/admin/leads?demo_filter=demo", auth=("admin", "secret"))
        review = self.client.get("/admin/review?demo_filter=demo", auth=("admin", "secret"))
        metrics = self.client.get("/admin/metrics?demo_filter=demo", auth=("admin", "secret"))

        self.assertEqual(leads.status_code, 200)
        self.assertEqual(review.status_code, 200)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("tenant_is_demo", leads.text)
        self.assertIn("CALL_DEMO_ADMIN_FILTER", review.text)
        self.assertIn("is_demo", metrics.text)

    def test_demo_fallback_records_show_successful_demo_leads(self):
        self._create_demo_lead("CALL_DEMO_FALLBACK", emergency=True)

        fallback = repository.list_demo_successful_leads()
        page = self.client.get("/admin/demo", auth=("admin", "secret"))

        self.assertEqual(fallback[0]["call_sid"], "CALL_DEMO_FALLBACK")
        self.assertIn("dog in backyard", fallback[0]["summary_text"])
        self.assertIn("Fallback Demo Records", page.text)
        self.assertIn("CALL_DEMO_FALLBACK", page.text)

    def test_demo_archive_marks_only_demo_records_as_test(self):
        demo_tenant, demo_lead = self._create_demo_lead("CALL_DEMO_ARCHIVE")
        default_tenant = repository.get_default_tenant()
        repository.create_or_update_call("CALL_REAL_ARCHIVE", "+19135550123", "+15551234567", tenant_id=default_tenant["id"])
        real_lead = repository.create_lead("CALL_REAL_ARCHIVE", VALID_ARGS, tenant_id=default_tenant["id"])

        result = repository.archive_demo_records()

        archived_demo = repository.get_lead(demo_lead["id"])
        untouched_real = repository.get_lead(real_lead["id"])
        demo_detail = repository.get_call_detail("CALL_DEMO_ARCHIVE")
        real_detail = repository.get_call_detail("CALL_REAL_ARCHIVE")

        self.assertGreaterEqual(result["calls_marked"], 1)
        self.assertEqual(archived_demo["lead_quality"], "test")
        self.assertEqual(untouched_real["lead_quality"], "unknown")
        self.assertEqual(demo_detail["review"]["review_status"], "good")
        self.assertEqual(real_detail["review"]["review_status"], "unreviewed")
        self.assertTrue(repository.get_tenant(demo_tenant["id"])["is_demo"])

    def test_admin_demo_archive_route_does_not_delete_records(self):
        self._create_demo_lead("CALL_DEMO_ARCHIVE_ROUTE")

        response = self.client.post("/admin/demo/archive", auth=("admin", "secret"), follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertIsNotNone(repository.get_call_detail("CALL_DEMO_ARCHIVE_ROUTE")["call"])
        self.assertIsNotNone(repository.get_lead_by_call_sid("CALL_DEMO_ARCHIVE_ROUTE"))


if __name__ == "__main__":
    unittest.main()
