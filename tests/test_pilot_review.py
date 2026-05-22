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


def test_settings(tmpdir):
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


class PilotReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db(test_settings(self.tmpdir.name))
        app = FastAPI()
        app.include_router(create_admin_router(test_settings(self.tmpdir.name)))
        self.client = TestClient(app)
        self.tenant = repository.get_default_tenant()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_call(self, call_sid="CALL_REVIEW", with_lead=False, emergency=False, sms_failed=False):
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
            return None
        args = dict(VALID_ARGS)
        if emergency:
            args["urgency"] = "Water is still coming out and caller cannot shut it off"
        lead = repository.create_lead(
            call_sid,
            args,
            tenant_id=self.tenant["id"],
            priority="emergency" if emergency else "normal",
            priority_reason="Matched emergency keyword" if emergency else "No emergency keywords matched.",
        )
        notification = repository.create_notification_attempt(
            lead["id"],
            "+15557654321",
            tenant_id=self.tenant["id"],
            recipient_type="emergency" if emergency else "normal",
        )
        if sms_failed:
            repository.mark_notification_failed(notification["id"], "forced failure")
            repository.record_call_event(call_sid, "sms_failed", {"notification_id": notification["id"]})
        else:
            repository.mark_notification_sent(notification["id"], "SM_TEST")
            repository.record_call_event(call_sid, "sms_sent", {"notification_id": notification["id"]})
        return lead

    def test_new_call_defaults_to_unreviewed_and_can_save_review(self):
        self._create_call("CALL_REVIEW_DEFAULT")

        detail = repository.get_call_detail("CALL_REVIEW_DEFAULT")
        self.assertEqual(detail["review"]["review_status"], "unreviewed")
        self.assertEqual(detail["review"]["review_tags"], [])

        review = repository.save_call_review(
            "CALL_REVIEW_DEFAULT",
            review_status="needs_review",
            review_tags=["awkward_ai", "repeated_question"],
            internal_notes="AI repeated the callback question.",
        )

        self.assertEqual(review["review_status"], "needs_review")
        self.assertIn("awkward_ai", review["review_tags"])
        self.assertIn("AI repeated", review["internal_notes"])

    def test_lead_review_fields_can_be_saved(self):
        lead = self._create_call("CALL_LEAD_REVIEW", with_lead=True)

        updated = repository.update_lead_review(
            lead["id"],
            lead_quality="incomplete",
            lead_notes="Missing unit number.",
        )

        self.assertEqual(updated["lead_quality"], "incomplete")
        self.assertEqual(updated["lead_notes"], "Missing unit number.")

    def test_feedback_can_be_added_and_resolved(self):
        self._create_call("CALL_FEEDBACK")

        feedback = repository.add_call_feedback(
            "CALL_FEEDBACK",
            feedback_source="plumber",
            feedback_text="Customer said the AI sounded rushed.",
            action_needed="conversation_polish",
        )
        resolved = repository.set_call_feedback_resolved(feedback["id"], True)

        self.assertEqual(feedback["feedback_source"], "plumber")
        self.assertEqual(feedback["action_needed"], "conversation_polish")
        self.assertTrue(resolved["resolved"])
        self.assertIsNotNone(resolved["resolved_at"])

    def test_review_queue_includes_attention_calls(self):
        self._create_call("CALL_NO_LEAD")
        self._create_call("CALL_SMS_FAILED", with_lead=True, sms_failed=True)
        self._create_call("CALL_EMERGENCY", with_lead=True, emergency=True)
        self._create_call("CALL_GOOD", with_lead=True)
        repository.save_call_review("CALL_GOOD", review_status="good", review_tags=["good_call"])

        rows = repository.list_call_review_queue()
        call_sids = {row["call_sid"] for row in rows}

        self.assertIn("CALL_NO_LEAD", call_sids)
        self.assertIn("CALL_SMS_FAILED", call_sids)
        self.assertIn("CALL_EMERGENCY", call_sids)
        self.assertNotIn("CALL_GOOD", call_sids)

    def test_review_page_renders_and_filters(self):
        self._create_call("CALL_REVIEW_PAGE")

        response = self.client.get("/admin/review", auth=("admin", "secret"))
        filtered = self.client.get("/admin/review?has_lead=no", auth=("admin", "secret"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pilot Review Queue", response.text)
        self.assertIn("CALL_REVIEW_PAGE", response.text)
        self.assertIn("Calls Needing Attention", response.text)
        self.assertEqual(filtered.status_code, 200)
        self.assertIn("CALL_REVIEW_PAGE", filtered.text)

    def test_call_detail_shows_review_form_feedback_summary_and_notifications(self):
        self._create_call("CALL_DETAIL_REVIEW", with_lead=True, emergency=True)
        repository.record_call_event("CALL_DETAIL_REVIEW", "media_stream_done", {"media_stream_exit_reason": "twilio_stop"})

        response = self.client.get("/admin/calls/CALL_DETAIL_REVIEW", auth=("admin", "secret"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("Copy-Friendly PM Summary", response.text)
        self.assertIn("Kitchen sink leaking underneath", response.text)
        self.assertIn("Homeowner or renter", response.text)
        self.assertIn("dog in backyard", response.text)
        self.assertIn("Notification Attempts", response.text)
        self.assertIn("Review", response.text)
        self.assertIn("Save Review", response.text)
        self.assertIn("Add Feedback", response.text)
        self.assertIn("realtime_model", response.text)
        self.assertIn("prompt_version_id", response.text)
        self.assertIn("twilio_stop", response.text)

    def test_admin_can_save_review_and_feedback_from_call_detail(self):
        self._create_call("CALL_ADMIN_REVIEW")

        review_response = self.client.post(
            "/admin/calls/CALL_ADMIN_REVIEW/review",
            data={
                "review_status": "bad",
                "review_tags": ["caller_hung_up", "realtime_model_issue"],
                "internal_notes": "Caller hung up after a parse failure.",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        feedback_response = self.client.post(
            "/admin/calls/CALL_ADMIN_REVIEW/feedback",
            data={
                "feedback_source": "internal",
                "feedback_text": "Investigate Realtime 2 behavior.",
                "action_needed": "bug_fix",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )

        self.assertEqual(review_response.status_code, 303)
        self.assertEqual(feedback_response.status_code, 303)
        detail = repository.get_call_detail("CALL_ADMIN_REVIEW")
        self.assertEqual(detail["review"]["review_status"], "bad")
        self.assertIn("realtime_model_issue", detail["review"]["review_tags"])
        self.assertEqual(detail["feedback"][0]["action_needed"], "bug_fix")

    def test_rule_based_summary_handles_no_lead_gracefully(self):
        self._create_call("CALL_NO_LEAD_SUMMARY")
        repository.record_call_event("CALL_NO_LEAD_SUMMARY", "validation_failed", {"missing_fields": ["address"]})

        detail = repository.get_call_detail("CALL_NO_LEAD_SUMMARY")
        summary = detail["summary_text"]

        self.assertIn("No lead created", summary)
        self.assertIn("validation_failed", summary)

    def test_metrics_page_renders_basic_counts(self):
        self._create_call("CALL_METRICS_NO_LEAD")
        self._create_call("CALL_METRICS_LEAD", with_lead=True)
        self._create_call("CALL_METRICS_FAILED", with_lead=True, sms_failed=True)

        response = self.client.get("/admin/metrics", auth=("admin", "secret"))
        metrics = repository.pilot_metrics()

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pilot Metrics", response.text)
        self.assertGreaterEqual(metrics["total_calls"], 3)
        self.assertGreaterEqual(metrics["calls_with_leads"], 2)
        self.assertGreaterEqual(metrics["sms_failed"], 1)


if __name__ == "__main__":
    unittest.main()
