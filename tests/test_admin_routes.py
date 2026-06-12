import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from admin.routes import create_admin_router
from config.settings import Settings
from storage import repository
from storage.database import configure_database, init_db


class AdminRoutesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db()
        app = FastAPI()
        app.include_router(
            create_admin_router(
                Settings(
                    openai_api_key="",
                    twilio_account_sid="",
                    twilio_auth_token="",
                    twilio_phone_number="",
                    plumber_phone_number="",
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
        self.client = TestClient(app)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_call_detail_shows_validation_failure_payload_and_realtime_model(self):
        repository.create_or_update_call(
            "CALL_ADMIN_FAIL",
            "+19135550123",
            "+19135550124",
            prompt_version_id=42,
            realtime_model="gpt-realtime-2",
            realtime_reasoning_effort="low",
        )
        repository.record_call_event(
            "CALL_ADMIN_FAIL",
            "validation_failed",
            {
                "errors": {"address": "Address is required and cannot be a placeholder."},
                "missing_fields": ["address"],
                "args": {"address": "my house"},
            },
        )
        repository.record_call_event(
            "CALL_ADMIN_FAIL",
            "twilio_websocket_disconnected",
            {
                "disconnect_code": 1006,
                "snapshot": {"media_stream_exit_reason": "websocket_disconnect"},
            },
        )
        repository.record_call_event(
            "CALL_ADMIN_FAIL",
            "media_stream_done",
            {
                "media_stream_exit_reason": "websocket_disconnect",
                "last_ai_transcript": "Okay, what's the service address?",
            },
        )

        response = self.client.get("/admin/calls/CALL_ADMIN_FAIL", auth=("admin", "secret"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("Call Summary", response.text)
        self.assertIn("prompt_version_id", response.text)
        self.assertIn("realtime_model", response.text)
        self.assertIn("realtime_reasoning_effort", response.text)
        self.assertIn("gpt-realtime-2", response.text)
        self.assertIn("low", response.text)
        self.assertIn("Lifecycle Debug", response.text)
        self.assertIn("twilio_websocket_disconnected", response.text)
        self.assertIn("media_stream_exit_reason", response.text)
        self.assertIn("validation_failed", response.text)
        self.assertIn("missing_fields", response.text)
        self.assertIn("my house", response.text)

    def test_tenant_pages_render(self):
        tenant = repository.get_default_tenant()

        tenants_response = self.client.get("/admin/tenants", auth=("admin", "secret"))
        detail_response = self.client.get(f"/admin/tenants/{tenant['id']}", auth=("admin", "secret"))

        self.assertEqual(tenants_response.status_code, 200)
        self.assertIn("Default Plumbing", tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant["id"]}"', tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant["id"]}/prompt"', tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant["id"]}/intake-policy"', tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant["id"]}/notification-policy"', tenants_response.text)
        self.assertIn("Details", tenants_response.text)
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn("Back to tenants", detail_response.text)
        self.assertIn("Tenant Summary", detail_response.text)
        self.assertIn("Onboarding / Telephony", detail_response.text)
        self.assertIn("Customer keeps their Google Maps number.", detail_response.text)
        self.assertIn("Live calls are blocked until Go Live is enabled", detail_response.text)
        self.assertIn("Phone Numbers", detail_response.text)
        self.assertIn("Prompt/persona settings", detail_response.text)
        self.assertIn("Intake policy", detail_response.text)
        self.assertIn("Notification policy", detail_response.text)

    def test_newly_created_tenant_is_clickable_and_has_prompt_profile(self):
        create_response = self.client.post(
            "/admin/tenants",
            data={
                "name": "New Tenant Plumbing",
                "slug": "new-tenant-plumbing",
                "business_name": "New Tenant Plumbing",
                "greeting": "New Tenant Plumbing, what's going on?",
                "notification_sms_number": "+15550001111",
                "backup_notification_sms_number": "",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )

        self.assertEqual(create_response.status_code, 303)
        detail_path = create_response.headers["location"]
        self.assertRegex(detail_path, r"^/admin/tenants/\d+$")

        tenant_id = int(detail_path.rsplit("/", 1)[-1])
        active_prompt = repository.get_active_prompt_profile(tenant_id)
        self.assertIsNotNone(active_prompt)
        self.assertTrue(active_prompt["is_active"])

        tenants_response = self.client.get("/admin/tenants", auth=("admin", "secret"))
        self.assertEqual(tenants_response.status_code, 200)
        self.assertIn("New Tenant Plumbing", tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant_id}"', tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant_id}/prompt"', tenants_response.text)
        self.assertIn(f'href="/admin/tenants/{tenant_id}/intake-policy"', tenants_response.text)

        detail_response = self.client.get(detail_path, auth=("admin", "secret"))
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn("Back to tenants", detail_response.text)
        self.assertIn("Prompt/persona settings", detail_response.text)
        self.assertIn("Intake policy", detail_response.text)
        self.assertIn("Onboarding / Telephony", detail_response.text)

        prompt_response = self.client.get(f"/admin/tenants/{tenant_id}/prompt", auth=("admin", "secret"))
        self.assertEqual(prompt_response.status_code, 200)
        self.assertIn("Generated Prompt Preview", prompt_response.text)
        self.assertIn("Back to tenant detail", prompt_response.text)
        self.assertIn("Back to tenants", prompt_response.text)
        self.assertIn("Realtime Model", prompt_response.text)
        self.assertIn("Realtime Model:", prompt_response.text)
        self.assertIn('select name="realtime_model"', prompt_response.text)
        self.assertIn('option value="" selected>env/default</option>', prompt_response.text)
        self.assertIn('option value="gpt-realtime-1.5">gpt-realtime-1.5</option>', prompt_response.text)
        self.assertIn('option value="gpt-realtime-2">gpt-realtime-2</option>', prompt_response.text)
        self.assertIn("realtime_model", prompt_response.text)
        self.assertIn("realtime_reasoning_effort", prompt_response.text)
        self.assertIn("env/default", prompt_response.text)
        self.assertIn("OPENAI_REALTIME_MODEL", prompt_response.text)
        self.assertIn("reasoning.effort=low", prompt_response.text)
        self.assertIn("gpt-realtime-2", prompt_response.text)
        self.assertIn("New Tenant Plumbing, what&#x27;s going on?", prompt_response.text)

        create_prompt_response = self.client.post(
            f"/admin/tenants/{tenant_id}/prompt",
            data={
                "label": "Realtime 2 test",
                "business_name": "New Tenant Plumbing",
                "greeting": "New Tenant Plumbing, what's going on?",
                "tone": "calm",
                "verbosity": "brief",
                "closing_line": "You're all set. We'll call back soon.",
                "avoid_phrases": "",
                "preferred_terms": "",
                "extra_instructions_text": "",
                "realtime_model": "gpt-realtime-2",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(create_prompt_response.status_code, 303)
        self.assertEqual(repository.get_active_prompt_profile(tenant_id)["realtime_model"], "gpt-realtime-2")

        intake_policy = repository.get_intake_policy(tenant_id)
        self.assertIsNotNone(intake_policy)
        self.assertTrue(intake_policy["enabled"])

        intake_response = self.client.get(f"/admin/tenants/{tenant_id}/intake-policy", auth=("admin", "secret"))
        self.assertEqual(intake_response.status_code, 200)
        self.assertIn("Generated Prompt Preview", intake_response.text)
        self.assertIn("Core workflow is locked", intake_response.text)
        self.assertIn("Add Extra Question", intake_response.text)

    def test_admin_can_update_telephony_status_test_callers_and_phone_live_switch(self):
        tenant = repository.create_tenant(
            "Admin Onboarding Plumbing",
            "admin-onboarding-plumbing",
            "Admin Onboarding Plumbing",
            "Admin Onboarding Plumbing, what's going on?",
            "+15550002222",
        )
        phone = repository.add_tenant_phone_number(tenant["id"], "+15551112222", "AI forwarding")

        telephony_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/telephony",
            data={
                "status": "testing",
                "public_business_number": "+19135550000",
                "ai_ingress_twilio_number": "+15551112222",
                "forwarding_setup_status": "customer_configured",
                "test_mode_enabled": "1",
                "allowed_test_callers": "+19135550123\n913-555-9999",
                "notes": "Customer configured after-hours forwarding.",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(telephony_response.status_code, 303)

        updated_tenant = repository.get_tenant(tenant["id"])
        telephony_profile = repository.get_telephony_profile(tenant["id"])
        self.assertEqual(updated_tenant["status"], "testing")
        self.assertEqual(telephony_profile["public_business_number"], "+19135550000")
        self.assertEqual(telephony_profile["ai_ingress_twilio_number"], "+15551112222")
        self.assertEqual(telephony_profile["forwarding_setup_status"], "customer_configured")
        self.assertTrue(telephony_profile["test_mode_enabled"])
        self.assertIn("+19135550123", telephony_profile["allowed_test_callers_json"])

        phone_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/phones/{phone['id']}/live",
            data={"accepts_live_calls": "1"},
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(phone_response.status_code, 303)
        phone_after = repository.list_tenant_phone_numbers(tenant["id"])[0]
        self.assertTrue(phone_after["accepts_live_calls"])

        go_live_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/go-live",
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(go_live_response.status_code, 303)
        self.assertEqual(repository.get_tenant(tenant["id"])["status"], "live")
        self.assertIsNotNone(repository.get_telephony_profile(tenant["id"])["live_enabled_at"])

        pause_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/pause",
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(pause_response.status_code, 303)
        self.assertEqual(repository.get_tenant(tenant["id"])["status"], "paused")

    def test_prompt_page_renders_preview_and_can_activate_previous_version(self):
        tenant = repository.get_default_tenant()
        original = repository.get_active_prompt_profile(tenant["id"])

        prompt_response = self.client.get(f"/admin/tenants/{tenant['id']}/prompt", auth=("admin", "secret"))
        self.assertEqual(prompt_response.status_code, 200)
        self.assertIn("Back to tenant detail", prompt_response.text)
        self.assertIn("Back to tenants", prompt_response.text)
        self.assertIn("Core workflow is locked", prompt_response.text)
        self.assertIn("Generated Prompt Preview", prompt_response.text)
        self.assertIn("First name is enough", prompt_response.text)
        self.assertIn("Intake policy", prompt_response.text)

        create_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/prompt",
            data={
                "label": "Custom greeting",
                "business_name": "Admin Test Plumbing",
                "greeting": "Admin Test Plumbing, what's going on?",
                "tone": "plainspoken",
                "verbosity": "brief",
                "closing_line": "You're all set. We'll call back soon.",
                "avoid_phrases": "certainly\nI apologize",
                "preferred_terms": "service address\ncallback number",
                "extra_instructions_text": "Mention weekend availability if the caller asks.",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(create_response.status_code, 303)
        active = repository.get_active_prompt_profile(tenant["id"])
        self.assertNotEqual(active["id"], original["id"])
        self.assertEqual(active["business_name"], "Admin Test Plumbing")

        activate_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/prompt/{original['id']}/activate",
            auth=("admin", "secret"),
            follow_redirects=False,
        )
        self.assertEqual(activate_response.status_code, 303)
        self.assertEqual(repository.get_active_prompt_profile(tenant["id"])["id"], original["id"])

    def test_intake_policy_page_renders_and_can_add_question(self):
        tenant = repository.get_default_tenant()

        response = self.client.get(f"/admin/tenants/{tenant['id']}/intake-policy", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("TENANT INTAKE POLICY", response.text)
        self.assertIn("Add Extra Question", response.text)
        self.assertIn("Ask once means the AI will ask this question before submitting", response.text)
        self.assertIn("collection_mode", response.text)
        self.assertIn("Prompt/persona settings", response.text)

        add_response = self.client.post(
            f"/admin/tenants/{tenant['id']}/intake-policy/extra",
            data={
                "key": "property_role",
                "label": "Homeowner or renter",
                "question_text": "Are you the homeowner or are you renting?",
                "collection_mode": "ask_once",
                "include_in_sms": "1",
                "include_in_admin": "1",
                "active": "1",
            },
            auth=("admin", "secret"),
            follow_redirects=False,
        )

        self.assertEqual(add_response.status_code, 303)
        policy = repository.get_intake_policy(tenant["id"])
        self.assertIn("property_role", policy["extra_questions_json"])
        self.assertIn("ask_once", policy["extra_questions_json"])
        updated_response = self.client.get(f"/admin/tenants/{tenant['id']}/intake-policy", auth=("admin", "secret"))
        self.assertIn("Homeowner or renter", updated_response.text)

    def test_intake_policy_page_rejects_invalid_json_cleanly(self):
        tenant = repository.get_default_tenant()

        response = self.client.post(
            f"/admin/tenants/{tenant['id']}/intake-policy",
            data={
                "enabled": "1",
                "extra_questions_json": "not-json",
                "conditional_questions_json": "[]",
                "sms_include_extra_fields": "",
                "admin_display_fields": "",
                "notes": "",
            },
            auth=("admin", "secret"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid Intake Policy JSON", response.text)
        self.assertNotIn("Traceback", response.text)

    def test_invalid_tenant_pages_show_clean_admin_error(self):
        detail_response = self.client.get("/admin/tenants/999999", auth=("admin", "secret"))
        prompt_response = self.client.get("/admin/tenants/999999/prompt", auth=("admin", "secret"))

        self.assertEqual(detail_response.status_code, 404)
        self.assertIn("Tenant 999999 was not found.", detail_response.text)
        self.assertIn("Back to tenants", detail_response.text)
        self.assertNotIn("Traceback", detail_response.text)
        self.assertEqual(prompt_response.status_code, 404)
        self.assertIn("Tenant 999999 was not found.", prompt_response.text)
        self.assertNotIn("Traceback", prompt_response.text)

        invalid_response = self.client.get("/admin/tenants/not-a-number", auth=("admin", "secret"))
        self.assertEqual(invalid_response.status_code, 404)
        self.assertIn("Tenant not-a-number was not found.", invalid_response.text)
        self.assertNotIn("Traceback", invalid_response.text)


if __name__ == "__main__":
    unittest.main()
