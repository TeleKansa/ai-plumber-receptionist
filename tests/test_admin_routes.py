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

    def test_call_detail_shows_validation_failure_payload(self):
        repository.create_or_update_call("CALL_ADMIN_FAIL", "+19135550123", "+19135550124")
        repository.record_call_event(
            "CALL_ADMIN_FAIL",
            "validation_failed",
            {
                "errors": {"address": "Address is required and cannot be a placeholder."},
                "missing_fields": ["address"],
                "args": {"address": "my house"},
            },
        )

        response = self.client.get("/admin/calls/CALL_ADMIN_FAIL", auth=("admin", "secret"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("validation_failed", response.text)
        self.assertIn("missing_fields", response.text)
        self.assertIn("my house", response.text)

    def test_tenant_pages_render(self):
        tenant = repository.get_default_tenant()

        tenants_response = self.client.get("/admin/tenants", auth=("admin", "secret"))
        detail_response = self.client.get(f"/admin/tenants/{tenant['id']}", auth=("admin", "secret"))

        self.assertEqual(tenants_response.status_code, 200)
        self.assertIn("Default Plumbing", tenants_response.text)
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn("Phone Numbers", detail_response.text)
        self.assertIn("Prompt/persona settings", detail_response.text)

    def test_prompt_page_renders_preview_and_can_activate_previous_version(self):
        tenant = repository.get_default_tenant()
        original = repository.get_active_prompt_profile(tenant["id"])

        prompt_response = self.client.get(f"/admin/tenants/{tenant['id']}/prompt", auth=("admin", "secret"))
        self.assertEqual(prompt_response.status_code, 200)
        self.assertIn("Core workflow is locked", prompt_response.text)
        self.assertIn("Generated Prompt Preview", prompt_response.text)
        self.assertIn("First name is enough", prompt_response.text)

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


if __name__ == "__main__":
    unittest.main()
