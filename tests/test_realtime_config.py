import tempfile
import unittest

import main
from config.settings import Settings
from storage import repository
from storage.database import configure_database, init_db
from workflow.realtime_config import (
    build_realtime_url,
    effective_realtime_model,
    realtime_reasoning_effort,
    realtime_session_overrides,
)


class RealtimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_model_remains_realtime_15(self):
        settings = Settings(
            openai_api_key="",
            twilio_account_sid="",
            twilio_auth_token="",
            twilio_phone_number="",
            plumber_phone_number="",
            host="example.test",
            oai_url="wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5",
            database_url=f"sqlite:///{self.tmpdir.name}/test.db",
            admin_password="secret",
            default_tenant_name="Default Plumbing",
            default_tenant_slug="default",
            default_tenant_greeting="Plumbing office, what's going on?",
        )

        self.assertEqual(effective_realtime_model(settings=settings), "gpt-realtime-1.5")
        self.assertEqual(
            build_realtime_url("gpt-realtime-1.5", settings.oai_url),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5",
        )

    def test_realtime_2_builds_url_and_low_reasoning_config(self):
        url = build_realtime_url(
            "gpt-realtime-2",
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5",
        )
        session_update = main.build_session_update(
            "913-555-0123",
            realtime_model="gpt-realtime-2",
        )

        self.assertEqual(url, "wss://api.openai.com/v1/realtime?model=gpt-realtime-2")
        self.assertEqual(realtime_reasoning_effort("gpt-realtime-2"), "low")
        self.assertEqual(realtime_session_overrides("gpt-realtime-2"), {"reasoning": {"effort": "low"}})
        self.assertEqual(session_update["session"]["reasoning"]["effort"], "low")

    def test_realtime_15_session_config_does_not_include_reasoning(self):
        session_update = main.build_session_update(
            "913-555-0123",
            realtime_model="gpt-realtime-1.5",
        )

        self.assertNotIn("reasoning", session_update["session"])
        self.assertIsNone(realtime_reasoning_effort("gpt-realtime-1.5"))

    def test_prompt_profile_model_overrides_env_default(self):
        tenant = repository.get_default_tenant()
        profile = repository.create_prompt_profile(
            tenant["id"],
            label="Realtime 2 prompt",
            business_name="Default Plumbing",
            greeting="Plumbing office, what's going on?",
            tone="casual",
            verbosity="short",
            closing_line="You're all set. We'll call back soon.",
            avoid_phrases=[],
            preferred_terms=[],
            realtime_model="gpt-realtime-2",
        )
        settings = Settings(
            openai_api_key="",
            twilio_account_sid="",
            twilio_auth_token="",
            twilio_phone_number="",
            plumber_phone_number="",
            host="example.test",
            oai_url="wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5",
            database_url=f"sqlite:///{self.tmpdir.name}/test.db",
            admin_password="secret",
            default_tenant_name="Default Plumbing",
            default_tenant_slug="default",
            default_tenant_greeting="Plumbing office, what's going on?",
        )

        self.assertEqual(profile["realtime_model"], "gpt-realtime-2")
        self.assertEqual(effective_realtime_model(profile, settings), "gpt-realtime-2")

    def test_call_record_stores_realtime_model(self):
        repository.create_or_update_call(
            "CALL_MODEL",
            "+19135550123",
            "+15551234567",
            realtime_model="gpt-realtime-2",
            realtime_reasoning_effort="low",
        )
        repository.update_call_stream_started(
            "CALL_MODEL",
            "MZ123",
            realtime_model="gpt-realtime-2",
            realtime_reasoning_effort="low",
        )

        detail = repository.get_call_detail("CALL_MODEL")

        self.assertEqual(detail["call"]["realtime_model"], "gpt-realtime-2")
        self.assertEqual(detail["call"]["realtime_reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
