import os
import unittest
from unittest.mock import patch

from config.settings import DEFAULT_HOST, DEFAULT_OAI_URL, get_settings


ENV_NAMES = (
    "OPENAI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "PLUMBER_PHONE_NUMBER",
    "HOST",
    "PUBLIC_HOST",
    "OAI_URL",
    "OPENAI_REALTIME_URL",
    "OPENAI_REALTIME_MODEL",
    "DATABASE_URL",
    "ADMIN_PASSWORD",
    "DEFAULT_TENANT_NAME",
    "DEFAULT_TENANT_SLUG",
    "DEFAULT_TENANT_GREETING",
)


class SettingsTests(unittest.TestCase):
    def test_settings_keep_current_fallbacks(self):
        clean_env = {key: value for key, value in os.environ.items() if key not in ENV_NAMES}

        with patch.dict(os.environ, clean_env, clear=True):
            settings = get_settings()

        self.assertEqual(settings.host, DEFAULT_HOST)
        self.assertEqual(settings.oai_url, DEFAULT_OAI_URL)
        self.assertEqual(settings.openai_realtime_model, "gpt-realtime-1.5")
        self.assertEqual(settings.database_url, "sqlite:///./local_dev.db")
        self.assertEqual(settings.default_tenant_name, "Default Plumbing")
        self.assertEqual(settings.default_tenant_slug, "default")

    def test_settings_read_host_and_oai_url_from_env(self):
        with patch.dict(
            os.environ,
            {
                "HOST": "example.test",
                "OAI_URL": "wss://example.test/realtime",
                "OPENAI_REALTIME_MODEL": "gpt-realtime-2",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.host, "example.test")
        self.assertEqual(settings.oai_url, "wss://example.test/realtime")
        self.assertEqual(settings.openai_realtime_model, "gpt-realtime-2")

    def test_settings_support_documented_aliases(self):
        with patch.dict(
            os.environ,
            {
                "PUBLIC_HOST": "public.example.test",
                "OPENAI_REALTIME_URL": "wss://public.example.test/realtime",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.host, "public.example.test")
        self.assertEqual(settings.oai_url, "wss://public.example.test/realtime")

    def test_settings_read_database_url_and_admin_password(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://user:pass@example.test:5432/db",
                "ADMIN_PASSWORD": "secret",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.database_url, "postgresql://user:pass@example.test:5432/db")
        self.assertEqual(settings.admin_password, "secret")

    def test_settings_read_default_tenant_values(self):
        with patch.dict(
            os.environ,
            {
                "DEFAULT_TENANT_NAME": "Acme Plumbing",
                "DEFAULT_TENANT_SLUG": "acme",
                "DEFAULT_TENANT_GREETING": "Acme plumbing, what's going on?",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.default_tenant_name, "Acme Plumbing")
        self.assertEqual(settings.default_tenant_slug, "acme")
        self.assertEqual(settings.default_tenant_greeting, "Acme plumbing, what's going on?")


if __name__ == "__main__":
    unittest.main()
