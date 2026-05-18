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
)


class SettingsTests(unittest.TestCase):
    def test_settings_keep_current_fallbacks(self):
        clean_env = {key: value for key, value in os.environ.items() if key not in ENV_NAMES}

        with patch.dict(os.environ, clean_env, clear=True):
            settings = get_settings()

        self.assertEqual(settings.host, DEFAULT_HOST)
        self.assertEqual(settings.oai_url, DEFAULT_OAI_URL)

    def test_settings_read_host_and_oai_url_from_env(self):
        with patch.dict(
            os.environ,
            {
                "HOST": "example.test",
                "OAI_URL": "wss://example.test/realtime",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.host, "example.test")
        self.assertEqual(settings.oai_url, "wss://example.test/realtime")

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


if __name__ == "__main__":
    unittest.main()
