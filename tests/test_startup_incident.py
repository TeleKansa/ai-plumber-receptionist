import tempfile
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import text

import main
from config.settings import Settings
from storage import database, repository
from storage.database import configure_database, init_db


def startup_settings(tmpdir):
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


class StartupIncidentTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_is_demo_migration_handles_existing_tenants_table_and_is_idempotent(self):
        with database.engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE tenants ("
                    "id INTEGER PRIMARY KEY, "
                    "name VARCHAR(255) NOT NULL, "
                    "slug VARCHAR(128) NOT NULL UNIQUE, "
                    "status VARCHAR(64) NOT NULL, "
                    "created_at DATETIME NOT NULL, "
                    "updated_at DATETIME NOT NULL)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO tenants (name, slug, status, created_at, updated_at) "
                    "VALUES ('Existing Plumbing', 'default', 'live', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
                )
            )

        init_db(startup_settings(self.tmpdir.name))
        init_db(startup_settings(self.tmpdir.name))

        tenant = repository.get_default_tenant()
        self.assertIn("is_demo", tenant)
        self.assertFalse(tenant["is_demo"])

    def test_startup_does_not_auto_create_demo_tenant(self):
        init_db(startup_settings(self.tmpdir.name))

        self.assertIsNone(repository.get_demo_tenant())

    def test_health_serves_after_startup(self):
        main.sessions.clear()
        with TestClient(main.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
