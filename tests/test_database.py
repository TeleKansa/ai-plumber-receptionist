import unittest

from config.settings import DEFAULT_DATABASE_URL
from storage.database import normalize_database_url


class DatabaseSettingsTests(unittest.TestCase):
    def test_default_database_is_local_sqlite(self):
        self.assertEqual(DEFAULT_DATABASE_URL, "sqlite:///./local_dev.db")

    def test_postgres_url_uses_psycopg_driver(self):
        self.assertEqual(
            normalize_database_url("postgresql://user:pass@example.test/db"),
            "postgresql+psycopg://user:pass@example.test/db",
        )

    def test_railway_postgres_url_alias_uses_psycopg_driver(self):
        self.assertEqual(
            normalize_database_url("postgres://user:pass@example.test/db"),
            "postgresql+psycopg://user:pass@example.test/db",
        )


if __name__ == "__main__":
    unittest.main()
