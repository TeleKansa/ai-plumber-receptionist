import unittest

from admin.auth import admin_enabled, check_admin_credentials


class AdminAuthTests(unittest.TestCase):
    def test_admin_disabled_without_password(self):
        self.assertFalse(admin_enabled(""))
        self.assertFalse(check_admin_credentials("", "admin", "anything"))

    def test_admin_accepts_configured_password(self):
        self.assertTrue(check_admin_credentials("secret", "admin", "secret"))

    def test_admin_rejects_wrong_credentials(self):
        self.assertFalse(check_admin_credentials("secret", "admin", "wrong"))
        self.assertFalse(check_admin_credentials("secret", "not-admin", "secret"))


if __name__ == "__main__":
    unittest.main()
