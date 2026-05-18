import unittest

from workflow.validation import (
    looks_like_address,
    looks_like_name,
    looks_like_phone,
    validate_service_request_args,
)


def valid_args():
    return {
        "issue": "Kitchen sink is leaking",
        "urgency": "Actively leaking, caller can shut off water",
        "address": "6100 West 120th Street",
        "callback": "732-789-0675",
        "name": "Sam Rivera",
    }


class ValidationTests(unittest.TestCase):
    def test_looks_like_phone_accepts_common_us_formats(self):
        self.assertTrue(looks_like_phone("732-789-0675"))
        self.assertTrue(looks_like_phone("(732) 789-0675"))
        self.assertTrue(looks_like_phone("+1 732 789 0675"))

    def test_looks_like_phone_rejects_short_values(self):
        self.assertFalse(looks_like_phone("12345"))
        self.assertFalse(looks_like_phone(""))

    def test_looks_like_address_rejects_placeholders(self):
        for value in ("", "home", "my house", "same place", "unknown"):
            self.assertFalse(looks_like_address(value))

    def test_looks_like_address_accepts_non_placeholder_text(self):
        self.assertTrue(looks_like_address("6100 West 120th Street"))

    def test_looks_like_name_rejects_placeholders(self):
        for value in ("", "unknown", "caller", "customer"):
            self.assertFalse(looks_like_name(value))

    def test_validate_service_request_args_accepts_valid_payload(self):
        self.assertEqual(validate_service_request_args(valid_args()), {})

    def test_validate_service_request_args_reports_missing_and_placeholder_fields(self):
        args = valid_args()
        args.update(
            {
                "issue": "",
                "urgency": "",
                "address": "my house",
                "callback": "123",
                "name": "unknown",
            }
        )

        errors = validate_service_request_args(args)

        self.assertEqual(set(errors), {"issue", "urgency", "address", "callback", "name"})


if __name__ == "__main__":
    unittest.main()
