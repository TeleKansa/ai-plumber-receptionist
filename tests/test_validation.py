import unittest

from workflow.validation import (
    looks_like_address,
    looks_like_name,
    looks_like_phone,
    name_supported_by_caller_text,
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
        for value in (
            "",
            "home",
            "my house",
            "at my house",
            "same place",
            "the same place",
            "unknown",
            "caller's house",
            "their house",
            "the address on file",
        ):
            self.assertFalse(looks_like_address(value))

    def test_looks_like_address_rejects_city_or_vague_location(self):
        for value in ("Overland Park", "near Walmart", "Kansas City", "123", "123 Overland Park"):
            self.assertFalse(looks_like_address(value))

    def test_looks_like_address_accepts_non_placeholder_text(self):
        self.assertTrue(looks_like_address("6100 West 120th Street"))
        self.assertTrue(looks_like_address("123 Main St"))
        self.assertTrue(looks_like_address("7420 W 135th St, Overland Park"))

    def test_looks_like_name_rejects_placeholders(self):
        for value in ("", "unknown", "caller", "customer"):
            self.assertFalse(looks_like_name(value))

    def test_name_support_requires_caller_text_when_available(self):
        self.assertTrue(name_supported_by_caller_text("Sam Rivera", "My name is Sam Rivera"))
        self.assertTrue(name_supported_by_caller_text("Sam", "This is Sam"))
        self.assertFalse(name_supported_by_caller_text("Thomas", "The sink is leaking at 123 Main St"))

    def test_single_name_without_caller_text_is_supported(self):
        self.assertTrue(name_supported_by_caller_text("Sam", ""))
        self.assertTrue(name_supported_by_caller_text("Sam Rivera", ""))

    def test_validate_service_request_args_accepts_valid_payload(self):
        self.assertEqual(validate_service_request_args(valid_args(), caller_text="My name is Sam Rivera"), {})

    def test_validate_service_request_args_accepts_first_name_only(self):
        args = valid_args()
        args["name"] = "Sam"

        self.assertEqual(validate_service_request_args(args, caller_text="My name is Sam"), {})

    def test_validate_service_request_args_accepts_first_name_without_transcript(self):
        args = valid_args()
        args["name"] = "Sam"

        self.assertEqual(validate_service_request_args(args, caller_text=""), {})

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

        errors = validate_service_request_args(args, caller_text="The sink is leaking")

        self.assertEqual(set(errors), {"issue", "urgency", "address", "callback", "name"})

    def test_validate_service_request_args_rejects_unsupported_name(self):
        args = valid_args()
        args["name"] = "Thomas"

        errors = validate_service_request_args(args, caller_text="The sink is leaking at 6100 West 120th Street")

        self.assertEqual(set(errors), {"name"})


if __name__ == "__main__":
    unittest.main()
