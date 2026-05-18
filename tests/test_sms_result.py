import json
import unittest

from workflow.sms_result import build_service_request_output


class SmsResultTests(unittest.TestCase):
    def test_service_request_output_reports_sms_success(self):
        output = json.loads(build_service_request_output(True))

        self.assertEqual(output, {"success": True})

    def test_service_request_output_reports_sms_failure(self):
        output = json.loads(build_service_request_output(False))

        self.assertEqual(output, {"success": False})


if __name__ == "__main__":
    unittest.main()
