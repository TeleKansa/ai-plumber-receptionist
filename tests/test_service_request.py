import tempfile
import unittest

from storage import repository
from storage.database import configure_database, init_db
from workflow.notifications import SmsSendResult
from workflow.service_request import process_service_request


VALID_ARGS = {
    "issue": "Kitchen sink is leaking",
    "urgency": "Actively leaking, caller can shut off water",
    "address": "6100 West 120th Street",
    "callback": "732-789-0675",
    "name": "Sam Rivera",
    "extra_fields": {"additional_notes": "gate code 1234"},
}

FIRST_NAME_ARGS = {
    "issue": "Kitchen sink is leaking",
    "urgency": "Actively leaking, caller can shut off water",
    "address": "6100 West 120th Street",
    "callback": "732-789-0675",
    "name": "Sam",
    "extra_fields": {"additional_notes": "gate code 1234"},
}

VALID_CALLER_TEXT = (
    "Kitchen sink is leaking. Water is still coming out. "
    "The address is 6100 West 120th Street. My name is Sam Rivera."
)


class FakeSmsSender:
    def __init__(self, success=True):
        self.success = success
        self.calls = []
        self.lead_exists_during_send = False

    async def __call__(self, call_sid, args, from_number, to_number=None):
        self.calls.append((call_sid, args, from_number, to_number))
        self.lead_exists_during_send = repository.get_lead_by_call_sid(call_sid) is not None
        if self.success:
            return SmsSendResult(success=True, provider_message_sid="SM_TEST")
        return SmsSendResult(success=False, error="forced failure")


class ServiceRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db()

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_invalid_submit_does_not_create_lead_or_send_sms(self):
        repository.create_or_update_call("CALL_INVALID", "+19135550123", "+19135550124")
        sender = FakeSmsSender()
        args = dict(VALID_ARGS)
        args["address"] = "my house"

        result = await process_service_request(
            "CALL_INVALID",
            args,
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "validation_failed")
        self.assertIn("address", result.output["missing_fields"])
        self.assertFalse(result.should_hangup)
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_INVALID"))
        events = [event for event in repository.list_recent_call_events() if event["call_sid"] == "CALL_INVALID"]
        self.assertIn("validation_failed", {event["event_type"] for event in events})

    async def test_valid_submit_creates_lead_before_sms_and_records_notification(self):
        repository.create_or_update_call("CALL_VALID", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)

        result = await process_service_request(
            "CALL_VALID",
            dict(VALID_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertTrue(result.output["success"])
        self.assertTrue(result.output["lead_saved"])
        self.assertTrue(result.should_hangup)
        self.assertTrue(sender.lead_exists_during_send)
        lead = repository.get_lead_by_call_sid("CALL_VALID")
        self.assertIsNotNone(lead)
        self.assertIsNotNone(lead["tenant_id"])
        notification = repository.get_notification_for_lead(lead["id"])
        self.assertEqual(notification["tenant_id"], lead["tenant_id"])
        self.assertEqual(notification["status"], "sent")
        self.assertEqual(notification["provider_message_sid"], "SM_TEST")
        events = [event for event in repository.list_recent_call_events() if event["call_sid"] == "CALL_VALID"]
        self.assertTrue(events)
        self.assertTrue(all(event["tenant_id"] == lead["tenant_id"] for event in events))

    async def test_first_name_only_valid_submit_creates_lead_and_sends_sms(self):
        repository.create_or_update_call("CALL_FIRST_NAME", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)

        result = await process_service_request(
            "CALL_FIRST_NAME",
            dict(FIRST_NAME_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text="My name is Sam. The address is 6100 West 120th Street.",
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)
        lead = repository.get_lead_by_call_sid("CALL_FIRST_NAME")
        self.assertEqual(lead["name"], "Sam")
        events = [event for event in repository.list_recent_call_events() if event["call_sid"] == "CALL_FIRST_NAME"]
        self.assertNotIn("validation_failed", {event["event_type"] for event in events})

    async def test_first_name_only_without_transcript_is_accepted_and_marked_unverified(self):
        repository.create_or_update_call("CALL_FIRST_NAME_NO_TRANSCRIPT", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)

        result = await process_service_request(
            "CALL_FIRST_NAME_NO_TRANSCRIPT",
            dict(FIRST_NAME_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text="",
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)
        lead = repository.get_lead_by_call_sid("CALL_FIRST_NAME_NO_TRANSCRIPT")
        self.assertEqual(lead["name"], "Sam")
        events = [
            event
            for event in repository.list_recent_call_events()
            if event["call_sid"] == "CALL_FIRST_NAME_NO_TRANSCRIPT"
        ]
        self.assertIn("name_provenance_unverified", {event["event_type"] for event in events})

    async def test_sms_failure_still_saves_lead_and_failed_notification(self):
        repository.create_or_update_call("CALL_SMS_FAIL", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=False)

        result = await process_service_request(
            "CALL_SMS_FAIL",
            dict(VALID_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "notification_failed")
        self.assertTrue(result.output["lead_saved"])
        self.assertTrue(result.should_hangup)
        lead = repository.get_lead_by_call_sid("CALL_SMS_FAIL")
        self.assertIsNotNone(lead)
        notification = repository.get_notification_for_lead(lead["id"])
        self.assertEqual(notification["status"], "failed")
        self.assertEqual(notification["error"], "forced failure")

    async def test_duplicate_submit_does_not_send_second_sms(self):
        repository.create_or_update_call("CALL_DUP", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)

        first = await process_service_request(
            "CALL_DUP",
            dict(VALID_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )
        second = await process_service_request(
            "CALL_DUP",
            dict(VALID_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertTrue(first.output["success"])
        self.assertEqual(second.output["reason"], "already_submitted")
        self.assertEqual(len(sender.calls), 1)
        leads = [lead for lead in repository.list_recent_leads() if lead["call_sid"] == "CALL_DUP"]
        self.assertEqual(len(leads), 1)

    async def test_callback_alias_is_normalized_to_caller_number(self):
        repository.create_or_update_call("CALL_CALLBACK_ALIAS", "+17327890675", "+19135550124")
        sender = FakeSmsSender(success=True)
        args = dict(VALID_ARGS)
        args["callback"] = "this number is good"

        result = await process_service_request(
            "CALL_CALLBACK_ALIAS",
            args,
            "+17327890675",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)
        lead = repository.get_lead_by_call_sid("CALL_CALLBACK_ALIAS")
        self.assertEqual(lead["callback"], "+17327890675")
        self.assertEqual(sender.calls[0][1]["callback"], "+17327890675")
        events = [event for event in repository.list_recent_call_events() if event["call_sid"] == "CALL_CALLBACK_ALIAS"]
        self.assertIn("callback_alias_normalized", {event["event_type"] for event in events})

    async def test_callback_alias_without_caller_number_fails_validation(self):
        repository.create_or_update_call("CALL_CALLBACK_ALIAS_FAIL", "unknown", "+19135550124")
        sender = FakeSmsSender(success=True)
        args = dict(VALID_ARGS)
        args["callback"] = "this number is good"

        result = await process_service_request(
            "CALL_CALLBACK_ALIAS_FAIL",
            args,
            "unknown",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "validation_failed")
        self.assertEqual(result.output["next_question_key"], "callback")
        self.assertIn("callback", result.output["missing_fields"])
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_CALLBACK_ALIAS_FAIL"))

    async def test_validation_guidance_returns_one_next_question(self):
        repository.create_or_update_call("CALL_GUIDANCE_ONE", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)
        args = {
            "issue": "",
            "urgency": "",
            "address": "",
            "callback": "",
            "name": "",
        }

        result = await process_service_request(
            "CALL_GUIDANCE_ONE",
            args,
            "+19135550123",
            "+19135559999",
            sender,
            caller_text="",
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["missing_fields"], ["issue"])
        self.assertEqual(result.output["next_question_key"], "issue")
        self.assertIn("What's going on with the plumbing?", result.output["guidance"])
        self.assertNotIn("service address", result.output["guidance"].lower())
        self.assertEqual(sender.calls, [])

    async def test_invalid_submit_followed_by_valid_submit_sends_one_sms_total(self):
        repository.create_or_update_call("CALL_RETRY", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)
        invalid_args = dict(VALID_ARGS)
        invalid_args["address"] = "same place"

        invalid = await process_service_request(
            "CALL_RETRY",
            invalid_args,
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )
        valid = await process_service_request(
            "CALL_RETRY",
            dict(VALID_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text=VALID_CALLER_TEXT,
        )

        self.assertFalse(invalid.output["success"])
        self.assertEqual(invalid.output["reason"], "validation_failed")
        self.assertFalse(invalid.should_hangup)
        self.assertTrue(valid.output["success"])
        self.assertEqual(len(sender.calls), 1)

    async def test_unsupported_name_does_not_create_lead_or_send_sms(self):
        repository.create_or_update_call("CALL_BAD_NAME", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)
        args = dict(VALID_ARGS)
        args["name"] = "Thomas"

        result = await process_service_request(
            "CALL_BAD_NAME",
            args,
            "+19135550123",
            "+19135559999",
            sender,
            caller_text="The sink is leaking at 6100 West 120th Street.",
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "validation_failed")
        self.assertIn("name", result.output["missing_fields"])
        self.assertIn("A first name is enough", result.output["guidance"])
        self.assertIn("Do not ask for last name", result.output["guidance"])
        self.assertFalse(result.should_hangup)
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_BAD_NAME"))

    async def test_unsupported_name_followed_by_supported_first_name_sends_one_sms_total(self):
        repository.create_or_update_call("CALL_NAME_RETRY", "+19135550123", "+19135550124")
        sender = FakeSmsSender(success=True)
        invalid_args = dict(FIRST_NAME_ARGS)
        invalid_args["name"] = "Thomas"

        invalid = await process_service_request(
            "CALL_NAME_RETRY",
            invalid_args,
            "+19135550123",
            "+19135559999",
            sender,
            caller_text="The sink is leaking at 6100 West 120th Street.",
        )
        valid = await process_service_request(
            "CALL_NAME_RETRY",
            dict(FIRST_NAME_ARGS),
            "+19135550123",
            "+19135559999",
            sender,
            caller_text="The sink is leaking at 6100 West 120th Street. My name is Sam.",
        )

        self.assertFalse(invalid.output["success"])
        self.assertEqual(invalid.output["reason"], "validation_failed")
        self.assertIn("name", invalid.output["missing_fields"])
        self.assertTrue(valid.output["success"])
        self.assertEqual(len(sender.calls), 1)
        leads = [lead for lead in repository.list_recent_leads() if lead["call_sid"] == "CALL_NAME_RETRY"]
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["name"], "Sam")

    async def test_tenant_specific_notification_number_is_used(self):
        tenant_a = repository.create_tenant(
            "Tenant A",
            "tenant-a",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        tenant_b = repository.create_tenant(
            "Tenant B",
            "tenant-b",
            "Tenant B Plumbing",
            "Tenant B plumbing, what's going on?",
            "+15550000002",
        )
        repository.create_or_update_call("CALL_TENANT_A", "+19135550123", "+15551110001", tenant_id=tenant_a["id"])
        sender = FakeSmsSender(success=True)

        result = await process_service_request(
            "CALL_TENANT_A",
            dict(FIRST_NAME_ARGS),
            "+19135550123",
            tenant_a["notification_sms_number"],
            sender,
            caller_text="My name is Sam. The address is 6100 West 120th Street.",
            tenant_id=tenant_a["id"],
        )

        self.assertTrue(result.output["success"])
        lead = repository.get_lead_by_call_sid("CALL_TENANT_A")
        notification = repository.get_notification_for_lead(lead["id"])
        self.assertEqual(lead["tenant_id"], tenant_a["id"])
        self.assertEqual(notification["tenant_id"], tenant_a["id"])
        self.assertEqual(notification["to_number"], "+15550000001")
        self.assertNotEqual(notification["to_number"], tenant_b["notification_sms_number"])


if __name__ == "__main__":
    unittest.main()
