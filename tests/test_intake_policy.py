import json
import tempfile
import unittest

from main import build_sms_body
from storage import repository
from storage.database import configure_database, init_db
from workflow.intake_policy import extra_questions, sms_extra_field_rows
from workflow.notifications import SmsSendResult
from workflow.prompt_builder import PromptBuilder
from workflow.service_request import process_service_request


BASE_ARGS = {
    "issue": "Kitchen sink is leaking",
    "urgency": "Active leak, caller can shut off water",
    "address": "6100 West 120th Street",
    "callback": "732-789-0675",
    "name": "Sam",
}

CALLER_TEXT = "My name is Sam. The kitchen sink is leaking at 6100 West 120th Street."


def property_role_question(collection_mode="ask_once", include_in_sms=True, active=True):
    return {
        "key": "property_role",
        "label": "Homeowner or renter",
        "question_text": "Are you the homeowner or are you renting?",
        "collection_mode": collection_mode,
        "include_in_sms": include_in_sms,
        "include_in_admin": True,
        "active": active,
    }


class FakeSmsSender:
    def __init__(self):
        self.calls = []

    async def __call__(self, call_sid, args, from_number):
        self.calls.append((call_sid, args, from_number))
        return SmsSendResult(success=True, provider_message_sid="SM_INTAKE")


class IntakePolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_tenant_has_default_intake_policy(self):
        tenant = repository.get_default_tenant()

        policy = repository.get_intake_policy(tenant["id"])

        self.assertIsNotNone(policy)
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["extra_questions_json"], "[]")
        self.assertEqual(policy["conditional_questions_json"], "[]")

    def test_extra_and_conditional_questions_appear_in_prompt(self):
        tenant = repository.get_default_tenant()
        policy = repository.update_intake_policy(
            tenant["id"],
            extra_questions=[
                property_role_question("ask_once")
            ],
            conditional_questions=[
                {
                    "key": "can_shut_water_off",
                    "label": "Can shut water off",
                    "condition_type": "urgency_contains",
                    "condition_keywords": ["active leak", "flooding"],
                    "question_text": "Can you shut the water off there?",
                    "collection_mode": "ask_once",
                    "include_in_sms": True,
                    "active": True,
                }
            ],
        )

        prompt = PromptBuilder().build(
            "913-555-0123",
            tenant=tenant,
            profile=repository.get_active_prompt_profile(tenant["id"]),
            intake_policy=policy,
        )

        self.assertIn("TENANT INTAKE POLICY", prompt)
        self.assertIn("Homeowner or renter", prompt)
        self.assertIn("Can shut water off", prompt)
        self.assertIn("extra_fields", prompt)
        self.assertIn("Before calling submit_service_request, ask every active required and ask_once", prompt)
        self.assertIn("Do not silently skip ask_once questions", prompt)
        self.assertIn("Ask exactly one question per turn", prompt)
        self.assertIn("ask only the backend-guided missing question", prompt)
        self.assertIn("declined", prompt)
        self.assertIn("unknown", prompt)
        self.assertIn("A first name is enough", prompt)
        self.assertIn("Last name is not required", prompt)
        self.assertIn("Collect exactly: issue, urgency, address, callback, name", prompt)

    def test_legacy_required_false_maps_to_ask_once(self):
        tenant = repository.get_default_tenant()
        policy = repository.update_intake_policy(
            tenant["id"],
            extra_questions=[
                {
                    "key": "property_role",
                    "label": "Homeowner or renter",
                    "question_text": "Are you the homeowner or are you renting?",
                    "required": False,
                    "include_in_sms": True,
                    "active": True,
                }
            ],
        )

        questions = extra_questions(policy)

        self.assertEqual(questions[0]["collection_mode"], "ask_once")

    async def test_ask_once_extra_question_blocks_premature_submit(self):
        tenant = repository.get_default_tenant()
        repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("ask_once")],
        )
        repository.create_or_update_call("CALL_ASK_ONCE_MISSING", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()
        intake_state = {}

        result = await process_service_request(
            "CALL_ASK_ONCE_MISSING",
            dict(BASE_ARGS),
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
            intake_state=intake_state,
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "intake_policy_missing_extra_fields")
        self.assertEqual(result.output["missing_extra_fields"][0]["key"], "property_role")
        self.assertIn("Ask exactly this one question and wait for the caller's answer", result.output["guidance"])
        self.assertIn("Are you the homeowner or are you renting?", result.output["guidance"])
        self.assertFalse(result.should_hangup)
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_ASK_ONCE_MISSING"))
        self.assertEqual(intake_state["pending_extra_question"]["key"], "property_role")
        events = [
            event for event in repository.list_recent_call_events()
            if event["call_sid"] == "CALL_ASK_ONCE_MISSING"
        ]
        self.assertIn("intake_policy_missing_extra_fields", {event["event_type"] for event in events})
        self.assertIn("intake_question_required", {event["event_type"] for event in events})

    async def test_ask_once_extra_field_answer_is_saved_and_sms_can_include_it(self):
        tenant = repository.get_default_tenant()
        policy = repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("ask_once")],
        )
        args = dict(BASE_ARGS)
        args["extra_fields"] = {"property_role": "homeowner"}
        repository.create_or_update_call("CALL_EXTRA", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_EXTRA",
            args,
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)
        lead = repository.get_lead_by_call_sid("CALL_EXTRA")
        self.assertEqual(lead["extra_fields"], {"property_role": "homeowner"})
        self.assertEqual(json.loads(lead["extra_fields_json"]), {"property_role": "homeowner"})
        body = build_sms_body(args, "+19135550123", policy)
        self.assertIn("Extra:", body)
        self.assertIn("Homeowner or renter: homeowner", body)

    async def test_ask_once_unknown_without_caller_response_is_rejected(self):
        tenant = repository.get_default_tenant()
        repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("ask_once")],
        )
        args = dict(BASE_ARGS)
        args["extra_fields"] = {"property_role": "unknown"}
        repository.create_or_update_call("CALL_ASK_ONCE_UNKNOWN", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_ASK_ONCE_UNKNOWN",
            args,
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
            intake_state={
                "pending_extra_question": {
                    "key": "property_role",
                    "question_text": "Are you the homeowner or are you renting?",
                    "asked": True,
                    "caller_response_after_pending": False,
                }
            },
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "intake_policy_unanswered_extra_field")
        self.assertFalse(result.should_hangup)
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_ASK_ONCE_UNKNOWN"))

    async def test_ask_once_unknown_after_caller_response_is_accepted(self):
        tenant = repository.get_default_tenant()
        policy = repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("ask_once")],
        )
        args = dict(BASE_ARGS)
        args["extra_fields"] = {"property_role": "unknown"}
        repository.create_or_update_call("CALL_ASK_ONCE_UNKNOWN_AFTER_RESPONSE", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_ASK_ONCE_UNKNOWN_AFTER_RESPONSE",
            args,
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
            intake_state={
                "pending_extra_question": {
                    "key": "property_role",
                    "question_text": "Are you the homeowner or are you renting?",
                    "asked": True,
                    "caller_response_after_pending": True,
                    "caller_response_text": "I don't know.",
                }
            },
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)
        lead = repository.get_lead_by_call_sid("CALL_ASK_ONCE_UNKNOWN_AFTER_RESPONSE")
        self.assertEqual(lead["extra_fields"]["property_role"], "unknown")
        self.assertIn("Homeowner or renter: unknown", build_sms_body(args, "+19135550123", policy))

    async def test_required_extra_field_blocks_unknown_value(self):
        tenant = repository.get_default_tenant()
        repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("required")],
        )
        args = dict(BASE_ARGS)
        args["extra_fields"] = {"property_role": "unknown"}
        repository.create_or_update_call("CALL_REQUIRED_EXTRA", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_REQUIRED_EXTRA",
            args,
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertFalse(result.output["success"])
        self.assertEqual(result.output["reason"], "intake_policy_missing_extra_fields")
        self.assertEqual(result.output["missing_extra_fields"][0]["collection_mode"], "required")
        self.assertFalse(result.should_hangup)
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_REQUIRED_EXTRA"))

    async def test_passive_extra_question_does_not_block_submit(self):
        tenant = repository.get_default_tenant()
        repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("passive")],
        )
        repository.create_or_update_call("CALL_PASSIVE_MISSING", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_PASSIVE_MISSING",
            dict(BASE_ARGS),
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)

    async def test_conditional_required_extra_field_only_blocks_when_applicable(self):
        tenant = repository.get_default_tenant()
        repository.update_intake_policy(
            tenant["id"],
            conditional_questions=[
                {
                    "key": "can_shut_water_off",
                    "label": "Can shut water off",
                    "condition_type": "urgency_contains",
                    "condition_keywords": ["active leak"],
                    "question_text": "Can you shut the water off there?",
                    "collection_mode": "required",
                    "include_in_sms": True,
                    "active": True,
                }
            ],
        )
        sender = FakeSmsSender()

        repository.create_or_update_call("CALL_COND_BLOCK", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        blocked = await process_service_request(
            "CALL_COND_BLOCK",
            dict(BASE_ARGS),
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        not_applicable_args = dict(BASE_ARGS)
        not_applicable_args["urgency"] = "Not actively leaking"
        repository.create_or_update_call("CALL_COND_PASS", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        passed = await process_service_request(
            "CALL_COND_PASS",
            not_applicable_args,
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertFalse(blocked.output["success"])
        self.assertIn("can_shut_water_off", [field["key"] for field in blocked.output["missing_extra_fields"]])
        self.assertTrue(passed.output["success"])
        self.assertEqual(len(sender.calls), 1)

    async def test_invalid_core_address_still_blocks_even_with_extra_fields(self):
        tenant = repository.get_default_tenant()
        repository.update_intake_policy(
            tenant["id"],
            extra_questions=[property_role_question("required")],
        )
        args = dict(BASE_ARGS)
        args["address"] = "my house"
        args["extra_fields"] = {"property_role": "homeowner"}
        repository.create_or_update_call("CALL_BAD_CORE", "+19135550123", "+15551234567", tenant_id=tenant["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_BAD_CORE",
            args,
            "+19135550123",
            "+15557654321",
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant["id"],
        )

        self.assertFalse(result.output["success"])
        self.assertIn("address", result.output["missing_fields"])
        self.assertEqual(sender.calls, [])
        self.assertIsNone(repository.get_lead_by_call_sid("CALL_BAD_CORE"))

    def test_sms_includes_only_active_include_in_sms_fields(self):
        tenant = repository.get_default_tenant()
        policy = repository.update_intake_policy(
            tenant["id"],
            extra_questions=[
                {
                    "key": "property_role",
                    "label": "Homeowner or renter",
                    "question_text": "Are you the homeowner or are you renting?",
                    "collection_mode": "ask_once",
                    "include_in_sms": True,
                    "active": True,
                },
                {
                    "key": "parking_note",
                    "label": "Parking note",
                    "question_text": "Any parking notes?",
                    "collection_mode": "ask_once",
                    "include_in_sms": False,
                    "active": True,
                },
                {
                    "key": "inactive_note",
                    "label": "Inactive note",
                    "question_text": "Inactive question?",
                    "collection_mode": "ask_once",
                    "include_in_sms": True,
                    "active": False,
                },
            ],
        )
        args = dict(BASE_ARGS)
        args["extra_fields"] = {
            "property_role": "homeowner",
            "parking_note": "use driveway",
            "inactive_note": "ignore me",
        }

        body = build_sms_body(args, "+19135550123", policy)
        rows = sms_extra_field_rows(policy, args)

        self.assertEqual(rows, [("Homeowner or renter", "homeowner")])
        self.assertIn("Homeowner or renter: homeowner", body)
        self.assertNotIn("Parking note", body)
        self.assertNotIn("Inactive note", body)

    async def test_tenant_a_intake_policy_does_not_affect_tenant_b(self):
        tenant_a = repository.create_tenant(
            "Tenant A",
            "tenant-a-intake",
            "Tenant A Plumbing",
            "Tenant A plumbing, what's going on?",
            "+15550000001",
        )
        tenant_b = repository.create_tenant(
            "Tenant B",
            "tenant-b-intake",
            "Tenant B Plumbing",
            "Tenant B plumbing, what's going on?",
            "+15550000002",
        )
        repository.update_intake_policy(
            tenant_a["id"],
            extra_questions=[property_role_question("required")],
        )
        repository.create_or_update_call("CALL_TENANT_B_INTAKE", "+19135550123", "+15552220002", tenant_id=tenant_b["id"])
        sender = FakeSmsSender()

        result = await process_service_request(
            "CALL_TENANT_B_INTAKE",
            dict(BASE_ARGS),
            "+19135550123",
            tenant_b["notification_sms_number"],
            sender,
            caller_text=CALLER_TEXT,
            tenant_id=tenant_b["id"],
        )

        self.assertTrue(result.output["success"])
        self.assertEqual(len(sender.calls), 1)
        lead = repository.get_lead_by_call_sid("CALL_TENANT_B_INTAKE")
        self.assertEqual(lead["tenant_id"], tenant_b["id"])


if __name__ == "__main__":
    unittest.main()
