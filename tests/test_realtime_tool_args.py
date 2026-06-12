import json
import unittest

import main


class RealtimeToolArgumentTests(unittest.TestCase):
    def test_function_call_arguments_delta_done_accumulates_full_args(self):
        records = {}

        main.register_function_call_delta(
            records,
            {
                "type": "response.function_call_arguments.delta",
                "call_id": "fc_1",
                "item_id": "item_1",
                "response_id": "resp_1",
                "delta": '{"issue":"Kitchen sink leak",',
            },
        )
        record = main.register_function_call_done(
            records,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "fc_1",
                "item_id": "item_1",
                "response_id": "resp_1",
                "arguments": '{"issue":"Kitchen sink leak","urgency":"active leak"}',
            },
        )

        result = main.tool_call_args_for_processing(record)

        self.assertEqual(result.state, "ready")
        self.assertEqual(result.source, "response.function_call_arguments.done")
        self.assertEqual(result.args["issue"], "Kitchen sink leak")
        self.assertEqual(result.args["urgency"], "active leak")

    def test_done_arguments_are_source_of_truth_over_partial_deltas(self):
        records = {}
        main.register_function_call_delta(
            records,
            {
                "call_id": "fc_2",
                "item_id": "item_2",
                "response_id": "resp_2",
                "delta": '{"issue":"Leaking tank in the kitchen","urgency',
            },
        )
        record = main.register_function_call_done(
            records,
            {
                "call_id": "fc_2",
                "item_id": "item_2",
                "response_id": "resp_2",
                "arguments": json.dumps(
                    {
                        "issue": "Leaking tank in the kitchen",
                        "urgency": "water still coming out",
                        "address": "6100 West 120th Street",
                        "callback": "+17327890675",
                        "name": "Sam",
                    }
                ),
            },
        )

        result = main.tool_call_args_for_processing(record)

        self.assertEqual(result.state, "ready")
        self.assertEqual(result.args["address"], "6100 West 120th Street")

    def test_output_item_done_partial_args_are_not_converted_to_empty_args(self):
        records = {}
        record = main.register_function_call_output_item(
            records,
            {
                "type": "response.output_item.done",
                "response_id": "resp_partial",
                "item": {
                    "id": "item_partial",
                    "type": "function_call",
                    "name": "submit_service_request",
                    "call_id": "fc_partial",
                    "arguments": '{"issue":"Leaking tank in the kitchen","urgency',
                },
            },
        )

        result = main.tool_call_args_for_processing(record, allow_output_item_fallback=True)

        self.assertEqual(result.state, "parse_failed")
        self.assertEqual(result.source, "response.output_item.done")
        self.assertIsNone(result.args)
        self.assertNotEqual(result.args, {})

    def test_malformed_final_args_do_not_become_validation_followup(self):
        records = {}
        record = main.register_function_call_done(
            records,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "fc_bad",
                "item_id": "item_bad",
                "response_id": "resp_bad",
                "arguments": '{"issue":"Leaking tank in the kitchen","urgency',
            },
        )

        result = main.tool_call_args_for_processing(record)
        output = main.build_tool_args_parse_failed_output()

        self.assertEqual(result.state, "parse_failed")
        self.assertEqual(output["reason"], "tool_args_parse_failed")
        self.assertIn("Retry submit_service_request with valid JSON", output["guidance"])
        self.assertNotIn("What's going on", output["guidance"])

    def test_duplicate_aliases_point_to_one_tool_record(self):
        records = {}
        by_delta = main.register_function_call_delta(
            records,
            {
                "call_id": "fc_dupe",
                "item_id": "item_dupe",
                "response_id": "resp_dupe",
                "delta": "{}",
            },
        )
        by_output = main.register_function_call_output_item(
            records,
            {
                "response_id": "resp_dupe",
                "item": {
                    "id": "item_dupe",
                    "type": "function_call",
                    "name": "submit_service_request",
                    "call_id": "fc_dupe",
                    "arguments": "{}",
                },
            },
        )

        self.assertIs(by_delta, by_output)
        self.assertEqual(len(main.unique_tool_call_records(records)), 1)

    def test_realtime_15_output_item_fallback_still_parses_clean_args(self):
        records = {}
        record = main.register_function_call_output_item(
            records,
            {
                "type": "response.output_item.done",
                "response_id": "resp_15",
                "item": {
                    "id": "item_15",
                    "type": "function_call",
                    "name": "submit_service_request",
                    "call_id": "fc_15",
                    "arguments": json.dumps(
                        {
                            "issue": "Kitchen sink leak",
                            "urgency": "active leak",
                            "address": "6100 West 120th Street",
                            "callback": "+17327890675",
                            "name": "Sam",
                        }
                    ),
                },
            },
        )

        result = main.tool_call_args_for_processing(record, allow_output_item_fallback=True)

        self.assertEqual(result.state, "ready")
        self.assertEqual(result.source, "response.output_item.done")
        self.assertEqual(result.args["name"], "Sam")

    def test_incomplete_done_event_is_not_processed(self):
        records = {}
        record = main.register_function_call_done(
            records,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "fc_cancelled",
                "item_id": "item_cancelled",
                "response_id": "resp_cancelled",
                "status": "incomplete",
                "arguments": '{"issue":"Kitchen sink leak"}',
            },
        )

        result = main.tool_call_args_for_processing(record)

        self.assertEqual(result.state, "incomplete")

    def test_response_create_guard_delays_parser_and_validation_followups(self):
        self.assertEqual(
            main.should_delay_response_create("validation_followup", response_active=True),
            "response_active",
        )
        self.assertEqual(
            main.should_delay_response_create("tool_args_retry", caller_speaking=True),
            "caller_speaking",
        )
        self.assertIsNone(main.should_delay_response_create("validation_followup"))

    def test_barge_in_payload_records_state(self):
        payload = main.barge_in_event_payload(
            {"last_ai_transcript": "Is water still coming out?", "last_response_create_reason": "other"},
            response_active=True,
            assistant_speaking=True,
            suppress_active=False,
        )

        self.assertTrue(payload["response_active"])
        self.assertTrue(payload["assistant_speaking"])
        self.assertEqual(payload["last_response_create_reason"], "other")


if __name__ == "__main__":
    unittest.main()
