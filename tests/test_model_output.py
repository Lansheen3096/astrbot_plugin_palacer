"""Empty reasoning-only responses must not silently become an ellipsis."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "palacer_output_under_test",
    Path(__file__).resolve().parents[1] / "palacer_harness.py",
)
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)


def response(text, finish="stop", reasoning=0, tokens=20):
    return {
        "choices": [{"finish_reason": finish, "message": {"content": text}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": tokens,
            "total_tokens": 100 + tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append(kwargs["json"])
        payload = next(self.replies)

        class Reply:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        return Reply()


class ModelOutputTests(unittest.TestCase):
    def test_time_tail_is_refreshed_for_retry_without_mutating_prefix_or_history(self):
        client = Client([response("", "length"), response('{"say":"嗯"}')])
        messages = [
            {"role": "system", "content": "固定人设"},
            {"role": "user", "content": "输入"},
        ]
        ticks = iter(["时间事实：0 秒", "时间事实：30 秒"])
        ph.call(
            client,
            "http://unit.test",
            "synthetic",
            messages,
            500,
            time_context=lambda: next(ticks),
        )
        self.assertEqual(
            client.requests[0]["messages"][-1]["content"], "输入\n\n时间事实：0 秒"
        )
        self.assertEqual(
            client.requests[1]["messages"][-1]["content"], "输入\n\n时间事实：30 秒"
        )
        self.assertEqual(messages[-1]["content"], "输入")
        self.assertTrue(
            all(r["messages"][0]["content"] == "固定人设" for r in client.requests)
        )

    def test_reasoning_only_length_response_gets_one_content_retry(self):
        client = Client(
            [
                response("", "length", reasoning=500, tokens=500),
                response('{"say":"我看到啦","children":[]}'),
            ]
        )
        text, usage, _ = ph.call(
            client,
            "http://unit.test",
            "synthetic",
            [],
            500,
            response_format={"type": "json_object"},
        )
        self.assertTrue(
            text.strip(), "reasoning-only truncation must not escape as empty content"
        )
        self.assertEqual(ph.parse_json(text)["say"], "我看到啦")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0]["thinking"]["type"], "enabled")
        self.assertEqual(client.requests[1]["thinking"]["type"], "disabled")
        self.assertEqual(usage["completion_tokens"], 520)
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 500)

    def test_normal_response_keeps_existing_reasoning_behavior(self):
        client = Client([response('{"say":"嗯","children":[]}')])
        text, _, _ = ph.call(client, "http://unit.test", "synthetic", [], 500)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["thinking"]["type"], "enabled")
        self.assertIn("嗯", text)

    def test_two_empty_responses_report_failure_without_looping(self):
        client = Client([response("", "length"), response("", "length")])
        with self.assertRaises(RuntimeError):
            ph.call(client, "http://unit.test", "synthetic", [], 500)
        self.assertEqual(len(client.requests), 2)


if __name__ == "__main__":
    unittest.main()
