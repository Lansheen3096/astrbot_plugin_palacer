"""Synthetic model payloads only; never build a real context bundle or send messages."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import palacer_engine as engine


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append(kwargs["json"])
        obj = next(self.replies)
        if isinstance(obj, Exception):
            raise obj
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(obj, ensure_ascii=False)},
                    }
                ],
                "usage": {},
            },
        )


class TaskContractTests(unittest.TestCase):
    def generate(self, plan, replies, tool_calls=None, **kwargs):
        client = Client([{"tool_calls": tool_calls or []}, plan, *replies])
        # Fail closed if a test accidentally attempts to load personal context.
        with patch.object(
            engine.cb, "build_bundle", return_value={"persona": "虚构角色资料"}
        ):
            result = engine.generate_turn(
                client, "http://synthetic", "synthetic", "test", "我到家了", **kwargs
            )
        return result, client.requests

    def test_distilled_situation_goal_and_individual_roles_reach_every_writer(self):
        plan = {
            "style": "轻快自然",
            "context": "虚构语境：对方结束排练回家，上一问指的是排练。",
            "goal": "接住回家的消息，再问排练是否顺利。",
            "children": [
                {"role": "迎接", "task": "只回应已经到家", "hard": [0]},
                {
                    "role": "追问",
                    "task": "只问排练",
                    "handoff": "接着上一句问排练，不重新迎接",
                    "depends_on": [0],
                },
            ],
        }
        result, requests = self.generate(
            plan,
            [{"say": "回来啦"}, {"say": "排练顺利嘛？"}],
            history=[{"role": "user", "content": "之前在排练"}],
            sent=["路上小心"],
            time_context=lambda: "虚构时间事实",
        )
        self.assertEqual(result["messages"], ["回来啦", "排练顺利嘛？"])
        for request in requests[2:]:
            system, user = [m["content"] for m in request["messages"]]
            self.assertIn("所选 persona", system)
            self.assertIn(plan["context"], user)
            self.assertIn(plan["goal"], user)
            self.assertIn("迎接", user)
            self.assertIn("追问", user)
            self.assertIn("路上小心", user)
            self.assertIn("只能使用", system)
            self.assertNotIn("children 最多", system)
            self.assertTrue(user.endswith("虚构时间事实"))
        last = requests[-1]["messages"][-1]["content"]
        self.assertIn("接着上一句问排练，不重新迎接", last)
        self.assertIn("尚未发送的草稿", last)
        self.assertIn("回来啦", last)

    def test_duplicate_sibling_is_rewritten_within_existing_call_budget(self):
        plan = {"children": [{"task": "迎接"}, {"task": "问排练"}, {"task": "结束"}]}
        result, requests = self.generate(
            plan,
            [
                {"say": "回来啦。"},
                {"say": "回来啦！♡"},
                {"say": "排练顺利嘛？"},
                {"say": "先歇一会儿叭"},
            ],
        )
        self.assertEqual(
            result["messages"], ["回来啦。", "排练顺利嘛？", "先歇一会儿叭"]
        )
        self.assertEqual(result["calls"], 6)
        self.assertIn("重复", requests[4]["messages"][-1]["content"])
        self.assertIn("排练顺利嘛", requests[5]["messages"][-1]["content"])
        self.assertNotIn("回来啦！♡", requests[5]["messages"][-1]["content"])

    def test_missing_predecessor_skips_only_dependent_messages(self):
        plan = {
            "children": [
                {"task": "先回应到家"},
                {"task": "接前句补一句", "depends_on": [0]},
                {"task": "独立问排练", "depends_on": []},
            ]
        }
        result, requests = self.generate(
            plan, [RuntimeError("synthetic failure"), {"say": "排练顺利嘛？"}]
        )
        self.assertEqual(result["messages"], ["排练顺利嘛？"])
        self.assertEqual(len(requests), 4)
        self.assertEqual(result["nodes"][1]["status"], "dependency_unavailable")

    def test_silence_and_exhausted_duplicates_do_not_create_filler(self):
        for kind in ("proactive", "watch"):
            with self.subTest(kind=kind):
                result, requests = self.generate({"children": []}, [], turn_kind=kind)
                self.assertEqual(result["messages"], [])
                self.assertEqual(len(requests), 2)
        result, requests = self.generate(
            {"children": [{"task": "迎接"}]},
            [{"say": "回来啦。"}, {"say": "回来啦！"}],
            sent=["回来啦"],
        )
        self.assertEqual(result["messages"], [])
        self.assertEqual(len(requests), 4)
        self.assertEqual(result["nodes"][0]["status"], "duplicate")
        result, _ = self.generate(
            {"children": [{"task": "有话再说"}]}, [{"say": "[[NO_MESSAGE]]"}]
        )
        self.assertEqual(result["messages"], [])

    def test_numbers_and_meaningful_symbols_are_not_erased_by_deduplication(self):
        for first, second in (
            ("数值是1.5", "数值是15"),
            ("使用C++", "使用C#"),
            ("数值是-2", "数值是2"),
        ):
            with self.subTest(first=first, second=second):
                result, requests = self.generate(
                    {"children": [{"task": "甲"}, {"task": "乙"}]},
                    [{"say": first}, {"say": second}],
                )
                self.assertEqual(result["messages"], [first, second])
                self.assertEqual(len(requests), 4)

    def test_internal_triggers_are_not_presented_as_user_statements(self):
        for kind in ("proactive", "watch"):
            with self.subTest(kind=kind):
                result, requests = self.generate({"children": []}, [], turn_kind=kind)
                self.assertEqual(result["hard_facts"][-1]["source"], "trigger")
                self.assertNotIn("用户刚说", result["hard_facts"][-1]["text"])
                self.assertTrue(
                    all(
                        "【用户消息】" not in r["messages"][-1]["content"]
                        for r in requests
                    )
                )

    def test_total_writer_failure_is_reported_instead_of_becoming_silence(self):
        with self.assertRaisesRegex(RuntimeError, "写手"):
            self.generate(
                {"children": [{"task": "回答"}]}, [RuntimeError("synthetic failure")]
            )

    def test_latest_input_is_indexed_for_mother_fact_selection(self):
        _, requests = self.generate(
            {"children": [{"task": "迎接", "hard": [0]}]}, [{"say": "回来啦"}]
        )
        self.assertIn("[H0] 用户刚说：我到家了", requests[1]["messages"][-1]["content"])

    def test_invalid_writer_payload_is_not_sent_as_text(self):
        for payload in ({}, ["broken"], {"say": {"unexpected": "object"}}):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(RuntimeError, "写手"),
            ):
                self.generate({"children": [{"task": "回答"}]}, [payload])

    def test_repeated_duplicates_use_only_one_rewrite_for_entire_turn(self):
        result, requests = self.generate(
            {"children": [{"task": "甲"}, {"task": "乙"}, {"task": "丙"}]},
            [{"say": "回来啦"}] * 4,
        )
        self.assertEqual(result["messages"], ["回来啦"])
        self.assertEqual(len(requests), 6)
        self.assertEqual(
            [n["status"] for n in result["nodes"]],
            ["accepted", "duplicate", "duplicate"],
        )

    def test_invalid_dependencies_do_not_generate_or_forward_orphaned_text(self):
        for depends_on in ([0], [1], [-1], [True], "0"):
            with self.subTest(depends_on=depends_on):
                result, requests = self.generate(
                    {"children": [{"task": "错误依赖", "depends_on": depends_on}]}, []
                )
                self.assertEqual(result["messages"], [])
                self.assertEqual(len(requests), 2)
                self.assertEqual(result["nodes"][0]["status"], "invalid_dependency")

    def test_legacy_plan_keeps_recent_context_and_malformed_plan_has_safe_fallback(
        self,
    ):
        for plan in ({"children": [{"task": "直接回答"}]}, {"children": "invalid"}):
            with self.subTest(plan=plan):
                result, requests = self.generate(
                    plan,
                    [{"say": "回来啦"}],
                    history=[{"role": "user", "content": "虚构前文：正在排练"}],
                )
                self.assertEqual(result["messages"], ["回来啦"])
                self.assertIn(
                    "虚构前文：正在排练", requests[-1]["messages"][-1]["content"]
                )

    def test_tool_results_are_indexed_and_only_selected_facts_reach_writer(self):
        with patch.dict(
            engine.pt.TOOLS, {"synthetic": lambda args: "虚构检索结果"}, clear=True
        ):
            result, requests = self.generate(
                {"children": [{"task": "回应检索结果", "hard": [1]}]},
                [{"say": "查到啦"}],
                tool_calls=[{"name": "synthetic", "args": {}}],
            )
        self.assertIn("[H1] 虚构检索结果", requests[1]["messages"][-1]["content"])
        self.assertEqual(result["nodes"][0]["hard_used"], [(1, "虚构检索结果")])
        self.assertNotIn("[H0]", requests[-1]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
