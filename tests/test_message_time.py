"""Deterministic receipts/model payloads and fake-send integration; no real chat."""

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import message_time as mt  # noqa: E402
import palacer_engine as engine  # noqa: E402


class Clock:
    boot = "test-boot"

    def __init__(self):
        self.w = 1793347200.0
        self.m = 100.0

    def wall(self):
        return self.w

    def monotonic(self):
        return self.m

    def advance(self, seconds):
        self.w += seconds
        self.m += seconds


class ModelClient:
    def __init__(self, clock):
        self.clock = clock
        self.requests = []

    def post(self, url, **kwargs):
        body = kwargs["json"]
        self.requests.append(body)
        n = len(self.requests)
        if n == 1:
            obj = {"tool_calls": []}
        elif n == 2:
            obj = {
                "style": "简短",
                "children": [{"task": "回应甲"}, {"task": "回应乙"}],
            }
        else:
            obj = {"say": "答" + str(n), "children": []}
        self.clock.advance(2)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(obj)}}
                ],
                "usage": {},
            },
        )


class TimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.path = Path(self.temp.name) / "times.sqlite3"
        self.times = mt.MessageTimes(self.path, self.clock)

    def tearDown(self):
        self.times.close()
        self.temp.cleanup()

    def test_persistence_clock_jump_restart_and_old_history(self):
        first = self.times.record("u", "user", "重复")
        self.clock.advance(3)
        second = self.times.record("u", "user", "重复")
        self.assertNotEqual(first.message_id, second.message_id)
        self.clock.w -= 86400
        text = self.times.context(
            "u", [second], [{"role": "user", "content": "旧历史"}]
        )
        self.assertIn("3.0 秒前", text)
        self.assertIn("本轮首条输入距前一次用户输入：3.0 秒", text)
        self.assertIn("时间未知", text)
        self.times.close()
        self.times = mt.MessageTimes(self.path, self.clock)
        self.assertEqual(self.times.get(first.message_id)["wall"], 1793347200.0)
        self.clock.boot = "another-boot"
        self.assertIn("单调时钟基准已变", self.times.context("u"))

    def test_exact_history_binding_does_not_match_repeated_text_or_reordered_history(
        self,
    ):
        msg = self.times.record("u", "user", "重复")
        item = {"role": "user", "content": [{"type": "text", "text": "重复"}]}
        self.times.bind(msg, "cid", 1, item)
        self.assertEqual(self.times.history_ids("u", "cid", 1, item), [msg.message_id])
        self.assertEqual(self.times.history_ids("u", "cid", 0, item), [])
        self.assertEqual(self.times.history_ids("u", "other-cid", 1, item), [])
        self.assertEqual(
            self.times.history_ids(
                "u", "cid", 1, {"role": "user", "content": "changed"}
            ),
            [],
        )

    def test_actual_mother_and_every_child_payload_get_fresh_tail(self):
        self.times.record("u", "user", "先前")
        self.clock.advance(5)
        self.times.record("u", "assistant", "之前回复")
        self.clock.advance(10800)
        a = self.times.record("u", "user", "回来啦")
        hist = [{"role": "user", "content": "旧历史"}]
        client = ModelClient(self.clock)
        with (
            patch.object(engine.cb, "build_bundle", return_value={}),
            patch.object(engine.cb, "render_bundle", return_value="固定资料"),
        ):
            result = engine.generate_turn(
                client,
                "http://test",
                "synthetic",
                "u",
                "回来啦",
                history=hist,
                time_context=lambda: self.times.context("u", [a], hist),
            )
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(result["messages"], ["答3", "答4"])
        for i, request in enumerate(client.requests):
            system, user = request["messages"]
            self.assertNotIn("程序时间事实", system["content"])
            self.assertIn("当前本地时间", user["content"])
            self.assertIn("Asia/Singapore", user["content"])
            self.assertIn("时间未知", user["content"])
            self.assertIn("本轮首条输入距前一次用户输入：10805.0 秒", user["content"])
            self.assertIn(f"{2 * i:.1f} 秒前", user["content"])
            self.assertTrue(
                user["content"].endswith("时间未知（无可靠逐条记录，不补造）")
            )
            self.assertEqual(set(system), {"role", "content"})
            self.assertEqual(set(user), {"role", "content"})
        last = client.requests[-1]["messages"][-1]["content"]
        self.assertIn("尚未发送的草稿", last)
        self.assertIn("答3", last)
        self.assertNotIn("前面已经发过", last)


if importlib.util.find_spec("astrbot") is None:
    Main = None
else:
    spec = importlib.util.spec_from_file_location("time_test_main", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Main = module.Main


class ConversationManager:
    def __init__(self):
        self.history = [
            {"role": "user", "content": "旧历史"},
            {"role": "assistant", "content": "旧回复"},
        ]

    async def get_curr_conversation_id(self, umo):
        return "cid"

    async def get_conversation(self, umo, cid):
        return SimpleNamespace(history=json.dumps(self.history))

    async def update_conversation(self, umo, cid, history):
        self.history = history


class Event:
    unified_msg_origin = "synthetic:test"

    def __init__(self, text, clock, sender=None):
        self.message_str = text
        self.clock = clock
        self.extra = {}
        self.sent = []
        self.sender = sender

    def set_extra(self, key, value):
        self.extra[key] = value

    async def send(self, chain):
        text = chain.chain[0].text
        if self.sender:
            result = await self.sender(text)
            if result is False:
                return False
        self.clock.advance(1)
        self.sent.append((text, self.clock.wall(), self.clock.monotonic()))


async def spin():
    for _ in range(30):
        await asyncio.sleep(0)


@unittest.skipIf(Main is None, "requires AstrBot runtime")
class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.cm = ConversationManager()
        self.p = object.__new__(Main)
        self.p.times = mt.MessageTimes(
            Path(self.temp.name) / "times.sqlite3", self.clock
        )
        self.p.context = SimpleNamespace(conversation_manager=self.cm)
        self.p.images = module.ib.ImageBridge(self.p)
        self.p.session_distill = module.sd.SessionDistill(
            self.p.context, self.p.times, lambda **row: None
        )
        self.p.voice_decisions = {}
        self.p.tier_marks = {}
        self.p.engine_workers = set()
        self.p.memory_shortlist = False
        self.p._voice_card = lambda persona_id: "虚构声音卡"
        self.p.history_locks = {}
        self.p.sessions = {}
        self.p.proactive_sessions = {}
        self.p.events = {}
        self.p.turn_done = {}
        self.p.enabled = True
        self.p.burst_started = {}
        self.p.mode = "canary"
        self.p.startup_delay = 0
        self.p.message_window = 0
        self.p._log = lambda **row: None
        self.p._plugin = lambda name: None
        self.p._persona_for = lambda umo: "synthetic"
        self.p.generation_sem = asyncio.Semaphore(1)
        self.p.base = "http://test"
        self.p.key = "synthetic"
        self.p.client = ModelClient(self.clock)

        async def model_post(*args, **kwargs):
            return self.p.client.post(*args, **kwargs)

        self.p.async_client = SimpleNamespace(post=model_post)
        self.umo = Event.unified_msg_origin
        self.p.umo_allow = {self.umo}
        self.calls = []

        async def gen(umo, batch, sent):
            self.calls.append((list(batch), list(sent)))
            return ["回复一。", "回复二。"]

        self.p._generate = gen

        async def media(event):
            return event.message_str

        self.p._text_with_media = media

    async def asyncTearDown(self):
        for s in [*self.p.sessions.values(), *self.p.proactive_sessions.values()]:
            await s.close()
        self.p.times.close()
        self.temp.cleanup()

    def rows(self):
        return [
            dict(r)
            for r in self.p.times.db.execute("SELECT * FROM events ORDER BY seq")
        ]

    async def test_repeated_batch_receives_before_slow_media_and_keeps_order(self):
        gate = asyncio.Event()

        async def media(event):
            if event is first:
                await gate.wait()
            return event.message_str + (
                " 视频观察一" if event is first else " 视频观察二"
            )

        first, second = Event("相同", self.clock), Event("相同", self.clock)
        self.p._text_with_media = media
        t = asyncio.create_task(self.p._accept_input(first))
        await spin()
        self.assertEqual(self.rows()[0]["wall"], self.clock.wall())
        self.clock.advance(4)
        await self.p._accept_input(second)
        await spin()
        self.assertEqual(self.calls, [])
        self.clock.advance(20)
        gate.set()
        await t
        await self.p.sessions[self.umo].task
        batch, _ = self.calls[0]
        self.assertEqual(list(batch), ["相同 视频观察一", "相同 视频观察二"])
        self.assertNotEqual(batch[0].message_id, batch[1].message_id)
        rows = self.rows()
        self.assertEqual([r["wall"] for r in rows[:2]], [1793347200, 1793347204])
        self.assertEqual([r["wall"] for r in rows[2:]], [x[1] for x in second.sent])
        self.assertEqual(
            [r["text"] for r in rows], [self.p._text_of(m) for m in self.cm.history[2:]]
        )
        self.assertEqual(
            self.p.sessions[self.umo].sent_at, [(x[0], x[2]) for x in second.sent]
        )
        self.assertTrue(all(set(m) == {"role", "content"} for m in self.cm.history))
        self.assertFalse(any("程序时间事实" in x[0] for x in second.sent))
        hist = await self.p._recent_history(self.umo)
        tail = self.p.times.context(self.umo, history=hist)
        self.assertIn("时间未知", tail)
        self.assertEqual(len(hist[-1]["_time_ids"]), 1)

    async def test_generation_interruption_cancels_unsent_and_preserves_identity(self):
        started = asyncio.Event()

        async def gen(umo, batch, sent):
            self.calls.append((list(batch), list(sent)))
            if len(self.calls) == 1:
                started.set()
                await asyncio.Event().wait()
            return ["重生成。"]

        self.p._generate = gen
        await self.p._accept_input(Event("重复", self.clock))
        await started.wait()
        self.clock.advance(2)
        e = Event("重复", self.clock)
        await self.p._accept_input(e)
        await self.p.sessions[self.umo].task
        self.assertEqual([str(x) for x in self.calls[-1][0]], ["重复", "重复"])
        self.assertEqual(
            [x.message_id for x in self.calls[0][0]], [self.calls[-1][0][0].message_id]
        )
        self.assertEqual(
            [r["role"] for r in self.rows()], ["user", "user", "assistant"]
        )
        self.assertEqual([x[0] for x in e.sent], ["重生成"])

    async def test_interruption_after_first_send_exposes_actual_elapsed_reply(self):
        original = self.p._send
        first_sent = asyncio.Event()
        hold = asyncio.Event()

        async def send(umo, text, delivery=None):
            result = await original(umo, text, delivery)
            first_sent.set()
            await hold.wait()
            return result

        self.p._send = send
        await self.p._accept_input(Event("甲", self.clock))
        await first_sent.wait()
        self.clock.advance(7)
        next_input = asyncio.create_task(self.p._accept_input(Event("乙", self.clock)))
        await spin()
        hold.set()
        await next_input
        self.p._send = original
        await self.p.sessions[self.umo].task
        self.assertEqual(self.calls[-1][1], ["回复一"])
        tail = self.p.times.context(self.umo, self.calls[-1][0])
        self.assertIn("9.0 秒前", tail)
        self.assertEqual(sum(r["text"] == "甲" for r in self.rows()), 1)
        self.assertEqual(len(self.cm.history), 7)

    async def test_multiple_interruptions_during_external_send_settle_once(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(text):
            entered.set()
            await release.wait()
            return True

        first = Event("相同", self.clock, delayed)
        await self.p._accept_input(first)
        await entered.wait()
        second = Event("相同", self.clock)
        t2 = asyncio.create_task(self.p._accept_input(second))
        await spin()
        self.clock.advance(3)
        third = Event("相同", self.clock)
        t3 = asyncio.create_task(self.p._accept_input(third))
        await spin()
        release.set()
        await asyncio.gather(t2, t3)
        await self.p.sessions[self.umo].task
        self.assertEqual(len(first.sent), 1)
        self.assertEqual(len(second.sent), 0)
        self.assertEqual(len(third.sent), 2)
        batch, sent = self.calls[-1]
        self.assertEqual(len({m.message_id for m in batch}), 3)
        self.assertEqual(sent, ["回复一"])
        self.assertEqual(len(self.rows()), 6)
        self.assertEqual(len(self.cm.history), 8)
        self.assertEqual(
            [self.p._text_of(m) for m in self.cm.history[2:]],
            ["相同", "回复一", "相同", "相同", "回复一", "回复二"],
        )

    async def test_partial_failure_retains_only_successful_reply_and_history(self):
        count = 0

        async def partial(text):
            nonlocal count
            count += 1
            return count == 1

        e = Event("甲", self.clock, partial)
        await self.p._accept_input(e)
        await self.p.sessions[self.umo].task
        self.assertEqual([r["text"] for r in self.rows()], ["甲", "回复一"])
        self.assertEqual(
            [self.p._text_of(m) for m in self.cm.history[2:]], ["甲", "回复一"]
        )
        self.assertEqual(
            [r["wall"] for r in self.rows() if r["role"] == "assistant"], [e.sent[0][1]]
        )

    async def test_send_false_and_exception_are_not_successful(self):
        for mode in ("false", "raise"):

            async def fail(text):
                if mode == "raise":
                    raise RuntimeError("fake send failed")
                return False

            e = Event("失败" + mode, self.clock, fail)
            await self.p._accept_input(e)
            await self.p.sessions[self.umo].task
            self.assertEqual(e.sent, [])
        self.assertFalse(any(r["role"] == "assistant" for r in self.rows()))
        self.assertEqual(len(self.cm.history), 4)

    async def test_proactive_success_failure_and_cancel_do_not_create_user_receipts(
        self,
    ):
        outgoing = []

        async def send(umo, chain):
            self.clock.advance(2)
            outgoing.append(chain.chain[0].text)
            return True

        self.p.context.send_message = send
        self.assertTrue(await self.p.send_proactive(self.umo, "主动一。主动二。"))
        await self.p.proactive_sessions[self.umo].task
        self.assertEqual(outgoing, ["主动一", "主动二"])
        rows = self.rows()
        self.assertEqual([r["role"] for r in rows], ["assistant", "assistant"])
        self.assertTrue(all(r["proactive"] for r in rows))
        self.assertIn(
            "主动开口，没有新用户输入", self.p.times.context(self.umo, proactive=True)
        )

        async def fail(umo, chain):
            return False

        self.p.context.send_message = fail
        await self.p.send_proactive(self.umo, "失败。")
        await self.p.proactive_sessions[self.umo].task
        self.assertEqual(len(self.rows()), 2)
        self.p.startup_delay = 60
        await self.p.send_proactive(self.umo, "取消。")
        await self.p._cancel_proactive(self.umo)
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual([self.p._text_of(m) for m in self.cm.history[2:]], outgoing)

    async def test_main_preserves_intentional_empty_generation(self):
        with patch.object(
            module.engine, "generate_turn", return_value={"messages": []}
        ):
            self.assertEqual(
                await Main._generate(self.p, self.umo, ["虚构输入"], []), []
            )

    async def test_proactive_and_watch_pass_distinct_generation_purposes(self):
        with patch.object(
            module.engine, "generate_turn", return_value={"messages": ["虚构消息"]}
        ) as generate:
            self.assertEqual(
                await Main.generate_proactive(self.p, self.umo), ["虚构消息"]
            )
            self.assertEqual(generate.call_args.kwargs.get("turn_kind"), "proactive")
            self.assertEqual(
                await Main.generate_watch_comment(self.p, self.umo, "虚构画面", 10),
                "虚构消息",
            )
            self.assertEqual(generate.call_args.kwargs.get("turn_kind"), "watch")

    async def test_main_generation_delivers_timing_to_actual_model_payloads(self):
        a = self.p.times.record(self.umo, "user", "你好")
        with (
            patch.object(module.engine.cb, "build_bundle", return_value={}),
            patch.object(module.engine.cb, "render_bundle", return_value="资料"),
        ):
            await Main._generate(self.p, self.umo, [a], [])
        self.assertEqual(len(self.p.client.requests), 4)
        for request in self.p.client.requests:
            self.assertIn(a.message_id, request["messages"][-1]["content"])
            self.assertIn("程序时间事实", request["messages"][-1]["content"])

    async def test_proactive_generation_delivers_timing_to_mother_and_children(self):
        with (
            patch.object(module.engine.cb, "build_bundle", return_value={}),
            patch.object(module.engine.cb, "render_bundle", return_value="资料"),
        ):
            await self.p.generate_proactive(self.umo)
        self.assertEqual(len(self.p.client.requests), 4)
        self.assertTrue(
            all(
                "主动开口，没有新用户输入" in r["messages"][-1]["content"]
                for r in self.p.client.requests
            )
        )

    async def test_regeneration_does_not_duplicate_already_persisted_reply(self):
        a = self.p.times.record(self.umo, "user", "甲")
        reply = self.p.times.record(self.umo, "assistant", "已发送")
        await self.p._write_history(self.umo, [a, reply])
        b = self.p.times.record(self.umo, "user", "乙")
        with (
            patch.object(module.engine.cb, "build_bundle", return_value={}),
            patch.object(module.engine.cb, "render_bundle", return_value="资料"),
        ):
            await Main._generate(self.p, self.umo, [a, b], ["已发送"])
        for request in self.p.client.requests[:2]:
            recent = (
                request["messages"][-1]["content"]
                .split("【最近对话】")[1]
                .split("【用户消息】")[0]
            )
            self.assertEqual(recent.count("你: 已发送"), 1)

    async def test_watch_draft_timing_and_send_do_not_invent_user_input(self):
        with (
            patch.object(module.engine.cb, "build_bundle", return_value={}),
            patch.object(module.engine.cb, "render_bundle", return_value="资料"),
        ):
            text = await self.p.generate_watch_comment(self.umo, "只看到红色画面", 45)
        self.assertTrue(text)
        self.assertEqual(self.rows(), [])
        self.assertTrue(
            all(
                "视频 45 秒时才发送" in r["messages"][-1]["content"]
                for r in self.p.client.requests
            )
        )
        outgoing = []

        async def send(umo, chain):
            self.clock.advance(1)
            outgoing.append(chain.chain[0].text)
            return True

        self.p.context.send_message = send
        self.assertFalse(
            await self.p.send_watch_comment(self.umo, "不可发。", guard=lambda: False)
        )
        self.assertTrue(
            await self.p.send_watch_comment(self.umo, "看到了。", guard=lambda: True)
        )
        self.assertEqual(outgoing, ["看到了"])
        self.assertEqual([r["role"] for r in self.rows()], ["assistant"])
        self.assertEqual(self.rows()[0]["wall"], self.clock.wall())
        self.assertEqual(self.p._text_of(self.cm.history[-1]), "看到了")

        async def fail(*args):
            return False

        self.p.context.send_message = fail
        self.assertFalse(
            await self.p.send_watch_comment(self.umo, "失败。", guard=lambda: True)
        )
        self.assertEqual(len(self.rows()), 1)

    async def test_watch_context_bypasses_full_video_observation(self):
        watch = SimpleNamespace(
            watch_api_version=1,
            observe_event=AsyncMock(return_value="【陪看状态】等待播放进度"),
        )
        video = SimpleNamespace(observe_event=AsyncMock(return_value="不应整片分析"))
        self.p._plugin = lambda name: (
            watch
            if name == "astrbot_plugin_watch_together"
            else video
            if name == "astrbot_plugin_video_storyboard"
            else None
        )
        event = SimpleNamespace(message_str="陪我看视频", unified_msg_origin=self.umo)
        text = await self.p._media_text(event)
        self.assertIn("等待播放进度", text)
        video.observe_event.assert_not_awaited()

    async def test_registered_watch_command_is_not_treated_as_palacer_chat_after_prefix_removal(
        self,
    ):
        self.p.enabled = True
        self.p.platforms = set()
        event = SimpleNamespace(
            unified_msg_origin=self.umo,
            message_str="watch status",
            get_platform_name=lambda: "synthetic",
            get_extra=lambda key: [
                SimpleNamespace(
                    handler_name="watch_command",
                    handler_module_path="data.plugins.astrbot_plugin_watch_together.main",
                )
            ],
        )
        self.assertFalse(self.p._match(event))


if __name__ == "__main__":
    unittest.main()
