"""Voice-card persistence, generation drafts, and default LLM routing."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_cards import VoiceCards, VoiceCardView


class StorageTests(unittest.TestCase):
    def test_scoped_atomic_persistence_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cards.json"
            cards = VoiceCards(path)
            self.assertEqual(cards.get("a"), "")
            cards.save("a", " 简洁、具体。 ")
            cards.save("b", "正式表达。")
            self.assertEqual(VoiceCards(path).get("a"), "简洁、具体。")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                cards.save("a", "x" * 12001)
            self.assertEqual(VoiceCards(path).get("a"), "简洁、具体。")
            cards.save("a", "  ")
            self.assertEqual(VoiceCards(path).get("a"), "")
            self.assertEqual(VoiceCards(path).get("b"), "正式表达。")

    def test_failed_save_does_not_activate_unsaved_card(self):
        with tempfile.TemporaryDirectory() as directory:
            cards = VoiceCards(Path(directory) / "cards.json")
            cards.save("a", "原卡")
            with patch("voice_cards.os.replace", side_effect=OSError):
                with self.assertRaises(OSError):
                    cards.save("a", "新卡")
            self.assertEqual(cards.get("a"), "原卡")
            self.assertEqual(list(Path(directory).glob(".voice-cards-*")), [])


class ViewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.view = VoiceCardView()
        self.view.voice_cards = VoiceCards(Path(self.directory.name) / "cards.json")
        self.persona = SimpleNamespace(
            persona_id="demo", system_prompt="虚构 persona：使用简洁正式语气。"
        )
        self.provider = SimpleNamespace(meta=lambda: SimpleNamespace(id="model-demo"))
        self.view.context = SimpleNamespace(
            persona_manager=SimpleNamespace(
                get_persona=AsyncMock(return_value=self.persona),
                get_all_personas=AsyncMock(return_value=[self.persona]),
            ),
            get_provider_by_id=lambda identity: (
                self.provider if identity == "model-demo" else None
            ),
            get_all_providers=lambda: [self.provider],
            llm_generate=AsyncMock(
                return_value=SimpleNamespace(completion_text="生成的声音卡")
            ),
        )
        self.payload = {"persona_id": "demo", "provider_id": "model-demo"}
        self.view._view_request = SimpleNamespace(
            json=AsyncMock(side_effect=lambda **kw: self.payload)
        )
        self.view._view_json = lambda data: data
        self.view._view_error = lambda message, **kw: {"error": message, **kw}
        self.view.persona_id = self.view.persona_id_webchat = "demo"

    async def asyncTearDown(self):
        self.directory.cleanup()

    async def test_generate_draft_then_manual_save_and_clear(self):
        result = await self.view.api_voice_card_generate()
        self.assertFalse(result["saved"])
        self.assertEqual(self.view._voice_card("demo"), "")
        self.assertFalse(self.view.voice_cards.path.exists())
        call = self.view.context.llm_generate.call_args.kwargs
        self.assertIn(self.persona.system_prompt, call["prompt"])
        self.assertEqual(call["contexts"], [])
        self.payload = {"persona_id": "demo", "text": "手动调整的卡"}
        self.assertTrue((await self.view.api_voice_card_save())["has_card"])
        self.assertEqual(self.view._voice_card("demo"), "手动调整的卡")
        self.payload["text"] = ""
        self.assertFalse((await self.view.api_voice_card_save())["has_card"])
        self.assertEqual(self.view._voice_card("demo"), "")

    async def test_failed_generation_and_unknown_provider_preserve_active_card(self):
        self.view.voice_cards.save("demo", "有效卡")
        self.view.context.llm_generate.side_effect = RuntimeError(
            "synthetic provider failure"
        )
        self.assertIn("error", await self.view.api_voice_card_generate())
        self.assertEqual(self.view._voice_card("demo"), "有效卡")
        self.payload["provider_id"] = "unknown"
        self.assertIn("error", await self.view.api_voice_card_generate())
        self.assertEqual(self.view.context.llm_generate.call_count, 1)

    async def test_status_does_not_expose_persona_prompt_or_credentials(self):
        result = await self.view.api_voice_card_status()
        self.assertNotIn(
            self.persona.system_prompt, json.dumps(result, ensure_ascii=False)
        )
        self.assertEqual(result["personas"][0]["has_card"], False)

    async def test_generation_lock_returns_busy_without_model_call(self):
        async with self.view.voice_cards.generation_lock:
            result = await self.view.api_voice_card_generate()
        self.assertEqual(result["status_code"], 409)
        self.view.context.llm_generate.assert_not_awaited()


try:
    from main import Main
except ModuleNotFoundError as exc:
    if exc.name != "astrbot":
        raise
    Main = None


@unittest.skipIf(Main is None, "requires AstrBot runtime")
class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_card_leaves_default_llm_request_untouched(self):
        plugin = object.__new__(Main)
        plugin.enabled = True
        plugin.base, plugin.key = "https://unit.test/v1", "synthetic"
        plugin.mode = "canary"
        plugin.persona_id = plugin.persona_id_webchat = "demo"
        plugin.voice_cards = SimpleNamespace(get=lambda identity: "")
        event = SimpleNamespace(
            unified_msg_origin="synthetic:FriendMessage:test",
            get_extra=lambda key: None,
            stop_event=unittest.mock.Mock(),
        )
        plugin._accept_input = AsyncMock()
        await plugin.on_llm_request(event)
        event.stop_event.assert_not_called()
        plugin._accept_input.assert_not_awaited()

    async def test_saved_card_is_required_and_other_persona_falls_back(self):
        plugin = object.__new__(Main)
        plugin.enabled = True
        plugin.base, plugin.key = "https://unit.test/v1", "synthetic"
        plugin.platforms = set()
        plugin.persona_id = "demo"
        plugin.persona_id_webchat = "other"
        plugin.voice_cards = SimpleNamespace(
            get=lambda identity: "具体自然" if identity == "demo" else ""
        )
        plugin.umo_allow = {"synthetic:test", "webchat:test"}
        event = SimpleNamespace(
            unified_msg_origin="synthetic:test",
            message_str="测试消息",
            message_obj=None,
            get_platform_name=lambda: "synthetic",
            get_extra=lambda key: None,
        )
        self.assertTrue(plugin._match(event))
        event.unified_msg_origin = "webchat:test"
        self.assertFalse(plugin._match(event))

    async def test_no_voice_card_does_not_start_custom_engine(self):
        plugin = object.__new__(Main)
        plugin.generation_sem = asyncio.Semaphore(1)
        plugin.voice_cards = SimpleNamespace(get=lambda identity: "")
        result = await plugin._run_engine("url", "synthetic", "umo", "message", "demo")
        self.assertEqual(result["status"], "fallback")

    async def test_burst_state_relinquishes_missing_card_but_keeps_active_round(self):
        plugin = object.__new__(Main)
        plugin.enabled = True
        plugin.mode = "canary"
        plugin.umo_allow = {"synthetic:test"}
        plugin.sessions = {}
        plugin.burst_started = {}
        plugin._persona_for = lambda umo: "demo"
        plugin.voice_cards = SimpleNamespace(get=lambda identity: "")
        self.assertIsNone(plugin.burst_state("synthetic:test"))
        plugin.sessions["synthetic:test"] = SimpleNamespace(
            task=SimpleNamespace(done=lambda: False), sent=[]
        )
        self.assertTrue(plugin.burst_state("synthetic:test")[0])
        plugin.sessions.clear()
        plugin.voice_cards = SimpleNamespace(get=lambda identity: "有效卡")
        self.assertEqual(plugin.burst_state("synthetic:test")[:2], (False, 0))
