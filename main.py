"""Palacer main."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import marshal
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx

try:
    from . import message_time as mt, palacer_engine as engine, palacer_session as ps
    from . import session_distill as sd, image_bridge as ib, perception as pm
    from . import palacer_harness as ph_mod
    from .voice_cards import VoiceCardView
except ImportError:
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    import message_time as mt
    import palacer_engine as engine
    import palacer_session as ps
    import session_distill as sd
    import image_bridge as ib
    import perception as pm
    import palacer_harness as ph_mod
    from voice_cards import VoiceCardView

from astrbot.api import logger, star

try:
    from .runtime_config import DATA, PLUGIN_DATA, configure, integration_id, setting
except ImportError:
    from runtime_config import DATA, PLUGIN_DATA, configure, integration_id, setting
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.core.message.components import (
    File,
    Forward,
    Image,
    Node,
    Nodes,
    Plain,
    Record,
    Reply,
    Video,
)
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse

deepseek_config = ph_mod.deepseek_config

MODE_MARK_INSTR = (
    "输出第一行只写 〔档位:X〕（X 取 日常/暧昧/色情/狂野 之一），正文另起一行。"
    "这一行插件会删掉，他看不到。"
)
PALACER_MARK_INSTR = (
    '想换档就在 JSON 里加 "tier":"日常|暧昧|色情|狂野"；不换就省略这一项。'
    "插件会把它读走，他看不到，也不要写进 say。"
)

TIER_MARK_RE = re.compile(
    r"^[\s（(〔\[]*档位\s*[:：]\s*([^\s)）〕\]\n]+)[^\S\n]*[)）〕\]]?[^\S\n]*\n?", re.M
)


class Main(VoiceCardView, star.Star):
    def __init__(self, context, config=None):
        super().__init__(context, config)
        self.config = config or {}
        configure(self.config)
        self.register_voice_card_view()
        self.enabled = bool(self.config.get("enabled", True))
        self.mode = str(self.config.get("mode") or "canary").strip().lower()
        self.platforms = {
            x.strip()
            for x in str(
                self.config.get("platforms") or self.config.get("platform") or ""
            ).split(",")
            if x.strip()
        }
        self.persona_id = str(self.config.get("persona_id") or "")
        self.persona_id_webchat = str(
            self.config.get("persona_id_webchat") or self.persona_id
        )
        self.umo_allow = {
            x.strip()
            for x in str(self.config.get("umo_allowlist") or "").split(",")
            if x.strip()
        }

        self.startup_delay = float(self.config.get("startup_delay") or 0.0)
        self.message_window = float(self.config.get("message_window", 5.0))
        self.trace = bool(self.config.get("trace", False))

        self.memory_shortlist = bool(self.config.get("memory_shortlist", False))
        try:
            max_gen = max(1, int(self.config.get("max_concurrent_generations", 1)))
        except (TypeError, ValueError):
            max_gen = 1
        self.generation_sem = asyncio.Semaphore(max_gen)
        try:
            base, key = deepseek_config()
        except Exception:
            base, key = "", ""

        try:
            ph_mod.set_model(str(self.config.get("model") or "deepseek-flash"))
            logger.info("[palacer] writer model=%s", ph_mod.get_model())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[palacer] 模型名读取失败，回退 deepseek-flash：%s", exc)

        try:
            ph_mod.set_planner_model(
                str(self.config.get("planner_model") or "deepseek-flash")
            )
            ph_mod.set_planner_effort(str(self.config.get("planner_effort") or "high"))
            logger.info(
                "[palacer] planner model=%s effort=%s",
                ph_mod.get_planner_model(),
                ph_mod.PLANNER_EFFORT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[palacer] planner 模型读取失败，回退 flash/high：%s", exc)
        self.base = str(self.config.get("api_base") or base)
        self.key = str(self.config.get("api_key") or key)
        self.client = httpx.Client(timeout=300)
        self.async_client = httpx.AsyncClient(timeout=180)
        for harness in (engine.ph, engine.brain.ph, ph_mod):
            harness.set_model(str(self.config.get("model") or "deepseek-flash"))
            harness.set_planner_model(
                str(self.config.get("planner_model") or "deepseek-flash")
            )
            harness.set_planner_effort(str(self.config.get("planner_effort") or "high"))
        self.engine_workers: set[asyncio.Task] = set()
        PLUGIN_DATA.mkdir(parents=True, exist_ok=True)
        self.times = mt.MessageTimes(PLUGIN_DATA / "message_times.sqlite3")
        self.session_distill = sd.SessionDistill(self.context, self.times, self._log)
        self.history_locks: dict[str, asyncio.Lock] = {}
        self.sessions: dict[str, ps.PalacerSession] = {}
        self.burst_started: dict[str, float] = {}
        self.events: dict[str, AstrMessageEvent] = {}
        self.tier_marks: dict[str, str] = {}
        self.voice_decisions: dict[str, dict] = {}
        self.shadow_tasks: set[asyncio.Task] = set()
        self.turn_done: dict[str, asyncio.Event] = {}
        self.proactive_sessions: dict[str, ps.PalacerSession] = {}
        self.images = ib.ImageBridge(self)
        out = PLUGIN_DATA / "traces"
        out.mkdir(parents=True, exist_ok=True)
        self.log_path = out / f"webui_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.perception = pm.OmniObserver(self.context, self.config, self._log)
        logger.info(
            "[palacer] model_call_empty_retries=%d", engine.ph.EMPTY_OUTPUT_RETRIES
        )
        logger.info(
            "[palacer] planner budgets: combined=%d brain=%d thinking=enabled; queue_state=palacer-v1",
            engine.BRAIN_MAX_TOKENS,
            engine.BRAIN_MAX_TOKENS,
        )

    def _log(self, **row):
        if not self.trace:
            return
        try:
            row.setdefault("ts", time.time())
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _chain(self, event: AstrMessageEvent):
        msg_obj = getattr(event, "message_obj", None)
        return list(getattr(msg_obj, "message", None) or []) if msg_obj else []

    def _match(self, event: AstrMessageEvent) -> bool:
        if not self.enabled:
            return False
        persona_id = self._persona_for(event.unified_msg_origin)
        if (
            not self.base
            or not self.key
            or not persona_id
            or not self._voice_card(persona_id)
        ):
            return False
        platform = str(event.get_platform_name() or "")
        if self.platforms and platform not in self.platforms:
            return False
        umo = event.unified_msg_origin
        if self.umo_allow and umo not in self.umo_allow:
            return False
        if not self.umo_allow:
            return False
        text = (event.message_str or "").strip()
        handlers = (
            getattr(event, "get_extra", lambda *args: None)("activated_handlers") or []
        )
        if any(
            getattr(h, "handler_name", "") == "watch_command"
            and getattr(h, "handler_module_path", "").endswith(
                "astrbot_plugin_watch_together.main"
            )
            for h in handlers
        ):
            return False
        if any(
            getattr(h, "handler_name", "") == "draw_command"
            and getattr(h, "handler_module_path", "").endswith(
                "astrbot_plugin_image_studio.main"
            )
            for h in handlers
        ):
            return False
        if any(
            getattr(h, "handler_name", "") == "voice"
            and getattr(h, "handler_module_path", "").endswith(
                integration_id("voice") + ".main"
            )
            and bool(integration_id("voice"))
            for h in handlers
        ):
            return False
        if text.startswith("/"):
            return False
        chain = self._chain(event)
        has_media = any(
            isinstance(c, (Image, File, Record, Video, Forward, Node, Nodes))
            for c in chain
        )
        has_quote = any(
            isinstance(c, Reply)
            and (
                getattr(c, "message_str", None)
                or getattr(c, "text", None)
                or getattr(c, "chain", None)
            )
            for c in chain
        )
        if not text and not has_media and not has_quote:
            return False
        for comp in chain:
            if not isinstance(
                comp, (Plain, Image, File, Record, Video, Reply, Forward, Node, Nodes)
            ):
                return False
        return True

    def _image_mime(self, raw_b64: str) -> str:
        try:
            head = base64.b64decode(raw_b64[:80])
        except Exception:
            return "image/jpeg"
        if head.startswith(b"\x89PNG"):
            return "image/png"
        if head.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if head.startswith(b"GIF8"):
            return "image/gif"
        if head.startswith(b"RIFF") and b"WEBP" in head:
            return "image/webp"
        return "image/jpeg"

    @staticmethod
    def _normalize_image_b64(raw_b64: str) -> str:
        """Normalize QQ stickers (WebP/GIF/transparent) to a still PNG."""
        try:
            import io

            from PIL import Image as PILImage

            raw = base64.b64decode(raw_b64)
            with PILImage.open(io.BytesIO(raw)) as im:
                im.seek(0)
                if im.mode in ("RGBA", "LA"):
                    bg = PILImage.new("RGB", im.size, "white")
                    bg.paste(im, mask=im.split()[-1])
                    im = bg
                else:
                    im = im.convert("RGB")
                im.thumbnail((1280, 1280))
                out = io.BytesIO()
                im.save(out, format="PNG")
                return base64.b64encode(out.getvalue()).decode()
        except Exception:
            return raw_b64

    def _describe_image_sync(self, raw_b64: str, *, video: bool = False) -> dict:
        prompt = (
            "你是图像观察器，不是所选角色。只记录这张图里确实可见的内容。"
            "不要推断人物身份、地点、时间、关系、情绪、意图；不要读不存在的文字；"
            "看不清或看不全就写进 uncertain。\n"
            '只输出 JSON：{"summary":"一句话","visible_text":["..."],'
            '"objects":["..."],"uncertain":["..."]}'
        )
        if video:
            prompt += "\n这是一张带格号的视频分镜。summary 按格号顺序描述可见画面变化，可见文字仍放 visible_text。"
        raw_b64 = self._normalize_image_b64(raw_b64)
        body = {
            "model": "deepseek-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{self._image_mime(raw_b64)};base64,{raw_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1200 if video else 800,
            "thinking": {"type": "disabled" if video else "enabled"},
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }
        r = self.client.post(
            self.base + "/chat/completions",
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        text = str(
            (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        ).strip()
        self._log(
            ev="vision_raw",
            status=r.status_code,
            finish=((d.get("choices") or [{}])[0].get("finish_reason") or ""),
            chars=len(text),
            raw=text[:2000],
        )
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`").strip()
            if t.startswith("json"):
                t = t[4:].strip()
        try:
            return json.loads(t)
        except Exception:
            return {
                "summary": text[:600],
                "visible_text": [],
                "objects": [],
                "uncertain": ["观察结果不是结构化 JSON"],
            }

    @staticmethod
    def _render_vision(obj: dict) -> str:
        def join(x):
            if isinstance(x, list):
                return "；".join(str(i).strip() for i in x if str(i).strip()) or "无"
            return str(x or "").strip() or "无"

        return (
            f"摘要：{join(obj.get('summary'))}；"
            f"可见文字：{join(obj.get('visible_text'))}；"
            f"物体：{join(obj.get('objects'))}；"
            f"不确定：{join(obj.get('uncertain'))}"
        )

    @staticmethod
    def _extract_file_text(path: str, name: str) -> str:
        try:
            suffix = Path(path).suffix.lower()
            if suffix in (
                ".txt",
                ".md",
                ".json",
                ".csv",
                ".log",
                ".py",
                ".yaml",
                ".yml",
                ".ini",
                ".cfg",
            ):
                return Path(path).read_text(encoding="utf-8", errors="ignore")[:8000]
            if suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(path)
                chunks = []
                for page in reader.pages[:20]:
                    chunks.append(page.extract_text() or "")
                return "\n".join(chunks)[:8000]
            if suffix == ".docx":
                from docx import Document

                doc = Document(path)
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())[
                    :8000
                ]
            if suffix == ".xlsx":
                from openpyxl import load_workbook

                wb = load_workbook(path, read_only=True, data_only=True)
                rows = []
                for ws in wb.worksheets[:3]:
                    rows.append(f"[sheet: {ws.title}]")
                    for row in ws.iter_rows(max_rows=50, values_only=True):
                        vals = [str(v) for v in row if v is not None]
                        if vals:
                            rows.append(" | ".join(vals))
                return "\n".join(rows)[:8000]
            return f"（{suffix or '未知类型'} 文件已收到，但当前不能解析内容）"
        except Exception as exc:
            return f"（文件读取失败：{type(exc).__name__}）"

    @staticmethod
    def _node_lines(node: Node, depth: int = 0) -> list[str]:
        who = str(getattr(node, "name", "") or getattr(node, "uin", "") or "").strip()
        prefix = f"{who}: " if who else ""
        lines = []
        for comp in getattr(node, "content", None) or []:
            if isinstance(comp, Plain):
                t = str(getattr(comp, "text", "") or "").strip()
                if t:
                    lines.append(prefix + t[:1000])
            elif isinstance(comp, Node):
                lines.extend(
                    "  " * depth + x for x in Main._node_lines(comp, depth + 1)
                )
            elif isinstance(comp, Image):
                lines.append(prefix + "[图片]")
            elif isinstance(comp, Record):
                lines.append(prefix + "[语音]")
            elif isinstance(comp, File):
                lines.append(prefix + f"[文件 {getattr(comp, 'name', '') or ''}]")
        return lines

    @classmethod
    def _nodes_lines(cls, nodes) -> list[str]:
        lines = []
        for n in nodes or []:
            lines.extend(cls._node_lines(n))
            if len(lines) >= 80:
                break
        return lines[:80]

    @staticmethod
    def _onebot_lines(obj, out: list[str]) -> None:
        if isinstance(obj, list):
            for x in obj:
                Main._onebot_lines(x, out)
            return
        if not isinstance(obj, dict):
            return
        typ = str(obj.get("type") or "")
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        if typ in ("text", "plain"):
            t = str(data.get("text") or data.get("content") or "").strip()
            if t:
                out.append(t[:1000])
        elif typ in ("image", "record", "file"):
            out.append(
                {
                    "image": "[图片]",
                    "record": "[语音]",
                    "file": f"[文件 {data.get('name') or ''}]",
                }[typ]
            )
        elif typ in ("node", "forward"):
            content = obj.get("content") or data.get("content") or data.get("messages")
            if content:
                Main._onebot_lines(content, out)
        elif typ == "json":
            out.append("[JSON 卡片]")
        elif isinstance(obj.get("messages"), list):
            Main._onebot_lines(obj["messages"], out)
        elif isinstance(obj.get("message"), (list, dict)):
            Main._onebot_lines(obj["message"], out)
        elif isinstance(obj.get("content"), (list, dict)):
            Main._onebot_lines(obj["content"], out)
        elif isinstance(obj.get("data"), (list, dict)):
            Main._onebot_lines(obj["data"], out)

    async def _expand_forward(self, event: AstrMessageEvent, fid: str) -> str:
        bot = getattr(event, "bot", None)
        if bot is None:
            return "合并转发无法展开（当前平台没有可用的 QQ 拉取接口），不要猜内容。"
        last = None
        for action, kwargs in (
            ("get_forward_msg", {"message_id": fid}),
            ("get_forward_msg", {"id": fid}),
            ("get_forward_msg", {"message_id": str(fid)}),
        ):
            try:
                raw = await bot.call_action(action, **kwargs)
                lines: list[str] = []
                self._onebot_lines(raw, lines)
                if lines:
                    return "合并转发内容：\n" + "\n".join(lines[:80])
                last = "返回为空"
                self._log(ev="forward_raw", id=fid, action=action, raw=repr(raw)[:2000])
            except Exception as exc:  # noqa: BLE001
                last = repr(exc)
        return f"合并转发无法展开（{last}），不要猜内容。"

    async def _describe_video(self, umo: str, result: dict, query: str = "") -> str:
        """Observe the existing video service's exact current window.

        Args:
            umo: Conversation origin.
            result: Video service analysis with a resolved local path and window.
            query: The user's current media question.

        Returns:
            Omni evidence for this video window.
        """
        window = result.get("window") or {}
        return await self.perception.observe(
            result.get("info", {}).get("path", ""),
            "video",
            umo,
            start=float(window.get("from", 0)),
            end=float(window.get("to", 0)),
            query=query,
        )

    async def _media_text(self, event: AstrMessageEvent) -> str:
        chain = self._chain(event)
        question = "\n".join(
            str(c.text or "") for c in chain if isinstance(c, Plain)
        ).strip()
        work: list = []
        quote_lines: list[str] = []
        forwards: list = []
        for comp in chain:
            if isinstance(comp, Forward):
                forwards.append(comp)
            elif isinstance(comp, Nodes):
                work.extend(getattr(comp, "nodes", None) or [])
            elif isinstance(comp, Node):
                work.append(comp)
            elif isinstance(comp, Reply):
                qtext = str(
                    getattr(comp, "message_str", "") or getattr(comp, "text", "") or ""
                ).strip()
                if not qtext:
                    qtext = "".join(
                        str(getattr(x, "text", "") or "")
                        for x in (getattr(comp, "chain", None) or [])
                        if isinstance(x, Plain)
                    ).strip()
                if qtext:
                    quote_lines.append(qtext[:2000])
                for sub in getattr(comp, "chain", None) or []:
                    if not isinstance(sub, Reply):
                        work.append(sub)
            else:
                work.append(comp)

        expanded_work = []
        for c in work:
            if isinstance(c, Nodes):
                expanded_work.extend(getattr(c, "nodes", None) or [])
            else:
                expanded_work.append(c)
        work = expanded_work
        images = [c for c in work if isinstance(c, Image)][:3]
        files = [c for c in work if isinstance(c, File)][:3]
        records = [c for c in work if isinstance(c, Record)][:3]
        node_roots = [c for c in work if isinstance(c, Node)]
        video_text = ""
        watch = self._plugin("astrbot_plugin_watch_together")
        watch_context = (
            await watch.observe_event(event)
            if getattr(watch, "watch_api_version", None) == 1
            else None
        )
        if watch_context is not None:
            video_text = watch_context
        video_plugin = self._plugin("astrbot_plugin_video_storyboard")
        if (
            watch_context is None
            and video_plugin is not None
            and hasattr(video_plugin, "observe_event")
        ):
            video_text = await video_plugin.observe_event(
                event,
                lambda image: self._describe_image_sync(image, video=True),
                describe_video=lambda result: self._describe_video(
                    event.unified_msg_origin, result, question
                ),
                max_seconds=self.perception.video_limit,
            )
            if video_text:
                self._log(
                    ev="video_observed",
                    umo=event.unified_msg_origin,
                    chars=len(video_text),
                )
        if (
            not images
            and not files
            and not records
            and not quote_lines
            and not forwards
            and not node_roots
            and not video_text
        ):
            return ""

        lines = [
            "【附件观察｜只依据这里可见/可听/可读取的内容；没有写到的不能猜，不确定就说明】"
        ]
        lines.append(
            "以下是本次通过工具实际取得的观察，可自然描述自己看到或听到的内容；用户猜测不等于观察，未提供的环境不能声称正在感知。"
        )
        if video_text:
            lines.append(video_text)
        for q in quote_lines:
            lines.append(f"[引用消息] {q}")
        if node_roots:
            nl = self._nodes_lines(node_roots)
            if nl:
                lines.append("[合并转发]\n" + "\n".join(nl))
                self._log(ev="forward_nodes", umo=event.unified_msg_origin, n=len(nl))
        for i, fw in enumerate(forwards, 1):
            fid = str(getattr(fw, "id", "") or "")
            if fid:
                expanded = await self._expand_forward(event, fid)
                lines.append(f"[合并转发{i}] " + expanded)
                self._log(
                    ev="forward_done",
                    umo=event.unified_msg_origin,
                    id=fid,
                    text=expanded[:500],
                )
            else:
                lines.append(f"[合并转发{i}] 没有可展开的 id，不要猜内容。")
                self._log(
                    ev="forward_error", umo=event.unified_msg_origin, error="missing id"
                )
        for i, img in enumerate(images, 1):
            try:
                raw = await img.convert_to_base64()
                obj = await asyncio.to_thread(self._describe_image_sync, raw)
                lines.append(f"[图片{i}] {self._render_vision(obj)}")
                self._log(
                    ev="vision_done",
                    umo=event.unified_msg_origin,
                    index=i,
                    summary=obj.get("summary"),
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"[图片{i}] 图片无法识别，不能描述内容。")
                self._log(
                    ev="vision_error",
                    umo=event.unified_msg_origin,
                    index=i,
                    error=repr(exc),
                )
        for i, f in enumerate(files, 1):
            name = str(getattr(f, "name", "") or f"file{i}")
            try:
                suffix = Path(name).suffix.lower()
                if suffix in pm.VIDEO_SUFFIXES:
                    continue
                path = await f.get_file()
                if suffix in pm.AUDIO_SUFFIXES:
                    observed = await self.perception.observe(
                        path, "audio", event.unified_msg_origin, query=question
                    )
                    lines.append(f"[音乐文件{i}] {observed}")
                    continue
                text = await asyncio.to_thread(self._extract_file_text, path, name)
                lines.append(f"[文件 {name}] 内容摘录：{text}")
                self._log(
                    ev="file_read",
                    umo=event.unified_msg_origin,
                    name=name,
                    chars=len(text),
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(
                    f"[文件 {name}] 文件已收到，但本轮理解失败（{type(exc).__name__}）；不能声称已经听过或看过，请说明失败并请他重发较短片段。"
                )
                self._log(
                    ev="file_error",
                    umo=event.unified_msg_origin,
                    name=name,
                    error=type(exc).__name__,
                )
        for i, rec in enumerate(records, 1):
            try:
                text = str(getattr(rec, "text", "") or "").strip()
                if not text:
                    stt = await self.context.get_using_stt_provider_async(
                        event.unified_msg_origin
                    )
                    if stt is None:
                        raise RuntimeError("no stt provider")
                    path = await rec.convert_to_file_path()
                    text = str(await stt.get_text(audio_url=path) or "").strip()
                if text:
                    lines.append(f"[语音{i}] 转写：{text}")
                    self._log(
                        ev="voice_done",
                        umo=event.unified_msg_origin,
                        index=i,
                        chars=len(text),
                    )
                else:
                    lines.append(f"[语音{i}] 没有转写出来，不能猜内容。")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"[语音{i}] 语音无法转写，不能猜内容。")
                self._log(
                    ev="voice_error",
                    umo=event.unified_msg_origin,
                    index=i,
                    error=repr(exc),
                )
        return "\n".join(lines)

    async def _text_with_media(self, event: AstrMessageEvent) -> str:
        text = (event.message_str or "").strip()

        try:
            files = [
                c
                for c in (getattr(event.message_obj, "message", []) or [])
                if isinstance(c, File)
            ]
        except Exception:
            files = []
        if files:
            names = "、".join(str(getattr(c, "name", "") or "文件") for c in files[:3])
            text = f"[他发来文件：{names}]"
        media = await self._media_text(event)
        if not media:
            return text
        if not text:
            text = "（用户发来附件）"
        return text + "\n\n" + media

    def _persona_for(self, umo: str) -> str:
        if umo.startswith("webchat:"):
            return self.persona_id_webchat
        return self.persona_id

    def _media_read_tool(self, args):
        """把 integration_media.read_now（async）从 worker 线程桥回主循环执行。"""
        media = self._plugin("integration_media")
        if media is None or not hasattr(media, "read_now"):
            return "读书工具不可用：integration_media 没加载。"
        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            return "读书工具不可用：没有可用的事件循环。"
        args = args if isinstance(args, dict) else {}
        try:
            chunks = int(args.get("chunks") or 1)
        except (TypeError, ValueError):
            chunks = 1
        fut = asyncio.run_coroutine_threadsafe(
            media.read_now(book_id=str(args.get("book_id") or ""), chunks=chunks),
            loop,
        )
        try:
            return fut.result(timeout=150)
        except Exception as exc:  # noqa: BLE001
            return f"读书失败：{type(exc).__name__}: {exc}"

    def _plugin(self, name: str):
        try:
            if name.startswith("integration_"):
                name = integration_id(name.removeprefix("integration_"))
                if not name:
                    return None
            md = self.context.get_registered_star(name)
            return getattr(md, "star_cls", None) if md else None
        except Exception:
            return None

    def _tier_ctx(self, umo: str) -> dict:
        """档位语料：常驻写法 + 该档语料 + 本轮口径，全部从 integration_mode 现取。

        integration_mode 原本把这些拼进默认 LLM 的 system_prompt，而 palacer 会 stop 掉那条请求，
        所以从 10-29 起这批语料一次都没进过模型。这里直接问它要：
        resident（常驻）与该档语料放写手 system 的固定段，换档只让这一段之后失效；
        口径块用 mode 的 _tail_block / _daily_tail，只把标记那一句换成 JSON 字段版。
        """
        tier, resident, tier_tail, tier_state = "日常", "", "", ""
        mode = self._plugin("integration_mode")
        if mode is not None:
            try:
                st = mode._st(umo)
                tier = str(st.get("tier") or "日常")
                resident = mode._corpus_text(getattr(mode, "BASE_FILE", "00_常驻.md"))
                block = (
                    mode._daily_tail(st)
                    if tier == "日常"
                    else mode._tail_block(tier, st)
                )

                keep, volatile = [], []
                for line in block.splitlines():
                    if line.startswith("· 上一轮你做的动作") or line.startswith(
                        "· 刚从亲密状态出来"
                    ):
                        volatile.append(line)
                    else:
                        keep.append(line)
                tier_tail = "\n".join(keep).replace(MODE_MARK_INSTR, PALACER_MARK_INSTR)
                tier_state = "\n".join(volatile)
                if tier != "日常":
                    if tier not in tuple(
                        getattr(mode, "NO_ACTION_TIERS", ("日常", "暧昧"))
                    ):
                        tier_tail += "\n· 这一档可以写括号动作：单独成条，括号里只写动作，不写解释。"
                    corpus = mode._tier_corpus(tier)
                    if corpus:
                        tier_tail += f"\n\n【本轮档位语料｜{tier}｜只当写法示范，别复述】\n{corpus}"
                    if tier in ("色情", "狂野"):
                        shared = mode._corpus_text(
                            getattr(mode, "SHARED_FILE", "04_共用.md")
                        )
                        if shared:
                            tier_tail += f"\n\n【共用写法｜{tier}档】\n{shared}"
            except Exception as exc:  # noqa: BLE001
                self._log(ev="tier_ctx_error", umo=umo, error=repr(exc))
        return {
            "tier": tier,
            "resident": resident,
            "tier_tail": tier_tail,
            "tier_state": tier_state,
        }

    def burst_state(self, umo: str):
        """Authoritative (busy, sent, age) for owned routes; None uses legacy pacer."""
        if not self.enabled or self.mode != "canary" or umo not in self.umo_allow:
            return None
        session = self.sessions.get(umo)
        busy = bool(session and session.task and not session.task.done())
        if not busy and not self._voice_card(self._persona_for(umo)):
            return None
        sent = len(session.sent) if busy else 0
        now = time.monotonic()
        age = max(0.0, now - self.burst_started.get(umo, now))
        return busy, sent, age

    def _release_burst(self, umo: str, reason: str) -> None:
        """Only release messages held for a round, preserving normal debounce."""
        burst = self._plugin("astrbot_plugin_burst")
        if burst is not None:
            try:
                burst.release_pending(umo, reason, held_only=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[palacer] burst release failed: %s", type(exc).__name__)

    async def _cancel_proactive(self, umo: str) -> None:
        old = self.proactive_sessions.pop(umo, None)
        if old is not None:
            await old.close()
            self._log(ev="proactive_cancelled", umo=umo)

    def _session(self, umo: str) -> ps.PalacerSession:
        s = self.sessions.get(umo)
        if s is None:
            s = ps.PalacerSession(
                generate=lambda batch, sent: self._generate(umo, batch, sent),
                send=lambda msg, delivery: self._send(umo, msg, delivery),
                prepare=lambda messages: self._prepare_voice(umo, messages),
                config=ps.SessionConfig(self.startup_delay, self.message_window),
                on_complete=lambda batch, sent: self._complete(umo, batch, sent),
                clock=self.times.clock,
            )
            self.sessions[umo] = s
        return s

    async def _recent_history(self, umo: str) -> list[dict]:
        await self.session_distill.prepare(umo)
        try:
            cm = self.context.conversation_manager
            cid = await cm.get_curr_conversation_id(umo)
            if not cid:
                return []
            conv = await cm.get_conversation(umo, cid)
            if conv is None:
                return []
            h = conv.history
            if isinstance(h, str):
                h = json.loads(h or "[]")
            if not isinstance(h, list):
                return []
            out = []
            for index, m in enumerate(h[-8:], max(0, len(h) - 8)):
                role = str(m.get("role") or "")
                if role not in ("user", "assistant"):
                    continue
                content = m.get("content")
                if isinstance(content, list):
                    text = "".join(
                        str(x.get("text") or "")
                        for x in content
                        if isinstance(x, dict) and x.get("type") == "text"
                    )
                else:
                    text = str(content or "")
                if text.strip():
                    out.append(
                        {
                            "role": role,
                            "content": text.strip(),
                            "_time_ids": self.times.history_ids(umo, cid, index, m),
                        }
                    )
            return out
        except Exception as exc:  # noqa: BLE001
            self._log(ev="history_read_error", umo=umo, error=repr(exc))
            return []

    async def _memory_shortlist(self, text):
        """Reuse the running KB manager's embedding provider and FAISS index."""
        try:
            cfg = engine.cb._read_json(
                DATA / "config" / "astrbot_plugin_memory_recall_config.json", {}
            )
            helper = await self.context.kb_manager.get_kb_by_name(
                str(setting("memory_kb_name", "") or cfg.get("kb_name") or "")
            )
            if helper is None or helper.init_error:
                return None
            hits = await asyncio.wait_for(
                helper.vec_db.retrieve(
                    query=text,
                    k=80,
                    fetch_k=120,
                    rerank=False,
                    metadata_filters={"kb_id": helper.kb.kb_id},
                ),
                timeout=10,
            )
            ids = []
            for hit in hits:
                md = hit.data.get("metadata") or {}
                if isinstance(md, str):
                    md = json.loads(md)
                if md.get("kind", "event") in ("preference", "event") and md.get(
                    "suspect"
                ) not in (1, "1", True):
                    ids.append(int(hit.data["id"]))
            return ids[:40] if ids else None
        except Exception as exc:
            self._log(ev="memory_shortlist_fallback", error=type(exc).__name__)
            return None

    async def _persist_request_stats(self, umo, records):
        """Use the framework API so the existing provider dashboard sees every attempt."""
        if not records:
            return
        for rec in records:
            try:
                providers = self.context.get_config().get("provider", [])
                model = str(rec.get("model") or "")
                pid = next(
                    (p["id"] for p in providers if p.get("model") == model),
                    "palacer/" + model,
                )
                prompt, hit = int(rec.get("prompt") or 0), int(rec.get("hit") or 0)
                await self.context.get_db().insert_provider_stat(
                    umo=umo,
                    provider_id=pid,
                    provider_model=model,
                    status=rec.get("status") or "error",
                    agent_type="internal",
                    stats={
                        "token_usage": {
                            "input_other": max(0, prompt - hit),
                            "input_cached": hit,
                            "output": int(rec.get("out") or 0),
                        },
                        "start_time": float(rec["ts"]) - float(rec.get("dt") or 0),
                        "end_time": float(rec["ts"]),
                    },
                )
            except Exception as exc:
                self._log(ev="stats_persist_error", error=type(exc).__name__)

    async def _run_engine(self, *args, **kwargs):
        """A slot owns the worker and HTTP cleanup, including during cancellation."""
        async with self.generation_sem:
            persona_id = args[4] if len(args) > 4 else kwargs.get("persona_id", "")
            card = self._voice_card(persona_id)
            if not card:
                return {"messages": [], "status": "fallback"}
            kwargs["voice_card"] = card
            if len(args) >= 4 and "memory_candidate_ids" not in kwargs:
                kwargs["memory_candidate_ids"] = (
                    await self._memory_shortlist(args[3])
                    if self.memory_shortlist
                    else None
                )
            if len(args) >= 4 and "resident" not in kwargs:
                tier_ctx = self._tier_ctx(str(args[2]))
                kwargs.setdefault("resident", tier_ctx["resident"])
                kwargs.setdefault("tier_label", tier_ctx["tier"])
                kwargs.setdefault("tier_tail", tier_ctx["tier_tail"])
                kwargs.setdefault("tier_state", tier_ctx["tier_state"])
            voice = (
                self.context.get_registered_star(integration_id("voice"))
                if integration_id("voice")
                else None
            )
            kwargs.setdefault(
                "voice_available",
                bool(
                    voice
                    and voice.activated
                    and callable(getattr(voice.star_cls, "prepare_reply", None))
                ),
            )
            bridge = engine.AsyncRequestBridge(
                self.async_client, asyncio.get_running_loop()
            )
            worker = asyncio.create_task(
                asyncio.to_thread(engine.generate_turn, bridge, *args, **kwargs)
            )
            self.engine_workers.add(worker)
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                bridge.cancel()

                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        bridge.cancel()
                    except BaseException:
                        break
                if worker.done():
                    try:
                        worker.result()
                    except BaseException:
                        pass
                raise
            finally:

                async def finish():
                    await bridge.close()
                    await self._persist_request_stats(
                        args[2] if len(args) >= 3 else "palacer:unknown", bridge.records
                    )

                cleanup = asyncio.create_task(finish())
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        bridge.cancel()
                self.engine_workers.discard(worker)

    async def _generate(self, umo: str, batch: list[str], sent: list[str]):
        text = "\n".join(batch).strip()
        hist = await self._recent_history(umo)
        self._log(ev="generate_start", umo=umo, batch=batch, sent=sent)
        try:
            tool_guide, tool_handlers = await self.images.prepare(umo, batch)
            image_context = await self.images.context(umo)
        except Exception as exc:
            self._log(ev="image_prepare_failed", error=type(exc).__name__)
            tool_guide, tool_handlers = "", {}
            image_context = []
        tool_handlers = dict(tool_handlers or {})
        tool_handlers.setdefault("book_read", self._media_read_tool)
        cancelled = threading.Event()

        def temporal():
            if cancelled.is_set():
                raise RuntimeError("Palacer generation cancelled")
            return self.times.context(umo, batch, hist)

        res = await self._run_engine(
            self.base,
            self.key,
            umo,
            text,
            self._persona_for(umo),
            hist,
            sent,
            temporal,
            handoff=self.session_distill.current(umo),
            tool_guide=tool_guide,
            tool_handlers=tool_handlers,
            image_context=image_context,
        )
        self.images.stage(umo, batch, res)
        messages = list(res.get("messages") or [])

        mark = str(res.get("tier_mark") or "").strip()
        cleaned = []
        for msg in messages:
            m = TIER_MARK_RE.search(msg)
            if m and not mark:
                mark = m.group(1).strip()
            cleaned.append(TIER_MARK_RE.sub("", msg).strip())
        messages = [x for x in cleaned if x]
        self.tier_marks[umo] = mark if mark in ("日常", "暧昧", "色情", "狂野") else ""
        self.voice_decisions[umo] = engine.parse_voice_decision(res.get("voice")) or {
            "use": False,
            "reason": "语音决策无效",
        }
        self._log(
            ev="voice_decision",
            umo=umo,
            **self.voice_decisions[umo],
            status=res.get("voice_status", "invalid"),
            error=res.get("voice_error", ""),
        )
        event = self.events.get(umo)
        notes = self._plugin("integration_notes")
        if event is not None and notes is not None and hasattr(notes, "verify"):
            fixed = []
            for msg in messages:
                try:
                    resp = LLMResponse(role="assistant", completion_text=msg)
                    await notes.verify(event, resp)
                    fixed.append(getattr(resp, "completion_text", "") or msg)
                except Exception as exc:  # noqa: BLE001
                    self._log(ev="notes_verify_error", umo=umo, error=repr(exc))
                    fixed.append(msg)
            messages = fixed
        self._log(
            ev="generate_done",
            umo=umo,
            messages=messages,
            calls=res.get("calls"),
            actual_requests=res.get("actual_requests"),
            turn_id=res.get("turn_id"),
            usage=res.get("usage"),
            style_flags=[n.get("style_flags", []) for n in res.get("nodes", [])],
            style_attempts=[n.get("style_attempts", []) for n in res.get("nodes", [])],
        )
        return messages

    async def generate_proactive(self, umo: str) -> list[str] | None:
        """None declines/fails; [] is an explicit decision to remain silent."""
        if not self.enabled or self.mode != "canary" or umo not in self.umo_allow:
            return None
        prompt = (
            "（现在没有新的用户消息。这一轮是你主动开口。"
            "按声音卡写 1-2 条很短的消息；不要提系统、定时、提醒、插件；"
            "如果不适合打扰，只输出 [[NO_MESSAGE]]。）"
        )
        hist = await self._recent_history(umo)
        try:
            res = await self._run_engine(
                self.base,
                self.key,
                umo,
                prompt,
                self._persona_for(umo),
                hist,
                [],
                lambda: self.times.context(umo, history=hist, proactive=True),
                turn_kind="proactive",
                handoff=self.session_distill.current(umo),
            )
            msgs = [m for m in (res.get("messages") or []) if "[[NO_MESSAGE]]" not in m]
            self._log(
                ev="proactive_generate",
                umo=umo,
                messages=msgs,
                usage=res.get("usage"),
                style_flags=[n.get("style_flags", []) for n in res.get("nodes", [])],
                style_attempts=[
                    n.get("style_attempts", []) for n in res.get("nodes", [])
                ],
            )
            return msgs
        except Exception as exc:  # noqa: BLE001
            self._log(ev="proactive_error", umo=umo, error=repr(exc))
            return None

    def _watch_busy(self, umo):
        session = self.sessions.get(umo)
        proactive = self.proactive_sessions.get(umo)
        return any(s and s.task and not s.task.done() for s in (session, proactive))

    async def generate_watch_comment(self, umo: str, evidence: str, at: float) -> str:
        """Draft for a playback landmark; never create a fictional user receipt."""
        if umo not in self.umo_allow or self._watch_busy(umo):
            return "[[NO_MESSAGE]]"
        hist = await self._recent_history(umo)
        cancelled = threading.Event()

        def temporal():
            if cancelled.is_set():
                raise RuntimeError("Watch draft cancelled")
            return (
                self.times.context(umo, history=hist, proactive=True)
                + f"\n陪看预写：脚本将在对方看到视频 {at:g} 秒时才发送；这不是新的用户输入。"
                + "只写一句很短的自然感受，或 [[NO_MESSAGE]]；只谈已给出的画面，不猜后续剧情。"
            )

        if self._watch_busy(umo):
            return "[[NO_MESSAGE]]"
        result = await self._run_engine(
            self.base,
            self.key,
            umo,
            "（陪看脚本请求预写一句，不是用户刚说的话。）\n" + evidence,
            self._persona_for(umo),
            hist,
            [],
            temporal,
            turn_kind="watch",
            handoff=self.session_distill.current(umo),
        )
        messages = result.get("messages") or []
        self._log(
            ev="watch_generate",
            umo=umo,
            style_flags=[n.get("style_flags", []) for n in result.get("nodes", [])],
            style_attempts=[
                n.get("style_attempts", []) for n in result.get("nodes", [])
            ],
        )
        return messages[0] if messages else "[[NO_MESSAGE]]"

    async def send_watch_comment(
        self, umo: str, text: str, *, guard, audio_path=None
    ) -> bool:
        """No debounce on a timed line; still use the private receipts and history."""
        if umo not in self.umo_allow or self._watch_busy(umo) or not guard():
            return False
        text = ps.TAIL_PERIOD.sub("", text)
        if not text:
            return False
        chain = MessageChain(
            [Record.fromFileSystem(audio_path)] if audio_path else [Plain(text)]
        )
        try:
            ok = await self.context.send_message(umo, chain)
        except Exception as exc:
            self._log(ev="watch_send_error", umo=umo, error=type(exc).__name__)
            return False
        if ok is False:
            return False
        voice = self._plugin("integration_voice")
        if voice is not None and callable(getattr(voice, "remember_sent", None)):
            voice.remember_sent(umo, text)
        receipt = self.times.record(umo, "assistant", text, proactive=True)
        self.session_distill.sent(umo)
        await self._write_history(umo, [receipt])
        self._log(ev="watch_sent", umo=umo, message_id=receipt.message_id)
        return True

    @staticmethod
    def _split_proactive(text: str) -> list[str]:
        parts = re.findall(r"[^。？！~…\n]+[。？！~…]*", text or "")
        return [p.strip() for p in parts if p.strip()] or [str(text or "").strip()]

    async def send_proactive(self, umo: str, text: str) -> bool:
        """Schedule a proactive message through the same 15s/5s PalacerSession timers."""
        if umo not in self.umo_allow:
            return False
        parts = [p for p in self._split_proactive(text) if p]
        if not parts:
            return False
        old = self.proactive_sessions.get(umo)
        if old is not None:
            await old.close()

        async def gen(batch, sent):
            return list(parts)

        async def send_one(msg):
            try:
                ok = await self.context.send_message(umo, MessageChain([Plain(msg)]))
                if ok is False:
                    return False
                voice = self._plugin("integration_voice")
                if voice is not None and callable(
                    getattr(voice, "remember_sent", None)
                ):
                    voice.remember_sent(umo, msg)
                receipt = self.times.record(umo, "assistant", msg, proactive=True)
                self.session_distill.sent(umo)
                await self._write_history(umo, [receipt])
                self._log(
                    ev="proactive_sent",
                    umo=umo,
                    message=msg,
                    message_id=receipt.message_id,
                )
                return receipt
            except Exception as exc:  # noqa: BLE001
                self._log(
                    ev="proactive_send_error", umo=umo, message=msg, error=repr(exc)
                )
                return False

        async def done(batch, sent):
            self.proactive_sessions.pop(umo, None)

        sess = ps.PalacerSession(
            generate=gen,
            send=send_one,
            config=ps.SessionConfig(self.startup_delay, self.message_window),
            on_complete=done,
            clock=self.times.clock,
        )
        self.proactive_sessions[umo] = sess
        await sess.on_user_message("（主动消息）")
        self._log(ev="proactive_scheduled", umo=umo, messages=parts)
        return True

    async def _prepare_voice(self, umo: str, messages: list[str]):
        """Dispatch a validated model choice while cancellation is safe.

        Args:
            umo: Conversation origin.
            messages: Final generated text segments.

        Returns:
            The outgoing text segments and an optional prepared voice component.
        """
        decision = self.voice_decisions.pop(umo, None)
        if not isinstance(decision, dict) or decision.get("use") is not True:
            return messages, None
        voice = self._plugin("integration_voice")
        if voice is not None and messages:
            try:
                audio = await voice.prepare_reply(
                    umo,
                    "\n".join(messages),
                    tier_hint=self.tier_marks.get(umo),
                )
                if audio is not None:
                    self._log(ev="voice_prepared", umo=umo)
                    return [audio.text], audio
                self._log(ev="voice_text_fallback", umo=umo)
            except Exception as exc:
                self._log(ev="voice_prepare_error", umo=umo, error=type(exc).__name__)
        return messages, None

    async def _send(self, umo: str, msg: str, delivery=None) -> bool | mt.TrackedText:
        event = self.events.get(umo)
        if event is None:
            self._log(ev="send_no_event", umo=umo, message=msg)
            return False
        session = self._session(umo)
        batch = list(session.inflight_batch)
        voice = self._plugin("integration_voice")
        if delivery is not None and (
            voice is None or voice._tier(umo) not in ("日常", "暧昧")
        ):
            delivery = None
        try:
            ok = await event.send(
                MessageChain([delivery] if delivery else [Plain(msg)])
            )
            if ok is False:
                return False
        except Exception as exc:  # noqa: BLE001
            self._log(ev="send_error", umo=umo, message=msg, error=repr(exc))
            return False
        history_text = "[语音] " + msg if delivery is not None else msg
        receipt = self.times.record(umo, "assistant", history_text)
        self.session_distill.sent(umo)
        await self._write_history(umo, [*batch, receipt])
        if voice is not None and callable(getattr(voice, "remember_sent", None)):
            voice.remember_sent(umo, "\n".join([*session.sent, msg]))
        self._log(ev="sent", umo=umo, message=msg, message_id=receipt.message_id)
        self._release_burst(umo, "palacer 已成功发送")
        return receipt

    def _text_of(self, item) -> str:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, list):
            return "".join(
                str(x.get("text") or "")
                for x in content
                if isinstance(x, dict) and x.get("type") == "text"
            )
        return str(content or "")

    async def _write_history(self, umo: str, messages, expected_cid=None):
        """Append each occurrence once; identity is never inferred from text."""
        async with self.history_locks.setdefault(umo, asyncio.Lock()):
            try:
                cm = self.context.conversation_manager
                cid = await cm.get_curr_conversation_id(umo)
                if not cid or (expected_cid is not None and cid != expected_cid):
                    return
                conv = await cm.get_conversation(umo, cid)
                if conv is None:
                    return
                h = (
                    json.loads(conv.history or "[]")
                    if isinstance(conv.history, str)
                    else list(conv.history)
                )
                bindings = []
                for message in messages:
                    mid = getattr(message, "message_id", None)
                    row = self.times.get(mid) if mid else None
                    if row is None or row["cid"] is not None:
                        continue
                    item = {
                        "role": row["role"],
                        "content": [{"type": "text", "text": row["text"]}],
                    }
                    bindings.append((message, len(h), item))
                    h.append(item)
                if not bindings:
                    return
                await cm.update_conversation(umo, cid, history=h)
                for message, index, item in bindings:
                    self.times.bind(message, cid, index, item)
                self._log(ev="history_written", umo=umo, messages=len(bindings))
            except Exception as exc:
                self._log(ev="history_error", umo=umo, error=repr(exc))

    async def record_voice_delivery(self, umo: str, text: str):
        """Append a receipt after an explicit voice command was actually sent.

        Args:
            umo: Conversation origin.
            text: Text spoken in the delivered recording.
        """
        receipt = self.times.record(umo, "assistant", "[语音] " + text)
        self.session_distill.sent(umo)
        await self._write_history(umo, [receipt])

    async def record_image_delivery(self, umo: str, cid: str, text: str):
        """Use the existing history lock and occurrence IDs for an actual image send."""
        receipt = self.times.record(umo, "assistant", text)
        await self._write_history(umo, [receipt], expected_cid=cid)

    async def _complete(self, umo: str, batch: list[str], sent: list[str]):
        self._log(ev="turn_complete", umo=umo, batch=batch, sent=sent)
        try:
            await self._write_history(umo, batch)
            await self.images.commit(umo, batch, sent)
            await self._post_turn(umo, sent)
        finally:
            done = self.turn_done.get(umo)
            if done is not None:
                done.set()
            self._log(ev="turn_finished", umo=umo, sent=len(sent))
            self._release_burst(umo, "palacer 本轮已结束")

    async def _accept_input(self, event):

        umo = event.unified_msg_origin
        self.voice_decisions.pop(umo, None)
        self.images.discard(umo)
        receipt = self.times.record(umo, "user", (event.message_str or "").strip())
        event.set_extra("palacer_input_handled", True)
        old = self.turn_done.get(umo)
        if old is not None:
            old.set()
        self.turn_done[umo] = asyncio.Event()
        self.events[umo] = event
        self.burst_started[umo] = time.monotonic()
        ready = asyncio.get_running_loop().create_future()
        session = self._session(umo)

        await session.on_user_message(receipt, preparation=ready)
        try:
            await self._cancel_proactive(umo)
            text = await self._text_with_media(event)
        finally:
            text = locals().get("text") or str(receipt) or "（用户发来附件）"
            prepared = self.times.prepared(receipt, text)
            if not ready.done():
                ready.set_result(prepared)
        self._log(
            ev="input",
            umo=umo,
            text=text[:3000],
            message_id=receipt.message_id,
            mode=self.mode,
        )

    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def on_input(self, event: AstrMessageEvent):
        if self.mode == "shadow" or not self._match(event):
            return
        await self._accept_input(event)

    async def _post_turn(self, umo: str, sent: list[str]) -> None:
        event = self.events.get(umo)
        if event is None:
            return
        text = "\n".join(sent).strip()
        if not text:
            return

        mark = self.tier_marks.pop(umo, "")
        settle_text = f"〔档位:{mark}〕\n{text}" if mark else text
        resp = LLMResponse(role="assistant", completion_text=settle_text)
        mode = self._plugin("integration_mode")
        if mode is not None and hasattr(mode, "settle"):
            try:
                await mode.settle(event, resp)
                self._log(ev="mode_settled", umo=umo)
            except Exception as exc:  # noqa: BLE001
                self._log(ev="mode_settle_error", umo=umo, error=repr(exc))
        life = self._plugin("integration_life")
        if life is not None and hasattr(life, "note_book_wish"):
            try:
                wish = None
                if hasattr(life, "_pending_wish_for_life"):
                    wish = life._pending_wish_for_life()
                if wish:
                    life._wish_hint[umo] = wish
                await life.note_book_wish(event, resp)
                self._log(ev="life_wish_checked", umo=umo)
            except Exception as exc:  # noqa: BLE001
                self._log(ev="life_wish_error", umo=umo, error=repr(exc))

    @filter.on_llm_request(priority=100000)
    async def on_llm_request(self, event: AstrMessageEvent, *args, **kwargs):
        """Stop the default LLM before it can stream/send a second reply."""
        owned = bool(event.get_extra("palacer_input_handled"))
        if self.mode != "canary" or (not owned and not self._match(event)):
            return

        if event.get_extra("media_file_handled"):
            event.stop_event()
            return
        umo = event.unified_msg_origin
        self.events[umo] = event
        if not event.get_extra("palacer_input_handled"):
            await self._accept_input(event)
        done = self.turn_done.get(umo)
        if done is not None:
            try:
                await asyncio.wait_for(done.wait(), timeout=120)
            except asyncio.TimeoutError:
                self._log(ev="turn_timeout", umo=umo)
        event.stop_event()
        self._log(ev="llm_stopped", umo=umo, text=event.message_str)

    @filter.on_decorating_result(priority=100)
    async def on_result(self, event: AstrMessageEvent):
        if not self._match(event):
            return
        result = event.get_result()
        if result is None or not result.chain or not result.is_llm_result():
            return
        if self.mode == "shadow":
            text = (event.message_str or "").strip()
            task = asyncio.create_task(self._shadow_turn(event, text))
            self.shadow_tasks.add(task)
            task.add_done_callback(self.shadow_tasks.discard)
            return

        result.chain.clear()
        if not event.get_extra("palacer_input_handled"):
            await self._accept_input(event)

    async def _shadow_turn(self, event: AstrMessageEvent, text: str):
        try:
            umo = event.unified_msg_origin
            res = await self._run_engine(
                self.base,
                self.key,
                umo,
                text,
                self._persona_for(umo),
                await self._recent_history(umo),
                [],
                handoff=self.session_distill.current(umo),
            )
            self._log(
                ev="shadow",
                umo=umo,
                text=text,
                messages=res.get("messages"),
                usage=res.get("usage"),
                style_flags=[n.get("style_flags", []) for n in res.get("nodes", [])],
                style_attempts=[
                    n.get("style_attempts", []) for n in res.get("nodes", [])
                ],
            )
        except Exception as exc:  # noqa: BLE001
            self._log(ev="shadow_error", error=repr(exc), tb=traceback.format_exc())

    async def initialize(self):

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        root = Path(__file__).resolve().parent
        files = (
            "main.py",
            "perception.py",
            "palacer_session.py",
            "message_time.py",
            "palacer_engine.py",
            "palacer_harness.py",
            "palacer_brain_harness.py",
            "image_bridge.py",
            "metadata.yaml",
        )
        functions = {
            "input": self._accept_input,
            "session": ps.PalacerSession.on_user_message,
            "time": mt.MessageTimes.context,
            "mother": engine.generate_turn,
            "children": engine.brain.run_brain_children,
            "model_call": engine.ph.call,
            "voice_decision": engine.parse_voice_decision,
            "voice_dispatch": self._prepare_voice,
            "omni_observe": pm.OmniObserver.observe,
        }
        voice = self._plugin("integration_voice")
        if voice is not None and callable(getattr(voice, "prepare_reply", None)):
            functions["voice_prepare"] = voice.prepare_reply
        runtime = {
            "timezone": "Asia/Singapore",
            "task_contract": "context-goal-role-v1",
            "prompt_hashes": {
                name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in {
                    "mother": engine.brain.BRAIN_PLANNER,
                    "writer": engine.brain.CHILD_RULES,
                    "goal": engine.brain.COMPANION_GOAL,
                    "perception": engine.ph.PERCEPTION_CARD,
                }.items()
            },
            "files": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in files
            },
            "code": {
                name: hashlib.sha256(marshal.dumps(fn.__code__, 0)).hexdigest()
                for name, fn in functions.items()
            },
            "code_files": {
                name: fn.__code__.co_filename for name, fn in functions.items()
            },
        }
        self._log(ev="time_context_ready", **runtime)
        logger.info("[palacer] time_context_ready %s", json.dumps(runtime))
        logger.info(
            "[palacer] WebUI canary ready: mode=%s allow=%s",
            self.mode,
            self.umo_allow,
        )

    async def terminate(self):
        for s in [*self.sessions.values(), *self.proactive_sessions.values()]:
            await s.close()
        for task in list(self.shadow_tasks):
            task.cancel()
        if self.shadow_tasks:
            await asyncio.gather(*self.shadow_tasks, return_exceptions=True)
        await self.async_client.aclose()
        self.client.close()
