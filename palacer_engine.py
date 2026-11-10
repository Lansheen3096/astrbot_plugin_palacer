"""Palacer palacer engine."""

from __future__ import annotations

import importlib.util
import asyncio
import concurrent.futures
import threading
import os
import sys
import time
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent


TOOL_MAX_TOKENS = int(os.environ.get("PALACER_TOOL_MAX_TOKENS") or 4096)
BRAIN_MAX_TOKENS = int(os.environ.get("PALACER_BRAIN_MAX_TOKENS") or 8192)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from . import palacer_harness as ph, palacer_tool_harness as pt
    from . import context_bundle as cb, palacer_brain_harness as brain
except ImportError:
    import palacer_harness as ph
    import palacer_tool_harness as pt
    import context_bundle as cb
    import palacer_brain_harness as brain


class GenerationCancelled(BaseException):
    """Control flow; never turn cancellation into a writer/tool failure."""


class AsyncRequestBridge:
    """The synchronous planner uses cancellable HTTP tasks on its owning loop."""

    def __init__(self, client, loop):
        self.client, self.loop = client, loop
        self.cancelled = threading.Event()
        self.tasks = set()
        self.records = []
        self.turn_id = f"{time.time_ns():x}-{id(self):x}"

    def checkpoint(self):
        if self.cancelled.is_set():
            raise GenerationCancelled()

    async def _request(self, args, kwargs):
        self.checkpoint()
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            return await self.client.post(*args, **kwargs)
        finally:
            self.tasks.discard(task)

    def post(self, *args, **kwargs):
        self.checkpoint()
        future = asyncio.run_coroutine_threadsafe(
            self._request(args, kwargs), self.loop
        )
        try:
            return future.result(timeout=kwargs.get("timeout", 180))
        except concurrent.futures.CancelledError:
            raise GenerationCancelled() from None
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("Palacer request deadline exceeded") from None

    def record_call(self, record):
        self.records.append(record)

    def cancel(self):
        self.cancelled.set()
        for task in tuple(self.tasks):
            if not task.cancelling():
                task.cancel()

    async def close(self):
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)


class RequestBudget:
    """Count actual HTTP attempts, including harness retries, across one turn."""

    def __init__(self, client, max_requests=6, timeout=180):
        self.client = client
        self.max_requests = max_requests
        self.deadline = time.monotonic() + timeout
        self.count = 0
        self.turn_id = (
            getattr(client, "turn_id", "") or f"{time.time_ns():x}-{id(self):x}"
        )

    def record_call(self, record):
        sink = getattr(self.client, "record_call", None)
        if sink is not None:
            sink(record)

    def checkpoint(self):
        check = getattr(self.client, "checkpoint", None)
        if check is not None:
            check()
        if time.monotonic() >= self.deadline:
            raise RuntimeError("Palacer turn deadline exceeded")

    def post(self, *args, **kwargs):
        self.checkpoint()
        left = self.deadline - time.monotonic()
        if self.count >= self.max_requests or left <= 0:
            raise RuntimeError("Palacer turn request/time budget exhausted")
        self.count += 1
        kwargs["timeout"] = left
        return self.client.post(*args, **kwargs)


def parse_voice_decision(value: object) -> dict | None:
    """Validate a model decision without coercing strings to booleans.

    Args:
        value: The voice field from a planner JSON object.

    Returns:
        The validated short decision, or None for a missing or invalid field.
    """
    if not isinstance(value, dict) or type(value.get("use")) is not bool:
        return None
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    return {"use": value["use"], "reason": reason.strip()[:160]}


def generate_turn(
    client: httpx.Client,
    base: str,
    key: str,
    umo: str,
    user_text: str,
    persona_id: str = "",
    history: list[dict] | None = None,
    sent: list[str] | None = None,
    time_context=None,
    turn_kind: str = "reply",
    handoff: str = "",
    tool_guide: str = "",
    tool_handlers: dict | None = None,
    image_context: list[dict] | None = None,
    memory_candidate_ids: list[int] | None = None,
    resident: str = "",
    tier_label: str = "",
    tier_tail: str = "",
    tier_state: str = "",
    voice_available: bool = False,
    voice_card: str = "",
) -> dict:
    """Run one brain/effector turn and return messages + trace."""
    client = RequestBudget(client, max_requests=ph.MAX_CALLS_PER_TURN)
    history = list(history or [])

    if sent and time_context is None:
        history.append({"role": "assistant", "content": "\n".join(sent)})
    bundle = cb.build_bundle(
        umo, user_text, persona_id=persona_id, memory_candidate_ids=memory_candidate_ids
    )
    hard_facts = [dict(x) for x in (bundle.get("hard_facts") or [])]
    hard_facts.extend(
        dict(x)
        for x in (image_context or [])
        if x.get("source") in {"image_studio.observation", "image_studio.status"}
        and x.get("type") == "hard"
        and isinstance(x.get("text"), str)
    )
    soft_facts = [dict(x) for x in (bundle.get("soft_facts") or [])]
    boundary_facts = [dict(x) for x in (bundle.get("boundary_facts") or [])]
    hard_facts.append(
        {
            "type": "hard",
            "source": "user" if turn_kind == "reply" else "trigger",
            "text": f"用户刚说：{user_text}"
            if turn_kind == "reply"
            else f"内部触发（不是用户发言）：{user_text}",
        }
    )
    user_fact_index = len(hard_facts) - 1
    resident = (
        str(bundle.get("persona") or "")
        + "\n\n"
        + voice_card
        + ("\n\n" + resident if resident else "")
    )
    bundle = {
        **bundle,
        "hard_facts": hard_facts,
        "soft_facts": soft_facts,
        "boundary_facts": boundary_facts,
    }
    planner_prefix = (
        ph.PERCEPTION_CARD + "\n\n" + voice_card + "\n\n" + ph.DECISION_CARD
    )
    planner_prefix += "\n\n【所选 persona】\n" + str(bundle.get("persona") or "")
    bundle_text = cb.render_bundle({**bundle, "persona": ""})

    usage = {}
    calls = {"n": 0}
    turn_brief = (
        "\n\n【本次用途】\n"
        + brain.TURN_PURPOSES.get(turn_kind, brain.TURN_PURPOSES["reply"])
        + "\n\n【此前实际已发送，本次只续接】\n"
        + str(sent or [])
        + f"\n\n【本轮语音能力】档位={tier_label or '未知'}；"
        f"允许调用={'是' if voice_available and turn_kind == 'reply' and tier_label in ('日常', '暧昧') else '否'}。"
        "voice.use 与 reason 都必须填写。"
    )
    available_tools = {**pt.TOOLS, **(tool_handlers or {})}
    tool_user = (
        bundle_text
        + turn_brief
        + "\n\n【最近对话】\n"
        + brain.fmt_history(history)
        + "\n\n【本次触发内容】\n"
        + user_text
        + "\n\n【输出】只输出 JSON。"
    )
    calls["n"] += 1
    raw_tool, u_tool, _ = ph.call(
        client,
        base,
        key,
        [
            {
                "role": "system",
                "content": planner_prefix
                + "\n\n"
                + brain.BRAIN_PLANNER
                + "\n\n【先判断工具需求】\n"
                + getattr(pt, "TOOL_CATALOG", pt.TOOL_PLANNER)
                + '\n先输出统一规划 JSON：{"tool_calls":[],"reply_plan":{母进程规划字段}}。'
                "不需要工具且现有事实足够时，tool_calls 为空数组，reply_plan 按上面的母进程格式给出完整规划。"
                "需要工具或尚不能可靠分工时，按目录给出最多2个 tool_calls，reply_plan 为 null，待工具结果后再规划。"
                "不要为减少调用而省略必要工具、猜测事实或编造事实编号。只输出一个 JSON 对象。"
                + (f"\n\n【本轮档位】{tier_label}" if tier_label else ""),
            },
            {
                "role": "user",
                "content": (tool_guide + "\n\n" + tool_user)
                if tool_guide
                else tool_user,
            },
        ],
        BRAIN_MAX_TOKENS,
        response_format={"type": "json_object"},
        time_context=time_context,
        kind="tool",
    )
    ph.add_usage(usage, u_tool)
    inline_plan = None
    try:
        tool_plan = ph.parse_json(raw_tool)
        requested = (
            tool_plan.get("tool_calls")
            if isinstance(tool_plan.get("tool_calls"), list)
            else []
        )
        candidate = tool_plan.get("reply_plan")
        if (
            tool_plan.get("tool_calls") == []
            and isinstance(candidate, dict)
            and isinstance(candidate.get("children"), list)
            and isinstance(candidate.get("style"), str)
            and isinstance(candidate.get("context"), str)
            and isinstance(candidate.get("goal"), str)
        ):
            inline_plan = candidate
    except Exception:
        requested = []
    requested = requested[:2]
    tool_runs = []
    for req in requested:
        client.checkpoint()
        if not isinstance(req, dict):
            continue
        name = str(req.get("name") or "")
        args = req.get("args") if isinstance(req.get("args"), dict) else {}
        fn = available_tools.get(name)
        if fn is None:
            tool_runs.append({"name": name, "args": args, "result": "未知工具"})
            continue
        try:
            result = fn(args)
            client.checkpoint()
        except Exception as exc:  # noqa: BLE001
            result = f"工具失败：{type(exc).__name__}: {exc}"
        tool_runs.append({"name": name, "args": args, "result": result})
    for t in tool_runs:
        hard_facts.append(
            {
                "type": "hard",
                "source": f"tool:{t['name']}",
                "text": str(t.get("result") or ""),
            }
        )
    tool_text = (
        "\n\n".join(f"[{x['name']}] {x['result']}" for x in tool_runs)
        or "（没有调用工具）"
    )

    brain_user = (
        cb.render_bundle({**bundle, "persona": ""})
        + turn_brief
        + "\n\n【最近对话】\n"
        + brain.fmt_history(history)
        + "\n\n【本次触发内容】\n"
        + user_text
        + "\n\n【工具结果】\n"
        + tool_text
        + "\n\n【输出】只输出 JSON。"
    )
    if inline_plan is not None:
        raw_brain = __import__("json").dumps(inline_plan, ensure_ascii=False)
    else:
        calls["n"] += 1
        raw_brain, u_brain, _ = ph.call(
            client,
            base,
            key,
            [
                {
                    "role": "system",
                    "content": planner_prefix + "\n\n" + brain.BRAIN_PLANNER,
                },
                {"role": "user", "content": brain_user},
            ],
            BRAIN_MAX_TOKENS,
            response_format={"type": "json_object"},
            kind="brain",
            time_context=time_context,
        )
        ph.add_usage(usage, u_brain)
    plan = {}
    try:
        plan = ph.parse_json(raw_brain)
        if not isinstance(plan, dict):
            raise ValueError("母进程计划必须是对象")
        style = brain.clean(str(plan.get("style") or ""))
        tasks = plan["children"]
        if not isinstance(tasks, list):
            raise ValueError("母进程 children 必须是数组")
        tasks = tasks[: ph.ROOT_MAX_CHILDREN]
    except Exception:
        plan = {}
        style = "按所选 persona 自然回复，不编造事实。"
        tasks = [
            {
                "role": "直接回应",
                "task": "只回应本次触发有依据的一件事",
                "hard": [user_fact_index],
                "soft": [],
                "depends_on": [],
            }
        ]
    if not style:
        style = "按所选 persona 自然回复，不编造事实。"

    tasks = [
        t for t in tasks if isinstance(t, dict) and str(t.get("task") or "").strip()
    ]
    if turn_kind == "reply" and not tasks:
        tasks = [
            {
                "role": "直接回应",
                "task": "只回应本次触发有依据的一件事",
                "handoff": "",
                "depends_on": [],
                "hard": [user_fact_index],
                "soft": [],
            }
        ]
        plan = {**plan, "children": tasks}

    decision = parse_voice_decision(plan.get("voice"))
    voice_status = "decided"
    voice_error = ""
    voice_allowed = (
        voice_available and turn_kind == "reply" and tier_label in ("日常", "暧昧")
    )
    if not voice_allowed:
        decision = {"use": False, "reason": "本轮语音能力或档位不允许"}
        voice_status = "blocked"
    elif decision is None:
        if client.count + len(tasks) + 2 <= client.max_requests:
            calls["n"] += 1
            try:
                raw_voice, u_voice, _ = ph.call(
                    client,
                    base,
                    key,
                    [
                        {
                            "role": "system",
                            "content": planner_prefix
                            + "\n\n"
                            + brain.VOICE_DECISION_RULES
                            + '\n上一份规划遗漏或错误填写 voice。只补交 {"voice":{"use":布尔值,"reason":"本轮具体依据"}}，不要重写规划或正文。',
                        },
                        {"role": "user", "content": brain_user},
                    ],
                    2048,
                    response_format={"type": "json_object"},
                    kind="brain",
                    time_context=time_context,
                    retry_context="missing_or_invalid_voice_decision",
                )
                ph.add_usage(usage, u_voice)
                repaired = ph.parse_json(raw_voice)
                decision = (
                    parse_voice_decision(repaired.get("voice"))
                    if isinstance(repaired, dict)
                    else None
                )
            except Exception as exc:
                voice_error = type(exc).__name__
            voice_status = "repaired" if decision is not None else "invalid"
        else:
            voice_status = "budget_exhausted"
        if decision is None:
            decision = {"use": False, "reason": "语音决策未有效完成，保留文字回复"}
    if decision["use"]:
        style += "\n本轮选择语音：整轮适合口述、合计不超过200字；不宣告发送方式，不说已经发出语音。"

    client.checkpoint()
    nodes = brain.run_brain_children(
        client,
        base,
        key,
        tasks,
        style,
        user_text,
        hard_facts,
        soft_facts,
        boundary_facts,
        usage,
        calls,
        time_context=time_context,
        context=str(plan.get("context") or ""),
        goal=str(plan.get("goal") or ""),
        history=history,
        sent=sent,
        turn_kind=turn_kind,
        handoff=handoff,
        resident=resident,
        tier_tail=tier_tail,
        tier_state=tier_state,
    )

    messages = [str(x.get("say") or "").strip() for x in nodes]
    messages = [x for x in messages if x]
    if not messages and any(n.get("status") == "error" for n in nodes):
        raise RuntimeError("palacer 写手生成失败，没有可发送的消息")
    return {
        "messages": messages,
        "hard_facts": hard_facts,
        "soft_facts": soft_facts,
        "boundary_facts": boundary_facts,
        "tasks": tasks,
        "context": str(plan.get("context") or ""),
        "goal": str(
            plan.get("goal")
            or brain.TURN_PURPOSES.get(turn_kind, brain.TURN_PURPOSES["reply"])
        ),
        "turn_kind": turn_kind,
        "planning_path": "combined" if inline_plan is not None else "tools_then_brain",
        "voice": decision,
        "voice_status": voice_status,
        "voice_error": voice_error,
        "nodes": nodes,
        "tier_marks": [n.get("tier") for n in nodes if n.get("tier")],
        "tier_mark": next(
            (n.get("tier") for n in reversed(nodes) if n.get("tier")), ""
        ),
        "tool_runs": tool_runs,
        "usage": usage,
        "calls": calls["n"],
        "actual_requests": client.count,
        "turn_id": client.turn_id,
    }
