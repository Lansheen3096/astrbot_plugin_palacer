"""Palacer palacer harness."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
import re
import time


try:
    from .runtime_config import PLUGIN_DATA, integration_dir, setting
except ImportError:
    from runtime_config import PLUGIN_DATA, integration_dir, setting


ROOT_MAX_CHILDREN = 3
CHILD_MAX_CHILDREN = 2
MAX_DEPTH = 2
MAX_CALLS_PER_TURN = 6
FORCE_RECURSION = os.environ.get("PALACER_FORCE_RECURSION") == "1"


ROOT_MAX_TOKENS = int(os.environ.get("PALACER_ROOT_MAX_TOKENS") or 0)
CHILD_MAX_TOKENS = int(os.environ.get("PALACER_CHILD_MAX_TOKENS") or 0)
EMPTY_OUTPUT_RETRIES = 1

STATE_INSTRUCTION = (
    "你会收到一个 <persona_context> 块。那是你自己的内部状态，不是他说的。"
    "按它调整，但不要复述它，也不要用助手口吻解释它。"
)

ROOT_INSTRUCTION = """你是所选角色的调度器，不负责写正文。
根据完整人格、当前状态和用户消息，决定这一轮发几条消息（1-3 条），并给每条一个具体任务。
输出 JSON：
{"style":"1-2句，概括这一轮的语气、节奏、边界和禁止项","children":[{"task":"这条消息要做什么，具体"}]}
不要写消息正文。只输出 JSON。"""

VOICE_CARD = "【表达约束】\n称呼、语气、身份和关系均遵循用户所选 persona。只写当前任务需要的内容，不复述内部卡片。"


PERCEPTION_CARD = """【视听能力边界】
当本轮有明确的附件观察、音乐听取或视频视听观察记录时，那是你通过已接入工具实际获得的观察，不只是用户转述。
你可以自然说自己听到或看到记录中的内容，不要在已有观察时否认这次的视听能力；不要复述内部工具名称。
能力只覆盖用户主动提供并已分析的内容与时间窗。未提供的媒体、未分析的片段和现实环境不能声称看见或听见。
用户猜测、文件名、标题不等于实际观察；乐器、歌词与时间不确定就保留不确定性。"""
COMMON_PREFIX = VOICE_CARD + "\n\n" + PERCEPTION_CARD


DECISION_CARD = "【规划约束】\n依据用户所选 persona 和给定边界进行规划；不预设身份、关系、身体能力或称呼。事实不足时保留不确定性。"

CHILD_INSTRUCTION = '你是消息写手。按所选 persona 完成一条消息。输出 JSON：{"say":"消息","children":[{"task":"确有必要的补充任务，否则空数组"}]}。不重复已发送内容，不编造事实，不复述内部信息；children 最多两条。'


def load_state_core():
    """Load an explicitly configured state integration."""
    path = integration_dir("state") / "core.py"
    spec = importlib.util.spec_from_file_location("palacer_state_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def persona(persona_id: str = "") -> str:
    """Read only the explicitly selected persona."""
    from context_bundle import load_persona

    return load_persona(persona_id)


def deepseek_config():
    """Optional explicit environment settings; never inspect unrelated credentials."""
    return (
        os.environ.get("PALACER_API_BASE", "").rstrip("/"),
        os.environ.get("PALACER_API_KEY", ""),
    )


def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    i = t.find("{")
    if i < 0:
        raise ValueError("no json")
    return json.JSONDecoder(strict=False).raw_decode(t[i:])[0]


def add_usage(acc, usage):
    for k in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        acc[k] = acc.get(k, 0) + int(usage.get(k) or 0)
    rmodel = usage.get("resp_model")
    if rmodel:
        models = acc.setdefault("resp_models", {})
        models[str(rmodel)] = models.get(str(rmodel), 0) + 1
    det = usage.get("completion_tokens_details") or {}
    acc["reasoning_tokens"] = acc.get("reasoning_tokens", 0) + int(
        det.get("reasoning_tokens") or 0
    )


def _model_from_config() -> str:
    return str(setting("model", "") or "").strip()


MODEL = os.environ.get("PALACER_MODEL") or _model_from_config() or "deepseek-flash"


def set_model(name: str) -> None:
    global MODEL
    name = str(name or "").strip()
    if name:
        MODEL = name


def get_model() -> str:
    return MODEL


PLANNER_MODEL = os.environ.get("PALACER_PLANNER_MODEL") or "deepseek-flash"
PLANNER_EFFORT = os.environ.get("PALACER_PLANNER_EFFORT") or "high"
WRITER_EFFORT = os.environ.get("PALACER_WRITER_EFFORT") or "low"
PLANNER_KINDS = {"tool", "brain", "planner"}


def set_planner_model(name: str) -> None:
    global PLANNER_MODEL
    name = str(name or "").strip()
    if name:
        PLANNER_MODEL = name


def set_planner_effort(name: str) -> None:
    global PLANNER_EFFORT
    name = str(name or "").strip()
    if name:
        PLANNER_EFFORT = name


def get_planner_model() -> str:
    return PLANNER_MODEL


def route(kind: str) -> tuple[str, str]:
    """按调用类型选 (模型, 思考强度)。planner 走 flash/high，写手走配置模型/low。"""
    if str(kind) in PLANNER_KINDS:
        return PLANNER_MODEL, PLANNER_EFFORT
    return MODEL, WRITER_EFFORT


CALL_LOG = PLUGIN_DATA / "call_log.jsonl"
CALL_LOG_MAX_MB = 8.0


def _log_call(
    kind,
    model,
    usage,
    finish,
    attempt,
    dt,
    *,
    turn_id="",
    request_id="",
    status="completed",
    error_type="",
    retry_reason="",
    sys_hash="",
    sys_chars=0,
):
    """One physical request per row; no prompt, account id, key or response body.

    sys_hash / sys_chars：system prompt 的指纹与长度。同一轮多个写手的 system 应该完全一致
    （前缀缓存靠它），不一致就是有变动内容混进了固定段。
    """
    if not setting("trace", False):
        return None
    PLUGIN_DATA.mkdir(parents=True, exist_ok=True)
    det = usage.get("completion_tokens_details") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    rec = {
        "ts": round(time.time(), 6),
        "kind": str(kind),
        "model": str(model or ""),
        "prompt": prompt,
        "hit": hit,
        "miss": int(usage.get("prompt_cache_miss_tokens", max(0, prompt - hit)) or 0),
        "out": int(usage.get("completion_tokens") or 0),
        "reason": int(det.get("reasoning_tokens") or 0),
        "dt": round(float(dt), 6),
        "try": int(attempt),
        "finish": str(finish or ""),
        "turn_id": str(turn_id),
        "request_id": str(request_id),
        "status": status,
        "error_type": error_type,
        "retry_reason": retry_reason,
        "sys_hash": str(sys_hash or ""),
        "sys_chars": int(sys_chars or 0),
        "usage_unknown": not bool(usage),
    }
    try:
        if (
            CALL_LOG.exists()
            and CALL_LOG.stat().st_size > CALL_LOG_MAX_MB * 1024 * 1024
        ):
            CALL_LOG.replace(CALL_LOG.with_suffix(".jsonl.1"))
        with CALL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[palacer] request ledger write failed: %s", type(exc).__name__
        )
    return rec


def call(
    client,
    base,
    key,
    messages,
    max_tokens,
    response_format=None,
    time_context=None,
    kind="other",
    retry_context="",
):

    t = time.perf_counter()
    call_model, call_effort = route(kind)
    payload = {
        "model": call_model,
        "messages": messages,
        "stream": False,
        **({"max_tokens": int(max_tokens)} if max_tokens else {}),
        "thinking": {"type": "enabled"},
        "reasoning_effort": call_effort,
        **({"response_format": response_format} if response_format else {}),
    }
    usage = {}
    retry_reason = retry_context
    turn_id = getattr(client, "turn_id", "") or f"standalone-{time.time_ns():x}"

    sys_text = str(messages[0].get("content") or "") if messages else ""
    sys_hash = hashlib.sha1(sys_text.encode("utf-8")).hexdigest()[:12]
    sys_chars = len(sys_text)
    for attempt in range(EMPTY_OUTPUT_RETRIES + 1):
        if time_context is not None:
            current_messages = [dict(message) for message in messages]
            current_messages[-1]["content"] += "\n\n" + time_context()
            payload = {**payload, "messages": current_messages}
        attempt_start = time.perf_counter()
        request_id = f"{turn_id}-{time.time_ns():x}"
        before = getattr(client, "count", None)
        try:
            r = client.post(
                base + "/chat/completions",
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            d = r.json()
        except BaseException as exc:
            if before is None or getattr(client, "count", None) != before:
                rec = _log_call(
                    kind,
                    call_model,
                    {},
                    "",
                    attempt,
                    time.perf_counter() - attempt_start,
                    turn_id=turn_id,
                    request_id=request_id,
                    sys_hash=sys_hash,
                    sys_chars=sys_chars,
                    status="aborted"
                    if type(exc).__name__ in ("GenerationCancelled", "CancelledError")
                    else "error",
                    error_type=type(exc).__name__,
                    retry_reason=retry_reason,
                )
                if isinstance(rec, dict) and hasattr(client, "record_call"):
                    client.record_call(rec)
            raise
        current_usage = d.get("usage") or {}
        choice = (d.get("choices") or [{}])[0]
        content = str((choice.get("message") or {}).get("content") or "")
        finish = choice.get("finish_reason")
        rec = _log_call(
            kind,
            d.get("model") or call_model,
            current_usage,
            finish,
            attempt,
            time.perf_counter() - attempt_start,
            turn_id=turn_id,
            request_id=request_id,
            sys_hash=sys_hash,
            sys_chars=sys_chars,
            status="length"
            if finish == "length"
            else "completed"
            if content.strip()
            else "empty",
            retry_reason=retry_reason,
        )
        if isinstance(rec, dict) and hasattr(client, "record_call"):
            client.record_call(rec)
        retry_reason = "length" if finish == "length" else "empty"
        if d.get("model"):
            usage["resp_model"] = str(d["model"])
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ):
            if name in current_usage:
                usage[name] = usage.get(name, 0) + int(current_usage[name] or 0)
        detail = current_usage.get("completion_tokens_details") or {}
        previous = usage.setdefault("completion_tokens_details", {})
        previous["reasoning_tokens"] = previous.get("reasoning_tokens", 0) + int(
            detail.get("reasoning_tokens") or 0
        )
        choice = (d.get("choices") or [{}])[0]
        content = str((choice.get("message") or {}).get("content") or "")
        finish = choice.get("finish_reason")
        if content.strip() and finish != "length":
            return content, usage, time.perf_counter() - t
        if attempt == 0:
            payload = {**payload, "thinking": {"type": "disabled"}}
    raise RuntimeError(f"模型未返回完整正文（finish_reason={finish}，已重试1次）")


def child_messages(
    style, task, user_text, previous, depth, extra_context="", extra_label="工具结果"
):
    prev = "\n".join(f"- {x}" for x in previous[-4:]) or "（还没有）"
    return [
        {
            "role": "system",
            "content": CHILD_INSTRUCTION
            + "\n\n"
            + VOICE_CARD
            + "\n\n风格卡：\n"
            + style
            + f"\n\n当前层级：{depth}",
        },
        {
            "role": "user",
            "content": (
                f"用户刚说：{user_text}\n\n"
                f"{extra_label}（只作参考，不要编）：\n{extra_context or '（无）'}\n\n"
                f"前面已经发过：\n{prev}\n\n这一条的任务：{task}"
            ),
        },
    ]


def run_tree(
    client,
    base,
    key,
    tasks,
    style,
    user_text,
    previous,
    depth,
    budget,
    usage,
    calls,
    extra_context="",
    extra_label="工具结果",
):
    nodes = []
    for task in tasks:
        if budget["left"] <= 0:
            break
        budget["left"] -= 1
        calls["n"] += 1
        task = str(task.get("task") if isinstance(task, dict) else task).strip()
        if not task:
            continue
        try:
            content, u, dt = call(
                client,
                base,
                key,
                child_messages(
                    style,
                    task,
                    user_text,
                    previous,
                    depth,
                    extra_context,
                    extra_label,
                ),
                CHILD_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            nodes.append({"task": task, "say": "", "error": str(exc), "children": []})
            continue
        add_usage(usage, u)
        try:
            obj = parse_json(content)
            say = str(obj.get("say") or "").strip()
            children = (
                obj.get("children") if isinstance(obj.get("children"), list) else []
            )
        except Exception:
            say = "" if content.lstrip().startswith("{") else content.strip()
            children = []
        if say:
            previous.append(say)
        node = {
            "task": task,
            "say": say,
            "latency": dt,
            "usage": u,
            "children": [],
            "depth": depth,
        }

        if (
            FORCE_RECURSION
            and depth == 1
            and not children
            and not budget.get("forced")
            and budget["left"] > 0
        ):
            children = [{"task": "补一句更轻的收尾，不要重复前一条"}]
            node["forced_recursion"] = True
            budget["forced"] = True
        nodes.append(node)
        if children and depth < MAX_DEPTH and budget["left"] > 0:
            node["children"] = run_tree(
                client,
                base,
                key,
                children[:CHILD_MAX_CHILDREN],
                style,
                user_text,
                previous,
                depth + 1,
                budget,
                usage,
                calls,
                extra_context,
            )
    return nodes


def flatten(nodes):
    out = []
    for n in nodes:
        if n.get("say"):
            out.append(n["say"])
        out.extend(flatten(n.get("children") or []))
    return out
