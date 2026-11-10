"""Palacer palacer brain harness."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


COMPANION_GOAL = "根据所选 persona 与已有对话保持一致的交流。\n1. 回应当前输入。\n2. 不重复询问已知事实。\n3. 适当接续未完话题。\n4. 不作无关状态汇报。\n5. 主动发言要有具体语境。\n6. 信息量与用户需要相称。\n7. 不添加空泛收束。\n8. 身份、关系、称呼与边界由所选 persona 决定。"

TURN_PURPOSES = {
    "reply": "回应最新用户输入；如果被插话，只续接尚未完成的内容，已发送的消息不重说。",
    "proactive": "现在没有新用户输入，结合已有语境主动开口；不适合打扰时可以不发消息。",
    "watch": "为陪看预写一句，只回应给定的画面证据，不预测后续；没有合适的话可以不发消息。",
}

VOICE_DECISION_RULES = """【本轮语音决策｜必须完成】
已配置且可用的语音集成可以把本轮最终回复合成为一条语音。你负责选择是否调用，系统在写手完成后执行。
每份完整规划必须包含 "voice":{"use":true或false,"reason":"基于本轮具体语境的一句短依据"}。
use 必须是 JSON 布尔值；选择文字也要填。不要把漏填当成选择文字，不输出分析过程。

判断声音相对于文字是否更适合当前沟通目标，并结合听取条件、内容形态、最近实际发送记录：
- 用户要求语音，或本轮表达更适合用声音且符合所选 persona 时，考虑 use=true。
- 他明确要求文字、不方便播放，或回复含代码、链接、地址、精确数据、需复制查阅的步骤时，选择文字。
- 普通聊天按这一轮的表达需要选择；最近已连续用语音要考虑打扰和节奏，但不按固定次数、比例或随机数决定。
- 不能只以「他没明确要求语音」或「文字更省事」作为拒绝依据，也不能见到情绪词就机械发语音。
- 语音只用于日常/暧昧，且本轮能力状态必须允许；更高档位、不可用、主动/陪看用途用 false。
- 若选择语音，整轮安排适合口述的短回复，合计不超过 200 字；不要承诺「已发语音」，实际发送可能回退文字。
reason 指出这一次的具体需要或限制，不照抄规则、不提模型懒惰；它仅供内部决策记录，不进聊天正文。"""

BRAIN_PLANNER = (
    """你是对话的母进程，负责理解语境、确定目标、蒸馏依据并给写手分工。
你看完整资料，写手只看你蒸馏的语境、目标、分工和获准的事实。你不写最终消息。
输入包括角色资料、状态卡、typed facts、最近对话、本次触发和工具结果。

按共同目标的 1–8 排分工：同一件事不让两个写手都说；第 3 条的接续和第 5 条的具体物由分工决定谁来落。

输出 JSON：
{
  "context":"蒸馏当前语境：在谈什么、代词指谁、对方现在要什么、哪些已回应、哪些尚待回应；区分对方说过、她说过和待核实的事",
  "goal":"这一轮具体要达成的沟通目标，服从本次触发用途",
  "style":"1-2句，这一轮的语气、节奏、边界和禁止项",
  "voice":{"use":false,"reason":"根据本轮具体需要选择声音或文字，此处布尔值只是格式示例，不是默认选择"},
  "children":[
    {"role":"本条职责，如直接回答、接住情绪、补充、追问或收束",
     "task":"本条独有的内容与完成条件，只做一件事",
     "handoff":"怎样接前一句；独立成句时写明独立，不重复开场",
     "depends_on":[],"hard":[0,1],"soft":[0]}
  ]
}

typed facts：
- hard_facts：外部/身体事实，来自工具、health、状态卡进度、用户刚说的话。
- soft_facts：review 里角色的想法，不是外部事实。
- boundary_facts：必须遵守的边界，不需要在 children 里引用，所有写手都会收到。

硬规则：
1. hard/soft 只能引用给定数组的下标；不能引用不存在的事实。
2. context 用来理解指代和关系，不扩充事实许可；外部事实仍须在对应任务的 hard 中选入。
3. 每条承担不同的信息或对话动作。相同意思只安排一次，不把一句话的换说法分给多个写手。
4. depends_on 是本轮前序任务的零基下标，仅在缺少该前句就接不上的时候填写；独立任务用 []。
5. 能一条说完就一条，最多三条；主动开口或陪看没有合适内容时 children 可为空。
只输出一个 JSON 对象，不要代码围栏。"""
    + "\n\n"
    + VOICE_DECISION_RULES
)


TIERS = ("日常", "暧昧", "色情", "狂野")

CHILD_RULES = (
    """你按用户所选 persona，负责本轮分工中分给你的那一条私聊消息。
共同目标："""
    + COMPANION_GOAL
    + """
母进程已经定好语境、目标和顺序。先读整轮分工，再用你的声音完成本条职责。
规则：
1. 只能使用任务允许的 hard/soft；hard 是外部事实，soft 是你自己的想法。语境、旧话和其他任务不扩大许可，不能补造事件、数字、引文或承诺；遵守 boundary facts。
2. 本条只完成自己的任务，其他任务留给对应写手；已发送内容和前序草稿说过的意思，不换措辞再说一遍。
3. 顺着实际存在的前句衔接，代词与话题要接得上。草稿尚未发送，对方还没有回应它。
4. 一条说一件事，保持自然。语气按声音卡和本轮风格，不复述内部语境、分工或系统信息；是否写括号动作按【本轮档位】那一段的规则，那段没说就不写。
5. 只输出 JSON：{"say":"这一条消息"}。没有独立新内容时 say 用空字符串；不派生子任务。
6. 想换档就在 JSON 里加 "tier":"日常|暧昧|色情|狂野"；不换就省略这一项。它不会出现在消息里。"""
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from . import palacer_tool_harness as pt
except ImportError:
    import palacer_tool_harness as pt
ph = pt.ph


def clean(text):
    t = (text or "").strip()
    t = re.sub(r"^[（(\[【〔]?\s*档位[:：][^\n]*[)）\]】〕]?\s*\n?", "", t)
    t = re.sub(r"[（(][^）)]{1,40}[）)]", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def fmt_history(hist):
    rows = []
    for m in hist[-8:]:
        who = "他" if m.get("role") == "user" else "你"
        rows.append(f"{who}: {str(m.get('content') or '')[:500]}")
    return "\n".join(rows) or "（还没有）"


def grounding_flags(text, allowed_text):
    flags = {}
    quotes = re.findall(r"[「“\\\"]([^」”\\\"]{2,80})[」”\\\"]", text or "")
    flags["unverified_quote"] = any(q not in allowed_text for q in quotes)
    nums = re.findall(r"\\d+(?:\\.\\d+)?", text or "")
    flags["unverified_number"] = any(n not in allowed_text for n in nums)
    flags["book_claim"] = bool(
        re.search(r"书里|原文|那句|读到", text or "")
    ) and not bool(allowed_text.strip())
    return flags


def _pick(items, idxs):
    out = []
    for idx in idxs if isinstance(idxs, list) else []:
        try:
            i = int(idx)
        except Exception:
            continue
        if 0 <= i < len(items):
            out.append((i, items[i]["text"]))
    return out


def _message_key(text):
    """Ignore presentation differences, without guessing semantic equivalence."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", text).rstrip("。.!?！？…♡❤♥~～\ufe0f")


STYLE_NAMES = {
    "H1": "二元对照（本轮第二次起）",
    "H2": "破折号插入语",
    "H3": "三项排比",
    "H4": "关怀术语",
    "H5": "预设对方感受",
    "H6": "总结收束/升华",
    "H7": "空泛关心",
    "H8": "元叙述开头",
    "H9": "求认同反问",
    "S1": "抽象套语",
    "S2": "连续问句",
    "S3": "恒常自述",
}

_SENTENCE_END = r"[。！？!?；;\n…]+|\.(?!\d)"
_H1 = re.compile(
    r"(?:不是|并非)[^，,。！？!?；;\n…]*?[，,]?\s*而是|不只是[^，,。！？!?；;\n…]*?[，,]?\s*更是"
)
_ADJECTIVE = r"(?:温柔|善良|漂亮|美丽|可爱|聪明|勇敢|坚强|独立|真诚|细腻|敏感|孤独|疲惫|难过|开心|安静|平静|柔软|坚定|温暖|清醒|可靠)"
_ADJECTIVE_TRIPLE = re.compile(
    rf"{_ADJECTIVE}(?:[、，,\s]|又|而|且)*{_ADJECTIVE}(?:[、，,\s]|又|而|且)*{_ADJECTIVE}"
)


def h1_count(text: str) -> int:
    text = unicodedata.normalize("NFKC", text or "")
    return sum(len(_H1.findall(s)) for s in re.split(_SENTENCE_END, text))


def style_breaches(text: str, *, h1_before: int = 0) -> list[str]:
    """Pure detector; the caller supplies H1 occurrences from this turn's outputs.

    H* require repair; S* are diagnostic only. No global/per-message counters.
    H3 uses literal short parallel clauses plus a conservative adjective lexicon.
    """
    text = unicodedata.normalize("NFKC", text or "")
    sentences = [s.strip() for s in re.split(_SENTENCE_END, text) if s.strip()]
    flags = set()
    if h1_count(text) and h1_before + h1_count(text) >= 2:
        flags.add("H1")
    if "——" in text:
        flags.add("H2")
    for sentence in sentences:
        clauses = [s.strip() for s in re.split(r"[、，,]", sentence)]
        if any(
            all(0 < len(c) <= 8 for c in clauses[i : i + 3])
            for i in range(len(clauses) - 2)
        ) or _ADJECTIVE_TRIPLE.search(sentence):
            flags.add("H3")
        patterns = {
            "H4": r"接住(?:你的?(?:情绪|感受|难过|委屈|疲惫)|你|它)",
            "H5": r"你不是[^。！？!?；;\n…]*你只是|其实你并不|你真正想要的是",
            "H6": r"总的来说|说到底|归根结底|换句话说|反正|这就够了|就很好|也挺好",
            "H7": r"好好照顾自己|记得吃饭|早点休息|多喝热水",
            "H8": r"^(?:我想了想|让我想想|怎么说呢|说实话)",
            "H9": r"(?:对不对|是吧|你也是吧)[\s♡❤♥~～]*$",
            "S1": r"其实|本质上|某种意义上",
            "S3": r"我一直在|我一直觉得",
        }
        for flag, pattern in patterns.items():
            if re.search(pattern, sentence):
                flags.add(flag)

    utterances = re.findall(r"[^。！？!?；;\n….]+[。！？!?；;\n….]*", text)
    questions = [bool(re.search(r"[?？]", s)) for s in utterances]
    if any(a and b for a, b in zip(questions, questions[1:])):
        flags.add("S2")
    return [flag for flag in STYLE_NAMES if flag in flags]


def run_brain_children(
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
    time_context=None,
    context="",
    goal="",
    history=None,
    sent=None,
    turn_kind="reply",
    handoff="",
    resident: str = "",
    tier_tail: str = "",
    tier_state: str = "",
):
    """Each palacer gets only its selected hard/soft facts; boundaries are global."""
    previous = []
    nodes = []
    repair_used = False
    h1_seen = sum(h1_count(s) for s in (sent or []))
    assignments = []
    for index, task in enumerate(tasks[: ph.ROOT_MAX_CHILDREN]):
        if not isinstance(task, dict):
            task = {"task": str(task)}
        task_text = str(task.get("task") or "").strip()
        if task_text:
            assignments.append(
                {
                    **task,
                    "index": index,
                    "task": task_text,
                    "role": str(task.get("role") or "回应"),
                }
            )
    division = json.dumps(
        [
            {
                k: t.get(k, "")
                for k in ("index", "role", "task", "handoff", "depends_on")
            }
            for t in assignments
        ],
        ensure_ascii=False,
    )
    boundary_text = (
        "\n".join(f"[B{i}] {x['text']}" for i, x in enumerate(boundary_facts))
        or "（无）"
    )
    for position, task in enumerate(assignments):
        if calls["n"] >= ph.MAX_CALLS_PER_TURN:
            break
        task_text = task["task"]
        dependencies = task.get("depends_on", [])
        valid_dependencies = isinstance(dependencies, list) and all(
            type(i) is int and 0 <= i < task["index"] for i in dependencies
        )
        accepted = {n["index"] for n in nodes if n.get("say")}
        if not valid_dependencies or any(i not in accepted for i in dependencies):
            nodes.append(
                {
                    "index": task["index"],
                    "role": task["role"],
                    "task": task_text,
                    "say": "",
                    "style_flags": [],
                    "status": "dependency_unavailable"
                    if valid_dependencies
                    else "invalid_dependency",
                }
            )
            continue
        hard = _pick(hard_facts, task.get("hard"))
        soft = _pick(soft_facts, task.get("soft"))
        allowed_text = (
            "hard facts：\n"
            + ("\n".join(f"[H{i}] {text}" for i, text in hard) or "（无）")
            + "\n\nsoft facts：\n"
            + ("\n".join(f"[S{i}] {text}" for i, text in soft) or "（无）")
        )
        messages = [
            {
                "role": "system",
                "content": ph.COMMON_PREFIX
                + "\n\n"
                + CHILD_RULES
                + ("\n\n" + resident if resident else "")
                + ("\n\n" + tier_tail if tier_tail else "")
                + "\n\nboundary facts（必须遵守）：\n"
                + boundary_text,
            },
            {
                "role": "user",
                "content": (
                    f"【本次用途】\n{TURN_PURPOSES.get(turn_kind, TURN_PURPOSES['reply'])}\n\n"
                    f"{('【本轮状态】' + chr(10) + tier_state + chr(10) + chr(10)) if tier_state else ''}"
                    f"{handoff + chr(10) + chr(10) if handoff else ''}"
                    f"【当前语境（仅供理解，不扩大事实许可）】\n{context or fmt_history(history or [])}\n\n"
                    f"【本轮目标】\n{goal or TURN_PURPOSES.get(turn_kind, TURN_PURPOSES['reply'])}\n\n"
                    f"【本轮风格】\n{style}\n\n【本次触发内容】\n{user_text}\n\n"
                    f"【整轮分工（下标从0开始）】\n{division}\n\n"
                    f"【此前实际已发送，本次只续接】\n{json.dumps(sent or [], ensure_ascii=False)}\n\n"
                    f"【本轮前面已生成、尚未发送的草稿（无实际发送时间）】\n{json.dumps(previous, ensure_ascii=False)}\n\n"
                    f"【前序任务结果】\n{json.dumps([{'index': n['index'], 'status': n['status']} for n in nodes], ensure_ascii=False)}\n\n"
                    f"【本条事实许可】\n{allowed_text}\n\n"
                    f"【你的职责】第{task['index']}条：{task['role']}\n"
                    f"这一条的任务：{task_text}\n衔接：{task.get('handoff') or '独立成句，顺着现有话题，不重新开场'}"
                ),
            },
        ]
        node = {
            "index": task["index"],
            "role": task["role"],
            "task": task_text,
            "say": "",
            "hard_used": hard,
            "soft_used": soft,
            "style_flags": [],
        }
        writer_prompt = messages[-1]["content"]
        style_retries = 0
        while calls["n"] < ph.MAX_CALLS_PER_TURN:
            calls["n"] += 1
            try:
                content, u, dt = ph.call(
                    client,
                    base,
                    key,
                    messages,
                    ph.CHILD_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    time_context=time_context,
                    kind=("child_repair" if node.get("rewritten") else "child"),
                    retry_context="style"
                    if style_retries
                    else "duplicate"
                    if node.get("rewritten")
                    else "",
                )
            except Exception as exc:  # noqa: BLE001
                node.update(status="error", error=str(exc))
                break
            ph.add_usage(usage, u)
            node["latency"] = node.get("latency", 0) + dt
            try:
                obj = ph.parse_json(content)
                if not isinstance(obj, dict) or not isinstance(obj.get("say"), str):
                    raise ValueError("写手输出缺少文本 say")
                say = obj["say"].strip()
            except Exception:
                node.update(status="error", error="invalid_writer_output")
                break
            _raw_say = say
            say = clean(say)
            if not say and _raw_say:
                say = re.sub(r"\n{3,}", "\n\n", _raw_say).strip()

            tier_mark = str(obj.get("tier") or "").strip()
            if tier_mark in TIERS:
                node["tier"] = tier_mark
            if say == "[[NO_MESSAGE]]":
                say = ""
            duplicate = bool(say) and _message_key(say) in {
                _message_key(x) for x in [*(sent or []), *previous]
            }
            breaches = style_breaches(say, h1_before=h1_seen)
            node["style_flags"] = breaches
            node.setdefault("style_attempts", []).append(breaches)

            hard_breaches = [
                flag for flag in breaches if flag.startswith("H") and flag != "H3"
            ]
            if hard_breaches:
                node["status"] = "style_rejected"
                node["rewritten"] = True
                logging.getLogger(__name__).warning(
                    "[palacer] style_retry task=%d attempt=%d flags=%s",
                    task["index"],
                    len(node["style_attempts"]),
                    hard_breaches,
                )

                remaining = len(assignments) - position - 1
                if (
                    style_retries >= 1
                    or calls["n"] + remaining >= ph.MAX_CALLS_PER_TURN
                ):
                    break
                style_retries += 1
                messages[-1]["content"] = (
                    writer_prompt
                    + "\n\n【修正要求】候选句踩了："
                    + "、".join(STYLE_NAMES[f] for f in hard_breaches)
                    + "。\n按所选 persona 的语气重写，别只是删掉那几个词。出现时按第一条判据检查是否已经说过同样意思。"
                    + "\n刚才的候选："
                    + json.dumps(say, ensure_ascii=False)
                )
                continue
            if duplicate:
                node["status"] = "duplicate"

                remaining = len(assignments) - position - 1
                if not repair_used and calls["n"] + remaining < ph.MAX_CALLS_PER_TURN:
                    repair_used = True
                    node["rewritten"] = True
                    if duplicate:
                        messages[-1]["content"] = writer_prompt + (
                            "\n\n【修正要求】刚才的候选与已发送内容或前序草稿重复："
                            + json.dumps(say, ensure_ascii=False)
                            + "。重新完成本条独有任务，增加新的意思；不要仅换措辞。"
                            + '本条已无新内容就输出 {"say":""}。'
                        )
                    continue
                break
            node.update(say=say, status="accepted" if say else "empty")
            if say:
                h1_seen += h1_count(say)
                previous.append(say)
            break
        nodes.append(node)
    return nodes
