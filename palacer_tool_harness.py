"""Palacer palacer tool harness."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
try:
    from .runtime_config import integration_dir, integration_path, setting
except ImportError:
    from runtime_config import integration_dir, integration_path, setting


TOOL_CATALOG = """可用工具：
1. book_quote(book_id, query)：在已读范围里找原文。book_id 用 book_list 里的 id。
2. book_associate(query)：在所有书的已读摘要里找相似内容。
3. book_read(book_id?, chunks?)：现在就读下一块（1–2 块），返回这一块的摘要、摘句、当时的反应和进度。
   用户要求继续阅读且已配置兼容阅读集成时可用；实际可用时间由集成决定。"""

TOOL_PLANNER = (
    """你是所选角色的工具调度器，不是所选角色本人。只判断这一轮是否需要工具，不要写正文。
"""
    + TOOL_CATALOG
    + """
输出 JSON：
{"tool_calls":[{"name":"book_quote","args":{"book_id":"book-example","query":"沉默"}}]}
规则：
1. 最多 2 个工具调用；不需要就输出 {"tool_calls":[]}。
2. 只能使用列出的工具，不要编 book_id。
3. 只输出一个 JSON 对象，不要代码围栏。"""
)

REPLY_PLANNER = """你是所选角色的对话调度器，不是所选角色本人。
根据角色资料、当前状态、用户消息和工具结果，输出这一轮的风格卡、工具结果蒸馏和子任务。
输出 JSON：
{"style":"1-2句，这一轮的语气、节奏、边界和禁止项","tool_context":"把工具结果蒸馏成1-2句给写手；没有就空字符串","children":[{"task":"这一条要做什么，具体"}]}
规则：
1. tool_context 只能写工具结果里真实存在的信息，不要编。
2. children 1-3 条。
3. 只输出一个 JSON 对象，不要代码围栏。"""


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from . import palacer_harness as ph
except ImportError:
    import palacer_harness as ph


def media_core():
    """Load the optional integration only when a book tool is invoked."""
    path = integration_dir("media") / "core.py"
    if not path.is_file():
        raise RuntimeError("未配置兼容的阅读集成")
    return load("palacer_media_core", path)


def load_book_data():
    chunks = [
        json.loads(x)
        for x in (
            integration_path("media_data")
            / "books"
            / str(setting("default_book_id", ""))
            / "chunks.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    progress = json.loads(
        (integration_path("media_data") / "progress.json").read_text(encoding="utf-8")
    )
    sessions = [
        json.loads(x)
        for x in (
            integration_path("media_data")
            / "books"
            / str(setting("default_book_id", ""))
            / "sessions.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    catalog = [
        json.loads(x)
        for x in (integration_path("media_data") / "catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    title = next(
        (
            b.get("title")
            for b in catalog
            if b.get("book_id") == str(setting("default_book_id", ""))
        ),
        str(setting("default_book_id", "")),
    )
    return chunks, progress, sessions, title


_BOOK_CACHE = None


def book_data():
    global _BOOK_CACHE
    if _BOOK_CACHE is None:
        _BOOK_CACHE = load_book_data()
    return _BOOK_CACHE


def tool_book_quote(args):
    query = str(args.get("query") or "").strip()
    bid = str(args.get("book_id") or str(setting("default_book_id", "")))
    if not query:
        return "缺少 query。"
    chunks, progress, _sessions, title = book_data()
    hits = media_core().quote_search(chunks, progress, bid, query, k=2)
    if not hits:
        return "读过的部分里没找到。"
    return "\n\n".join(
        f"《{title}》{h.get('chapter_title', '')}（块 {h.get('chunk_index')}）：{str(h.get('text', ''))[:220]}"
        for h in hits
    )


def tool_book_associate(args):
    query = str(args.get("query") or "").strip()
    if not query:
        return "缺少 query。"
    _chunks, _progress, sessions, title = book_data()
    scored = []
    for s in sessions:
        summary = str(s.get("summary") or "")
        if not summary:
            continue
        score = (
            100 if query in summary else sum(1 for ch in set(query) if ch in summary)
        )
        if score >= max(2, len(set(query)) // 2):
            scored.append((score, s))
    scored.sort(key=lambda x: (x[0], int(x[1].get("ts") or 0)), reverse=True)
    if not scored:
        return "已读摘要里没找到。"
    return "\n\n".join(
        f"《{title}》{s.get('chapter_title', '')}（块 {s.get('chunk_from')}-{s.get('chunk_to')}）：{s.get('summary')}"
        for _, s in scored[:3]
    )


TOOLS = {"book_quote": tool_book_quote, "book_associate": tool_book_associate}


def clean_message(text):
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


__all__ = ["ph", "TOOLS", "TOOL_CATALOG", "TOOL_PLANNER"]
