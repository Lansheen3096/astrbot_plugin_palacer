"""Palacer context bundle."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

try:
    from .runtime_config import (
        DATA,
        PLUGIN_DATA,
        integration_dir,
        integration_path,
        setting,
    )
except ImportError:
    from runtime_config import (
        DATA,
        PLUGIN_DATA,
        integration_dir,
        integration_path,
        setting,
    )


RERANK_MAX_DOCS = int(os.environ.get("PALACER_RERANK_MAX_DOCS") or 80)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_persona(persona_id: str = "") -> str:
    """Load an explicitly selected persona without a private fallback."""
    if not persona_id:
        return ""
    con = sqlite3.connect(f"file:{DATA / 'data_v4.db'}?mode=ro", uri=True)
    try:
        row = con.execute(
            "select system_prompt from personas where persona_id=?", (persona_id,)
        ).fetchone()
        return str(row[0]).strip() if row else ""
    finally:
        con.close()


def load_state_card(umo: str) -> str:
    path = integration_dir("state") / "core.py"
    if not path.is_file():
        return ""
    sc = _load("bundle_state_core", path)
    snap = sc.collect_snapshot(DATA, umo)
    return sc.compose_context(sc.build_sections(snap), 700)


def _ledger(name: str, row: dict, max_mb: float = 8.0) -> None:
    """召回/反思账本：一行一条，超过上限就滚动一份 .1 备份。写入失败绝不影响主流程。"""
    if not setting("trace", False):
        return
    PLUGIN_DATA.mkdir(parents=True, exist_ok=True)
    path = PLUGIN_DATA / name
    try:
        row.setdefault("ts", round(time.time(), 3))
        if path.exists() and path.stat().st_size > max_mb * 1024 * 1024:
            path.replace(path.with_suffix(".jsonl.1"))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_reflections(umo: str, user_text: str, max_items: int = 2) -> list[str]:
    if not (integration_dir("review") / "main.py").is_file():
        return []
    sys.path.insert(0, str(integration_dir("review")))
    try:
        rm = _load("bundle_review_main", integration_dir("review") / "main.py")
    finally:
        try:
            sys.path.pop(0)
        except Exception:
            pass
    log_path = integration_path("review_log")
    rows = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception:
        return []
    recent = [x for r in rows[-5:] for x in r.get("reflections", [])]
    recent = [x for x in recent if not rm.is_pure_event_reflection(x)]
    hits = rm.pick_relevant(user_text, recent, k=max_items)

    day_by_text = {}
    for r in rows[-5:]:
        for x in r.get("reflections", []):
            day_by_text[str(x.get("text") or "")] = r.get("day")
    out = []
    for h in hits:
        h = dict(h)
        h.setdefault("day", day_by_text.get(str(h.get("text") or "")))
        out.append(rm.format_reflection_line(h, time_day()))
    _ledger(
        "palacer_reflection_log.jsonl",
        {
            "umo": umo,
            "query": str(user_text or "")[:200],
            "candidates": len(recent),
            "picked": len(hits),
            "picked_text": [str(h.get("text") or "")[:120] for h in hits],
            "picked_days": [day_by_text.get(str(h.get("text") or "")) for h in hits],
        },
    )
    return out


def time_day() -> str:
    import time

    return time.strftime("%Y-%m-%d")


def load_health() -> str:
    if not (integration_dir("health") / "main.py").is_file():
        return ""
    try:
        hm = _load("bundle_health_main", integration_dir("health") / "main.py")
        obj = object.__new__(hm.Main)
        obj._cache = json.loads(
            (integration_path("health_cache")).read_text(encoding="utf-8")
        )
        obj._state = json.loads(
            (integration_path("health_state")).read_text(encoding="utf-8")
        )
        block, _fp, _tag = obj._compose()
        return block or ""
    except Exception as exc:  # noqa: BLE001
        return f"（health 读取失败：{type(exc).__name__}: {exc}）"


def extract_health_facts(health_block: str) -> list[dict]:
    if "还没同步上来" in (health_block or ""):
        return [{"type": "hard", "source": "health.stage", "text": "还没同步上来"}]
    out = []
    for line in (health_block or "").splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append({"type": "hard", "source": "health", "text": line[2:].strip()})
    return out


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _preselect(query: str, pool: list[dict], limit: int) -> list[int]:
    """粗筛：字符 bigram 重合度排序后取前 limit 个（保留原下标）。

    只为了把 rerank 的输入从几百条压到几十条；排序质量由后面的 rerank 决定。
    """

    def grams(text: str) -> set[str]:
        t = str(text or "")
        if len(t) < 2:
            return {t} if t else set()
        return {t[i : i + 2] for i in range(len(t) - 1)}

    qg = grams(query)
    scored = []
    for index, item in enumerate(pool):
        dg = grams(item.get("text"))
        if not qg or not dg:
            score = 0.0
        else:
            score = len(qg & dg) / len(qg | dg)
        scored.append((score, -index, index))
    scored.sort(reverse=True)
    return sorted({index for _score, _neg, index in scored[:limit]})


def _rerank(query: str, documents: list[str]) -> list[dict]:
    if not documents or not setting("rerank_url", ""):
        return []
    body = json.dumps(
        {
            "model": "bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        str(setting("rerank_url", "")),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("results") or []


def _mem_decay(
    event_date: str | None, halflife_days: float, decay_floor: float
) -> float:
    if not event_date:
        return 1.0
    try:
        days = (date.today() - date.fromisoformat(str(event_date))).days
    except Exception:
        return 1.0
    if days <= 0:
        return 1.0
    return max(decay_floor, 0.5 ** (days / halflife_days))


def _memory_terms(query):
    try:
        from astrbot.core.knowledge_base.retrieval import tokenizer
        from pathlib import Path

        stops = tokenizer.load_stopwords(
            Path(tokenizer.__file__).with_name("hit_stopwords.txt")
        )
        return list(dict.fromkeys(tokenizer.tokenize_text(query, stops)))[:64]
    except ImportError:
        return list(dict.fromkeys(re.findall(r"\w+", query.lower())))[:64]


def _memory_candidates(con, query, limit=96, dense_ids=()):
    """Reuse the FTS index; reserve recent preferences/events before reranking."""
    limit = max(1, min(int(limit), 256))
    md = "CASE WHEN json_valid(d.metadata) THEN d.metadata ELSE '{}' END"
    kind = f"coalesce(json_extract({md}, '$.kind'), 'event')"
    eligible = f"coalesce(json_extract({md}, '$.suspect'), 0) NOT IN (1, '1', 'true')"
    states = con.execute(
        f"SELECT d.text, d.metadata FROM documents d WHERE {eligible} AND {kind}='state' ORDER BY d.id"
    ).fetchall()
    pool_where = f"{eligible} AND {kind} IN ('preference','event')"
    terms = _memory_terms(query)
    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
    rows = []
    reserve = max(2, limit // 4)
    if match:
        try:
            rows = con.execute(
                f"""SELECT d.id, d.text, d.metadata FROM documents_fts
                JOIN documents d ON d.id=documents_fts.rowid
                WHERE documents_fts MATCH ? AND {pool_where}
                ORDER BY bm25(documents_fts), d.id LIMIT ?""",
                (match, max(1, limit - reserve)),
            ).fetchall()
        except Exception:
            pool = con.execute(
                f"SELECT d.id,d.text,d.metadata FROM documents d WHERE {pool_where}"
            ).fetchall()
            rows = sorted(
                pool,
                key=lambda r: (-sum(t.lower() in r[1].lower() for t in terms), -r[0]),
            )[: max(1, limit - reserve)]
    dense = []
    ids = list(dict.fromkeys(int(i) for i in dense_ids))[:40]
    if ids:
        dense_rows = con.execute(
            f"SELECT d.id,d.text,d.metadata FROM documents d WHERE {pool_where} AND d.id IN ({','.join('?' for _ in ids)})",
            ids,
        ).fetchall()
        by_id = {row[0]: row for row in dense_rows}
        dense = [by_id[i] for i in ids if i in by_id]
    selected = {row[0]: row for row in dense[: max(1, limit - reserve)]}
    for row in rows:
        if len(selected) < limit - reserve:
            selected.setdefault(row[0], row)
    for category in ("preference", "event"):
        recent = con.execute(
            f"SELECT d.id,d.text,d.metadata FROM documents d WHERE {pool_where} AND {kind}=? ORDER BY d.id DESC LIMIT ?",
            (category, max(1, reserve // 2)),
        ).fetchall()
        for row in recent:
            if len(selected) < limit:
                selected.setdefault(row[0], row)
    if len(selected) < limit:
        recent = con.execute(
            f"SELECT d.id,d.text,d.metadata FROM documents d WHERE {pool_where} ORDER BY d.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        for row in recent:
            if len(selected) < limit:
                selected.setdefault(row[0], row)
    return [*states, *((r[1], r[2]) for r in selected.values())]


def load_memory(user_text: str, candidate_ids=None, umo: str = "") -> list[dict]:
    """Reuse the memory_recall rules against the configured KB."""
    try:
        cfg = _read_json(
            DATA / "config" / "astrbot_plugin_memory_recall_config.json", {}
        )
        kb_name = str(setting("memory_kb_name", "") or cfg.get("kb_name") or "")
        if not kb_name:
            return []
        floor = float(cfg.get("floor", 0.05))
        lead = float(cfg.get("lead", 2.0))
        state_max = int(cfg.get("state_max", 40))
        recall_max = int(cfg.get("recall_max", 3))
        halflife = float(cfg.get("halflife_days", 30.0))
        decay_floor = float(cfg.get("decay_floor", 0.35))
        new_discount = float(cfg.get("new_discount", 0.5))

        kb_db = DATA / "knowledge_base" / "kb.db"
        con = sqlite3.connect(f"file:{kb_db}?mode=ro", uri=True)
        try:
            row = con.execute(
                "select kb_id from knowledge_bases where kb_name=?", (kb_name,)
            ).fetchone()
        finally:
            con.close()
        if not row:
            return []
        doc_db = DATA / "knowledge_base" / str(row[0]) / "doc.db"
        con = sqlite3.connect(f"file:{doc_db}?mode=ro", uri=True)
        try:
            rows = (
                con.execute("select text, metadata from documents").fetchall()
                if candidate_ids is None
                else _memory_candidates(
                    con, user_text, limit=96, dense_ids=candidate_ids
                )
            )
        finally:
            con.close()

        items = []
        for text, md in rows:
            try:
                md = json.loads(md or "{}")
            except Exception:
                md = {}
            if md.get("suspect") in (1, "1", True):
                continue
            items.append(
                {
                    "text": str(text or ""),
                    "kind": md.get("kind") or "event",
                    "event_date": md.get("event_date"),
                    "status": md.get("status"),
                }
            )
        states = [it for it in items if it["kind"] == "state"][:state_max]
        pool = [it for it in items if it["kind"] in ("preference", "event")]

        recalled = []

        query = ""
        presel_n = 0
        rerank_n = 0
        top1 = top2 = 0.0
        passed = False
        drop_reason = "no_pool" if not pool else ""
        score_by_text = {}
        if pool:
            try:
                mr = _load(
                    "bundle_memory_recall",
                    DATA / "plugins" / "astrbot_plugin_memory_recall" / "main.py",
                )
                query = mr.clean_query(user_text)
            except Exception:
                query = user_text
            if query:
                try:
                    results = []

                    sel = list(range(len(pool)))
                    if len(pool) > RERANK_MAX_DOCS:
                        sel = _preselect(query, pool, RERANK_MAX_DOCS)
                    presel_n = len(sel)
                    for offset in range(0, len(sel), 16):
                        chunk = sel[offset : offset + 16]
                        batch = _rerank(query, [pool[i]["text"] for i in chunk])
                        results.extend(
                            {**r, "index": chunk[int(r["index"])]}
                            for r in batch
                            if 0 <= int(r.get("index") or -1) < len(chunk)
                        )
                    rerank_n = len(results)
                except Exception as exc:  # noqa: BLE001
                    results = []
                    drop_reason = f"rerank_failed:{type(exc).__name__}"
                scored = []
                for r in results:
                    try:
                        i = int(r.get("index"))
                    except Exception:
                        continue
                    if not (0 <= i < len(pool)):
                        continue
                    it = dict(pool[i])
                    s = float(r.get("relevance_score") or 0.0)
                    if it["kind"] == "event":
                        s *= _mem_decay(it["event_date"], halflife, decay_floor)
                    if it["status"] == "new":
                        s *= new_discount
                    scored.append((s, it))
                scored.sort(key=lambda x: -x[0])
                score_by_text = {
                    it["text"]: round(s, 4) for s, it in scored[:recall_max]
                }
                if scored:
                    top1 = scored[0][0]
                    top2 = scored[1][0] if len(scored) > 1 else 0.0
                    if top1 >= floor and top1 >= lead * top2:
                        passed = True
                        recalled = [it for s, it in scored[:recall_max] if s >= floor]
                    else:
                        drop_reason = "below_floor" if top1 < floor else "not_leading"
                else:
                    drop_reason = "no_scored"
        out = [
            {"type": "hard", "source": "memory.state", "text": it["text"]}
            for it in states
        ]
        out += [
            {
                "type": "hard",
                "source": f"memory.{it['kind']}",
                "text": it["text"],
            }
            for it in recalled
        ]
        _ledger(
            "palacer_recall_log.jsonl",
            {
                "umo": umo,
                "query": str(query or "")[:200],
                "pool": len(pool),
                "states": len(states),
                "preselect": presel_n,
                "reranked": rerank_n,
                "top1": round(top1, 4),
                "top2": round(top2, 4),
                "floor": floor,
                "lead": lead,
                "passed": passed,
                "injected": [
                    {
                        "text": str(it["text"])[:160],
                        "kind": it["kind"],
                        "event_date": it.get("event_date"),
                        "score": score_by_text.get(it["text"]),
                    }
                    for it in recalled
                ],
                "drop": drop_reason,
            },
        )
        return out
    except Exception:
        return []


BOOK_INTENT = ("书", "读", "故事", "内容", "写了", "讲了", "剧情", "章节", "卷")


READING_FRESH_HOURS = 6
READING_QUOTE_DAYS = 30


def _book_overview(rows: list[dict], progress: dict, title: str, limit: int = 8) -> str:
    """按章取每章最近一条摘要；章数超过 limit 时均匀采样，保住开头和结尾。"""
    by_chapter = {}
    for r in rows:
        try:
            idx = int(r.get("chapter_index"))
        except (TypeError, ValueError):
            continue
        summary = str(r.get("summary") or "").strip()
        if not summary:
            continue
        label = str(r.get("chapter_title") or f"第{idx}章")
        by_chapter[idx] = (label, summary)
    idxs = sorted(by_chapter)
    if len(idxs) > limit:
        step = len(idxs) / limit
        idxs = sorted({idxs[min(len(idxs) - 1, int(i * step))] for i in range(limit)})
    return "\n".join(f"[{by_chapter[i][0]}] {by_chapter[i][1][:70]}" for i in idxs)


def load_reading(user_text: str) -> list[dict]:
    """读书事实：每本书各自一条（最多 3 本），「想读但没到手」单独标出来。

    以前只取全场最新的一条会话 —— 两本书交替读的时候，A 的情节会被说成 B 的。
    现在按书分组，每本自己一条，书名写在前面。
    """
    try:
        sessions: dict[str, list[dict]] = {}
        for path in (integration_path("media_data") / "books").glob("*/sessions.jsonl"):
            try:
                rows = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            except Exception:
                continue
            if rows:
                bid = str(rows[-1].get("book_id") or path.parent.name)
                sessions[bid] = rows

        def _has_body(x):
            summary = str(x.get("summary") or "")
            return bool(summary) and not any(
                k in summary for k in ("没有正文", "无正文", "只有卷名", "没有任何叙述")
            )

        progress = (
            _read_json(integration_path("media_data") / "progress.json", {}) or {}
        )
        books = progress.get("books") or {}

        total_chunks: dict[str, int] = {}
        for bid in sessions:
            try:
                chunk_file = (
                    integration_path("media_data") / "books" / bid / "chunks.jsonl"
                )
                with chunk_file.open(encoding="utf-8") as f:
                    total_chunks[bid] = sum(1 for line in f if line.strip())
            except Exception:
                total_chunks[bid] = 0

        order = sorted(
            sessions,
            key=lambda b: max(float(x.get("ts") or 0) for x in sessions[b]),
            reverse=True,
        )[:3]

        book_intent = any(k in (user_text or "") for k in BOOK_INTENT)
        keep = order if book_intent else order[:1]

        out: list[dict] = []
        titles: dict[str, str] = {}
        for bid in keep:
            rows = [x for x in sessions[bid] if _has_body(x)] or list(sessions[bid])
            rows.sort(key=lambda x: float(x.get("ts") or 0))
            last = rows[-1]
            title = str(last.get("title") or "书")
            titles[bid] = title
            st = books.get(bid) or {}
            done = int(st.get("chunk_index") or 0)
            total = total_chunks.get(bid) or 0
            head = f"《{title}》："
            if total:
                head += f"进度 {done}/{total}（{100 * done / total:.0f}%），"
                head += "这本已经读完了" if st.get("finished") else "还没读完"
                head += "；"
            age_h = (time.time() - float(last.get("ts") or 0)) / 3600
            if age_h <= READING_FRESH_HOURS:
                when = "刚读过"
            elif age_h <= 24:
                when = "今天早些时候读过"
            else:
                when = f"{age_h / 24:.0f} 天前读过"
            parts = [f"{head}{when}{str(last.get('chapter_title') or '')}。"]
            summary = str(last.get("summary") or "").strip()
            if summary:
                parts.append(f"那一段是：{summary[:140]}")
            if last.get("quote_is_striking") and last.get("quote"):
                parts.append(f"她记住的一句：{str(last.get('quote'))[:100]}")
            reaction = str(last.get("reaction") or "").strip()
            if reaction:
                parts.append(f"当时的反应：{reaction[:80]}")
            out.append(
                {"type": "hard", "source": "reading.session", "text": " ".join(parts)}
            )

        wish_state = (
            _read_json(integration_path("media_data") / "wish_state.json", {}) or {}
        )
        active = wish_state.get("active") or {}
        wish_title = (
            str(active.get("title") or "") if active.get("status") == "pending" else ""
        )
        unread_named = (
            bool(wish_title)
            and wish_title in (user_text or "")
            and wish_title not in {t for t in titles.values() if t}
        )
        if keep and not unread_named:
            named = next(
                (b for b in keep if titles.get(b) and titles[b] in (user_text or "")),
                None,
            )
            want = named
            if want is None and book_intent:
                want = keep[0]
            if want:
                rows = [x for x in sessions[want] if _has_body(x)] or list(
                    sessions[want]
                )
                overview = _book_overview(
                    rows, books.get(want) or {}, titles.get(want) or "书"
                )
                if overview:
                    out.append(
                        {
                            "type": "hard",
                            "source": "reading.overview",
                            "text": f"《{titles.get(want) or '书'}》她读过的部分按章顺下来是（只这一本）：\n"
                            + overview,
                        }
                    )

        if active.get("status") == "pending" and active.get("title"):
            bid = str(active.get("book_id") or "")
            if not (
                integration_path("media_data") / "books" / bid / "chunks.jsonl"
            ).exists():
                out.append(
                    {
                        "type": "hard",
                        "source": "reading.wish",
                        "text": (
                            f"她最近想去读《{active['title']}》，这本还没到手 —— 她只知道自己想读，"
                            "没读过任何一页，不要当成读过，也不要编这本书的情节。"
                        ),
                    }
                )
        return out
    except Exception:
        return []


def load_life(umo: str, user_text: str) -> list[dict]:
    """Reuse integration_life's current state, fresh event, wish and weather."""
    try:
        state = _read_json(integration_path("life_state"), {})
        st = state.get(umo) if isinstance(state, dict) else None
        if not isinstance(st, dict):
            return []
        out = []
        cur = str(st.get("state") or "SOLO")
        away = (
            bool((st.get("away") or {}).get("away"))
            if isinstance(st.get("away"), dict)
            else False
        )
        state_cn = {
            "SOLO": "他暂时没有和你互动，可能仍在家，你自己待着",
            "NEAR": "用户当前在电脑前",
            "CHATTING": "你正跟他说着话",
            "DROWSY": "状态集成报告低活跃状态",
            "BLACKOUT": "状态集成报告中断恢复状态",
        }.get(cur, "你自己待着")
        if cur == "SOLO" and away:
            state_cn = "他可能刚刚出门，也可能仍在家休息；去向尚未确认，你自己待着"
        out.append({"type": "hard", "source": "life.state", "text": state_cn})

        now = time.time()
        events = []
        try:
            for line in (
                (integration_path("life_events"))
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        except Exception:
            events = []
        for ev in reversed(events):
            if ev.get("umo") != umo or ev.get("archived"):
                continue
            if ev.get("kind") in ("gap", "digest"):
                continue
            age = now - float(ev.get("ts") or now)
            if age > 90 * 60:
                break
            minutes = max(0, int(age // 60))
            out.append(
                {
                    "type": "soft",
                    "source": "life.event",
                    "text": f"{minutes} 分钟前的模型生成体验（同一事件重复读取不算新证据，不证明用户现状）：{ev.get('text', '')}",
                }
            )
            break

        active = st.get("active")
        if isinstance(active, dict) and active.get("title"):
            reason = str(active.get("reason") or "").strip()
            out.append(
                {
                    "type": "hard",
                    "source": "life.wish",
                    "text": f"你最近想读《{active['title']}》。{reason} 还没跟他说。",
                }
            )

        try:
            w = _load("bundle_life_weather", integration_dir("life") / "weather.py")
            sig = _load("bundle_life_signals", integration_dir("life") / "signals.py")
            if w.match_keyword(user_text):
                facts, _src = sig.read_snapshot()
                block = w.from_snapshot(facts)
                if block:
                    out.append(
                        {
                            "type": "hard",
                            "source": "life.weather",
                            "text": block.replace("【此刻外面的天气】", "").strip(),
                        }
                    )
        except Exception:
            pass
        return out
    except Exception:
        return []


def extract_state_facts(state_card: str) -> list[dict]:
    out = []
    for m in re.finditer(r"<reading>(.*?)</reading>", state_card or "", re.S):
        text = re.sub(r"\s+", " ", m.group(1)).strip()
        if text:
            out.append({"type": "hard", "source": "state.reading", "text": text})
    return out


def extract_boundary_facts(state_card: str, persona: str) -> list[dict]:
    boundaries = []
    for m in re.finditer(r"<self>(.*?)</self>", state_card or "", re.S):
        text = re.sub(r"\s+", " ", m.group(1)).strip()
        provisional = text.startswith(
            ("当次体验", "暂定认识", "既有认识（来源未复核）")
        )
        if (
            text
            and not provisional
            and any(
                k in text for k in ("碰不到", "不能", "不要", "只能", "不是", "边界")
            )
        ):
            boundaries.append(
                {"type": "boundary", "source": "state.self", "text": text}
            )
    fixed = [
        "身份、称呼、关系和行为边界按所选 persona，不预设角色身份。",
        "不要编造没读过的原文、没发生过的事件、没记住的细节；不确定就说不确定。",
        "健康信息仅在与当前话题相关且用户已提供时使用。",
        "图片和文件只能依据附件观察；观察里没有的细节不能猜，看不清就说看不清。",
        "用户发来的截图/文件是直接证据；和健康缓存冲突时以截图/文件为准，并说明缓存可能没同步。",
        "语音只能依据转写内容；听不清就说听不清，不要猜语音里说了什么。",
        "引用消息只能依据引用内容；引用里没有的细节不要补。",
        "合并转发只能依据展开的内容；无法展开就说无法展开，不要猜。",
    ]
    for text in fixed:
        if all(text != x["text"] for x in boundaries):
            boundaries.append({"type": "boundary", "source": "persona", "text": text})
    return boundaries


def build_bundle(
    umo: str,
    user_text: str,
    include_health: bool = True,
    persona_id: str = "",
    memory_candidate_ids=None,
) -> dict:
    persona = load_persona(persona_id)
    state_card = load_state_card(umo)
    reflections = load_reflections(umo, user_text)
    health = load_health() if include_health else ""
    life_facts = load_life(umo, user_text)
    hard_facts = (
        extract_health_facts(health)
        + extract_state_facts(state_card)
        + load_memory(user_text, candidate_ids=memory_candidate_ids, umo=umo)
        + load_reading(user_text)
        + [f for f in life_facts if f.get("type") == "hard"]
    )
    soft_facts = [
        {"type": "soft", "source": "review", "text": x} for x in reflections
    ] + [f for f in life_facts if f.get("type") == "soft"]
    boundary_facts = extract_boundary_facts(state_card, persona)
    return {
        "meta": {
            "umo": umo,
            "user_text": user_text,
            "persona_id": persona_id,
            "ts": int(time.time()),
        },
        "persona": persona,
        "state_card": state_card,
        "reflections": reflections,
        "health": health,
        "hard_facts": hard_facts,
        "soft_facts": soft_facts,
        "boundary_facts": boundary_facts,
        "chars": {
            "persona": len(persona),
            "state_card": len(state_card),
            "reflections": sum(len(x) for x in reflections),
            "health": len(health),
            "hard_facts": sum(len(x["text"]) for x in hard_facts),
            "soft_facts": sum(len(x["text"]) for x in soft_facts),
            "boundary_facts": sum(len(x["text"]) for x in boundary_facts),
        },
    }


def render_bundle(bundle: dict) -> str:
    parts = []
    if bundle.get("persona"):
        parts.append("【persona】\n" + bundle["persona"])
    if bundle.get("state_card"):
        parts.append("【state_card】\n" + bundle["state_card"])
    hard = bundle.get("hard_facts") or []
    if hard:
        parts.append(
            "【hard_facts｜外部/身体事实】\n"
            + "\n".join(f"[H{i}] {x['text']}" for i, x in enumerate(hard))
        )
    soft = bundle.get("soft_facts") or []
    if soft:
        parts.append(
            "【soft_facts｜角色的想法，不是外部事实】\n"
            + "\n".join(f"[S{i}] {x['text']}" for i, x in enumerate(soft))
        )
    boundary = bundle.get("boundary_facts") or []
    if boundary:
        parts.append(
            "【boundary_facts｜必须遵守】\n"
            + "\n".join(f"[B{i}] {x['text']}" for i, x in enumerate(boundary))
        )
    return "\n\n".join(parts)
