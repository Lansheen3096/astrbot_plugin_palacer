"""Palacer palacer session."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

try:
    from .message_time import Clock, TrackedText
except ImportError:
    from message_time import Clock, TrackedText

_log = logging.getLogger("palacer.session")

TAIL_PERIOD = re.compile(r"。(?=.{0,2}\Z)", re.DOTALL)


@dataclass
class SessionConfig:
    startup_delay: float = 15.0
    message_window: float = 5.0


class PalacerSession:
    def __init__(
        self,
        generate: Callable[[list[str], list[str]], Awaitable[list[str]]],
        send: Callable[[str], Awaitable[bool | None]],
        config: SessionConfig | None = None,
        on_complete: Callable[[list[str], list[str]], Awaitable[None]] | None = None,
        clock=None,
        prepare=None,
    ) -> None:
        self.generate = generate
        self.send = send
        self.prepare = prepare
        self.on_complete = on_complete
        self.config = config or SessionConfig()
        self.pending_user: list[str] = []
        self.inflight_batch: list[str] = []
        self.sent: list[str] = []
        self.gen_id = 0
        self.task: asyncio.Task | None = None
        self.generate_calls: list[tuple[list[str], list[str]]] = []
        self.sent_at: list[tuple[str, float]] = []
        self.clock = clock or Clock()
        self.preparations: dict[str, asyncio.Future] = {}

    async def on_user_message(
        self, text: str, preparation: asyncio.Future | None = None
    ) -> None:
        """Debounce; cancel any in-flight generation/plan."""
        busy = self.task is not None and not self.task.done()
        if busy and self.inflight_batch:
            self.pending_user = list(self.inflight_batch) + self.pending_user
            self.inflight_batch = []
        if not busy:
            self.sent.clear()
        self.gen_id += 1
        gen = self.gen_id
        previous = self.task
        if previous is not None and not previous.done() and not previous.cancelling():
            previous.cancel()
        self.pending_user.append(text)
        if preparation is not None:
            self.preparations[text.message_id] = preparation

        if previous is not None and not previous.done():
            await asyncio.gather(previous, return_exceptions=True)
        if gen == self.gen_id:
            self.task = asyncio.create_task(self._startup(gen))
            self.task.add_done_callback(self._watch)

    @staticmethod
    def _watch(task: asyncio.Task) -> None:
        """Log a turn task that dies with an exception instead of failing silently."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _log.error("[palacer] 回合任务异常退出：%r", exc, exc_info=exc)

    def _cancel_task(self) -> None:
        if self.task and not self.task.done() and not self.task.cancelling():
            self.task.cancel()
        self.task = None

    async def _startup(self, gen: int) -> None:
        try:
            await asyncio.sleep(self.config.startup_delay)
        except asyncio.CancelledError:
            return
        if gen != self.gen_id:
            return
        await self._generate_and_send(gen)

    async def _generate_and_send(self, gen: int) -> None:
        batch = list(self.pending_user)
        self.pending_user.clear()
        self.inflight_batch = list(batch)
        try:
            try:
                prepared = []
                for item in batch:
                    future = self.preparations.get(getattr(item, "message_id", None))
                    value = await asyncio.shield(future) if future is not None else item
                    prepared.append(
                        TrackedText(value, item.message_id)
                        if hasattr(item, "message_id")
                        else value
                    )
                batch = prepared
                self.inflight_batch = list(batch)
                messages = await self.generate(batch, list(self.sent))
                delivery = None
                if self.prepare is not None:
                    messages, delivery = await self.prepare(messages)
            except asyncio.CancelledError:
                return
            if gen != self.gen_id:
                return
            self.generate_calls.append((batch, list(self.sent)))
            _log.info(
                "[palacer] gen=%s 准备发送 %d 条（delivery=%s）",
                gen,
                len(messages),
                type(delivery).__name__,
            )
            for index, msg in enumerate(messages):
                msg = TAIL_PERIOD.sub("", msg)
                if not msg:
                    continue
                try:
                    await asyncio.sleep(
                        0.0 if index == 0 else self.config.message_window
                    )
                except asyncio.CancelledError:
                    return
                if gen != self.gen_id:
                    return
                try:
                    outgoing = (
                        self.send(msg, delivery) if self.prepare else self.send(msg)
                    )
                    sending = asyncio.create_task(outgoing)
                    _log.info(
                        "[palacer] gen=%s 第 %d/%d 条开始发送",
                        gen,
                        index + 1,
                        len(messages),
                    )
                    ok = await asyncio.shield(sending)
                    _log.info(
                        "[palacer] gen=%s 第 %d/%d 条返回 %r",
                        gen,
                        index + 1,
                        len(messages),
                        ok,
                    )
                except asyncio.CancelledError:
                    try:
                        ok = await sending
                    except (Exception, asyncio.CancelledError):
                        return
                    if ok is not False:
                        self.sent.append(msg)
                        self.sent_at.append(
                            (msg, getattr(ok, "monotonic_at", self.clock.monotonic()))
                        )
                    return
                except Exception:
                    break
                if ok is False:
                    break

                self.sent.append(msg)
                self.sent_at.append(
                    (msg, getattr(ok, "monotonic_at", self.clock.monotonic()))
                )
                if gen != self.gen_id:
                    return
        finally:
            if gen == self.gen_id:
                self.inflight_batch = []
                if self.on_complete is not None:
                    await self.on_complete(list(batch), list(self.sent))
                for item in batch:
                    self.preparations.pop(getattr(item, "message_id", None), None)

    async def close(self) -> None:
        self.gen_id += 1
        previous = self.task
        self._cancel_task()
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
