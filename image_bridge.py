"""Palacer image bridge."""

from __future__ import annotations

import asyncio
import hashlib
import json


CATALOG = """
3. image_library(category, query, id)：查人物/服饰/场景/道具参考库。
4. image_generate(prompt, characters, outfits, scene, props, situation, expression, pose, shot, light, aspect, extra_images, user_requested)：
   规划一张画；characters/outfits/props 是名字或 ID 数组，scene 是字符串；extra_images 只能填本会话已有图片编号。
   此处仅规划，不会提交 API。回复实际发出后才开始画，图片完成后单独发送；不得声称图片已生成或已送达。
   本轮最多规划一张图。用户明确要求画图时 user_requested=true；其他情况遵守发图策略和冷却。
"""


class ImageBridge:
    def __init__(self, owner):
        self.owner = owner
        self.staged = {}
        self.planning = {}

    def plugin(self):
        meta = self.owner.context.get_registered_star("astrbot_plugin_image_studio")
        return (
            getattr(meta, "star_cls", None)
            if meta and getattr(meta, "activated", False)
            else None
        )

    @staticmethod
    def batch_key(batch):
        ids = [str(getattr(x, "message_id", "")) for x in batch]
        if not ids or any(not x for x in ids):
            return ""
        return "palacer:" + hashlib.sha256("\0".join(ids).encode()).hexdigest()

    def discard(self, scope):
        self.staged.pop(scope, None)
        self.planning.pop(scope, None)

    async def context(self, scope):
        plugin = self.plugin()
        if plugin is None or not hasattr(plugin.image_service, "visual_context"):
            return []
        cid = await self.owner.context.conversation_manager.get_curr_conversation_id(
            scope
        )
        return plugin.image_service.visual_context(scope, cid)

    async def prepare(self, scope, batch):
        plugin = self.plugin()
        key = self.batch_key(batch)
        if plugin is None or not key:
            return "", {}
        guide = await plugin.image_service.guide()
        if not guide.get("enabled"):
            return "", {}
        loop = asyncio.get_running_loop()
        planned = []
        self.planning[scope] = (key, planned)

        def lookup(args):
            future = asyncio.run_coroutine_threadsafe(
                plugin.image_service.library(args), loop
            )
            try:
                return future.result(timeout=10)
            except Exception:
                future.cancel()
                return "参考库查询失败，不能编造素材。"

        def plan(args):

            if not isinstance(args, dict):
                return "无效作画参数。"
            if not isinstance(args.get("prompt"), str) or not args["prompt"].strip():
                return "作画计划缺少画面描述。"
            if not planned:
                planned.append({"args": json.loads(json.dumps(args))})
            return "已记录本轮作画意向；尚未提交、尚未生成、尚未发送。先自然回应，回复实际发出后才提交。"

        return CATALOG + "\n" + guide["text"], {
            "image_library": lookup,
            "image_generate": plan,
        }

    def stage(self, scope, batch, result):
        key, plans = self.planning.get(scope, ("", []))
        if key and key == self.batch_key(batch):
            self.planning.pop(scope, None)
            if plans:
                self.staged[scope] = (key, plans[:1])

    async def commit(self, scope, batch, sent):
        key, plans = self.staged.pop(scope, ("", []))
        if not sent or not key or key != self.batch_key(batch) or not plans:
            return
        plugin = self.plugin()
        if plugin is None:
            return
        cid = await self.owner.context.conversation_manager.get_curr_conversation_id(
            scope
        )
        args = dict(plans[0]["args"])
        user_requested = args.pop("user_requested", False) is True
        try:
            job = await plugin.commit_plan(
                scope, cid, key, args, user_requested=user_requested
            )
            self.owner._log(
                ev="image_committed", umo=scope, job_id=job["id"], status=job["status"]
            )
        except Exception as exc:
            self.owner._log(
                ev="image_commit_failed", umo=scope, error=type(exc).__name__
            )
