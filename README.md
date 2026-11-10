# Palacer

AstrBot 的母进程 / 写手回复插件：母进程整理上下文、选择事实并分工，每个写手完成一条消息；消息按窗口投递，用户插话会取消尚未发送的生成，并依据实际已发送内容继续。

适用于指定的 WebUI 与 QQ 私聊会话。声音卡按 persona 独立维护；没有保存有效声音卡时，插件不接管消息，保留 AstrBot 默认 `on_llm_request` 管线。

## 安装与启用

1. QQ / OneBot 部署请先安装或升级 [astrbot_plugin_burst >=0.3.0](https://github.com/Lansheen3096/astrbot_plugin_burst)，再安装 Palacer。使用 AstrBot **4.29.1 或以上版本**，先配置正常可用的聊天模型与 persona。在插件页通过仓库地址安装：`https://github.com/Lansheen3096/astrbot_plugin_palacer`。
2. 在插件配置中填写 `api_base`、`api_key`、写手 `model` 与 `planner_model`，选择 `persona_id`，填写 `umo_allowlist`。WebUI 可以另选 `persona_id_webchat`，留空继承 `persona_id`。白名单留空时不接管任何会话。
3. 打开插件的 **声音卡** 页面，选择相应 persona，手动填写，或选择 AstrBot 中已配置的聊天模型点击「根据 persona 生成草稿」。检查、编辑并保存后，下一条新消息即可由 Palacer 处理。WebUI 使用不同 persona 时，需要分别保存声音卡。

生成仅提炼表达方式，不会合成音频。生成草稿会将所选 persona 发送给你在页面上选择的模型；生成不会自动保存或覆盖已有卡片。保存空文本会清除当前 persona 的卡片，新消息恢复默认 LLM 流程。清空时已经接管的消息仍由原流程收尾，避免重复回复。

修改 persona 内容后，可在声音卡页面重新生成并保存。页面编辑的 persona 不会自动改变插件配置中的接管 persona。

## 模型与媒体

- 主生成链使用 DeepSeek 兼容的 `/chat/completions` 协议，包含 `thinking`、`reasoning_effort` 与 JSON 输出选项。模型名默认保留 `deepseek-flash`，请按服务实际提供的名称配置。普通 OpenAI 兼容服务不保证支持这些字段。
- API 地址和密钥需显式配置；也可使用 `PALACER_API_BASE` 与 `PALACER_API_KEY` 环境变量。插件不会从其他提供商配置中自动取第一组凭据。
- 图片观察使用 `deepseek-flash`，需要所配置接口实际支持图片输入。
- 音乐与视频观察可在 `omni_provider_id` 中选择现有的音视频提供商，还需系统已有 `ffmpeg` 和 `ffprobe`。不配置时对应能力不可用，不影响普通文本回复。
- PDF / DOCX / XLSX 与图片处理依赖见 `requirements.txt`；安装由 AstrBot 管理。核心网络请求使用 `httpx`，同步规划运行在线程中，HTTP 请求由可取消的异步客户端执行。

## 主要配置

| 配置 | 用途 |
| --- | --- |
| `mode` | `canary` 接管指定会话；`shadow` 保留默认回复并运行影子诊断 |
| `umo_allowlist` | 逗号分隔的会话 UMO，空值不接管 |
| `persona_id` / `persona_id_webchat` | 选择 persona，无私人默认值 |
| `message_window` | 每条消息的发送窗口，默认 5 秒 |
| `max_concurrent_generations` | 生成并发上限，默认 1 |
| `startup_delay` | 可选聚合等待，默认 0；可与外部 burst 插件配合 |
| `memory_kb_name` | 可选事实知识库名称，空值不猜测知识库名 |
| `memory_shortlist` | 可选 embedding 短名单；需要先配置可用的知识库与 embedding 模型 |
| `rerank_url` | 可选 rerank 完整 URL，空值不请求任何 rerank 服务 |
| `trace` | 详细本地诊断，默认关闭；启用后可能含消息、检索事实及 persona 相关信息 |

## burst 配套依赖与配置

QQ / OneBot 的完整聚合与排队链需要 **astrbot_plugin_burst >=0.3.0**。旧版没有 `held_only` 释放接口，也不会读取公开版 Palacer 的注册名；仅更新 Palacer 无法获得完整协作行为。

- burst 负责适配器入口的静默聚合与长消息立即放行；Palacer 的 `startup_delay` 保持 `0`，避免重复等待。
- burst 的 `palacer_plugin` 保持 `astrbot_plugin_palacer`，`bypass_umos` 通常留空。只有刻意跳过 burst 时才填写 bypass。
- burst 优先读取 Palacer 的 `burst_state(umo)`：刚开始生成且尚未发出消息时，在 `queue_grace_seconds` 范围内暂存快速连发；已经发出消息或超过宽限后，按用户插话处理。
- Palacer 成功发送或本轮结束后调用 `release_pending(umo, held_only=True)`，只放行因上一轮生成而暂存的批次，不提前打断普通静默聚合。
- 没有声音卡时，Palacer 返回未接管状态，burst 自动回到默认管线 / pacer 的状态判断。

burst 当前只包装 `aiocqhttp` 适配器，**WebUI 不经过 burst**。WebUI 可直接使用 Palacer；需要自身聚合时单独调整 `startup_delay`。不装 burst 时 Palacer 核心生成仍能工作，但 QQ 缺少入口聚合与双方队列协作。

## 可选集成

独立文本回复与声音卡页面不要求安装其他插件。`integrations` 中所有注册名与数据路径默认留空；不扫描或猜测现有私人插件、账号或文件。

以下接口用于兼容已有扩展，不代表同名公开插件已经提供这些接口。注册名填写实际插件名；数据路径相对 AstrBot 的 `data` 目录。配置后相关插件必须实现约定的方法与数据结构。

| 配置 | 接口与数据约定 |
| --- | --- |
| `state_plugin` | `core.py` 提供 `collect_snapshot(data, umo)`、`build_sections(snapshot)`、`compose_context(sections, limit)` |
| `review_plugin` + `review_log` | `main.py` 提供 `is_pure_event_reflection`、`pick_relevant`、`format_reflection_line`；日志逐行含 `day`、`reflections` |
| `health_plugin` + `health_cache` / `health_state` | `Main._compose()` 使用 `_cache` 和 `_state` 返回上下文块 |
| `media_plugin` + `media_data` | `core.py.quote_search`；插件实例可提供 `read_now(book_id, chunks)`；数据含 `books`、`progress.json`、`catalog.jsonl` |
| `life_plugin` + `life_state` / `life_events` | 可选状态、事件、天气和阅读愿望；`note_book_wish` 等接口 |
| `mode_plugin` | 可选 `_st`、`_corpus_text`、`_daily_tail`、`_tail_block`、`_tier_corpus`、`settle`；语料均由用户本地提供 |
| `voice_plugin` | 可选 `prepare_reply`、`send_prepared`、`remember_sent` 等接口，按运行时能力检查 |
| `notes_plugin` | 可选 `verify(event, response)` |

事实记忆可复用 `astrbot_plugin_memory_recall` 的显式知识库配置。需要记忆时，先创建事实知识库并配好 embedding；再填写知识库名和可选 rerank 服务。不配置这些内容时，普通文本回复仍然可用，记忆召回可以为空。

## 本地数据与隐私

持久数据位于 AstrBot 的 `data/plugin_data/astrbot_plugin_palacer/`：

- `voice_cards.json`：本地 persona 声音卡，原子保存，新文件权限为 `0600`。
- `message_times.sqlite3`：实际消息收发时间和历史绑定，含消息文本。
- `traces/`、调用和检索日志：仅在启用 `trace` 时记录，可能含敏感对话。

配置由 AstrBot 保存在自己的配置目录。`secret: true` 只遮罩 WebUI 中的密钥，不加密落盘文件。不要将配置、数据目录、日志、备份或声音卡提交到公开仓库。

## 验证与限制

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
node --check views/voice-card/app.js
```

测试使用合成 persona、假模型响应和临时数据库，覆盖时间记录、分段取消、事实分工、声音卡保存/生成草稿和缺卡回退。部分消息投递测试需要 AstrBot 运行时；在普通 Python 环境会跳过。真实模型质量、平台投递和第三方插件接口需在自己的部署中验证。

本仓库提供独立的公开版本，不自动迁移其他插件的数据，不携带历史对话、角色卡、账号标识或原始开发历史。

## 开发参考

遵循 AstrBot 官方 [插件开发](https://docs.astrbot.app/dev/star/plugin-new.html)、[可视化视图](https://docs.astrbot.app/dev/star/guides/plugin-pages.html)、[插件配置](https://docs.astrbot.app/dev/star/guides/plugin-config.html)、[存储](https://docs.astrbot.app/dev/star/guides/storage.html) 与 [发布](https://docs.astrbot.app/dev/star/plugin-publish.html) 文档。可视化页面使用 AstrBot 内置 bridge 和 Dashboard 认证，不启动额外 Web 服务。
