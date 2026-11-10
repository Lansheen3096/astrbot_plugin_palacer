const $ = (id) => document.getElementById(id);
const bridge = window.AstrBotPluginView;
const drafts = new Map();
const saved = new Map();
let selected = "";
let busy = false;
let connected = false;
function message(text, error = false) {
  $("status").textContent = text;
  $("status").dataset.error = String(error);
}
function theme(context) {
  document.documentElement.classList.toggle("dark", Boolean(context.isDark));
}
function render() {
  const text = $("card").value;
  $("count").textContent = `${text.length} / 12000`;
  $("routing").textContent = saved.get(selected) ? "已保存 · Palacer 可接管" : "默认 LLM 流程";
  for (const id of ["persona", "provider", "card", "generate", "save", "reload"]) {
    $(id).disabled = busy || !connected || (id !== "persona" && !selected);
  }
  $("generate").disabled ||= !$("provider").value;
  $("save").disabled ||= $("card").value === (saved.get(selected) || "");
}
async function action(work) {
  if (busy) return;
  busy = true;
  render();
  try { await work(); }
  catch (error) { message(error.message || "操作失败，请重试。", true); }
  finally { busy = false; render(); }
}
async function load() {
  if (!selected) { $("card").value = ""; return; }
  const result = await bridge.apiPost("voice-card/load", { persona_id: selected });
  saved.set(selected, result.text || "");
  $("card").value = drafts.get(selected) ?? result.text ?? "";
  message(saved.get(selected) ? "已读取本地声音卡。" : "还没有声音卡。可手动填写或生成草稿；保存前使用默认回复流程。");
}
$("persona").addEventListener("change", () => action(async () => {
  if (selected) drafts.set(selected, $("card").value);
  selected = $("persona").value;
  await load();
}));
$("card").addEventListener("input", () => {
  drafts.set(selected, $("card").value);
  message("有未保存的修改。");
  render();
});
$("provider").addEventListener("change", render);
$("generate").addEventListener("click", () => action(async () => {
  message("正在生成草稿…");
  const result = await bridge.apiPost("voice-card/generate", {
    persona_id: selected, provider_id: $("provider").value,
  });
  $("card").value = result.text;
  drafts.set(selected, result.text);
  message("草稿已生成，可编辑后保存。当前生效的声音卡尚未改变。");
}));
$("save").addEventListener("click", () => action(async () => {
  const result = await bridge.apiPost("voice-card/save", {
    persona_id: selected, text: $("card").value,
  });
  saved.set(selected, result.text);
  drafts.delete(selected);
  $("card").value = result.text;
  message(result.has_card ? "声音卡已保存，下一条新消息即可使用。" : "声音卡已清空，下一条新消息使用默认回复流程。");
}));
$("reload").addEventListener("click", () => action(async () => {
  drafts.delete(selected);
  await load();
}));
async function start() {
  if (!bridge) throw new Error("请从 AstrBot 插件页面打开声音卡。 ");
  theme(await bridge.ready());
  bridge.onContext(theme);
  const data = await bridge.apiGet("voice-card/status");
  for (const [id, rows, placeholder] of [
    ["persona", data.personas, "请选择 persona"],
    ["provider", data.providers, "请选择生成模型"],
  ]) {
    $(id).replaceChildren(new Option(placeholder, ""));
    for (const row of rows) $(id).add(new Option(row.name, row.id));
  }
  $("configured").textContent = `接管使用的 persona 在插件配置中选择。当前：${data.selected_persona || "未选择"}；WebUI：${data.webchat_persona || "未选择"}。`;
  $("persona").value = data.selected_persona || data.webchat_persona || "";
  selected = $("persona").value;
  connected = true;
  await action(load);
}
start().catch((error) => message(error.message, true));
