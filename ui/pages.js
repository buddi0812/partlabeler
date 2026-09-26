// PartLabeler pages after the start screen, laid out as CVAT's: a project (labels, tasks grouped by subset,
// create task, export, backup), creating tasks, a task (details, media info, jobs with stage and state), and the
// Tasks and Jobs lists across projects. The annotator opens per job (/projects/<p>/tasks/<t>/jobs/<j>).
// Talks to ui/host_fastapi.py (/api/projects/..., /api/tasks, /api/annotation-jobs); shares the start screen's
// look and helpers (ui/home.js).
import { CSS, header, fileBrowser, followJob, labelsEditor, api, esc, ago, size, TASKS, ICON, prefs } from "/ui/home.js";
import { notificationCenter, assistant, webAssistant } from "/ui/canvas.js";

const PAGE_CSS = `
.pg { max-width:1240px; margin:0 auto; padding:10px 20px 110px; display:grid; gap:18px; }
.pg-bar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; min-height:36px; }
.pg-back { color:var(--teal-deep); text-decoration:none; font-weight:600; padding:4px 8px; margin-left:-8px; border-radius:8px; }
.pg-back:hover { background:var(--teal-soft); }
.pg-sp { flex:1; }
.pg-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.pg-title h1 { font:600 clamp(22px, 2.6vw, 28px)/1.15 var(--display); margin:0; overflow-wrap:anywhere; }
.pg-meta { color:var(--steel); font-size:13px; margin:2px 0 0; }
.pg-section { display:grid; gap:10px; }
.pg-section > h2 { font:600 19px/1.2 var(--display); margin:0; display:flex; gap:8px; align-items:baseline; }
.pg-section > h2 small { color:var(--steel); font:500 14px var(--numeric); }
.pg-labels { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.pg-lab { display:inline-flex; gap:6px; align-items:center; padding:3px 11px; border-radius:99px; background:var(--paper); border:1px solid var(--line); font-size:13px; }
.pg-lab i { width:11px; height:11px; border-radius:3px; flex:none; }
.pg-tools { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.pg-tools input[type=text] { width:min(240px, 100%); }
.pg-subset > h3 { font:600 15px var(--display); margin:6px 0 8px; color:var(--steel); display:flex; gap:8px; align-items:center; }
.pg-list { display:grid; gap:10px; }
.pg-task { display:grid; grid-template-columns:160px minmax(0,1fr) minmax(200px, 260px) auto; gap:16px; align-items:center; background:var(--paper);
  border:1px solid var(--line); border-radius:12px; padding:10px 12px 10px 10px; position:relative; transition:border-color .2s, box-shadow .2s; }
.pg-task:hover { border-color:var(--line-hi); box-shadow:0 8px 22px rgba(31,42,48,.08); }
.pg-thumb { width:160px; height:100px; border-radius:8px; background:var(--stage) center/cover no-repeat; display:block; }
.pg-thumb.big { width:100%; height:auto; aspect-ratio:16/10; max-width:420px; }
.pg-task h3 { margin:0; font:600 16px/1.3 var(--display); overflow-wrap:anywhere; }
.pg-task h3 a { color:inherit; text-decoration:none; } .pg-task h3 a:hover { color:var(--teal-deep); text-decoration:underline; }
.pg-task .pg-meta { font-size:12.5px; }
.pg-media { font-size:12px; color:var(--steel); margin-top:4px; overflow-wrap:anywhere; }
.pg-prog-text { font-size:12.5px; color:var(--steel); display:flex; flex-wrap:wrap; gap:2px 10px; }
.pg-prog-text b { color:var(--ink); font:600 13px var(--numeric); }
.pg-prog { height:8px; background:var(--alu-2); border-radius:4px; overflow:hidden; display:flex; margin:6px 0 4px; }
.pg-prog i { display:block; height:100%; } .pg-prog .done { background:var(--ok); } .pg-prog .review { background:var(--lab); } .pg-prog .work { background:color-mix(in srgb, var(--teal) 38%, var(--paper)); }
.pg-btns { display:flex; gap:6px; align-items:center; }
.pg-btns .h-menu { position:relative; top:auto; right:auto; }
.pg-status { font-size:11.5px; font-weight:600; padding:2px 9px; border-radius:99px; background:var(--alu-2); color:var(--steel); text-transform:capitalize; }
.pg-status.completed { background:var(--ok-soft); color:var(--ok-ink); } .pg-status.validation { background:var(--c-info-soft,#e3ecfb); color:var(--c-info-ink,#2b5aa6); }
.pg-top { display:grid; grid-template-columns:minmax(260px, 420px) minmax(0,1fr); gap:22px; align-items:start; }
.pg-name { font:600 clamp(20px, 2.4vw, 26px) var(--display); border:1px solid transparent; background:transparent; padding:2px 6px; margin-left:-7px; border-radius:8px; width:100%; }
.pg-name:hover { border-color:var(--line); background:var(--paper); }
.pg-kv { display:grid; grid-template-columns:max-content minmax(0,1fr); gap:5px 18px; font-size:13px; margin:0; }
.pg-kv dt { color:var(--steel); } .pg-kv dd { margin:0; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
.pg-jobs { display:grid; grid-template-columns:repeat(auto-fill, minmax(290px, 1fr)); gap:10px; }
.pg-job { background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:12px 14px; display:grid; gap:8px; position:relative; }
.pg-job:hover { border-color:var(--line-hi); }
.pg-job h3 { margin:0; font:600 16px var(--display); display:flex; gap:8px; align-items:center; }
.pg-job h3 a { color:inherit; } .pg-job .pg-kv { font-size:12.5px; }
.pg-job select { padding:4px 8px; font-size:13px; }
.pg-tile { display:grid; gap:6px; background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:10px; color:inherit; text-decoration:none; }
.pg-tile:hover { border-color:var(--teal); box-shadow:0 8px 22px rgba(31,42,48,.08); }
.pg-tile .pg-thumb { width:100%; height:130px; }
.pg-empty { color:var(--steel); padding:18px; border:1.5px dashed var(--line-hi); border-radius:12px; background:var(--c-empty,rgba(255,255,255,.5)); }
.pg-files { display:grid; gap:6px; }
.pg-form { max-width:760px; }
.pg-form .h-panel { display:grid; gap:14px; }
.pg-two { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:12px; }
.pg-dlg { width:min(520px, calc(100vw - 32px)); }
.pg-dlg h2 { margin:0; padding:18px 20px 4px; font:600 19px var(--display); }
.pg-dlg .h-form { padding:10px 20px 4px; }
.pg-dlg .pg-dlgjob { padding:0 20px; }
/* settings */
.st { display:grid; grid-template-columns:210px minmax(0,1fr); gap:22px; align-items:start; }
.st-nav { position:sticky; top:76px; display:grid; gap:2px; }
.st-nav a { padding:7px 12px; border-radius:8px; color:var(--steel); text-decoration:none; font-weight:600; font-size:13.5px; }
.st-nav a:hover { background:var(--alu-2); color:var(--ink); }
.st-body { display:grid; gap:18px; min-width:0; }
.st-sec { display:grid; scroll-margin-top:84px; }
.st-sec > h2 { margin-bottom:6px; }
.st-sec > .h-sub { margin:0 0 6px; font-size:13px; }
.st-row { display:flex; gap:16px; align-items:center; justify-content:space-between; padding:11px 0; border-top:1px solid var(--line); min-width:0; }
.st-row > span:first-child { display:grid; gap:1px; min-width:0; } .st-row b { font-weight:600; } .st-row small { color:var(--steel); font-size:12.5px; }
.st-row.st-col { flex-direction:column; align-items:stretch; gap:8px; }
.st-row input[type=number] { width:104px; } .st-row input[type=text] { width:min(260px, 100%); }
.st-switch { appearance:none; -webkit-appearance:none; width:42px; height:24px; border-radius:99px; background:var(--alu-2); border:1px solid var(--line-hi);
  position:relative; cursor:pointer; flex:none; margin:0; transition:background .2s, border-color .2s; }
.st-switch::after { content:""; position:absolute; top:2px; left:2px; width:18px; height:18px; border-radius:50%; background:var(--paper);
  box-shadow:0 1px 3px rgba(0,0,0,.3); transition:transform .25s var(--spring); }
.st-switch:checked { background:var(--teal); border-color:var(--teal); } .st-switch:checked::after { transform:translateX(18px); }
.st-switch:focus-visible { outline:2px solid var(--teal); outline-offset:2px; }
.st-swatches { display:flex; flex-wrap:wrap; gap:8px; }
.st-sw { display:inline-flex; align-items:center; gap:8px; padding:5px 13px 5px 6px; border-radius:99px; }
.st-sw i { width:20px; height:20px; border-radius:50%; background:var(--sw); flex:none; }
.st-sw[aria-checked=true] { border-color:var(--sw); box-shadow:0 0 0 2px var(--sw); font-weight:600; }
.st code, .st-code { font:12.5px ui-monospace,"Cascadia Mono",Consolas,monospace; background:var(--alu); padding:3px 7px; border-radius:6px; overflow-wrap:anywhere; }
.st-saved { color:var(--ok-ink); font-weight:600; font-size:13px; opacity:0; transition:opacity .3s; } .st-saved.on { opacity:1; }
.st-danger { color:var(--bad); } .st-danger:hover:not(:disabled) { border-color:var(--bad); }
.st-list { margin:2px 0 8px; padding-left:20px; } .st-list li { margin:3px 0; }
.st-more summary { cursor:pointer; color:var(--teal-deep); font-weight:600; font-size:13.5px; padding:10px 0 4px; border-top:1px solid var(--line); }
.st-more .h-form { max-width:420px; margin:8px 0 6px; }
@media (max-width: 900px) { .st { grid-template-columns:minmax(0,1fr); } .st-nav { position:static; display:flex; flex-wrap:wrap; }
  .st-row { flex-wrap:wrap; } }

/* sign in */
.lg { min-height:100vh; display:grid; place-items:center; padding:24px 16px; }
.lg-card { width:min(420px, 100%); display:grid; gap:14px; padding:26px 26px 22px; }
.lg-card h1 { font:600 24px/1.2 var(--display); margin:6px 0 0; }
.lg .h-logo { margin:0; }
.lg-card .h-form button.primary { justify-self:start; padding:8px 22px; }
.lg-link { border:0; background:none; padding:0; color:var(--teal-deep); font-weight:600; text-decoration:underline; }
.lg-link:hover:not(:disabled) { border:0; }
.lg-forgot summary { cursor:pointer; color:var(--steel); font-size:13px; }
.lg-forgot p { font-size:13px; color:var(--ink-2); margin:6px 0 0; }
.lg-forgot code { font:12.5px ui-monospace,"Cascadia Mono",Consolas,monospace; background:var(--alu); padding:2px 6px; border-radius:5px; overflow-wrap:anywhere; }
@media (max-width: 900px) {
  .pg-task { grid-template-columns:110px minmax(0,1fr); } .pg-thumb { width:110px; height:70px; }
  .pg-task .pg-progcol, .pg-task .pg-btns { grid-column:1 / -1; } .pg-top { grid-template-columns:minmax(0,1fr); } }
`;

const enc = encodeURIComponent;
const STAGES = ["annotation", "validation", "acceptance"], STATES = ["new", "in progress", "rejected", "completed"];
const jobUrl = (p, t, j) => `/projects/${enc(p)}/tasks/${t}/jobs/${j}`;
const when = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { dateStyle: "medium" }) : "");
const mb = (b) => (b ? size(b) : "");

function media(t) {                                   // one line: "1920×1080 · 15 fps · 3:48 · HEVC · 33 MB · 687 frames kept"
  const i = t.info || {}, d = i.duration;
  const out = [i.width ? `${i.width}×${i.height}` : null];
  if (t.kind === "video") out.push(i.fps ? `${+i.fps.toFixed(2)} fps` : null, d ? `${Math.floor(d / 60)}:${String(Math.floor(d % 60)).padStart(2, "0")}` : null,
    i.codec ? i.codec.toUpperCase() : null, mb(i.size), `${t.frames} frames kept (every ${t.every})`);
  else out.push(`${t.frames} images`, i.mixed_sizes ? "mixed sizes" : null, mb(i.size));
  return out.filter(Boolean).join(" · ");
}

function progress(t) {                                // CVAT's task row: done · on review · annotating · total
  const pct = (n) => (t.total ? (n / t.total) * 100 : 0);
  return `<div class="pg-prog-text"><span><b>${t.done}</b> done</span><span><b>${t.review}</b> on review</span><span><b>${t.annotating}</b> annotating</span><span><b>${t.total}</b> total</span></div>
    <div class="pg-prog" role="img" aria-label="${t.done} of ${t.total} jobs done, ${t.review} on review"><i class="done" style="width:${pct(t.done)}%"></i><i class="review" style="width:${pct(t.review)}%"></i></div>
    <div class="pg-prog-text">${t.confirmed} of ${t.frames} ${t.kind === "video" ? "frames" : "images"} confirmed, ${t.labeled} labeled</div>`;
}

function statusChip(s) { return `<span class="pg-status ${esc(s)}">${esc(s)}</span>`; }

const SECTIONS = [["account", "Account"], ["appearance", "Appearance"], ["helpers", "Helpers and messages"], ["annotating", "Annotating and export"],
                  ["tasks", "New tasks"], ["storage", "Projects folder"], ["transfer", "Unfinished work as a zip"]];
const THEMES = ["light", "dark", "system"], THEME_NAMES = { light: "Light", dark: "Dark", system: "Match my computer" };
const ACCENTS = { teal: ["Teal", "#0a7c78"], blue: ["Blue", "#2563c9"], indigo: ["Indigo", "#4f46c8"], violet: ["Violet", "#7c3aca"],   // ui/theme.css
                  rose: ["Rose", "#c2305a"], orange: ["Orange", "#b4530a"], green: ["Green", "#287a36"], graphite: ["Graphite", "#4b5a66"] };
const EXPORTS = { "": "The project's first format", yolo: "YOLO (Ultralytics)", coco: "COCO 1.0", cvat: "CVAT for images 1.1", voc: "Pascal VOC",
                  labelstudio: "Label Studio", folders: "Class folders (classification)", csv: "CSV list (classification)" };

// ---- sign in / create an account (before any other page) ------------------------------------------------
function loginPage(root, args) {
  let signup = !!args.first;
  const next = new URLSearchParams(location.search).get("next") || "/";
  const go = () => { location.href = next.startsWith("/") && !next.startsWith("//") ? next : "/"; };
  root.innerHTML = `<main class="lg"><div class="lg-card h-panel">
    <a class="h-logo" href="/login">${ICON.logo}<span>PartLabeler</span></a>
    <h1 class="lg-h"></h1><p class="h-sub lg-lede"></p>
    <form class="h-form lg-in" autocomplete="on">
      <label class="h-field"><span>Username</span><input type="text" name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required></label>
      <label class="h-field"><span>Password</span><input type="password" name="password" autocomplete="current-password" required></label>
      <label class="h-check-row"><input type="checkbox" name="keep" checked> Keep me signed in<small>For 30 days in this browser.</small></label>
      <button class="primary" type="submit">Sign in</button></form>
    <form class="h-form lg-up" autocomplete="on" hidden>
      <label class="h-field"><span>Username</span><input type="text" name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required
        pattern="[A-Za-z0-9][A-Za-z0-9._\\-]{0,31}" title="1 to 32 letters, digits, dots, dashes or underscores"><small>Letters, digits, dots, dashes or underscores.</small></label>
      <label class="h-field"><span>Your name</span><input type="text" name="name" autocomplete="name" maxlength="60" placeholder="Optional, shown in the top bar"></label>
      <label class="h-field"><span>Password</span><input type="password" name="password" autocomplete="new-password" minlength="8" required><small>At least 8 characters.</small></label>
      <label class="h-field"><span>Repeat the password</span><input type="password" name="repeat" autocomplete="new-password" minlength="8" required></label>
      <label class="h-check-row"><input type="checkbox" name="keep" checked> Keep me signed in<small>For 30 days in this browser.</small></label>
      <button class="primary" type="submit">Create account</button></form>
    <p class="h-err lg-err" role="alert"></p>
    <p class="h-sub lg-switch"></p>
    <details class="lg-forgot"><summary>Forgot your password?</summary><p>Passwords stay on this computer, so they can't be emailed.
      Anyone at this computer can set a new one in a command prompt in the PartLabeler folder:
      <code>.venv\\Scripts\\python -m engine.cli account reset USERNAME</code></p></details>
  </div></main>`;
  const $ = (s) => root.querySelector(s), err = $(".lg-err");
  const show = () => {
    $(".lg-in").hidden = signup; $(".lg-up").hidden = !signup; err.textContent = "";
    $(".lg-h").textContent = signup ? (args.first ? "Create your account" : "Create an account") : "Sign in";
    $(".lg-lede").textContent = signup ? "Accounts live on this computer. Yours keeps your theme, settings and projects folder."
      : "Welcome back. Your settings and projects folder come with your account.";
    $(".lg-switch").innerHTML = signup ? (args.first ? "" : `Already have an account? <button type="button" class="lg-link" data-lg>Sign in</button>`)
      : `New here? <button type="button" class="lg-link" data-lg>Create an account</button>`;
    (signup ? $(".lg-up") : $(".lg-in")).username.focus();
    document.title = `${$(".lg-h").textContent} · PartLabeler`;
  };
  root.addEventListener("click", (e) => { if (e.target.closest("[data-lg]")) { signup = !signup; show(); } });
  const submit = (form, url, body) => form.addEventListener("submit", async (e) => {
    e.preventDefault(); err.textContent = "";
    const b = body(form); if (!b) return;
    form.querySelector("button[type=submit]").disabled = true;
    try { await api(url, b); go(); }
    catch (x) { err.textContent = x.message; form.querySelector("button[type=submit]").disabled = false; }
  });
  submit($(".lg-in"), "/api/auth/login", (f) => ({ username: f.username.value.trim(), password: f.password.value, keep: f.keep.checked }));
  submit($(".lg-up"), "/api/auth/signup", (f) => {
    if (f.password.value !== f.repeat.value) { err.textContent = "The two passwords are different"; f.repeat.focus(); return null; }
    return { username: f.username.value.trim(), name: f.name.value.trim(), password: f.password.value, keep: f.keep.checked };
  });
  show();
}

export function page(root, args) {
  document.head.appendChild(Object.assign(document.createElement("style"), { textContent: CSS + PAGE_CSS }));
  if (args.view === "login") return loginPage(root, args);
  const active = { tasks: "tasks", jobs: "jobs", settings: "" }[args.view] ?? "projects";
  root.innerHTML = `${header(active)}<main class="pg"><p class="pg-meta">Loading…</p></main><div class="h-toasts" aria-live="polite"></div>
    <dialog class="h-confirm pg-dlg pg-ask"><h2></h2><p class="pg-ask-text"></p><div class="b-foot"><button type="button" data-a="no">Cancel</button><button type="button" class="danger" data-a="yes">OK</button></div></dialog>`;
  const $ = (s) => root.querySelector(s), main = $(".pg"), pick = fileBrowser();

  function toast(n) {
    if (prefs().toasts === false && n.level !== "error" && n.level !== "warning") return;   // Settings: only problems pop up
    const t = document.createElement("div");
    t.className = `h-toast ${n.level || "info"}`; t.setAttribute("role", n.level === "error" ? "alert" : "status");
    t.innerHTML = "<b></b><small></small>"; t.querySelector("b").textContent = n.title; t.querySelector("small").textContent = n.detail || "";
    $(".h-toasts").appendChild(t);
    while ($(".h-toasts").children.length > 4) $(".h-toasts").firstChild.remove();
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 260); }, n.level === "error" ? 10000 : 5000);
  }
  function ask(title, text, ok = "OK") {
    const d = $(".pg-ask");
    d.querySelector("h2").textContent = title; d.querySelector(".pg-ask-text").textContent = text; d.querySelector('[data-a="yes"]').textContent = ok;
    d.showModal();
    return new Promise((res) => { d.onclick = (e) => { const a = e.target.closest("[data-a]")?.dataset.a; if (!a) return; d.close(); res(a === "yes"); }; });
  }
  const runAction = async (a) => {
    if (a.type === "open") location.href = `/projects/${enc(a.project)}`;
    else if (a.type === "goto") location.href = `/p/${enc(a.project)}?item=${a.item}`;
    else if (a.type === "folder") await api("/api/open-folder", { path: a.path }).catch((e) => toast({ level: "error", title: "Could not open", detail: e.message }));
    else if (a.type === "restore") await api(`/api/trash/${enc(a.entry)}/restore`, {}).catch(() => {});
  };

  // notifications (shared history), device, Rivet: as on the start screen
  const center = notificationCenter({ mount: $(".h-bellmount"), onRead: () => api("/api/notifications/read", {}).catch(() => {}),
    onClear: () => api("/api/notifications/clear", {}).then(pollNotes).catch(() => {}), onAction: runAction });
  let seen = null;
  async function pollNotes() {
    const d = await api("/api/notifications").catch(() => null); if (!d) return;
    if (seen) d.items.filter((n) => !seen.has(`${n.id}:${n.count}`)).slice(0, 3).reverse().forEach(toast);
    seen = new Set(d.items.map((n) => `${n.id}:${n.count}`)); center.set(d.items, d.unread);
  }
  pollNotes(); setInterval(() => { if (!document.hidden) pollNotes(); }, 3000);
  api("/api/info").then((i) => {
    const dev = $(".h-device"); dev.hidden = false; dev.title = i.device; dev.querySelector("span").textContent = i.device;
    dev.classList.toggle("cpu", !/cuda|gpu|nvidia|rtx|gtx|tesla/i.test(i.device));
  }).catch(() => {});
  assistant({ mount: document.body, variant: "fab", transport: webAssistant("/"), firstTip: 60000,
    context: () => ({ page: args.view, project: undefined }),
    actions: { notifications: { label: "Open notifications", run: () => center.open() } },
    starters: ["How do I add a task to a project?", "What are jobs?", "How can I export a dataset?", { q: "What makes PartLabeler different?", local: "pros" }] });
  document.addEventListener("click", (e) => root.querySelectorAll("details.h-menu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; }));

  // ---- export and upload dialogs (CVAT's "Export … as a dataset", "Upload annotations") -------------
  function exportDialog(project, formats, what, scope) {
    const d = document.createElement("dialog");
    d.className = "h-confirm pg-dlg";
    d.innerHTML = `<h2>Export ${esc(what)} as a dataset</h2><form class="h-form" method="dialog">
      <label class="h-field"><span>Export format</span><select name="format">${formats.map((f) => `<option value="${f}"${f === prefs().export_format ? " selected" : ""}>${esc(FORMAT_NAMES[f] || f)}</option>`).join("")}</select></label>
      <label class="h-check-row"><input type="checkbox" name="images" checked> Save images<small>Off: the label files only.</small></label>
      <label class="h-check-row"><input type="checkbox" name="confirmed"${prefs().confirmed_only ? " checked" : ""}> Confirmed frames only<small>Only frames you confirmed (Enter).</small></label>
      <label class="h-field"><span>Custom name</span><input type="text" name="name" placeholder="Leave empty for the default name"></label></form>
      <div class="pg-dlgjob"></div>
      <div class="b-foot"><button type="button" data-a="cancel">Cancel</button><button type="button" class="primary" data-a="ok">OK</button></div>`;
    document.body.appendChild(d); d.showModal();
    const f = d.querySelector("form");
    d.addEventListener("close", () => d.remove());
    d.addEventListener("click", async (e) => {
      const a = e.target.closest("[data-a]")?.dataset.a; if (!a) return;
      if (a === "cancel") { d.close(); return; }
      e.target.disabled = true;
      try {
        const { job } = await api(`/api/projects/${enc(project)}/export`, { format: f.format.value, save_images: f.images.checked,
          reviewed_only: f.confirmed.checked, name: f.name.value.trim(), ...scope });
        followJob(job, d.querySelector(".pg-dlgjob"), { onDone: (res) => {
          d.querySelector('[data-a="ok"]').hidden = true; d.querySelector('[data-a="cancel"]').textContent = "Close";
          d.querySelector(".pg-dlgjob").insertAdjacentHTML("beforeend", `<p class="h-sub" style="margin:8px 0">${res.images} images in ${esc(res.folder)}
            <button type="button" data-open="${esc(res.folder)}">Open folder</button></p>`);
          pollNotes(); } });
      } catch (err) { d.querySelector(".pg-dlgjob").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; e.target.disabled = false; }
    });
    d.addEventListener("click", (e) => { const o = e.target.closest("[data-open]"); if (o) runAction({ type: "folder", path: o.dataset.open }); });
  }
  function uploadDialog(project, what, scope, after) {
    const d = document.createElement("dialog");
    d.className = "h-confirm pg-dlg";
    d.innerHTML = `<h2>Upload annotations to ${esc(what)}</h2><form class="h-form" method="dialog">
      <div class="h-field"><span>YOLO labels folder</span><div class="h-row"><input type="text" name="labels" placeholder="A labels/ folder: one .txt per frame or image…"><button type="button" data-a="browse">Browse…</button></div>
        <small>Files are matched to frames by name (or by frame number). Boxes, or polygons in a segmentation project.</small></div>
      <div class="h-field"><span>Import mode</span><div class="h-row"><label><input type="radio" name="mode" value="replace" checked> Replace</label><label><input type="radio" name="mode" value="append"> Append</label></div>
        <small>Replace removes the existing labels of this ${esc(what.split(" ")[0])} first.</small></div></form>
      <div class="pg-dlgjob"></div>
      <div class="b-foot"><button type="button" data-a="cancel">Cancel</button><button type="button" class="primary" data-a="ok">OK</button></div>`;
    document.body.appendChild(d); d.showModal();
    const f = d.querySelector("form");
    d.addEventListener("close", () => d.remove());
    d.addEventListener("click", async (e) => {
      const a = e.target.closest("[data-a]")?.dataset.a; if (!a) return;
      if (a === "cancel") { d.close(); return; }
      if (a === "browse") { const p = await pick({ kind: "dir", folder: true, start: f.labels.value }); if (p) f.labels.value = p; return; }
      const mode = f.mode.value;
      if (mode === "replace" && !(await ask(`Replace existing annotations?`, `The labels of ${what} are removed, then the uploaded ones added.`, "Replace"))) return;
      try {
        const r = await api(`/api/projects/${enc(project)}/annotations`, { labels: f.labels.value.trim(), mode, ...scope });
        d.close(); toast({ level: "success", title: `Uploaded ${r.boxes} labels`, detail: `${r.files_matched} files matched${r.files_unmatched ? `, ${r.files_unmatched} did not` : ""}` });
        after?.();
      } catch (err) { d.querySelector(".pg-dlgjob").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; }
    });
  }

  function taskRow(t, project, withProject = false) {
    const href = `/projects/${enc(project)}/tasks/${t.id}`;
    return `<article class="pg-task"><a href="${href}" aria-label="Open task ${esc(t.name)}"><span class="pg-thumb" style="background-image:url('${t.preview}')"></span></a>
      <div><h3><a href="${href}">#${t.id}: <span translate="no">${esc(t.name)}</span></a></h3>
        <p class="pg-meta">${withProject ? `Project <a href="/projects/${enc(project)}">${esc(project)}</a> · ` : ""}Created ${esc(when(t.created))}${t.updated ? ` · Last updated ${esc(ago(t.updated))}` : ""}
          ${t.subset ? ` · <span class="h-chip">${esc(t.subset)}</span>` : ""} ${statusChip(t.status)}</p>
        <div class="pg-media">${esc(media(t))}</div></div>
      <div class="pg-progcol">${progress(t)}</div>
      <div class="pg-btns"><a href="${href}"><button type="button" class="primary">Open</button></a>
        <details class="h-menu"><summary aria-label="Actions for task ${esc(t.name)}" title="Actions">⋯</summary><div class="h-menu-list">
          <button type="button" data-t-upload="${t.id}" data-p="${esc(project)}">Upload annotations</button>
          <button type="button" data-t-export="${t.id}" data-p="${esc(project)}">Export task dataset</button>
          <button type="button" data-t-backup="${t.id}" data-p="${esc(project)}">Backup task</button>
          <button type="button" class="danger" data-t-delete="${t.id}" data-p="${esc(project)}" data-name="${esc(t.name)}">Delete</button></div></details></div></article>`;
  }
  async function taskActions(e, reload, formatsOf) {
    const up = e.target.closest("[data-t-upload]"), ex = e.target.closest("[data-t-export]"), del = e.target.closest("[data-t-delete]");
    const bk = e.target.closest("[data-t-backup]");
    const b = up || ex || del || bk; if (!b) return false;
    b.closest("details").open = false;
    const project = b.dataset.p, tid = +(up?.dataset.tUpload ?? ex?.dataset.tExport ?? del?.dataset.tDelete ?? bk?.dataset.tBackup);
    if (bk) {
      try {
        await api(`/api/projects/${enc(project)}/tasks/${tid}/backup`, {});
        toast({ title: `Backing up task #${tid}…`, detail: "A zip with its frames, labels and progress goes to the _backups folder; a notification says when it is ready." });
        setTimeout(pollNotes, 1500);
      } catch (err) { toast({ level: "error", title: "Could not back up", detail: err.message }); }
    }
    else if (up) uploadDialog(project, `task #${tid}`, { task: tid }, reload);
    else if (ex) exportDialog(project, await formatsOf(project), `task #${tid}`, { tasks: [tid] });
    else if (await ask(`Delete task ${del.dataset.name}?`, "Its stored frames and all its labels are deleted. An image folder itself stays where it is. This cannot be undone.", "Delete")) {
      try { await api(`/api/projects/${enc(project)}/tasks/${tid}`, undefined, "DELETE"); toast({ title: `Deleted task ${del.dataset.name}` }); reload(); }
      catch (err) { toast({ level: "error", title: "Could not delete", detail: err.message }); }
    }
    return true;
  }

  // ---- project page ---------------------------------------------------------------------------------
  async function projectPage(name) {
    let d;
    try { d = await api(`/api/projects/${enc(name)}`); } catch (err) { main.innerHTML = `<p class="h-err">${esc(err.message)}</p>`; return; }
    let q = "", sort = "id", editing = null;
    const render = () => {
      const tasks = d.tasks.filter((t) => !q || t.name.toLowerCase().includes(q) || (t.subset || "").toLowerCase().includes(q))
        .sort((a, b) => sort === "name" ? a.name.localeCompare(b.name) : sort === "updated" ? String(b.updated || "").localeCompare(String(a.updated || "")) : sort === "subset" ? (a.subset || "~").localeCompare(b.subset || "~") : a.id - b.id);
      const groups = d.subsets.length ? [...new Set(tasks.map((t) => t.subset || ""))].sort((a, b) => (a || "~").localeCompare(b || "~")) : [null];
      const totals = { done: 0, total: 0 }; d.tasks.forEach((t) => { totals.done += t.done; totals.total += t.total; });
      main.innerHTML = `<div class="pg-bar"><a class="pg-back" href="/">‹ Back to projects</a><span class="pg-sp"></span>
          <details class="h-menu" style="position:relative;top:auto;right:auto"><summary style="width:auto;padding:0 12px;border:1px solid var(--line);background:var(--paper)">Actions ▾</summary><div class="h-menu-list">
            <button type="button" data-pa="export">Export dataset</button><button type="button" data-pa="backup">Backup project</button>
            <button type="button" data-pa="folder">Show in folder</button><button type="button" class="danger" data-pa="trash">Move to trash…</button></div></details></div>
        <div><div class="pg-title"><h1 translate="no">${esc(d.name)}</h1><span class="h-chip">${esc(TASKS[d.task] || d.task)}</span></div>
          <p class="pg-meta">Created ${esc(when(d.created))} · ${d.tasks.length} task${d.tasks.length === 1 ? "" : "s"} · ${d.items} frames or images · ${totals.done} of ${totals.total} jobs done</p></div>
        <section class="pg-section h-panel"><h2>Labels <small>${d.labels.length}</small></h2>
          ${editing ? `<div class="pg-labelsed"></div><div class="h-row"><button type="button" class="primary" data-pa="save-labels">Save labels</button><button type="button" data-pa="cancel-labels">Cancel</button><span class="h-sub pg-labmsg"></span></div>`
            : `<div class="pg-labels">${d.labels.map((l, i) => `<span class="pg-lab"><i style="background:${esc(l.color || colorOf(i))}"></i><span translate="no">${esc(l.name)}</span></span>`).join("") || `<span class="h-sub">No labels yet.</span>`}
              <button type="button" data-pa="edit-labels">Edit labels</button></div>`}</section>
        <section class="pg-section"><h2>Tasks <small>${d.tasks.length}</small></h2>
          <div class="pg-tools"><input type="text" class="pg-q" placeholder="Search…" value="${esc(q)}" aria-label="Search tasks">
            <select class="h-sort pg-sort" aria-label="Sort tasks">${[["id", "ID"], ["name", "Name"], ["updated", "Updated date"], ["subset", "Subset"]].map(([v, t]) => `<option value="${v}"${v === sort ? " selected" : ""}>${t}</option>`).join("")}</select>
            <span class="pg-sp"></span><a href="/projects/${enc(name)}/tasks/create"><button type="button" class="primary">+ Create a new task</button></a>
            <a href="/projects/${enc(name)}/tasks/create?multi=1"><button type="button">Create multi tasks</button></a>
            <button type="button" data-pa="import-task" title="Add a task backup (.zip) with its frames, labels and progress">Import task…</button></div>
          ${d.tasks.length ? groups.map((g) => {
            const ts = tasks.filter((t) => g === null || (t.subset || "") === g);
            return ts.length ? `<div class="pg-subset">${g !== null ? `<h3>${g ? `Subset: ${esc(g)}` : "No subset"} <small class="h-count">${ts.length}</small></h3>` : ""}<div class="pg-list">${ts.map((t) => taskRow(t, name)).join("")}</div></div>` : "";
          }).join("") || `<p class="pg-empty">No task matches “${esc(q)}”.</p>`
            : `<div class="pg-empty"><b>No tasks yet.</b> A task is one video or one image folder to label. <a href="/projects/${enc(name)}/tasks/create">Create the first task</a>.</div>`}</section>`;
      if (editing) editing = labelsEditor($(".pg-labelsed"), d.labels.map((l, i) => ({ ...l, color: l.color || colorOf(i), from: i })));
      const qi = $(".pg-q"); qi.oninput = () => { q = qi.value.trim().toLowerCase(); render(); const n = $(".pg-q"); n.focus(); n.setSelectionRange(n.value.length, n.value.length); };
      $(".pg-sort").onchange = (e) => { sort = e.target.value; render(); };
    };
    const reload = async () => { d = await api(`/api/projects/${enc(name)}`); render(); };
    render();
    if (location.hash === "#export") exportDialog(name, d.formats, `project ${name}`, {});
    main.addEventListener("click", async (e) => {
      if (await taskActions(e, reload, async () => d.formats)) return;
      const a = e.target.closest("[data-pa]")?.dataset.pa; if (!a) return;
      e.target.closest("details")?.removeAttribute("open");
      if (a === "export") exportDialog(name, d.formats, `project ${name}`, {});
      else if (a === "backup") {
        const { job } = await api(`/api/projects/${enc(name)}/backup`, {});
        toast({ title: `Backing up ${name}…`, detail: "The zip goes to the _backups folder; a notification says when it is ready." });
        const wait = async () => { const j = await api(`/api/jobs/${job}`).catch(() => null); if (j && !j.finished) setTimeout(wait, 1000); else pollNotes(); }; wait();
      } else if (a === "folder") { const info = await api("/api/info"); runAction({ type: "folder", path: `${info.home}${info.home.includes("\\") ? "\\" : "/"}${name}` }); }
      else if (a === "trash") {
        if (await ask(`Move ${name} to the trash?`, "It moves to the Trash with its frames and labels; restore it from the Trash on the start screen.", "Move to trash")) {
          await api(`/api/projects/${enc(name)}/trash`, {}); location.href = "/";
        }
      } else if (a === "import-task") {
        const path = await pick({ kind: "backup", folder: false }); if (!path) return;
        try {
          const { job } = await api(`/api/projects/${enc(name)}/tasks/import`, { path });
          toast({ title: "Adding the task…", detail: "Its frames, labels and progress are copied into this project." });
          const wait = async () => { const j = await api(`/api/jobs/${job}`).catch(() => null);
            if (j && !j.finished) return setTimeout(wait, 800);
            if (j?.error) toast({ level: "error", title: "Could not add the task", detail: j.error }); else await reload();
            pollNotes(); };
          wait();
        } catch (err) { toast({ level: "error", title: "Could not add the task", detail: err.message }); }
      } else if (a === "edit-labels") { editing = true; render(); }
      else if (a === "cancel-labels") { editing = null; render(); }
      else if (a === "save-labels") {
        let labels; try { labels = editing.get(); } catch { $(".pg-labmsg").innerHTML = `<span class="h-err">The raw labels are not valid JSON</span>`; return; }
        const kept = new Set(labels.map((l) => l.from).filter((x) => x != null)), gone = d.labels.filter((_, i) => !kept.has(i));
        if (gone.length && !(await ask(`Delete ${gone.length} label${gone.length === 1 ? "" : "s"}?`,
          `${gone.map((l) => l.name).join(", ")}: every annotation with ${gone.length === 1 ? "this label" : "these labels"} is deleted too.`, "Delete"))) return;
        try { d = await api(`/api/projects/${enc(name)}`, { labels }, "PATCH"); editing = null; render(); toast({ level: "success", title: "Labels saved" }); }
        catch (err) { $(".pg-labmsg").innerHTML = `<span class="h-err">${esc(err.message)}</span>`; }
      }
    });
  }

  // ---- create task ------------------------------------------------------------------------------------
  async function createTaskPage(name) {
    const d = await api(`/api/projects/${enc(name)}`);
    const multi = new URLSearchParams(location.search).has("multi");
    let files = [];
    main.innerHTML = `<div class="pg-bar"><a class="pg-back" href="/projects/${enc(name)}">‹ Back to project</a></div>
      <div class="pg-title"><h1>${multi ? "Create multi tasks" : "Create a new task"}</h1></div>
      <form class="pg-form h-form" autocomplete="off"><section class="h-panel"><h2>Basic configuration</h2>
        <label class="h-field"><span>Name</span><input type="text" name="name" ${multi ? 'value="{{file_name}}"' : 'placeholder="Leave empty: the file or folder name"'}>
          ${multi ? `<small>One task per file: {{file_name}} is each file's name, {{index}} its number.</small>` : ""}</label>
        <div class="h-field"><span>Project</span><span><a href="/projects/${enc(name)}">${esc(name)}</a> · ${esc(TASKS[d.task] || d.task)}</span></div>
        <label class="h-field"><span>Subset</span><input type="text" name="subset" list="pg-subsets" placeholder="Optional: Train, Test, Validation…">
          <datalist id="pg-subsets">${[...new Set(["Train", "Test", "Validation", ...d.subsets])].map((s) => `<option value="${esc(s)}">`).join("")}</datalist>
          <small>Exporting the project puts each subset in its own folders.</small></label>
        <div class="h-field"><span>Labels</span><span class="h-sub">Project labels will be used:</span>
          <div class="pg-labels">${d.labels.map((l, i) => `<span class="pg-lab"><i style="background:${esc(l.color || colorOf(i))}"></i>${esc(l.name)}</span>`).join("") || `<span class="h-sub">none yet</span>`}</div></div>
        <div class="h-field"><span>Select files *</span><div class="pg-files"></div>
          <div class="h-row"><button type="button" data-f="video">Add a video…</button><button type="button" data-f="folder">Add an image folder…</button>
            <input type="text" class="pg-path" placeholder="or type a path and press Enter" aria-label="Path of a video or image folder"></div>
          <small>${multi ? "Add several: each becomes a task." : "One video or one image folder."}</small></div></section>
        <details class="h-panel h-more"><summary>Advanced configuration</summary><div class="pg-two" style="margin-top:14px">
          <label class="h-field"><span>Image quality</span><input type="number" name="quality" min="5" max="100" value="${prefs().quality || 95}"><small>JPEG quality of a video's frames, 5–100.</small></label>
          <label class="h-check-row" style="align-self:center"><input type="checkbox" name="lossless" ${d.frame_format === "webp" || prefs().lossless ? "checked" : ""}> Lossless frames<small>Exact pixels, about twice the space.</small></label>
          <label class="h-field"><span>Frame step</span><input type="number" name="every" min="1" value="${prefs().every || 5}"><small>Keep every Nth frame of a video.</small></label>
          <label class="h-field"><span>Start frame</span><input type="number" name="start" min="0" placeholder="first"></label>
          <label class="h-field"><span>Stop frame</span><input type="number" name="stop" min="0" placeholder="last"></label>
          <label class="h-field"><span>Segment size</span><input type="number" name="segment" min="0" placeholder="whole task" value="${prefs().segment_size || ""}"><small>Frames per job; empty: one job.</small></label>
          <label class="h-field"><span>Sorting method</span><select name="sorting"><option value="lexicographical">Lexicographical</option><option value="natural">Natural</option></select><small>Order of an image folder's pictures.</small></label>
        </div></details>
        <div class="h-row"><button type="submit" class="primary" value="open">Submit &amp; Open</button><button type="submit" value="continue">Submit &amp; Continue</button><span class="h-sub pg-msg"></span></div>
        <div class="pg-job"></div></form>`;
    const f = main.querySelector("form");
    const renderFiles = () => {
      main.querySelector(".pg-files").innerHTML = files.map((p, i) => `<div class="h-src"><span title="${esc(p)}">${esc(p)}</span><button type="button" data-rm="${i}" aria-label="Remove">×</button></div>`).join("")
        || `<span class="h-sub">Nothing selected yet.</span>`;
      f.querySelector('[value="open"]').textContent = files.length > 1 ? `Submit ${files.length} tasks & Open` : "Submit & Open";
    };
    const add = (p) => { if (!p) return; files = multi ? [...files, p] : [p]; renderFiles(); };
    renderFiles();
    main.addEventListener("click", async (e) => {
      const b = e.target.closest("[data-f],[data-rm]"); if (!b) return;
      if (b.dataset.rm != null) { files.splice(+b.dataset.rm, 1); renderFiles(); return; }
      add(await pick(b.dataset.f === "video" ? { kind: "video", folder: false } : { kind: "images", folder: true }));
    });
    main.querySelector(".pg-path").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); add(e.target.value.trim()); e.target.value = ""; } });
    f.addEventListener("submit", async (e) => {
      e.preventDefault();
      const open = e.submitter?.value !== "continue", msg = main.querySelector(".pg-msg");
      if (!files.length) { msg.innerHTML = `<span class="h-err">Select a video or an image folder</span>`; return; }
      f.querySelectorAll("button[type=submit]").forEach((x) => (x.disabled = true));
      try {
        const { job } = await api(`/api/projects/${enc(name)}/tasks`, { sources: files, name: f.name.value.trim(), subset: f.subset.value.trim(),
          quality: f.quality.value, lossless: f.lossless.checked, every: f.every.value, start: f.start.value, stop: f.stop.value,
          segment_size: f.segment.value, sorting: f.sorting.value });
        followJob(job, main.querySelector(".pg-job"), {
          onDone: (res) => {
            if (open && res.tasks.length) location.href = `/projects/${enc(name)}/tasks/${res.tasks[0]}`;
            else { files = []; renderFiles(); msg.innerHTML = `Created ${res.tasks.length} task${res.tasks.length === 1 ? "" : "s"}. <a href="/projects/${enc(name)}">Back to the project</a>`; }
            pollNotes();
          },
          onEnd: () => f.querySelectorAll("button[type=submit]").forEach((x) => (x.disabled = false)) });
      } catch (err) { msg.innerHTML = `<span class="h-err">${esc(err.message)}</span>`; f.querySelectorAll("button[type=submit]").forEach((x) => (x.disabled = false)); }
    });
  }

  // ---- task page ----------------------------------------------------------------------------------------
  async function taskPage(name, tid) {
    let t;
    const load = async () => { t = await api(`/api/projects/${enc(name)}/tasks/${tid}`); };
    try { await load(); } catch (err) { main.innerHTML = `<p class="h-err">${esc(err.message)}</p>`; return; }
    const render = () => {
      const i = t.info || {}, d = i.duration;
      const rows = t.kind === "video" ? [["Source", t.source], ["Resolution", i.width ? `${i.width} × ${i.height}` : "?"], ["Frame rate", i.fps ? `${+i.fps.toFixed(3)} fps` : "?"],
        ["Duration", d ? `${Math.floor(d / 60)}:${String(Math.floor(d % 60)).padStart(2, "0")}` : "?"], ["Codec", i.codec ? i.codec.toUpperCase() : "?"],
        ["File size", mb(i.size) || "?"], ["Frames in the video", i.frames ?? "?"], ["Frame step", `every ${t.every}`],
        ["Start / stop frame", `${t.start ?? "first"} / ${t.stop ?? "last"}`], ["Frames kept", t.frames],
        ["Stored as", t.format === "webp" ? `lossless WebP, ${mb(i.stored)}` : `JPEG quality ${t.quality}, ${mb(i.stored)}`]]
        : [["Source", t.source], ["Resolution", i.width ? `${i.width} × ${i.height}${i.mixed_sizes ? " (mixed sizes)" : ""}` : "?"], ["Images", t.frames],
           ["Size", mb(i.size) || "?"], ["Sorting method", t.sorting || "lexicographical"]];
      main.innerHTML = `<div class="pg-bar"><a class="pg-back" href="/projects/${enc(name)}">‹ Back to project</a><span class="pg-sp"></span>
          <details class="h-menu" style="position:relative;top:auto;right:auto"><summary style="width:auto;padding:0 12px;border:1px solid var(--line);background:var(--paper)">Actions ▾</summary><div class="h-menu-list">
            <button type="button" data-t-upload="${t.id}" data-p="${esc(name)}">Upload annotations</button>
            <button type="button" data-t-export="${t.id}" data-p="${esc(name)}">Export task dataset</button>
            <button type="button" data-t-backup="${t.id}" data-p="${esc(name)}">Backup task</button>
            <button type="button" class="danger" data-t-delete="${t.id}" data-p="${esc(name)}" data-name="${esc(t.name)}">Delete</button></div></details></div>
        <div class="pg-top"><span class="pg-thumb big" style="background-image:url('${t.preview}')"></span>
          <div class="pg-section"><input class="pg-name" value="${esc(t.name)}" aria-label="Task name" title="Click to rename" translate="no">
            <p class="pg-meta">Task #${t.id} created ${esc(when(t.created))} in project <a href="/projects/${enc(name)}">${esc(name)}</a>${t.updated ? ` · Last updated ${esc(ago(t.updated))}` : ""} · ${statusChip(t.status)}</p>
            <label class="h-field" style="max-width:320px"><span>Subset</span><input type="text" class="pg-subset-in" list="pg-subsets" value="${esc(t.subset)}" placeholder="none">
              <datalist id="pg-subsets">${[...new Set(["Train", "Test", "Validation", ...t.subsets])].map((s) => `<option value="${esc(s)}">`).join("")}</datalist></label>
            ${progress(t)}
            <dl class="pg-kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("")}</dl></div></div>
        <section class="pg-section"><h2>Jobs <small>${t.jobs.length}</small></h2><div class="pg-jobs">${t.jobs.map((j) => `<article class="pg-job">
          <h3><a href="${jobUrl(name, t.id, j.id)}">Job #${j.id}</a> ${statusChip(j.state)}</h3>
          <div class="h-row"><label class="h-sub">Stage <select data-stage="${j.id}">${STAGES.map((s) => `<option${s === j.stage ? " selected" : ""}>${s}</option>`).join("")}</select></label>
            <label class="h-sub">State <select data-state="${j.id}">${STATES.map((s) => `<option${s === j.state ? " selected" : ""}>${s}</option>`).join("")}</select></label></div>
          <dl class="pg-kv"><dt>Frame range</dt><dd>${j.start - t.start_item + 1}–${j.end - t.start_item}</dd><dt>Frame count</dt><dd>${j.frames} (${t.frames ? Math.round(j.frames / t.frames * 100) : 0}%)</dd>
            <dt>Confirmed</dt><dd>${j.confirmed} of ${j.frames} (${j.labeled} labeled)</dd><dt>Updated</dt><dd>${j.updated ? esc(ago(j.updated)) : "not yet"}</dd></dl>
          <div class="h-row"><a href="${jobUrl(name, t.id, j.id)}"><button type="button" class="primary">Open</button></a></div></article>`).join("")}</div></section>`;
      const nm = $(".pg-name");
      nm.onkeydown = (e) => { if (e.key === "Enter") nm.blur(); if (e.key === "Escape") { nm.value = t.name; nm.blur(); } };
      nm.onchange = async () => { try { t = { ...t, ...(await api(`/api/projects/${enc(name)}/tasks/${tid}`, { name: nm.value }, "PATCH")) }; render(); toast({ title: "Renamed" }); } catch (err) { toast({ level: "error", title: "Could not rename", detail: err.message }); nm.value = t.name; } };
      $(".pg-subset-in").onchange = async (e) => { try { t = { ...t, ...(await api(`/api/projects/${enc(name)}/tasks/${tid}`, { subset: e.target.value }, "PATCH")) }; render(); toast({ title: `Subset: ${t.subset || "none"}` }); } catch (err) { toast({ level: "error", title: "Could not change the subset", detail: err.message }); } };
    };
    render();
    main.addEventListener("change", async (e) => {
      const st = e.target.closest("[data-stage],[data-state]"); if (!st) return;
      const body = st.dataset.stage ? { stage: st.value } : { state: st.value };
      try { await api(`/api/projects/${enc(name)}/jobs/${st.dataset.stage || st.dataset.state}`, body, "PATCH"); await load(); render(); }
      catch (err) { toast({ level: "error", title: "Could not change the job", detail: err.message }); }
    });
    main.addEventListener("click", async (e) => {
      await taskActions(e, async () => { try { await load(); render(); } catch { location.href = `/projects/${enc(name)}`; } }, async () => t.formats);
    });
  }

  // ---- settings (per account: engine/accounts.py) ------------------------------------------------------
  async function settingsPage() {
    let me;
    try { me = await api("/api/me"); } catch (err) { main.innerHTML = `<p class="h-err">${esc(err.message)}</p>`; return; }
    const projects = await api("/api/projects").catch(() => []);
    const P = me.prefs;
    const sw = (k, title, note) => `<label class="st-row"><span><b>${title}</b><small>${note}</small></span>
      <input type="checkbox" class="st-switch" role="switch" data-pref="${k}"${P[k] ? " checked" : ""}></label>`;
    const num = (k, title, note, min, max) => `<label class="st-row"><span><b>${title}</b><small>${note}</small></span>
      <input type="number" data-pref="${k}" min="${min}" max="${max}" value="${P[k]}"></label>`;
    main.innerHTML = `<div class="pg-title"><h1>Settings</h1><span class="st-saved" role="status" aria-live="polite"></span></div>
      <p class="pg-meta">Signed in as <b translate="no">${esc(me.user.name)}</b>. Changes are saved at once and follow your account on this computer.</p>
      <div class="st"><nav class="st-nav" aria-label="Settings sections">${SECTIONS.map(([id, t]) => `<a href="#${id}">${t}</a>`).join("")}</nav><div class="st-body">
      <section class="h-panel st-sec" id="account"><h2>Account</h2>
        <div class="st-row"><span><b>Username</b><small>You sign in with it; it can't be changed.</small></span><code translate="no">${esc(me.user.username)}</code></div>
        <form class="st-row st-name"><span><b>Your name</b><small>Shown in the top bar.</small></span>
          <span class="h-row"><input type="text" name="name" value="${esc(me.user.name)}" maxlength="60" autocomplete="name" aria-label="Your name"><button type="submit">Save</button></span></form>
        <details class="st-more"><summary>Change password</summary><form class="h-form st-pw" autocomplete="off">
          <label class="h-field"><span>Current password</span><input type="password" name="old" autocomplete="current-password" required></label>
          <label class="h-field"><span>New password</span><input type="password" name="new" autocomplete="new-password" minlength="8" required><small>At least 8 characters. Other browsers are signed out.</small></label>
          <label class="h-field"><span>Repeat the new password</span><input type="password" name="repeat" autocomplete="new-password" minlength="8" required></label>
          <div class="h-row"><button type="submit" class="primary">Change password</button><span class="h-sub st-pwmsg" role="status"></span></div></form></details>
        <details class="st-more"><summary>Delete this account</summary><form class="h-form st-del" autocomplete="off">
          <p class="h-sub">Removes the account and its settings. Projects stay where they are, for other accounts or for a new one.</p>
          <label class="h-field"><span>Password</span><input type="password" name="password" autocomplete="current-password" required></label>
          <div class="h-row"><button type="submit" class="danger">Delete my account</button><span class="h-err st-delmsg" role="alert"></span></div></form></details>
        <div class="h-row" style="margin-top:12px"><button type="button" data-signout>Sign out</button></div></section>

      <section class="h-panel st-sec" id="appearance"><h2>Appearance</h2>
        <div class="st-row st-col"><span><b>Theme</b><small>Match my computer follows Windows' light or dark mode.</small></span>
          <span class="h-seg three st-theme" role="group" aria-label="Theme" style="--x:${THEMES.indexOf(P.theme)}"><span class="h-seg-thumb" aria-hidden="true"></span>
            ${THEMES.map((t) => `<button type="button" data-theme="${t}" aria-pressed="${t === P.theme}">${THEME_NAMES[t]}</button>`).join("")}</span></div>
        <div class="st-row st-col"><span><b>Colour palette</b><small>Buttons, links and highlights. The label colours of your projects stay as they are.</small></span>
          <div class="st-swatches" role="radiogroup" aria-label="Colour palette">${Object.entries(ACCENTS).map(([a, [name, c]]) =>
            `<button type="button" class="st-sw" role="radio" aria-checked="${a === P.accent}" data-accent="${a}" style="--sw:${c}"><i aria-hidden="true"></i>${name}</button>`).join("")}</div></div>
        ${sw("motion", "Animations", "The start screen demo, Rivet's moves, sliding panels. Also off when Windows asks for less motion.")}</section>

      <section class="h-panel st-sec" id="helpers"><h2>Helpers and messages</h2>
        ${sw("rivet", "Rivet, the helper", "The robot at the bottom right that answers questions about PartLabeler. Applies to pages you open next.")}
        ${sw("tips", "Rivet's tips", "A “Did you know?” bubble now and then.")}
        ${sw("hints", "First-steps hint", "The short how-to over the frame when a job opens in the annotator.")}
        ${sw("toasts", "Pop-up messages", "Off: only problems pop up. The bell keeps every message either way.")}
        ${sw("update_check", "Look for updates", "Asks GitHub for a new version (at most every 6 hours). Off: only when you click Check for updates.")}</section>

      <section class="h-panel st-sec" id="annotating"><h2>Annotating and export</h2>
        ${num("track_n", "Frames per track", "How far T (ahead) and R (back) track. The annotator remembers changes you make there.", 1, 10000)}
        ${num("brush", "Brush size", "Outline brush and eraser, in pixels (segmentation projects).", 1, 120)}
        <label class="st-row"><span><b>Export format</b><small>Picked first in export dialogs, when the project offers it.</small></span>
          <select data-pref="export_format">${Object.entries(EXPORTS).map(([v, t]) => `<option value="${v}"${v === P.export_format ? " selected" : ""}>${t}</option>`).join("")}</select></label>
        ${sw("confirmed_only", "Confirmed frames only", "Ticked by default when exporting: only frames you confirmed.")}</section>

      <section class="h-panel st-sec" id="tasks"><h2>New tasks</h2><p class="h-sub">The starting values of Create a new task; each task can still change them.</p>
        ${num("every", "Frame step", "Keep every Nth frame of a video.", 1, 1000)}
        ${num("quality", "Image quality", "JPEG quality of a video's stored frames, 5 to 100.", 5, 100)}
        ${sw("lossless", "Lossless frames", "Exact pixels as WebP, about twice the disk space.")}
        ${num("segment_size", "Segment size", "Frames per job; 0 makes one job per task.", 0, 1000000)}</section>

      <section class="h-panel st-sec" id="storage"><h2>Projects folder</h2>
        <p class="h-sub">Where your projects are kept, with their frames and labels, the trash, backups and Teach runs. This setting is yours only.</p>
        <div class="st-row st-col"><span><b>Now</b></span><code class="st-code st-home" translate="no">${esc(me.home)}</code></div>
        <form class="h-form st-folder" autocomplete="off">
          <label class="h-field"><span>New folder</span><span class="h-row"><input type="text" name="path" placeholder="Full path, e.g. D:\\PartLabeler\\projects" aria-label="New projects folder">
            <button type="button" data-st="browse">Browse…</button></span></label>
          <label class="h-check-row"><input type="checkbox" name="move" checked> Move everything in the current folder there
            <small>Projects, trash, backups and Teach runs. Off: the new folder starts as it is and the old projects stay where they are.
              Other accounts using the current folder no longer see what moves.</small></label>
          <div class="h-row"><button type="submit" class="primary">Use this folder</button>
            <button type="button" data-st="default"${me.home === me.default_home ? " hidden" : ""}>Back to the default folder</button></div>
          <p class="h-sub">The app's default folder: <code class="st-code" translate="no">${esc(me.default_home)}</code></p></form>
        <div class="st-job"></div></section>

      <section class="h-panel st-sec" id="transfer"><h2>Unfinished work as a zip</h2>
        <p class="h-sub">Carry on elsewhere: a backup zip keeps the frames, labels, confirmed frames and jobs.</p>
        <ul class="st-list"><li><b>A whole project</b>: its ⋯ menu on the start screen, or Actions on its page, then <b>Backup project</b>.</li>
          <li><b>One task</b>: its ⋯ menu, then <b>Backup task</b>.</li>
          <li>The zips go to the <code>_backups</code> folder inside your projects folder.</li></ul>
        <div class="h-row"><button type="button" class="primary" data-st="import">Import a zip…</button><label class="h-sub">as
          <select class="st-into" aria-label="Import as"><option value="">a new project</option>${projects.map((p) =>
            `<option value="${esc(p.name)}">tasks of ${esc(p.name)} (${esc(TASKS[p.task] || p.task)})</option>`).join("")}</select></label></div>
        <div class="st-ijob"></div></section>
      </div></div>`;

    const saved = $(".st-saved");
    let savedTimer = 0;
    const flashSaved = (text = "Saved") => { saved.textContent = text; saved.classList.add("on"); clearTimeout(savedTimer); savedTimer = setTimeout(() => saved.classList.remove("on"), 1800); };
    const apply = (p) => {
      const r = document.documentElement.dataset;
      r.theme = p.theme; r.accent = p.accent;
      if (p.motion) delete r.motion; else r.motion = "off";
    };
    async function save(changes) {
      try {
        const p = await api("/api/me/prefs", changes);
        Object.assign(window.PL.prefs, p); apply(p); flashSaved();
        main.querySelectorAll("input[type=number][data-pref]").forEach((x) => { x.value = p[x.dataset.pref]; });
        main.querySelectorAll("[data-theme]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.theme === p.theme)));
        $(".st-theme").style.setProperty("--x", THEMES.indexOf(p.theme));
        main.querySelectorAll("[data-accent]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.accent === p.accent)));
      } catch (err) { toast({ level: "error", title: "Not saved", detail: err.message }); }
    }
    main.addEventListener("change", (e) => {
      const k = e.target.dataset?.pref; if (!k) return;
      save({ [k]: e.target.type === "checkbox" ? e.target.checked : e.target.type === "number" ? +e.target.value : e.target.value });
    });
    main.addEventListener("click", async (e) => {
      const th = e.target.closest("button[data-theme]"), ac = e.target.closest("button[data-accent]"), st = e.target.closest("[data-st]")?.dataset.st;
      if (th) save({ theme: th.dataset.theme });
      else if (ac) save({ accent: ac.dataset.accent });
      else if (st === "browse") { const p = await pick({ kind: "dir", folder: true, start: me.home }); if (p) $(".st-folder").path.value = p; }
      else if (st === "default") { $(".st-folder").path.value = me.default_home; $(".st-folder").path.focus(); }
      else if (st === "import") importZip();
    });
    $(".st-swatches").addEventListener("keydown", (e) => {                       // arrows move along the palette, as radio buttons do
      if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) return;
      e.preventDefault();
      const all = [...main.querySelectorAll("[data-accent]")], i = all.indexOf(document.activeElement);
      const b = all[(i + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : all.length - 1)) % all.length]; b.focus(); b.click();
    });
    $(".st-name").addEventListener("submit", async (e) => {
      e.preventDefault();
      try { const u = await api("/api/me", { name: e.target.name.value }, "PATCH"); window.PL.user = u; flashSaved(); root.querySelector(".h-acct-who b").textContent = u.name; }
      catch (err) { toast({ level: "error", title: "Not saved", detail: err.message }); }
    });
    $(".st-pw").addEventListener("submit", async (e) => {
      e.preventDefault(); const f = e.target, msg = $(".st-pwmsg");
      if (f.new.value !== f.repeat.value) { msg.innerHTML = `<span class="h-err">The two new passwords are different</span>`; return; }
      try { await api("/api/me/password", { old: f.old.value, new: f.new.value }); f.reset(); msg.textContent = "Password changed. Other browsers were signed out."; }
      catch (err) { msg.innerHTML = `<span class="h-err">${esc(err.message)}</span>`; }
    });
    $(".st-del").addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!(await ask("Delete your account?", "Your settings go; your projects stay in their folder.", "Delete account"))) return;
      try { await api("/api/me/delete", { password: e.target.password.value }); location.href = "/login"; }
      catch (err) { $(".st-delmsg").textContent = err.message; }
    });
    $(".st-folder").addEventListener("submit", async (e) => {
      e.preventDefault(); const f = e.target, path = f.path.value.trim();
      if (!path) { f.path.focus(); return; }
      if (f.move.checked && !(await ask("Move your projects?", `Everything in ${me.home} moves to ${path}. Keep PartLabeler open until it is done.`, "Move"))) return;
      try {
        const r = await api("/api/me/projects-folder", { path, move: f.move.checked });
        const done = async () => { me = await api("/api/me"); $(".st-home").textContent = me.home; f.path.value = "";
          main.querySelector('[data-st="default"]').hidden = me.home === me.default_home; flashSaved("Projects folder changed"); };
        if (r.job) followJob(r.job, $(".st-job"), { onDone: done }); else done();
      } catch (err) { $(".st-job").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; }
    });
    async function importZip() {
      const path = await pick({ kind: "backup", folder: false }); if (!path) return;
      const into = $(".st-into").value;
      try {
        const { job } = await api(into ? `/api/projects/${enc(into)}/tasks/import` : "/api/backups/restore", { path });
        followJob(job, $(".st-ijob"), { onDone: (res) => {
          const name = into || res.name;
          $(".st-ijob").insertAdjacentHTML("beforeend", `<p class="h-sub">Done: <a href="/projects/${enc(name)}">open ${esc(name)}</a>.</p>`);
          pollNotes();
        } });
      } catch (err) { $(".st-ijob").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; }
    }
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }

  // ---- all tasks, all jobs ----------------------------------------------------------------------------
  async function tasksPage() {
    const all = await api("/api/tasks");
    const formats = {};
    let q = "";
    const render = () => {
      const shown = all.filter((t) => !q || `${t.name} ${t.project} ${t.subset}`.toLowerCase().includes(q));
      main.innerHTML = `<div class="pg-title"><h1>Tasks</h1><span class="h-count">${all.length}</span></div>
        <div class="pg-tools"><input type="text" class="pg-q" placeholder="Search…" value="${esc(q)}" aria-label="Search tasks"></div>
        <div class="pg-list">${shown.map((t) => taskRow(t, t.project, true)).join("") || `<p class="pg-empty">${all.length ? "Nothing matches." : "No tasks yet: open a project and create one."}</p>`}</div>`;
      const qi = $(".pg-q"); qi.oninput = () => { q = qi.value.trim().toLowerCase(); render(); const n = $(".pg-q"); n.focus(); n.setSelectionRange(n.value.length, n.value.length); };
    };
    render();
    main.addEventListener("click", (e) => taskActions(e, () => location.reload(), async (p) => (formats[p] ||= (await api(`/api/projects/${enc(p)}`)).formats)));
  }
  async function jobsPage() {
    const all = await api("/api/annotation-jobs");
    main.innerHTML = `<div class="pg-title"><h1>Jobs</h1><span class="h-count">${all.length}</span></div>
      <div class="pg-jobs">${all.map((j) => `<a class="pg-tile" href="${jobUrl(j.project, j.task, j.id)}"><span class="pg-thumb" style="background-image:url('${j.preview}')"></span>
        <b>Job #${j.id} ${statusChip(j.state)}</b><span class="pg-meta">${esc(j.project)} · ${esc(j.task_name)}${j.subset ? ` · ${esc(j.subset)}` : ""}</span>
        <span class="pg-meta">Stage: ${esc(j.stage)} · ${j.frames} frames · ${j.confirmed} confirmed${j.updated ? ` · ${esc(ago(j.updated))}` : ""}</span></a>`).join("")
        || `<p class="pg-empty">No jobs yet: every task has at least one.</p>`}</div>`;
  }

  ({ project: () => projectPage(args.project), "create-task": () => createTaskPage(args.project), task: () => taskPage(args.project, args.task),
     tasks: tasksPage, jobs: jobsPage, settings: settingsPage })[args.view]?.();
}

const FORMAT_NAMES = { yolo: "YOLO (Ultralytics)", coco: "COCO 1.0", cvat: "CVAT for images 1.1", voc: "Pascal VOC", labelstudio: "Label Studio",
                       folders: "Class folders (ImageFolder)", csv: "CSV list" };
const colorOf = (i) => `hsl(${(i * 137.508) % 360} 78% 52%)`;
