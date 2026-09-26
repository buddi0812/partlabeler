// PartLabeler annotator. One ES module, two hosts: the local FastAPI page and a notebook widget
// (anywidget). `model` only needs send(msg) and on("msg:custom", fn); the web host also sets
// model.homeUrl (link back to the start screen and REST helpers) and model.startItem. Protocol: engine/api.py.
// Also exports notificationCenter(), shared with the start screen (ui/home.js).

const CSS = `
.pl { --bg:#eef1f4; --panel:#fff; --ink:#15202b; --muted:#56636f; --line:#d5dbe1; --accent:#0a7c78; --accent-ink:#075e5b;
  --accent-soft:#e1f0ef; --st0:#dfe4e9; --st1:#b8a8e6; --st2:#6aa6f2; --st3:#27a79b; --st4:#2f9e5b; --flag:#e09a2c; --bad:#c2412f;
  font:14px/1.45 "Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif; color:var(--ink); background:var(--bg);
  display:grid; grid-template-rows:auto auto minmax(0,1fr) auto; gap:8px; padding:0 12px 10px; outline:none; position:relative;
  -webkit-tap-highlight-color:transparent; }
.pl.web { min-height:100vh; }
.pl:fullscreen { overflow:auto; }
.pl.big { height:var(--plh); }
.pl.big .pl-stage canvas { max-height:calc(var(--plh) - 230px); }
.pl.big .pl-side { max-height:calc(var(--plh) - 150px); }
.pl * { box-sizing:border-box; }
.pl [hidden] { display:none !important; }
.pl-head { display:flex; align-items:center; gap:12px; margin:0 -12px; padding:8px 12px; background:var(--panel); border-bottom:1px solid var(--line); min-width:0; }
.pl-back { color:var(--accent-ink); text-decoration:none; font-weight:600; white-space:nowrap; padding:4px 6px; border-radius:6px; }
.pl-back:hover { background:var(--accent-soft); }
.pl-title { font-weight:600; font-size:15px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
.pl-where { color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
.pl-chip { font-size:12px; font-weight:600; padding:2px 9px; border-radius:99px; white-space:nowrap; }
.pl-chip.rev { background:#dcf1e4; color:#1f7a44; } .pl-chip.flag { background:#fbecd3; color:#8a5304; }
.pl-sp { flex:1; }
.pl-saved { color:var(--muted); font-size:13px; white-space:nowrap; display:inline-flex; gap:5px; align-items:center; }
.pl-saved svg { width:14px; height:14px; } .pl-saved.lost { color:var(--bad); font-weight:600; }
.pl-bar-tools { display:flex; flex-wrap:wrap; align-items:center; gap:6px 14px; }
.pl-group { display:inline-flex; gap:4px; align-items:center; }
.pl-group-label { color:var(--muted); font-size:13px; margin-right:2px; }
.pl button { font:inherit; font-size:13.5px; border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:7px;
  padding:5px 11px; cursor:pointer; white-space:nowrap; touch-action:manipulation; }
.pl button:hover:not(:disabled) { border-color:var(--accent); background:#f7fbfb; }
.pl button:disabled { opacity:.45; cursor:default; }
.pl button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
.pl button.primary:hover:not(:disabled) { background:var(--accent-ink); }
.pl button.icon { padding:5px 8px; }
.pl :focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.pl select, .pl input[type=number], .pl input[type=text], .pl input[type=search] { font:inherit; font-size:13.5px; border:1px solid var(--line);
  border-radius:7px; padding:4px 7px; background:var(--panel); color:var(--ink); }
.pl input[type=number] { width:64px; font-variant-numeric:tabular-nums; }
.pl-main { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:10px; min-height:0; }
.pl-stage { position:relative; background:#0d1217; border-radius:10px; overflow:hidden; display:flex; align-items:center; justify-content:center; min-height:320px; }
.pl-stage canvas { max-width:100%; max-height:calc(100vh - 250px); display:block; cursor:crosshair; }
.pl-hint { position:absolute; left:50%; top:16px; transform:translateX(-50%); background:rgba(21,32,43,.88); color:#fff; border-radius:10px;
  padding:10px 14px; max-width:min(560px, 90%); font-size:13.5px; display:flex; gap:12px; align-items:flex-start; }
.pl-hint button { background:transparent; color:#fff; border-color:rgba(255,255,255,.4); padding:2px 8px; }
.pl-hint kbd { color:var(--ink); }
.pl-side { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; display:flex; flex-direction:column;
  gap:14px; overflow:auto; max-height:calc(100vh - 150px); overscroll-behavior:contain; }
.pl-side h2 { margin:0 0 6px; font-size:13px; font-weight:600; color:var(--ink); }
.pl-side section { display:flex; flex-direction:column; }
.pl-tools { display:flex; gap:4px; flex-wrap:wrap; align-items:center; } .pl-tools button[aria-pressed=true] { background:var(--ink); color:#fff; border-color:var(--ink); }
.pl-list { display:flex; flex-direction:column; gap:2px; overflow:auto; overscroll-behavior:contain; }
.pl-classes { max-height:24vh; } .pl-boxes { max-height:18vh; }
.pl-row { display:grid; grid-template-columns:12px minmax(16px, auto) minmax(0,1fr) auto; gap:7px; align-items:center; width:100%;
  padding:4px 6px; border:1px solid transparent; border-radius:6px; background:none; text-align:left; font:inherit; font-size:13.5px; color:inherit; cursor:pointer; }
.pl-row:hover { background:var(--bg); } .pl-row[aria-pressed=true] { background:var(--accent-soft); border-color:var(--accent); }
.pl-rowwrap { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:center; }
.pl-sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
.pl-key { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
.pl-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pl-src { font-size:12px; color:var(--muted); }
.pl-x { border:none !important; padding:2px 7px !important; color:var(--muted); background:none !important; font-size:16px !important; }
.pl-x:hover { color:var(--bad); }
.pl-empty { color:var(--muted); font-size:13px; }
.pl-side .flash { outline:2px solid var(--accent); outline-offset:4px; border-radius:8px; }
.pl-settings summary { cursor:pointer; font-weight:600; font-size:13px; }
.pl-settings[open] summary { margin-bottom:8px; }
.pl-field { display:flex; flex-direction:column; gap:4px; font-size:13px; margin-bottom:6px; }
.pl-bottom { display:grid; gap:5px; }
.pl-strip { width:100%; height:28px; display:block; cursor:pointer; border-radius:5px; background:var(--st0); }
.pl-foot { display:flex; gap:14px; align-items:center; color:var(--muted); font-variant-numeric:tabular-nums; min-height:18px; flex-wrap:wrap; font-size:13px; }
.pl-bar { flex:0 0 160px; height:6px; background:var(--st0); border-radius:3px; overflow:hidden; }
.pl-bar i { display:block; height:100%; background:var(--accent); width:0; }
.pl-legend { display:flex; gap:12px; flex-wrap:wrap; } .pl-legend span { display:inline-flex; gap:5px; align-items:center; }
.pl-toasts { position:fixed; right:16px; bottom:16px; display:flex; flex-direction:column; gap:8px; z-index:30; max-width:400px; }
.pl-toast { background:var(--ink); color:#fff; padding:9px 12px; border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,.22); display:flex; gap:10px; align-items:flex-start; }
.pl-toast b { font-weight:600; display:block; } .pl-toast small { opacity:.8; display:block; margin-top:1px; overflow-wrap:anywhere; }
.pl-toast.error { background:var(--bad); } .pl-toast.warning { background:#8a5304; }
.pl-toast button { background:transparent; color:#fff; border-color:rgba(255,255,255,.45); padding:2px 9px; margin-left:auto; }
.pl-banner { background:#fde8e5; color:#8b2a1d; border:1px solid #f3b8ae; border-radius:8px; padding:8px 12px; font-weight:600; }
.pl-banner.info { background:#fdf1d2; color:#6b4700; border-color:#f0d08a; font-weight:500; }
.pl-help-dlg { border:1px solid var(--line); border-radius:12px; padding:0; width:min(720px, calc(100vw - 32px)); max-height:85vh; color:var(--ink); }
.pl-help-dlg::backdrop { background:rgba(21,32,43,.45); }
.pl-help-dlg header { display:flex; align-items:center; padding:14px 18px; border-bottom:1px solid var(--line); }
.pl-help-dlg h2 { margin:0; font-size:16px; } .pl-help-dlg header button { margin-left:auto; }
.pl-help-grid { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:4px 28px; padding:14px 18px 18px; overflow:auto; }
.pl-help-grid h3 { grid-column:1 / -1; margin:10px 0 2px; font-size:13px; color:var(--muted); font-weight:600; }
.pl-help-grid div { display:flex; justify-content:space-between; gap:12px; padding:3px 0; border-bottom:1px solid #eef1f4; font-size:13.5px; }
.pl kbd, .pl-help-dlg kbd { font:12px/1.2 ui-monospace,"Cascadia Mono",Consolas,monospace; border:1px solid var(--line); border-bottom-width:2px; border-radius:5px; padding:1px 5px; background:#f7f9fa; white-space:nowrap; }
.pl-brush { display:flex; gap:8px; align-items:center; font-size:13px; margin-top:6px; } .pl-brush input { flex:1; accent-color:var(--accent); }
.pl-gridwrap { display:flex; flex-direction:column; gap:8px; min-height:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.pl-gridbar { display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center; }
.pl-gridbar h2 { margin:0; font-size:15px; font-weight:600; }
.pl-filters { display:flex; gap:4px; flex-wrap:wrap; }
.pl-filters button { padding:3px 10px; border-radius:99px; font-size:12.5px; }
.pl-filters button[aria-pressed=true] { background:var(--ink); color:#fff; border-color:var(--ink); }
.pl-filters b { font-weight:600; font-variant-numeric:tabular-nums; margin-left:3px; opacity:.75; }
.pl-grid { overflow:auto; max-height:calc(100vh - 205px); min-height:260px; overscroll-behavior:contain; display:flex; flex-direction:column; gap:14px; padding:2px 4px 8px 2px; }
.pl.big .pl-grid { max-height:calc(var(--plh) - 165px); }
.pl-gsec h3 { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:0 0 6px; font-size:13.5px; font-weight:600; position:sticky; top:0; background:var(--panel); z-index:1; padding:5px 0; }
.pl-gsec h3 .pl-src { font-weight:400; }
.pl-gsec h3 select { font-size:12.5px; padding:2px 5px; }
.pl-cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(var(--card, 132px), 1fr)); gap:8px; }
.pl .pl-card { position:relative; display:flex; flex-direction:column; padding:0; border:2px solid var(--line); border-radius:9px; overflow:hidden; background:#0d1217; text-align:left; color:var(--ink); }
.pl .pl-card:hover:not(:disabled) { border-color:var(--accent); background:#0d1217; }
.pl-card img { width:100%; aspect-ratio:1; object-fit:contain; display:block; background:#0d1217; }
.pl-card-cls { display:flex; gap:5px; align-items:center; padding:3px 6px; background:var(--panel); font-size:12px; min-height:23px; min-width:0; }
.pl-card-cls span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
.pl-card-cls .pl-sw { flex:none; }
.pl .pl-card.sugg { border-style:dashed; } .pl-card.sugg .pl-card-cls { color:var(--muted); font-style:italic; }
.pl .pl-card[aria-pressed=true] { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent); }
.pl .pl-card.cur { outline:2px solid var(--ink); outline-offset:2px; }
.pl-card .pl-badge { position:absolute; top:4px; left:4px; font-size:11px; font-weight:600; padding:1px 6px; border-radius:99px; background:rgba(21,32,43,.82); color:#fff; max-width:calc(100% - 8px); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pl-card .pl-badge.odd { background:var(--flag); color:#1d1300; }
.pl-range { display:flex; gap:8px; align-items:center; font-size:13px; margin-top:8px; flex-wrap:wrap; } .pl-range input { flex:1; min-width:90px; accent-color:var(--accent); }
.pl-sortnote { font-size:12.5px; color:var(--muted); margin:6px 0 0; }
.pl-peek { border:1px solid var(--line); border-radius:12px; padding:0; width:min(1100px, calc(100vw - 32px)); max-height:92vh; color:var(--ink); background:var(--panel); }
.pl-peek::backdrop { background:rgba(13,18,23,.78); }
.pl-peek header { display:flex; gap:10px; align-items:center; padding:10px 14px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
.pl-peek header b { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; flex:1; }
.pl-peek img { display:block; width:100%; height:min(calc(92vh - 60px), 760px); object-fit:contain; background:#0d1217; }
@media (max-width: 900px) { .pl-main { grid-template-columns:minmax(0,1fr); } .pl-side { max-height:none; } .pl-where { display:none; } }
@media (prefers-reduced-motion: reduce) { .pl *, .pln-drawer { scroll-behavior:auto !important; transition:none !important; } }
`;

const NOTE_CSS = `
.pln-bell { position:relative; display:inline-flex; align-items:center; justify-content:center; width:36px; height:32px; border:1px solid #d5dbe1;
  border-radius:8px; background:#fff; color:#15202b; cursor:pointer; padding:0; }
.pln-bell:hover { border-color:#0a7c78; } .pln-bell:focus-visible { outline:2px solid #0a7c78; outline-offset:2px; }
.pln-bell svg { width:18px; height:18px; flex:none; transform-origin:50% 10%; }
.pln-bell.ring svg { animation:pln-ring .9s ease-in-out; }
.pln-badge.pop { animation:pln-pop .35s cubic-bezier(.2,.9,.3,1.5); }
@keyframes pln-ring { 15% { transform:rotate(18deg); } 35% { transform:rotate(-15deg); } 55% { transform:rotate(9deg); } 75% { transform:rotate(-5deg); } }
@keyframes pln-pop { from { transform:scale(.4); } }
.pln-badge { position:absolute; top:-6px; right:-7px; min-width:18px; height:18px; padding:0 5px; border-radius:9px; background:#c2412f; color:#fff;
  font:600 11px/18px system-ui,sans-serif; font-variant-numeric:tabular-nums; text-align:center; }
.pln-drawer { position:fixed; top:0; right:0; bottom:0; width:min(420px, 100vw); background:#fff; border-left:1px solid #d5dbe1; z-index:40;
  box-shadow:-10px 0 30px rgba(21,32,43,.14); display:flex; flex-direction:column; color:#15202b; overscroll-behavior:contain;
  font:14px/1.45 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif; transition:transform .18s ease-out; }
.pln-drawer[hidden] { display:flex; transform:translateX(105%); visibility:hidden; }
.pln-head { display:flex; align-items:center; gap:8px; padding:14px 16px 10px; }
.pln-head h2 { margin:0; font-size:16px; font-weight:600; margin-right:auto; }
.pln-drawer button { font:inherit; font-size:13px; border:1px solid #d5dbe1; background:#fff; color:#15202b; border-radius:7px; padding:4px 10px; cursor:pointer; }
.pln-drawer button:hover { border-color:#0a7c78; } .pln-drawer button:focus-visible { outline:2px solid #0a7c78; outline-offset:2px; }
.pln-filter { display:flex; gap:4px; padding:0 16px 10px; border-bottom:1px solid #e6eaee; }
.pln-filter button[aria-pressed=true] { background:#15202b; color:#fff; border-color:#15202b; }
.pln-list { list-style:none; margin:0; padding:6px 0 16px; overflow:auto; flex:1; }
.pln-item { display:grid; grid-template-columns:4px minmax(0,1fr); gap:12px; padding:10px 16px; border-bottom:1px solid #f0f2f4; }
.pln-item.unread { background:#f5fafa; }
.pln-mark { border-radius:2px; } .pln-item.success .pln-mark { background:#2f9e5b; } .pln-item.info .pln-mark { background:#6aa6f2; }
.pln-item.warning .pln-mark { background:#e09a2c; } .pln-item.error .pln-mark { background:#c2412f; }
.pln-title { font-weight:600; overflow-wrap:anywhere; } .pln-detail { color:#56636f; font-size:13px; overflow-wrap:anywhere; margin-top:1px; }
.pln-meta { color:#56636f; font-size:12px; margin-top:4px; font-variant-numeric:tabular-nums; }
.pln-actions { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
.pln-empty { color:#56636f; padding:28px 16px; text-align:center; }
.pln-item.unread { animation:pln-in .3s ease-out; } @keyframes pln-in { from { background:#e1f0ef; } }
@media (prefers-reduced-motion: reduce) { .pln-drawer { transition:none; } .pln-bell svg, .pln-badge, .pln-item { animation:none !important; } }
`;

const STATUS = ["No boxes", "Suggestions", "Tracked or imported", "Has your boxes", "Confirmed"];
const FORMAT_NAMES = { yolo: "YOLO", coco: "COCO", cvat: "CVAT", voc: "Pascal VOC", labelstudio: "Label Studio",
                       folders: "Class folders", csv: "CSV list" };
const TASK_NAMES = { detect: "Boxes", segment: "Outlines", classify: "Image classes" };
function rgbOf(i) {                                                   // colorOf(i) as [r, g, b] (hsl 78% 52%)
  const h = (i * 137.508) % 360, s = 0.78, l = 0.52, a = s * Math.min(l, 1 - l);
  const f = (n) => { const k = (n + h / 30) % 12; return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))); };
  return [f(0), f(8), f(4)];
}
const colorOf = (i) => `hsl(${(i * 137.508) % 360} 78% 52%)`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const BELL = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>`;
const CHECK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`;
const rtf = typeof Intl !== "undefined" && Intl.RelativeTimeFormat ? new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }) : null;
const dtf = typeof Intl !== "undefined" ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }) : null;
function when(t) {
  const s = Math.round(t - Date.now() / 1000);
  if (!rtf) return new Date(t * 1000).toLocaleString();
  if (Math.abs(s) < 45) return "just now";
  if (Math.abs(s) < 3600) return rtf.format(Math.round(s / 60), "minute");
  if (Math.abs(s) < 86400) return rtf.format(Math.round(s / 3600), "hour");
  return dtf ? dtf.format(new Date(t * 1000)) : new Date(t * 1000).toLocaleString();
}

// ---- notification center: bell + drawer, shared by the annotator and the start screen ----------
export function notificationCenter({ mount, onRead, onClear, onAction, actionAvailable = () => true }) {
  if (!document.getElementById("pln-css")) {
    document.head.appendChild(Object.assign(document.createElement("style"), { id: "pln-css", textContent: NOTE_CSS }));
  }
  const bell = Object.assign(document.createElement("button"), { type: "button", className: "pln-bell", title: "Notifications (N)" });
  bell.innerHTML = `${BELL}<span class="pln-badge" hidden></span>`;
  bell.setAttribute("aria-haspopup", "dialog");
  mount.appendChild(bell);
  const drawer = document.createElement("aside");
  drawer.className = "pln-drawer"; drawer.hidden = true;
  drawer.setAttribute("role", "dialog"); drawer.setAttribute("aria-label", "Notifications");
  drawer.innerHTML = `<div class="pln-head"><h2 tabindex="-1">Notifications</h2>
      <button type="button" data-n="read">Mark all as read</button><button type="button" data-n="clear">Clear all</button>
      <button type="button" data-n="close" aria-label="Close notifications">✕</button></div>
    <div class="pln-filter" role="group" aria-label="Show">
      <button type="button" data-f="all" aria-pressed="true">All</button><button type="button" data-f="error" aria-pressed="false">Problems</button></div>
    <ul class="pln-list" aria-live="polite"></ul>`;
  document.body.appendChild(drawer);
  const S = { items: [], unread: 0, filter: "all", fresh: new Set() };
  let timer = null;

  let shownUnread = 0;
  function renderBadge() {
    const badge = bell.querySelector(".pln-badge");
    if (S.unread > shownUnread) {                                  // something new: ring once
      bell.classList.remove("ring"); badge.classList.remove("pop"); void bell.offsetWidth;
      bell.classList.add("ring"); badge.classList.add("pop");
    }
    shownUnread = S.unread;
    badge.hidden = !S.unread; badge.textContent = S.unread > 99 ? "99+" : String(S.unread);
    bell.setAttribute("aria-label", S.unread ? `Notifications, ${S.unread} unread` : "Notifications");
  }
  function renderList() {
    const shown = S.items.filter((n) => S.filter === "all" || n.level === "error" || n.level === "warning");
    drawer.querySelector(".pln-list").innerHTML = shown.length ? shown.map((n, i) => {
      const acts = [];
      if (n.action && actionAvailable(n.action)) acts.push(`<button type="button" data-act="${i}">${esc(n.action.label || "Open")}</button>`);
      if (n.action?.type === "folder") acts.push(`<button type="button" data-copy="${i}">Copy path</button>`);
      return `<li class="pln-item ${esc(n.level)} ${S.fresh.has(n.id) ? "unread" : ""}" data-i="${i}"><span class="pln-mark" aria-hidden="true"></span><div>
        <div class="pln-title">${esc(n.title)}</div>${n.detail ? `<div class="pln-detail">${esc(n.detail)}</div>` : ""}
        <div class="pln-meta">${n.project ? `${esc(n.project)}, ` : ""}<time datetime="${new Date(n.time * 1000).toISOString()}">${esc(when(n.time))}</time></div>
        ${acts.length ? `<div class="pln-actions">${acts.join("")}</div>` : ""}</div></li>`;
    }).join("") : `<li class="pln-empty">${S.filter === "all" ? "Nothing yet. Project, export and review events appear here." : "No problems recorded."}</li>`;
    drawer.querySelector(".pln-list").dataset.shown = JSON.stringify(shown.map((n) => n.id));
    drawer._shown = shown;
  }
  function open() {
    drawer.hidden = false; bell.setAttribute("aria-expanded", "true");
    S.fresh = new Set(S.items.slice(0, S.unread).map((n) => n.id));      // highlighted until the drawer closes
    renderList(); drawer.querySelector("h2").focus();
    if (S.unread) { onRead?.(); S.unread = 0; renderBadge(); }
    timer = setInterval(renderList, 30000);
  }
  function close() {
    drawer.hidden = true; bell.setAttribute("aria-expanded", "false"); clearInterval(timer);
    S.fresh = new Set(); bell.focus();
  }
  bell.addEventListener("click", () => (drawer.hidden ? open() : close()));
  drawer.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } });
  drawer.addEventListener("click", async (e) => {
    const t = e.target.closest("button"); if (!t) return;
    if (t.dataset.n === "close") close();
    else if (t.dataset.n === "read") { onRead?.(); S.fresh = new Set(); S.unread = 0; renderBadge(); renderList(); }
    else if (t.dataset.n === "clear") { onClear?.(); }
    else if (t.dataset.f) { S.filter = t.dataset.f; drawer.querySelectorAll("[data-f]").forEach((b) => b.setAttribute("aria-pressed", b === t)); renderList(); }
    else if (t.dataset.act != null) { const n = drawer._shown[+t.dataset.act]; if (n) onAction?.(n.action, n); }
    else if (t.dataset.copy != null) {
      const n = drawer._shown[+t.dataset.copy];
      try { await navigator.clipboard.writeText(n.action.path); t.textContent = "Copied"; } catch { t.textContent = n.action.path; }
    }
  });
  renderBadge();
  return {
    set(items, unread) { S.items = items || []; S.unread = unread || 0; renderBadge(); if (!drawer.hidden) renderList(); },
    add(item, unread) {
      const i = S.items.findIndex((n) => n.id === item.id);
      if (i >= 0) S.items.splice(i, 1);
      S.items.unshift(item);
      if (drawer.hidden) S.unread = unread;
      else { S.fresh.add(item.id); S.unread = 0; renderList(); if (unread) onRead?.(); }
      renderBadge();
    },
    open, close, toggle: () => (drawer.hidden ? open() : close()),
    get isOpen() { return !drawer.hidden; },
  };
}

// ---- Rivet, the helper robot: a chat about using PartLabeler (engine/assistant.py) -------------
// Shared by the start screen (floating, bottom right), the annotator (in the header) and notebooks.
// Answers come from the local server, which asks Gemini Flash-Lite or, without a key, the guide.
const RIVET_CSS = `
.rv, .rv-panel { --rv-ink:#1f2a30; --rv-alu:#e9eef1; --rv-steel:#9aa9b1; --rv-teal:#0a7c78; --rv-teal-deep:#075e5b;
  --rv-amber:#f2a900; --rv-screen:#0f3534; --rv-glow:#8ff5e6; --rv-line:#d5dbe1; --rv-muted:#56636f; }
.rv [hidden], .rv-panel [hidden] { display:none !important; }
.rv { position:relative; display:inline-block; font:14px/1.45 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif; color:var(--rv-ink); }
.rv-fab { position:fixed; right:16px; bottom:12px; z-index:35; }
.rv-launch { position:relative; display:block; border:0; background:none; padding:0; margin:0; cursor:pointer; color:inherit; -webkit-tap-highlight-color:transparent; }
.rv-fab .rv-launch { width:76px; height:86px; }
.rv-top .rv-launch { width:38px; height:32px; border:1px solid var(--rv-line); border-radius:8px; background:#fff; display:inline-flex; align-items:center; justify-content:center; }
.rv-top .rv-launch:hover { border-color:var(--rv-teal); }
.rv-launch:focus-visible { outline:2px solid var(--rv-teal); outline-offset:3px; border-radius:12px; }
.rv-float { display:block; width:100%; height:100%; }
.rv-bot { width:100%; height:100%; overflow:visible; display:block; }
.rv-top .rv-bot { width:30px; height:27px; }
.rv-shadow { position:absolute; left:50%; bottom:-2px; width:46px; height:8px; margin-left:-23px; border-radius:50%; background:rgba(31,42,48,.25); filter:blur(2px); }
.rv-top .rv-shadow, .rv-top .rv-brk { display:none; }
.rv-shell { fill:var(--rv-alu); stroke:var(--rv-ink); stroke-width:2.5; }
.rv-screen { fill:var(--rv-screen); }
.rv-eyes { transform:translate(var(--ex,0px), var(--ey,0px)); transition:transform .15s ease-out; }
.rv-eye { fill:var(--rv-glow); transform-box:fill-box; transform-origin:center; animation:rv-blink 5.2s infinite; }
.rv-smile { fill:none; stroke:var(--rv-glow); stroke-width:2.2; stroke-linecap:round; }
.rv-talk { fill:var(--rv-glow); display:none; } .rv-talk rect { transform-box:fill-box; transform-origin:center; }
.rv-rivet { fill:var(--rv-steel); stroke:var(--rv-ink); stroke-width:2; }
.rv-neck { fill:var(--rv-steel); }
.rv-arm { fill:none; stroke:var(--rv-ink); stroke-width:3.2; stroke-linecap:round; }
.rv-wave { transform-box:view-box; transform-origin:55px 64px; }
.rv-core { fill:var(--rv-teal); transform-box:fill-box; transform-origin:center; }
.rv-bulb { fill:var(--rv-amber); animation:rv-glow 2.6s ease-in-out infinite; }
.rv-headg { transform-box:view-box; transform-origin:40px 56px; transition:transform .3s cubic-bezier(.2,.9,.3,1.4); }
.rv-brk { fill:none; stroke:var(--rv-amber); stroke-width:3; stroke-linecap:round; transform-box:view-box; transform-origin:40px 44px;
  opacity:0; transform:scale(1.14); transition:opacity .25s, transform .4s cubic-bezier(.2,.9,.3,1.3); }
.rv-fab .rv-float { animation:rv-bob 3.4s ease-in-out infinite; }
.rv-fab .rv-shadow { animation:rv-shadow 3.4s ease-in-out infinite; }
.rv-launch:hover .rv-headg, .rv-launch:focus-visible .rv-headg { transform:rotate(-7deg); }
.rv-launch:hover .rv-brk, .rv-launch:focus-visible .rv-brk, .rv[data-state=thinking] .rv-brk, .rv[data-state=talking] .rv-brk { opacity:1; transform:scale(1); }
[data-state=thinking] .rv-eyes { animation:rv-scan .7s ease-in-out infinite alternate; }
[data-state=thinking] .rv-brk { animation:rv-lock .8s ease-in-out infinite alternate; }
[data-state=thinking] .rv-bulb { animation-duration:.5s; }
[data-state=thinking] .rv-core, [data-state=talking] .rv-core { animation:rv-pulse .6s ease-in-out infinite alternate; }
[data-state=talking] .rv-smile { display:none; }
[data-state=talking] .rv-talk { display:inline; }
[data-state=talking] .rv-talk rect { animation:rv-eq .42s ease-in-out infinite alternate; }
[data-state=talking] .rv-talk rect:nth-child(2) { animation-duration:.3s; } [data-state=talking] .rv-talk rect:nth-child(3) { animation-duration:.52s; }
[data-state=happy] .rv-wave { animation:rv-wave 1.3s ease-in-out; }
[data-state=happy] .rv-eye { animation:rv-squint 1.3s ease-in-out; }
[data-state=sad] .rv-bulb { fill:#c2412f; animation:none; }
[data-state=sad] .rv-eyes { transform:translate(0, 2px); }
@keyframes rv-blink { 0%, 94%, 100% { transform:scaleY(1); } 96.5% { transform:scaleY(.1); } }
@keyframes rv-squint { 15%, 85% { transform:scaleY(.45); } }
@keyframes rv-glow { 50% { opacity:.4; } }
@keyframes rv-bob { 50% { transform:translateY(-5px); } }
@keyframes rv-shadow { 50% { transform:scaleX(.78); opacity:.55; } }
@keyframes rv-scan { from { transform:translateX(-3px); } to { transform:translateX(3px); } }
@keyframes rv-lock { from { transform:scale(1); } to { transform:scale(.92); } }
@keyframes rv-pulse { to { transform:scale(1.35); } }
@keyframes rv-eq { from { transform:scaleY(.35); } to { transform:scaleY(1.5); } }
@keyframes rv-wave { 0%, 100% { transform:rotate(0); } 20% { transform:rotate(-130deg); } 40% { transform:rotate(-98deg); }
  60% { transform:rotate(-134deg); } 80% { transform:rotate(-104deg); } }
.rv-tip { position:absolute; right:4px; bottom:94px; width:max-content; max-width:min(300px, calc(100vw - 36px)); background:#fff; color:var(--rv-ink);
  border:1px solid var(--rv-line); border-radius:16px 16px 4px 16px; padding:12px 14px; box-shadow:0 14px 34px rgba(31,42,48,.2);
  transform-origin:bottom right; animation:rv-pop .45s cubic-bezier(.2,.9,.3,1.35); z-index:1; }
.rv-top .rv-tip { top:42px; bottom:auto; right:0; border-radius:16px 4px 16px 16px; transform-origin:top right; }
.rv-tip.out { animation:rv-out .2s ease-in forwards; }
.rv-tip p { margin:0; font-size:13.5px; } .rv-tip p b { display:block; font-size:12.5px; color:var(--rv-teal-deep); margin-bottom:2px; }
.rv-tip-row { display:flex; gap:6px; margin-top:9px; align-items:center; }
.rv-tip button, .rv-panel button { font:inherit; font-size:13px; border:1px solid var(--rv-line); background:#fff; color:var(--rv-ink); border-radius:99px; padding:4px 12px; cursor:pointer; }
.rv-tip button:hover, .rv-panel button:hover { border-color:var(--rv-teal); }
.rv-tip button.go, .rv-panel button.go { background:var(--rv-teal); border-color:var(--rv-teal); color:#fff; font-weight:600; }
.rv-tip .x, .rv-ph .x { border:0; background:none; color:var(--rv-muted); padding:2px 8px; margin-left:auto; font-size:15px; }
.rv-tip :focus-visible, .rv-panel :focus-visible { outline:2px solid var(--rv-teal); outline-offset:2px; }
@keyframes rv-pop { from { opacity:0; transform:scale(.55) translateY(10px); } }
@keyframes rv-out { to { opacity:0; transform:scale(.85); } }
.rv-panel { position:fixed; right:16px; bottom:108px; width:min(410px, calc(100vw - 32px)); height:min(600px, calc(100vh - 136px)); z-index:45;
  display:flex; flex-direction:column; background:#fff; color:var(--rv-ink); border:1px solid var(--rv-line); border-radius:18px;
  box-shadow:0 26px 64px rgba(31,42,48,.26); font:14px/1.5 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif; overflow:hidden;
  transform-origin:bottom right; transition:opacity .2s ease-out, transform .3s cubic-bezier(.2,.9,.3,1.15), visibility 0s; }
.rv-panel.at-top { top:54px; bottom:auto; right:12px; transform-origin:top right; height:min(600px, calc(100vh - 72px)); }
.rv-panel[hidden] { display:flex; opacity:0; transform:scale(.9) translateY(14px); visibility:hidden; pointer-events:none;
  transition:opacity .2s ease-out, transform .3s cubic-bezier(.2,.9,.3,1.15), visibility 0s .3s; }
.rv-panel.at-top[hidden] { transform:scale(.9) translateY(-14px); }
.rv-ph { display:flex; align-items:center; gap:10px; padding:10px 10px 10px 12px; border-bottom:1px solid #e6eaee; background:#f3f6f7; }
.rv-avatar { width:40px; height:38px; flex:none; }
.rv-ph-t { min-width:0; flex:1; } .rv-ph-t b { font:600 16px/1.2 Bahnschrift,"DIN Alternate","Segoe UI",system-ui,sans-serif; letter-spacing:.01em; display:block; }
.rv-ph-t small { display:block; color:var(--rv-muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rv-ph .rv-ph-b { border-radius:8px; padding:3px 9px; font-size:12.5px; white-space:nowrap; flex:none; }
.rv-ph .rv-ph-b[aria-pressed=false] { color:var(--rv-muted); }
.rv-log { flex:1; overflow:auto; padding:14px; display:flex; flex-direction:column; gap:10px; overscroll-behavior:contain; }
.rv-msg { max-width:90%; padding:9px 12px; border-radius:16px; animation:rv-in .24s ease-out; overflow-wrap:anywhere; }
.rv-msg.bot { align-self:flex-start; background:#f0f3f5; border-bottom-left-radius:4px; }
.rv-msg.me { align-self:flex-end; background:var(--rv-teal); color:#fff; border-bottom-right-radius:4px; white-space:pre-wrap; }
.rv-msg.err { background:#fde8e5; color:#8b2a1d; }
.rv-msg p { margin:0 0 7px; } .rv-msg p:last-child { margin-bottom:0; }
.rv-msg ul, .rv-msg ol { margin:4px 0 7px; padding-left:20px; } .rv-msg li { margin:2px 0; }
.rv-msg code { font:12.5px ui-monospace,"Cascadia Mono",Consolas,monospace; background:#e2e7eb; padding:1px 4px; border-radius:4px; }
.rv-msg kbd { font:12px/1.2 ui-monospace,"Cascadia Mono",Consolas,monospace; border:1px solid #c9d2d8; border-bottom-width:2px; border-radius:5px; padding:1px 5px; background:#fff; }
.rv-msg h4 { margin:2px 0 4px; font-size:14px; }
.rv-note { font-size:11.5px; color:var(--rv-muted); margin-top:6px; }
.rv-dos { display:flex; gap:6px; flex-wrap:wrap; margin-top:9px; }
.rv-panel .rv-do { border-color:var(--rv-teal); color:var(--rv-teal-deep); font-weight:600; }
.rv-panel .rv-do:hover { background:#e1f0ef; }
.rv-typing { display:inline-flex; gap:4px; padding:4px 0; } .rv-typing i { width:7px; height:7px; border-radius:50%; background:#8a9aa3; animation:rv-dot 1s ease-in-out infinite; }
.rv-typing i:nth-child(2) { animation-delay:.15s; } .rv-typing i:nth-child(3) { animation-delay:.3s; }
@keyframes rv-dot { 0%, 60%, 100% { transform:translateY(0); opacity:.5; } 30% { transform:translateY(-5px); opacity:1; } }
@keyframes rv-in { from { opacity:0; transform:translateY(6px); } }
.rv-chips { display:flex; flex-wrap:wrap; gap:6px; padding:2px 14px 10px; }
.rv-panel .rv-chip { animation:rv-in .3s ease-out both; animation-delay:calc(var(--i, 0) * 70ms); text-align:left; }
.rv-keybox { margin:0 14px 10px; padding:10px 12px; border:1px dashed #c3ccd2; border-radius:12px; background:#fafbfc; font-size:12.5px; display:grid; gap:7px; }
.rv-keybox p { margin:0; color:var(--rv-muted); } .rv-keybox a { color:var(--rv-teal-deep); font-weight:600; }
.rv-keyrow { display:flex; gap:6px; } .rv-keyrow input { flex:1; min-width:0; font:inherit; border:1px solid var(--rv-line); border-radius:8px; padding:5px 9px; }
.rv-keyerr { color:#b3321f; }
.rv-form { display:flex; gap:8px; align-items:flex-end; padding:10px 12px 12px; border-top:1px solid #e6eaee; }
.rv-form textarea { flex:1; resize:none; font:inherit; color:inherit; border:1px solid var(--rv-line); border-radius:14px; padding:8px 12px; max-height:120px; min-height:40px; line-height:1.4; }
.rv-form textarea:focus-visible { outline:2px solid var(--rv-teal); outline-offset:0; border-color:transparent; }
.rv-panel .rv-send { width:40px; height:40px; border-radius:50%; padding:0; background:var(--rv-teal); border-color:var(--rv-teal); color:#fff; display:grid; place-items:center; flex:none; transition:transform .15s; }
.rv-panel .rv-send:active { transform:scale(.92); } .rv-panel .rv-send svg { width:18px; height:18px; }
.rv-panel .rv-send.stop { background:var(--rv-ink); border-color:var(--rv-ink); }
@media (max-width: 600px) {
  .rv-panel, .rv-panel.at-top { inset:0; width:auto; height:auto; border-radius:0; border:0; }
  .rv-fab .rv-launch { width:62px; height:70px; } .rv-tip { bottom:78px; } }
@media (prefers-reduced-motion: reduce) {
  .rv *, .rv-panel, .rv-panel * { animation:none !important; transition:none !important; } }
`;
const ROBOT = (head = false) => `<svg class="rv-bot" viewBox="${head ? "8 0 64 58" : "0 0 80 88"}" aria-hidden="true" focusable="false">
  <g class="rv-brk"><path d="M4 16V4h12"/><path d="M64 4h12v12"/><path d="M76 72v12H64"/><path d="M16 84H4V72"/></g>
  ${head ? "" : `<path class="rv-arm" d="M25 64q-8 2-9 10"/><path class="rv-arm rv-wave" d="M55 64q8 2 9 10"/>
  <rect x="34" y="53" width="12" height="6" class="rv-neck"/><rect x="24" y="58" width="32" height="20" rx="7" class="rv-shell"/>
  <circle cx="40" cy="68" r="3.5" class="rv-core"/>`}
  <g class="rv-headg"><line x1="40" y1="18" x2="40" y2="10" stroke="#1f2a30" stroke-width="2.5"/><circle class="rv-bulb" cx="40" cy="8" r="4"/>
    <circle cx="13" cy="36" r="3.5" class="rv-rivet"/><circle cx="67" cy="36" r="3.5" class="rv-rivet"/>
    <rect x="16" y="18" width="48" height="36" rx="12" class="rv-shell"/><rect x="22" y="24" width="36" height="24" rx="7" class="rv-screen"/>
    <g class="rv-eyes"><rect class="rv-eye" x="29" y="30" width="7" height="10" rx="3.5"/><rect class="rv-eye" x="44" y="30" width="7" height="10" rx="3.5"/></g>
    <path class="rv-smile" d="M35 43.5q5 3.2 10 0"/>
    <g class="rv-talk"><rect x="34" y="42" width="2.4" height="4" rx="1.2"/><rect x="38.8" y="41" width="2.4" height="6" rx="1.2"/><rect x="43.6" y="42" width="2.4" height="4" rx="1.2"/></g></g></svg>`;
const SEND = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h13M13 6l6 6-6 6"/></svg>`;
const STOP = `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>`;
const KEYLIKE = /^(((Ctrl|Shift|Alt|Cmd)\s?\+\s?)*(\S|F\d{1,2}|Enter|Esc|Escape|Del|Delete|Tab|Space|Backspace|Home|End))$/;

// Markdown subset for answers (the text is escaped first, so nothing the model writes becomes HTML)
function md(src) {
  const inline = (s) => s.replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, (_, t) => (KEYLIKE.test(t.trim()) ? `<kbd>${t.trim()}</kbd>` : `<b>${t}</b>`))
    .replace(/(^|[\s(])[*_]([^*_\n]+)[*_](?=[\s).,!?:;]|$)/g, "$1<i>$2</i>");
  const out = []; let list = null, para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("<br>"))}</p>`); para = []; } };
  const endList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; } };
  for (const line of esc(src).split("\n")) {
    const ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/), h = line.match(/^#{1,4}\s+(.*)/);
    if (ul || ol) { flush(); const tag = ul ? "ul" : "ol"; if (list?.tag !== tag) { endList(); list = { tag, items: [] }; } list.items.push((ul || ol)[1]); }
    else if (h) { flush(); endList(); out.push(`<h4>${inline(h[1])}</h4>`); }
    else if (!line.trim()) { flush(); endList(); }
    else if (list && /^\s{2,}\S/.test(line)) list.items[list.items.length - 1] += ` ${line.trim()}`;   // a wrapped list item
    else { endList(); para.push(line); }
  }
  flush(); endList();
  return out.join("");
}

// Transports: the web app asks the server over HTTP; a notebook widget asks through its message channel.
export function webAssistant(base = "/") {
  const post = (path, body, signal) => fetch(`${base}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });
  return {
    info: () => fetch(`${base}api/assistant`).then((r) => (r.ok ? r.json() : null)),
    async saveKey(key) {
      const r = await post("api/assistant/key", { key }); const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.detail || r.statusText); return d;
    },
    async chat(payload, onEvent, signal) {
      const r = await post("api/assistant/chat", payload, signal);
      if (!r.ok || !r.body) throw new Error(`The helper is not available right now (${r.status}). Reload the page.`);
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (line) onEvent(JSON.parse(line)); }
      }
      if (buf.trim()) onEvent(JSON.parse(buf));
    },
  };
}
function widgetAssistant(model) {
  const waiting = new Map(), infoWait = [];
  model.on("msg:custom", (m) => {
    if (m.type === "assistant_info") infoWait.splice(0).forEach((f) => f(m));
    else if (m.type === "assistant" && waiting.has(m.id)) {
      const w = waiting.get(m.id); w.onEvent(m);
      if (m.done || m.error) { waiting.delete(m.id); w.resolve(); }
    }
  });
  return {
    info: () => new Promise((res) => { infoWait.push(res); model.send({ type: "assistant_info" }); }),
    saveKey: (key) => new Promise((res, rej) => { infoWait.push((m) => (m.key_error ? rej(new Error(m.key_error)) : res(m))); model.send({ type: "assistant_key", key }); }),
    chat: (payload, onEvent, signal) => new Promise((resolve) => {
      const id = Math.random().toString(36).slice(2);
      waiting.set(id, { onEvent, resolve });
      signal?.addEventListener("abort", () => { waiting.delete(id); resolve(); });
      model.send({ type: "assistant", id, ...payload });
    }),
  };
}

export function assistant({ mount, transport, variant = "fab", context = () => ({}), actions = {}, starters = [], firstTip = 25000 }) {
  if (!document.getElementById("rv-css")) {
    document.head.appendChild(Object.assign(document.createElement("style"), { id: "rv-css", textContent: RIVET_CSS }));
  }
  const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const store = (k, v) => { try { v === undefined ? null : sessionStorage.setItem(k, v); return sessionStorage.getItem(k); } catch { return null; } };
  const local = (k, v) => { try { if (v !== undefined) localStorage.setItem(k, v); return localStorage.getItem(k); } catch { return null; } };
  const root = document.createElement("div");
  root.className = `rv ${variant === "fab" ? "rv-fab" : "rv-top"}`; root.dataset.state = "idle";
  root.innerHTML = `<div class="rv-tip" role="status" hidden></div>
    <button type="button" class="rv-launch" aria-haspopup="dialog" aria-expanded="false" aria-label="Ask Rivet, the PartLabeler helper" title="Ask Rivet (help)">
      <span class="rv-float">${ROBOT(variant !== "fab")}</span><span class="rv-shadow" aria-hidden="true"></span></button>`;
  mount.appendChild(root);
  const panel = document.createElement("section");
  panel.className = `rv-panel${variant === "fab" ? "" : " at-top"}`; panel.hidden = true; panel.dataset.state = "idle";
  panel.setAttribute("role", "dialog"); panel.setAttribute("aria-label", "Rivet, the PartLabeler helper");
  panel.innerHTML = `<header class="rv-ph"><span class="rv-avatar">${ROBOT(true)}</span>
      <div class="rv-ph-t"><b>Rivet</b><small class="rv-status">Ask me anything about PartLabeler</small></div>
      <button type="button" class="rv-ph-b" data-r="tips" title="Show a 'Did you know?' bubble now and then">Tips</button>
      <button type="button" class="rv-ph-b" data-r="new" title="Start a new conversation">New chat</button>
      <button type="button" class="x" data-r="close" aria-label="Close the helper">✕</button></header>
    <div class="rv-log" aria-live="polite"></div>
    <div class="rv-chips"></div>
    <div class="rv-keybox" hidden><p>Rivet is answering from the built-in guide. For full answers, add a free Gemini API key
      (<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">get one here</a>). It is saved on this computer only.</p>
      <form class="rv-keyrow"><input type="password" name="gemini-key" autocomplete="off" spellcheck="false" aria-label="Gemini API key" placeholder="Paste your Gemini API key…">
        <button type="submit" class="go">Save</button></form><span class="rv-keyerr" role="alert"></span></div>
    <form class="rv-form"><textarea rows="1" name="question" aria-label="Your question" placeholder="Ask how to export, track, use Teach & Transfer…"></textarea>
      <button type="submit" class="rv-send" aria-label="Send">${SEND}</button></form>`;
  document.body.appendChild(panel);
  const $ = (s) => panel.querySelector(s), launch = root.querySelector(".rv-launch"), tipEl = root.querySelector(".rv-tip");
  const S = { msgs: [], info: null, busy: null, tipsOn: local("pl-rivet-tips") !== "off", tipTimer: null, moodTimer: null };
  try { S.msgs = JSON.parse(store("pl-rivet-chat") || "[]").filter((m) => m && typeof m.text === "string"); } catch { S.msgs = []; }

  function mood(state, ms) {
    clearTimeout(S.moodTimer);
    root.dataset.state = panel.dataset.state = state;
    if (ms) S.moodTimer = setTimeout(() => mood(S.busy ? "thinking" : "idle"), ms);
  }
  function save() { store("pl-rivet-chat", JSON.stringify(S.msgs.filter((m) => !m.pending).slice(-30))); }
  function renderTips() { const b = $('[data-r="tips"]'); b.setAttribute("aria-pressed", String(S.tipsOn)); b.textContent = S.tipsOn ? "Tips on" : "Tips off"; }
  function renderStatus() {
    $(".rv-status").textContent = S.busy ? "Thinking…" : S.info?.ready ? "Powered by Gemini Flash-Lite" : S.info ? "Answering from the built-in guide" : "Ask me anything about PartLabeler";
    $(".rv-keybox").hidden = !(S.info && !S.info.ready && S.info.can_save);
  }
  const doButtons = (names) => names.filter((n) => actions[n]).map((n) => `<button type="button" class="rv-do" data-do="${esc(n)}">${esc(actions[n].label)}</button>`).join("");
  function msgHtml(m) {
    if (m.role === "user") return `<div class="rv-msg me">${esc(m.text)}</div>`;
    if (m.pending && !m.text) return `<div class="rv-msg bot"><span class="rv-typing" aria-label="Rivet is typing"><i></i><i></i><i></i></span></div>`;
    let text = m.text.replace(/\[\[do:[a-z-]*\]?\]?/g, "").trim();
    if (m.pending) text = text.replace(/\[\[?[a-z:-]*$/, "");      // a button marker still arriving
    return `<div class="rv-msg bot${m.error ? " err" : ""}">${m.error ? esc(text) : md(text)}
      ${m.offline ? `<div class="rv-note">${m.offline === "down" ? "Gemini could not be reached, so this answer is from the built-in guide." : "Answer from the built-in guide."}</div>` : ""}
      ${m.dos?.length ? `<div class="rv-dos">${doButtons(m.dos)}</div>` : ""}</div>`;
  }
  function renderLog() {
    const log = $(".rv-log");
    const hello = `<div class="rv-msg bot"><p>Hi, I'm <b>Rivet</b>. I know my way around PartLabeler: ask me how to do something, or what a button does.</p></div>`;
    log.innerHTML = hello + S.msgs.map(msgHtml).join("");
    $(".rv-chips").innerHTML = S.msgs.length ? "" : starters.map((q, i) => `<button type="button" class="rv-chip" style="--i:${i}" data-ask="${esc(q.q || q)}"
      ${q.local ? `data-local="${esc(q.local)}"` : ""}>${esc(q.q || q)}</button>`).join("");
    log.scrollTop = log.scrollHeight;
  }
  let raf = 0;
  const renderSoon = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; renderLog(); }); };
  function setBusy(ctrl) {
    S.busy = ctrl; renderStatus();
    const b = $(".rv-send"); b.classList.toggle("stop", Boolean(ctrl));
    b.innerHTML = ctrl ? STOP : SEND; b.setAttribute("aria-label", ctrl ? "Stop the answer" : "Send");
  }

  // `local`: a fixed answer (tips, the pros) shown without asking Gemini, so it costs no quota
  async function ask(q, local) {
    q = String(q || "").trim();
    if (!q) return;
    if (S.busy) S.busy.abort();
    open(false);
    S.msgs.push({ role: "user", text: q });
    if (local) {
      S.msgs.push({ role: "assistant", text: local, offline: "guide" });
      save(); renderLog(); mood("happy", 1400);
      return;
    }
    const m = { role: "assistant", text: "", pending: true };
    S.msgs.push(m); renderLog(); mood("thinking");
    const ctrl = new AbortController(); setBusy(ctrl);
    const slow = setTimeout(() => { if (S.busy === ctrl && !m.text) $(".rv-status").textContent = "Gemini is busy, still thinking…"; }, 6000);
    const history = S.msgs.filter((x) => !x.pending && !x.error).slice(-12).map((x) => ({ role: x.role, text: x.text.replace(/\[\[do:[a-z-]*\]\]/g, "") }));
    try {
      await transport.chat({ messages: history, context: context(), actions: Object.keys(actions) }, (ev) => {
        if (ev.offline) m.offline = ev.reason === "no_key" ? "no_key" : "down";
        if (ev.text) { m.text += ev.text; if (root.dataset.state !== "talking") mood("talking"); renderSoon(); }
        if (ev.error) { m.text = ev.error; m.error = true; }
      }, ctrl.signal);
    } catch (err) {
      if (err.name === "AbortError") m.text += m.text ? "\n\n_(stopped)_" : "_(stopped)_";
      else { m.text = String(err.message || err); m.error = true; }
    }
    clearTimeout(slow);
    m.pending = false;
    m.dos = [...m.text.matchAll(/\[\[do:([a-z-]+)\]\]/g)].map((x) => x[1]).filter((n, i, a) => a.indexOf(n) === i);
    if (!m.text.trim()) { m.text = "I didn't get an answer. Try asking again."; m.error = true; }
    if (S.busy === ctrl) setBusy(null);
    save(); renderLog();
    mood(m.error ? "sad" : "happy", m.error ? 2500 : 1400);
  }

  function open(focus = true) {
    hideTip();
    if (!panel.hidden) { if (focus) $("textarea").focus(); return; }
    panel.hidden = false; launch.setAttribute("aria-expanded", "true");
    renderLog(); renderStatus(); renderTips();
    if (!S.busy) mood("happy", 1400);
    if (focus) setTimeout(() => $("textarea").focus(), 60);
    if (!S.info) transport.info().then((i) => { S.info = i; renderStatus(); }).catch(() => {});
  }
  function close() { panel.hidden = true; launch.setAttribute("aria-expanded", "false"); launch.focus(); }
  const toggle = () => (panel.hidden ? open() : close());

  // "Did you know?" bubbles: advantages and tips, now and then, never while typing or in a dialog
  function hideTip() { if (tipEl.hidden) return; tipEl.classList.add("out"); setTimeout(() => { tipEl.hidden = true; tipEl.classList.remove("out"); }, 200); }
  function showTip(tip, lead = "Did you know?") {
    if (!tip || !panel.hidden) return;
    tipEl.innerHTML = `<p><b>${esc(lead)}</b>${esc(tip.text)}</p><div class="rv-tip-row">
      ${tip.ask ? `<button type="button" class="go" data-t="more">${tip.cta ? esc(tip.cta) : "Tell me more"}</button>` : ""}
      <button type="button" class="x" data-t="hide" aria-label="Hide this tip">✕</button></div>`;
    tipEl._tip = tip; tipEl.hidden = false; mood("happy", 1400);
    clearTimeout(tipEl._t); tipEl._t = setTimeout(hideTip, 14000);
  }
  function scheduleTip(ms) {
    clearTimeout(S.tipTimer);
    S.tipTimer = setTimeout(() => {
      const quiet = !document.hidden && panel.hidden && !document.querySelector("dialog[open]")
        && !document.activeElement?.matches?.("input,textarea,select") && !document.querySelector(".pln-drawer:not([hidden])");
      if (S.tipsOn && quiet && S.info?.tips?.length) showTip(S.info.tips[Math.floor(Math.random() * S.info.tips.length)]);
      scheduleTip(quiet ? 240000 + Math.random() * 180000 : 45000);
    }, ms);
  }
  tipEl.addEventListener("click", (e) => {
    const t = e.target.closest("[data-t]")?.dataset.t; if (!t) return;
    const tip = tipEl._tip; hideTip();
    if (t === "more") { if (tip.ask === true) open(); else ask(tip.ask, tip.more); }
  });

  // eyes follow the pointer (floating robot only)
  if (variant === "fab" && !reduce) {
    let px = 0, py = 0, pending = false;
    window.addEventListener("pointermove", (e) => {
      px = e.clientX; py = e.clientY; if (pending) return; pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const r = launch.getBoundingClientRect(), dx = px - (r.left + r.width / 2), dy = py - (r.top + r.height * 0.4);
        const k = Math.min(1, Math.hypot(dx, dy) / 240), a = Math.atan2(dy, dx);
        root.style.setProperty("--ex", `${(Math.cos(a) * 2.6 * k).toFixed(2)}px`); root.style.setProperty("--ey", `${(Math.sin(a) * 2 * k).toFixed(2)}px`);
      });
    }, { passive: true });
  }

  launch.addEventListener("click", toggle);
  panel.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } });
  panel.addEventListener("click", (e) => {
    const t = e.target.closest("button"); if (!t) return;
    if (t.dataset.r === "close") close();
    else if (t.dataset.r === "new") { S.busy?.abort(); S.msgs = []; save(); renderLog(); $("textarea").focus(); }
    else if (t.dataset.r === "tips") { S.tipsOn = !S.tipsOn; local("pl-rivet-tips", S.tipsOn ? "on" : "off"); renderTips(); }
    else if (t.dataset.ask) ask(t.dataset.ask, t.dataset.local ? S.info?.[t.dataset.local] : undefined);
    else if (t.dataset.do && actions[t.dataset.do]) {
      if (window.innerWidth <= 600 || variant !== "fab") close();
      actions[t.dataset.do].run();
    }
  });
  const ta = $("textarea");
  const grow = () => { ta.style.height = "auto"; ta.style.height = `${Math.min(120, ta.scrollHeight + 2)}px`; };
  ta.addEventListener("input", grow);
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $(".rv-form").requestSubmit(); } });
  $(".rv-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (S.busy && !ta.value.trim()) { S.busy.abort(); return; }
    const q = ta.value; ta.value = ""; grow(); ask(q);
  });
  $(".rv-keyrow").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = e.target.querySelector("input"), err = $(".rv-keyerr"), btn = e.target.querySelector("button");
    err.textContent = ""; btn.disabled = true; btn.textContent = "Checking…";
    try { S.info = await transport.saveKey(input.value.trim()); input.value = ""; renderStatus(); mood("happy", 1400); }
    catch (x) { err.textContent = String(x.message || x); mood("sad", 2000); }
    btn.disabled = false; btn.textContent = "Save";
  });

  transport.info().then((i) => {
    S.info = i; renderStatus();
    if (local("pl-rivet-met") !== "1" && variant === "fab") {
      local("pl-rivet-met", "1");
      setTimeout(() => showTip({ text: "I'm Rivet. Ask me anything about PartLabeler, like how to export or how Teach & Transfer works.", ask: true, cta: "Ask a question" }, "Hi there!"), 2500);
    }
    scheduleTip(firstTip);
  }).catch(() => {});
  return { open, close, toggle, ask, get isOpen() { return !panel.hidden; } };
}

// ---- the annotator ---------------------------------------------------------------------------
function render({ model, el }) {
  if (!document.getElementById("pl-css")) {
    document.head.appendChild(Object.assign(document.createElement("style"), { id: "pl-css", textContent: CSS }));
  }
  const web = Boolean(model.homeUrl);                                // running in the web app (REST helpers available)
  el.className = web ? "pl web" : "pl";
  el.tabIndex = 0;
  el.innerHTML = `
    <header class="pl-head">
      <a class="pl-back" href="/" hidden>← Projects</a>
      <span class="pl-title">PartLabeler</span><span class="pl-where"></span><span class="pl-chips"></span>
      <span class="pl-sp"></span>
      <span class="pl-saved" role="status" aria-live="polite"></span>
      <span class="pl-bellmount"></span><span class="pl-rivetmount"></span>
      <button type="button" class="icon pl-full" data-a="full" aria-label="Full screen" title="Full screen">⤢</button>
      <button type="button" class="icon" data-a="help" aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">?</button>
    </header>
    <div class="pl-bar-tools" role="toolbar" aria-label="Frame actions">
      <span class="pl-group">
        <button type="button" class="icon" data-a="prev" aria-label="Previous frame" title="Previous frame (←)">◀</button>
        <button type="button" class="icon" data-a="next" aria-label="Next frame" title="Next frame (→)">▶</button>
        <button type="button" data-a="nextTodo" title="Next frame that is flagged or not confirmed (Shift+→)">Next to check</button>
      </span>
      <span class="pl-group pl-video">
        <span class="pl-group-label" id="pl-track-label">Track</span>
        <button type="button" data-a="trackBack" title="Track this frame's boxes backwards (R; Shift+R to the start)">◀ Back</button>
        <input type="number" class="pl-n" min="1" value="20" name="track-frames" autocomplete="off" aria-label="Frames to track" title="Frames to track">
        <button type="button" data-a="track" title="Track this frame's boxes ahead (T; Shift+T to the end)">Ahead ▶</button>
        <button type="button" data-a="stop" title="Stop the running job (X)">Stop</button>
      </span>
      <span class="pl-group">
        <button type="button" data-a="undo" title="Undo the last change (Ctrl+Z)">Undo</button>
        <button type="button" data-a="exportJump" title="Export the dataset: YOLO, COCO, CVAT, Pascal VOC or Label Studio">Export</button>
      </span>
      <span class="pl-sp"></span>
      <button type="button" class="primary" data-a="review" title="Mark this frame as checked and go to the next one (Enter)">Confirm frame</button>
    </div>
    <div class="pl-banner" hidden></div>
    <div class="pl-main">
      <div class="pl-gridwrap" hidden><div class="pl-gridbar"><h2 class="pl-gtitle">Images</h2><span class="pl-src pl-gcount"></span>
          <div class="pl-filters" role="group" aria-label="Show"></div></div><div class="pl-grid" tabindex="-1"></div></div>
      <div class="pl-stage"><canvas class="pl-cv" aria-label="Frame image. Click a part to box it."></canvas>
        <div class="pl-hint" hidden><span><b>Click a part</b> to outline and box it, or press <kbd>B</kbd> and drag to draw a box.
          Pick the class first with <kbd>1</kbd>–<kbd>9</kbd>. Press <kbd>?</kbd> for all shortcuts.</span>
          <button type="button" data-a="hideHint">Got it</button></div></div>
      <aside class="pl-side" aria-label="Labeling tools">
        <section class="pl-toolsec"><h2>Tool</h2><div class="pl-tools" role="group" aria-label="Tool">
          <button type="button" data-tool="click" title="Click a part to outline and box it (C)">Click to outline</button>
          <button type="button" data-tool="box" title="Drag to draw a box (B)">Draw box</button>
          <button type="button" data-tool="brush" class="pl-seg" title="Paint onto the selected outline, or a new part (P)">Brush</button>
          <button type="button" data-tool="erase" class="pl-seg" title="Erase from the selected outline (E)">Eraser</button></div>
          <label class="pl-brush pl-seg">Brush size <input type="range" class="pl-bsize" min="1" max="120" value="12" name="brush-size" aria-label="Brush size in pixels"><span class="pl-bsize-v">12 px</span></label></section>
        <section><h2 id="pl-class-h">Class</h2>
          <input type="search" class="pl-filter" name="class-filter" autocomplete="off" placeholder="Filter classes…" aria-label="Filter classes" hidden>
          <div class="pl-list pl-classes" role="group" aria-labelledby="pl-class-h"></div>
          <form class="pl-addcls pl-tools" style="margin-top:6px"><input type="text" class="pl-newcls" name="new-class" autocomplete="off" placeholder="New class…" aria-label="New class name" style="flex:1;min-width:0">
            <button type="submit" title="Add a class; selected pictures in the grid get it">Add</button></form></section>
        <section><h2 id="pl-box-h">Boxes on this frame</h2><div class="pl-list pl-boxes" role="group" aria-labelledby="pl-box-h"></div></section>
        <section class="pl-find"><h2>Find parts</h2><div class="pl-tools">
          <button type="button" data-b="suggest" title="Suggest boxes that look like parts you labeled on other frames (S)">Suggest</button>
          <button type="button" data-b="findAll" title="Find more parts like the selected box on this frame (F)">Find similar</button>
          <button type="button" data-b="accept" title="Keep every suggestion on this frame (Y)">Accept all</button>
          <button type="button" data-b="sortParts" title="See every part as a picture, grouped by look, and name or fix classes in bulk (G)">Sort parts</button></div>
          <div class="pl-tools pl-seg" style="margin-top:6px">
          <button type="button" data-b="outlineHere" title="Outline every box on this frame that has no outline yet">Outline boxes</button>
          <button type="button" data-b="outlineAll" title="Outline every box without an outline, on every frame">…on every frame</button></div></section>
        <section class="pl-sort" hidden><h2>Sort</h2><div class="pl-tools">
          <button type="button" data-b="group" title="Put look-alikes together (G)">Group by look</button>
          <button type="button" data-b="suggestTags" class="pl-imgonly" title="Suggest a class for every image without one, from the ones you classed (S)">Suggest classes</button>
          <button type="button" data-b="acceptTags" class="pl-imgonly" title="Keep the suggested classes: of the selected images, or all (Y)">Accept suggestions</button>
          <button type="button" data-b="odd" title="Find labels that disagree with their look-alikes">Check labels</button>
          <button type="button" data-b="closeGrid" class="pl-partsonly" title="Back to labeling frames (Esc)">Back to frames</button></div>
          <label class="pl-range" hidden>Groups <input type="range" class="pl-k" min="1" value="1" name="groups" aria-label="Number of groups"><span class="pl-kv"></span></label>
          <p class="pl-sortnote">Select pictures (click, Shift+click, Ctrl+click, Ctrl+A), then press a class number. Double-click to look closer.</p></section>
        <section class="pl-export"><h2>Export dataset</h2><div class="pl-tools">
          <select class="pl-fmt" name="export-format" aria-label="Export format"></select>
          <label><input type="checkbox" class="pl-revonly" name="confirmed-only"> Confirmed frames only</label>
          <button type="button" class="primary" data-b="export">Export</button></div></section>
        <details class="pl-settings"><summary>Settings</summary>
          <label class="pl-field pl-tasksel">Label type
            <select class="pl-task" name="label-type"><option value="detect">Boxes</option><option value="segment">Outlines (masks)</option></select></label>
          <label class="pl-field pl-parentf">Parent object
            <input type="text" class="pl-parent" name="parent-object" autocomplete="off" placeholder="Optional, e.g. engine block…"></label>
          <span class="pl-src pl-parentf">What the parts sit on. Suggestions then search inside it, which helps when the camera or distance changes.
            Leave empty to search the whole image.</span>
          <div class="pl-tools" style="margin-top:8px"><button type="button" data-b="saveSettings">Save settings</button></div>
          <p class="pl-src pl-device"></p><p class="pl-src pl-version"></p></details>
      </aside>
    </div>
    <div class="pl-bottom">
      <canvas class="pl-strip" role="img" aria-label="Timeline of every frame; click to jump" title="Every frame; click to jump"></canvas>
      <div class="pl-foot"><span class="pl-counts"></span><span class="pl-bar" aria-hidden="true"><i></i></span><span class="pl-prog" role="status" aria-live="polite"></span>
        <span class="pl-sp"></span><span class="pl-legend"></span></div>
    </div>
    <div class="pl-toasts" aria-live="polite" aria-relevant="additions"></div>
    <dialog class="pl-help-dlg" aria-labelledby="pl-help-h"><header><h2 id="pl-help-h">Keyboard shortcuts</h2>
      <button type="button" data-a="closeHelp">Close</button></header>
      <div class="pl-help-grid">
        <h3>Label</h3>
        <div><span>New part of the current class</span><span>Click</span></div>
        <div><span>Next larger outline for it</span><kbd>M</kbd></div>
        <div><span>Grow / shrink the selected part</span><span><kbd>Shift</kbd>/<kbd>Alt</kbd> + click</span></div>
        <div><span>Select a box</span><span><kbd>Ctrl</kbd> + click</span></div>
        <div><span>Draw a box (redraws the selected one)</span><span><kbd>B</kbd> then drag</span></div>
        <div><span>Back to click-to-outline</span><kbd>C</kbd></div>
        <div><span>Pick class (re-labels the selected box)</span><span><kbd>1</kbd>–<kbd>9</kbd>, <kbd>0</kbd>, <kbd>[</kbd> <kbd>]</kbd></span></div>
        <div><span>Delete the selected box</span><kbd>Del</kbd></div>
        <div><span>Deselect</span><kbd>Esc</kbd></div>
        <div><span>Undo</span><span><kbd>Ctrl</kbd> + <kbd>Z</kbd></span></div>
        <h3>Frames</h3>
        <div><span>Previous / next frame</span><span><kbd>←</kbd> <kbd>→</kbd></span></div>
        <div><span>Next frame to check</span><span><kbd>Shift</kbd> + <kbd>→</kbd></span></div>
        <div><span>Confirm frame and go on</span><kbd>Enter</kbd></div>
        <div><span>Track ahead / to the end</span><span><kbd>T</kbd> / <kbd>Shift</kbd>+<kbd>T</kbd></span></div>
        <div><span>Track back / to the start</span><span><kbd>R</kbd> / <kbd>Shift</kbd>+<kbd>R</kbd></span></div>
        <div><span>Stop the running job</span><kbd>X</kbd></div>
        <h3>Find and review</h3>
        <div><span>Suggest boxes from labeled frames</span><kbd>S</kbd></div>
        <div><span>Find more like the selected box</span><kbd>F</kbd></div>
        <div><span>Accept all suggestions</span><kbd>Y</kbd></div>
        <div><span>Notifications</span><kbd>N</kbd></div>
        <div><span>Ask Rivet, the helper</span><kbd>H</kbd></div>
        <div><span>This list</span><kbd>?</kbd></div>
        <div><span>Full screen (top bar)</span><span>⤢</span></div>
        <h3>Outlines and sorting</h3>
        <div><span>Brush / eraser (outline projects)</span><span><kbd>P</kbd> / <kbd>E</kbd></span></div>
        <div><span>Smaller / larger brush</span><span><kbd>,</kbd> <kbd>.</kbd></span></div>
        <div><span>Sort parts / group by look</span><kbd>G</kbd></div>
        <div><span>Grid: give the selected pictures a class</span><span><kbd>1</kbd>–<kbd>9</kbd>, <kbd>0</kbd></span></div>
        <div><span>Grid: select all shown / clear class</span><span><kbd>Ctrl</kbd>+<kbd>A</kbd> / <kbd>Del</kbd></span></div>
        <div><span>Grid: look closer</span><span>Double-click, <kbd>Space</kbd></span></div>
      </div></dialog>
    <dialog class="pl-peek" aria-label="Picture"><header><b class="pl-peek-t"></b><span class="pl-peek-c"></span>
      <button type="button" data-a="peekPrev" aria-label="Previous">◀</button><button type="button" data-a="peekNext" aria-label="Next">▶</button>
      <button type="button" data-a="closePeek">Close</button></header><img class="pl-peek-img" alt=""></dialog>`;
  const $ = (s) => el.querySelector(s);
  const cv = $(".pl-cv"), ctx = cv.getContext("2d"), strip = $(".pl-strip"), sctx = strip.getContext("2d");
  const S = { project: null, statuses: [], flags: [], item: 0, target: 0, data: null, img: null, mask: null,
              maskItem: -1, cls: 0, tool: "click", sel: null, drag: null, busy: false, filter: "",
              saving: false, savedAt: null, hintHidden: false, started: false, brush: 12, hover: null, paint: null,
              grid: null };
  try { S.hintHidden = localStorage.getItem("pl-hint-hidden") === "1"; } catch {}
  const MUTATING = new Set(["box", "click", "cycle", "delete", "set_class", "review", "accept", "undo", "settings", "paint", "tag", "accept_tags", "add_class"]);
  const seg = () => S.project?.task === "segment", classify = () => S.project?.task === "classify";
  const send = (m) => { if (MUTATING.has(m.type)) { S.saving = true; renderSaved(); } model.send(m); };
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  // ---- notifications ---------------------------------------------------------------------
  const post = (url, body) => fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) })
    .then(async (r) => { const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.detail || r.statusText); return d; });
  const center = notificationCenter({
    mount: $(".pl-bellmount"),
    onRead: () => model.send({ type: "notifications_read" }),
    onClear: () => model.send({ type: "notifications_clear" }),
    actionAvailable: (a) => web || (a.type === "goto" && a.project === S.project?.name),   // notebooks: no REST helpers
    onAction: (a) => runAction(a),
  });
  // ---- Rivet, the helper ------------------------------------------------------------------
  const flash = (box) => {
    const side = $(".pl-side");
    side.scrollBy({ top: box.getBoundingClientRect().top - side.getBoundingClientRect().top - 60 });
    const r = box.getBoundingClientRect();
    if (r.top < 0 || r.bottom > window.innerHeight) box.scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
    box.classList.add("flash"); setTimeout(() => box.classList.remove("flash"), 1600);
  };
  const helper = assistant({
    mount: $(".pl-rivetmount"), variant: "top", firstTip: 90000,
    transport: web ? webAssistant(model.homeUrl) : widgetAssistant(model),
    context: () => ({ page: web ? "annotator" : "notebook", kind: S.project?.kind, frame: S.item + 1, frames: S.project?.count,
                      confirmed: S.statuses.filter((s) => s === 4).length, to_check: S.flags.length, boxes: S.data?.boxes?.length ?? 0,
                      classes: S.project?.classes?.length, tool: S.tool, busy: S.busy }),
    actions: {
      export: { label: "Show Export", run: () => actions.exportJump() },
      shortcuts: { label: "Show shortcuts", run: () => actions.help() },
      notifications: { label: "Open notifications", run: () => center.open() },
      find: { label: "Show Find parts", run: () => flash($(".pl-find")) },
      settings: { label: "Open Settings", run: () => { $(".pl-settings").open = true; flash($(".pl-settings")); } },
      ...(web ? { projects: { label: "Go to projects", run: () => { location.href = model.homeUrl; } } } : {}),
    },
    starters: ["How can I export?", "How do I track through the video?", "What does 'To check' mean?", { q: "What makes PartLabeler different?", local: "pros" }],
  });

  async function runAction(a) {
    try {
      if (a.type === "goto") {
        if (a.project === S.project?.name) { center.close(); goto(a.item); }
        else location.href = `${model.homeUrl}p/${encodeURIComponent(a.project)}?item=${a.item}`;
      } else if (a.type === "open") location.href = `${model.homeUrl}p/${encodeURIComponent(a.project)}`;
      else if (a.type === "folder") await post(`${model.homeUrl}api/open-folder`, { path: a.path });
      else if (a.type === "restore") { const r = await post(`${model.homeUrl}api/trash/${encodeURIComponent(a.entry)}/restore`); toast({ level: "success", title: `Restored ${r.name}` }); }
    } catch (err) { toast({ level: "error", title: "That did not work", detail: String(err.message || err) }); }
  }

  // ---- drawing ---------------------------------------------------------------------------
  // Outlines arrive as 1-bit PNG crops; each is tinted once in its class colour (edge solid, inside see-through)
  const tints = new Map();
  function tinted(b) {
    const [mx, my, src] = b.mask, key = `${b.cls}|${src.length}|${src.slice(-40)}`;
    let t = tints.get(key);
    if (t) return t.ready ? t : null;
    if (tints.size > 600) tints.clear();
    t = { ready: false }; tints.set(key, t);
    const img = new Image();
    img.onload = () => {
      const w = img.width, h = img.height, c = document.createElement("canvas"); c.width = w; c.height = h;
      const x = c.getContext("2d"); x.drawImage(img, 0, 0);
      const d = x.getImageData(0, 0, w, h), px = d.data, inside = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) inside[i] = px[i * 4] > 127 ? 1 : 0;
      const [r, g, bl] = rgbOf(b.cls);
      for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) {
        const i = yy * w + xx;
        const edge = inside[i] && (xx === 0 || yy === 0 || xx === w - 1 || yy === h - 1 || !inside[i - 1] || !inside[i + 1] || !inside[i - w] || !inside[i + w]);
        px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = bl; px[i * 4 + 3] = inside[i] ? (edge ? 255 : 95) : 0;
      }
      x.putImageData(d, 0, 0);
      Object.assign(t, { ready: true, canvas: c, inside, w, h });
      draw();
    };
    img.src = src;
    return null;
  }
  const inMask = (b, x, y) => {
    const t = b.mask && tinted(b); if (!t) return false;
    const xx = Math.floor(x - b.mask[0]), yy = Math.floor(y - b.mask[1]);
    return xx >= 0 && yy >= 0 && xx < t.w && yy < t.h && t.inside[yy * t.w + xx] === 1;
  };

  function draw() {
    if (!S.img) return;
    ctx.drawImage(S.img, 0, 0);
    if (S.mask && S.maskItem === S.item && !seg()) ctx.drawImage(S.mask, 0, 0);
    const lw = Math.max(2, cv.width / 700), fs = Math.max(12, cv.width / 95);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    if (seg()) {
      for (const b of S.data?.boxes || []) {
        if (!b.mask || (S.paint && b.obj === S.paint.obj)) continue;
        const t = tinted(b); if (!t) continue;
        ctx.globalAlpha = b.source === "suggested" ? 0.55 : b.obj === S.sel ? 1 : 0.85;
        ctx.drawImage(t.canvas, b.mask[0], b.mask[1]);
      }
      ctx.globalAlpha = 1;
      if (S.paint) { ctx.globalAlpha = 0.6; ctx.drawImage(S.paint.cv, 0, 0); ctx.globalAlpha = 1; }
    }
    for (const b of S.data?.boxes || []) {
      const [x1, y1, x2, y2] = b.box, col = colorOf(b.cls), on = b.obj === S.sel;
      const outlined = seg() && b.mask;
      ctx.setLineDash(b.source === "tracked" ? [lw * 4, lw * 2] : b.source === "suggested" ? [lw, lw * 1.5] : outlined ? [lw * 1.2, lw * 2.4] : []);
      if (on) { ctx.lineWidth = lw * 3; ctx.strokeStyle = "#fff"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); }
      ctx.lineWidth = on ? lw * 1.8 : outlined ? lw * 0.6 : lw; ctx.strokeStyle = col;
      if (!outlined || on) ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      const label = S.project.classes[b.cls] ?? b.cls, tw = ctx.measureText(label).width + fs * 0.6;
      const ty = y1 - fs * 1.35 < 0 ? y2 : y1 - fs * 1.35;
      ctx.fillStyle = col; ctx.fillRect(x1, ty, tw, fs * 1.35);
      ctx.fillStyle = "#fff"; ctx.fillText(label, x1 + fs * 0.3, ty + fs * 1.05);
    }
    if (S.drag?.moved && S.tool === "box") {
      const d = S.drag; ctx.setLineDash([lw * 3, lw * 2]); ctx.lineWidth = lw; ctx.strokeStyle = "#fff";
      ctx.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
      ctx.setLineDash([]);
    }
    if (S.hover && (S.tool === "brush" || S.tool === "erase")) {           // the brush's footprint
      ctx.beginPath(); ctx.arc(S.hover[0], S.hover[1], S.brush, 0, Math.PI * 2);
      ctx.lineWidth = lw; ctx.strokeStyle = "#fff"; ctx.setLineDash(S.tool === "erase" ? [lw * 2, lw * 2] : []); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  function drawStrip() {
    const dpr = window.devicePixelRatio || 1, w = strip.clientWidth * dpr, h = strip.clientHeight * dpr;
    strip.width = w; strip.height = h;
    const n = S.statuses.length; if (!n) return;
    const css = getComputedStyle(el), cw = w / n;
    for (let i = 0; i < n; i++) {
      sctx.fillStyle = css.getPropertyValue(`--st${S.statuses[i]}`);
      sctx.fillRect(Math.floor(i * cw), 0, Math.ceil(cw), h);
    }
    sctx.fillStyle = css.getPropertyValue("--flag");
    for (const f of S.flags) sctx.fillRect(Math.floor(f * cw), 0, Math.max(2, Math.ceil(cw)), h * 0.3);
    const x = (S.item + 0.5) * cw;
    sctx.fillStyle = "#15202b"; sctx.fillRect(x - dpr, 0, 2 * dpr, h);
    sctx.beginPath(); sctx.moveTo(x - 5 * dpr, h); sctx.lineTo(x + 5 * dpr, h); sctx.lineTo(x, h - 7 * dpr); sctx.fill();
  }

  function renderSide() {
    const cls = S.project?.classes || [];
    const filt = $(".pl-filter"); filt.hidden = cls.length <= 8;
    const q = S.filter.trim().toLowerCase();
    const shown = cls.map((c, i) => [c, i]).filter(([c]) => !q || c.toLowerCase().includes(q));
    $(".pl-classes").innerHTML = !S.project ? "" : !cls.length ? `<span class="pl-empty">No classes yet: type one below${S.grid ? "; the selected pictures get it" : ""}.</span>` : shown.length ? shown.map(([c, i]) => `<button type="button" class="pl-row" data-cls="${i}" aria-pressed="${i === S.cls}">
      <span class="pl-sw" style="background:${colorOf(i)}" aria-hidden="true"></span><span class="pl-key">${i < 10 ? (i + 1) % 10 : ""}</span>
      <span class="pl-name" translate="no">${esc(c)}</span><span></span></button>`).join("") : `<span class="pl-empty">No class matches “${esc(S.filter)}”.</span>`;
    const boxes = S.data?.boxes || [];
    $(".pl-boxes").innerHTML = boxes.length ? boxes.map((b) => `<div class="pl-rowwrap"><button type="button" class="pl-row" data-obj="${b.obj}" aria-pressed="${b.obj === S.sel}">
      <span class="pl-sw" style="background:${colorOf(b.cls)}" aria-hidden="true"></span><span class="pl-key">#${b.obj}</span>
      <span class="pl-name"><span translate="no">${esc(cls[b.cls] ?? b.cls)}</span> <span class="pl-src">${b.source}${b.score != null && b.source !== "manual" ? " " + b.score.toFixed(2) : ""}</span></span><span></span></button>
      <button type="button" class="pl-x" data-del="${b.obj}" aria-label="Delete ${esc(cls[b.cls] ?? "")} box #${b.obj}" title="Delete (Del)">×</button></div>`).join("")
      : `<span class="pl-empty">No boxes yet. Click a part in the image.</span>`;
    el.querySelectorAll("[data-tool]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.tool === S.tool));
    const grid = Boolean(S.grid), cl = classify();
    el.querySelectorAll(".pl-seg").forEach((n) => (n.hidden = !seg() || grid));
    $(".pl-toolsec").hidden = grid; $(".pl-boxes").closest("section").hidden = grid; $(".pl-find").hidden = grid;
    $(".pl-sort").hidden = !grid;
    el.querySelectorAll(".pl-imgonly").forEach((n) => (n.hidden = !cl));
    el.querySelectorAll(".pl-partsonly").forEach((n) => (n.hidden = cl));
    $(".pl-bsize-v").textContent = `${S.brush} px`;
    const hint = $(".pl-hint");
    hint.hidden = S.hintHidden || !S.project || grid || boxes.length > 0 || S.statuses.some((s) => s >= 2);
  }

  function renderSaved() {
    const s = $(".pl-saved");
    if (S.lost) { s.className = "pl-saved lost"; s.textContent = "Connection lost"; return; }
    s.className = "pl-saved";
    if (S.saving) s.textContent = "Saving…";
    else if (S.savedAt) { s.innerHTML = `${CHECK}<span>All changes saved</span>`; s.title = `Every change is saved automatically. Last save ${when(S.savedAt / 1000)}.`; }
    else { s.textContent = ""; }
  }

  function renderTop() {
    if (!S.project || !S.data) return;
    $(".pl-title").textContent = S.project.name;
    $(".pl-where").textContent = `${S.project.kind === "video" ? "Frame" : "Image"} ${S.item + 1} of ${S.project.count}`;
    $(".pl-where").title = S.data.name;
    const chips = [];
    if (S.data.reviewed) chips.push(`<span class="pl-chip rev">Confirmed</span>`);
    if (S.flags.includes(S.item)) chips.push(`<span class="pl-chip flag" title="A tracked box changed size, jumped or was lost">To check</span>`);
    $(".pl-chips").innerHTML = chips.join(" ");
    const rev = S.statuses.filter((s) => s === 4).length, lab = S.statuses.filter((s) => s >= 2).length;
    $(".pl-counts").textContent = `${lab} labeled, ${rev} confirmed, ${S.flags.length} to check`;
    const busy = S.busy;
    el.querySelectorAll('[data-a="track"],[data-a="trackBack"],[data-b="suggest"],[data-b="findAll"],[data-b="export"]').forEach((b) => (b.disabled = busy));
    $('[data-a="stop"]').disabled = !busy;
    $('[data-a="undo"]').disabled = busy || !S.undo;
    $(".pl-video").hidden = S.project.kind !== "video";
    $('[data-a="review"]').textContent = S.data.reviewed ? "Unconfirm frame" : "Confirm frame";
    $(".pl-device").textContent = [S.project.device, S.gpu && `GPU memory in use: ${S.gpu}`].filter(Boolean).join(". ");
    el.querySelectorAll('[data-b="group"],[data-b="suggestTags"],[data-b="odd"],[data-b="outlineHere"],[data-b="outlineAll"],[data-b="sortParts"]')
      .forEach((b) => (b.disabled = busy));
    const cl = classify();
    $('[data-a="prev"]').closest(".pl-group").hidden = cl; $('[data-a="review"]').hidden = cl;
    $(".pl-strip").hidden = cl; $(".pl-legend").hidden = cl;
    if (cl) {
      const n = (v) => S.statuses.filter((s) => s === v).length, given = n(4) + n(2), sugg = n(1);
      $(".pl-where").textContent = `${S.project.count} images`;
      $(".pl-counts").textContent = `${given} classed, ${sugg} suggested, ${S.statuses.length - given - sugg} without a class`;
    }
  }

  function renderProject() {
    const fmt = $(".pl-fmt"), keep = fmt.value;
    fmt.innerHTML = (S.project.formats || ["yolo", "coco"]).map((f) => `<option value="${f}">${FORMAT_NAMES[f] || f}</option>`).join("");
    if (keep) fmt.value = keep;
    $(".pl-parent").value = S.project.parent || "";
    $(".pl-task").value = seg() ? "segment" : "detect";
    $(".pl-revonly").closest("label").hidden = classify();              // suggested classes are never exported
    $(".pl-tasksel").hidden = classify();
    el.querySelectorAll(".pl-parentf").forEach((n) => (n.hidden = classify()));
    $('[data-a="exportJump"]').title = `Export the dataset: ${(S.project.formats || []).map((f) => FORMAT_NAMES[f] || f).join(", ")}`;
    $(".pl-version").innerHTML = (S.project.version ? `PartLabeler version ${esc(S.project.version)}.` : "")
      + (web ? ` <a href="${model.homeUrl}#update">Check for updates</a>` : " Update: rerun Step 3 of the notebook.");
    const back = $(".pl-back"); back.hidden = !web; if (web) back.href = model.homeUrl;
    if (web) document.title = `${S.project.name} · PartLabeler`;
    if (classify() && !S.grid) openGrid("images");
    if (!seg() && (S.tool === "brush" || S.tool === "erase")) S.tool = "click";
  }

  function legend() {
    $(".pl-legend").innerHTML = STATUS.map((t, i) => `<span><i class="pl-sw" style="background:var(--st${i})" aria-hidden="true"></i>${t}</span>`).join("")
      + `<span><i class="pl-sw" style="background:var(--flag)" aria-hidden="true"></i>To check</span>`;
  }

  function toast({ level = "info", title, detail = "", actionLabel, onClick, ms }) {
    const t = document.createElement("div");
    t.className = `pl-toast ${level}`; t.setAttribute("role", level === "error" ? "alert" : "status");
    t.innerHTML = `<div><b></b>${detail ? "<small></small>" : ""}</div>`;
    t.querySelector("b").textContent = title; if (detail) t.querySelector("small").textContent = detail;
    if (actionLabel) {
      const b = Object.assign(document.createElement("button"), { type: "button", textContent: actionLabel });
      b.addEventListener("click", () => { onClick(); t.remove(); }); t.appendChild(b);
    }
    $(".pl-toasts").appendChild(t);
    while ($(".pl-toasts").children.length > 4) $(".pl-toasts").firstChild.remove();
    setTimeout(() => t.remove(), ms || (level === "error" ? 10000 : actionLabel ? 7000 : 4500));
  }
  const hint = (text) => toast({ title: text });

  // ---- actions ---------------------------------------------------------------------------
  // `target` runs ahead of `item` (the frame on screen) so quick repeated steps are not lost
  const goto = (i) => {
    if (!S.project) return;
    i = Math.max(0, Math.min(S.project.count - 1, i));
    if (i === S.target && i === S.item) return;
    S.target = i; S.sel = null; send({ type: "goto", item: i });
  };
  function nextTodo() {
    const after = (arr) => arr.find((i) => i > S.target);
    const todo = after(S.flags) ?? after(S.statuses.map((s, i) => (s !== 4 ? i : -1)).filter((i) => i >= 0));
    if (todo != null) goto(todo); else hint("Nothing left to check after this frame");
  }
  function setClass(i) {
    if (!S.project || i < 0 || i >= S.project.classes.length) return;
    S.cls = i;
    if (S.grid) { tagSelected(i); renderSide(); return; }
    if (S.sel != null) send({ type: "set_class", obj: S.sel, cls: i, item: S.item });
    renderSide();
  }
  const trackN = () => Math.max(1, parseInt($(".pl-n").value, 10) || 20);
  const video = () => S.project?.kind === "video";
  const track = (count, direction) => { if (video()) send({ type: "track", item: S.item, count, direction }); };
  const actions = {
    prev: () => goto(S.target - 1), next: () => goto(S.target + 1), nextTodo,
    track: () => track(trackN(), 1), trackAll: () => track(-1, 1),
    trackBack: () => track(trackN(), -1), trackBackAll: () => track(-1, -1),
    stop: () => send({ type: "stop" }),
    undo: () => send({ type: "undo" }),
    suggest: () => send({ type: "suggest", item: S.item }),
    findAll: () => send({ type: "find_all", item: S.item, obj: S.sel }),
    accept: () => send({ type: "accept", item: S.item }),
    review: () => { send({ type: "review", item: S.item, value: !S.data?.reviewed }); if (!S.data?.reviewed) setTimeout(() => goto(S.item + 1), 50); },
    // (review targets the frame on screen; navigation after it starts from there)
    export: () => send({ type: "export", format: $(".pl-fmt").value, reviewed_only: $(".pl-revonly").checked }),
    exportJump: () => { flash($(".pl-export")); $(".pl-fmt").focus(); },   // the controls sit below a possibly long class list
    saveSettings: () => send({ type: "settings", parent: $(".pl-parent").value, ...(classify() ? {} : { task: $(".pl-task").value }) }),
    sortParts: () => openGrid("parts"),
    closeGrid: () => closeGrid(),
    group: () => send({ type: "sort", of: S.grid?.of || "images" }),
    suggestTags: () => send({ type: "suggest_tags" }),
    acceptTags: () => send({ type: "accept_tags", keys: S.grid?.sel.size ? [...S.grid.sel] : undefined }),
    odd: () => send({ type: "odd", of: S.grid?.of || "images" }),
    outlineHere: () => send({ type: "outline", item: S.item }),
    outlineAll: () => send({ type: "outline", item: S.item, all: true }),
    peekPrev: () => peekStep(-1), peekNext: () => peekStep(1),
    closePeek: () => $(".pl-peek").close(),
    full: () => toggleFull(),
    help: () => $(".pl-help-dlg").showModal(),
    closeHelp: () => $(".pl-help-dlg").close(),
    hideHint: () => { S.hintHidden = true; try { localStorage.setItem("pl-hint-hidden", "1"); } catch {} renderSide(); el.focus(); },
  };
  el.addEventListener("click", (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a; if (a && actions[a]) { actions[a](); return; }
    if (S.grid && e.target.closest(".pl-gridwrap") && gridClick(e)) return;
    if (!e.target.closest(".pl-side")) return;
    const b = e.target.closest("[data-b]")?.dataset.b; if (b) { actions[b](); return; }
    const t = e.target.closest("[data-tool],[data-cls],[data-obj],[data-del]"); if (!t) return;
    if (t.dataset.del) { send({ type: "delete", item: S.item, obj: +t.dataset.del }); if (S.sel === +t.dataset.del) S.sel = null; }
    else if (t.dataset.tool) { S.tool = t.dataset.tool; renderSide(); }
    else if (t.dataset.cls) setClass(+t.dataset.cls);
    else if (t.dataset.obj) { S.sel = +t.dataset.obj; renderSide(); draw(); }
  });
  $(".pl-filter").addEventListener("input", (e) => { S.filter = e.target.value; renderSide(); });
  $(".pl-bsize").addEventListener("input", (e) => { S.brush = +e.target.value; renderSide(); });
  $(".pl-addcls").addEventListener("submit", (e) => {
    e.preventDefault();
    const name = $(".pl-newcls").value.trim(); if (!name) return;
    const G = S.grid, keys = G?.sel.size ? [...G.sel] : undefined;
    send({ type: "add_class", name, of: G?.of, keys }); $(".pl-newcls").value = "";
  });
  let kTimer = 0;
  $(".pl-k").addEventListener("input", (e) => {
    $(".pl-kv").textContent = e.target.value;
    clearTimeout(kTimer); kTimer = setTimeout(() => send({ type: "sort", of: S.grid.of, k: +e.target.value }), 120);
  });
  el.addEventListener("change", (e) => {
    const sel = e.target.closest("[data-gname]"); if (!sel || sel.value === "" || !S.grid) return;
    const G = S.grid, cls = +sel.value, keys = sections().find((x) => x.group === +sel.dataset.gname)?.keys || [];
    const todo = G.of === "images" ? keys.filter((k) => !isGiven(G.cards.get(k)) || G.cards.get(k).cls === cls) : keys;
    if (todo.length) send({ type: "tag", of: G.of, keys: todo, cls });
    if (keys.length > todo.length) hint(`${keys.length - todo.length} kept the class you gave them before`);
    sel.value = "";
  });
  el.addEventListener("dblclick", (e) => { const c = S.grid && e.target.closest("[data-card]"); if (c) peek(+c.dataset.card); });

  // ---- the grid: image classes, and sorting the parts of a box or outline project --------------
  const FILTERS = { all: "All", none: "No class", sugg: "Suggested", given: "Classed", odd: "To check", unsure: "Unsure", dups: "Duplicates" };
  const isGiven = (c) => c && c.cls != null && c.source !== "suggested";
  const PASS = {
    all: () => true, none: (c) => c.cls == null, sugg: (c) => c.source === "suggested", given: isGiven,
    odd: (c) => S.grid.odd.has(c.key), unsure: (c) => S.grid.unsure.has(c.key), dups: (c) => S.grid.dupOf.has(c.key),
  };
  function openGrid(of) {
    S.grid = { of, cards: new Map(), order: [], groups: null, unsure: new Set(), dups: [], dupOf: new Set(), odd: new Map(),
               filter: "all", sel: new Set(), cur: null, anchor: null, shown: 400, visible: [], thumbs: new Map(), want: new Set() };
    S.sel = null; S.tool = S.tool === "brush" || S.tool === "erase" ? "click" : S.tool;
    $(".pl-gtitle").textContent = of === "images" ? "Images" : "Parts";
    $(".pl-gridwrap").hidden = false; $(".pl-stage").hidden = true; $(".pl-range").hidden = true;
    send({ type: "grid", of });
    renderGrid(); renderSide(); renderTop();
    $(".pl-grid").focus?.();
  }
  function closeGrid() {
    if (!S.grid || classify()) return;
    S.grid = null;
    $(".pl-gridwrap").hidden = true; $(".pl-stage").hidden = false;
    renderSide(); renderTop();
    send({ type: "goto", item: S.target });                        // classes may have changed
  }
  function sections() {
    const G = S.grid, pass = PASS[G.filter], keep = (keys) => keys.filter((k) => G.cards.has(k) && pass(G.cards.get(k)));
    if (G.filter === "sugg") return [{ title: "Suggested, least sure first", keys: keep(G.order).sort((a, b) => (G.cards.get(a).score ?? 0) - (G.cards.get(b).score ?? 0)) }];
    if (G.filter === "odd") return [{ title: "These look like another class (the one on the badge)", keys: [...G.odd.keys()].filter((k) => G.cards.has(k)) }];
    if (G.filter === "dups") return G.dups.map((d, i) => ({ title: `Near-duplicates ${i + 1}`, keys: d.filter((k) => G.cards.has(k)) }));
    if (G.groups) return G.groups.map((g, i) => ({ title: `Group ${i + 1}`, keys: keep(g), group: i })).filter((x) => x.keys.length);
    return [{ title: "", keys: keep(G.order) }];
  }
  function cardHtml(c) {
    const G = S.grid, cls = S.project.classes, name = c.cls != null ? cls[c.cls] ?? c.cls : "No class";
    const badge = G.odd.has(c.key) ? `<span class="pl-badge odd">Looks like ${esc(cls[G.odd.get(c.key)[0]])}</span>`
      : G.dupOf.has(c.key) ? `<span class="pl-badge">Near-duplicate</span>` : G.unsure.has(c.key) ? `<span class="pl-badge">Unsure</span>` : "";
    const src = c.src || G.thumbs.get(c.key);
    const sure = c.source === "suggested" && c.score != null ? ` ${Math.round(c.score * 100)}%` : "";
    return `<button type="button" class="pl-card${c.source === "suggested" ? " sugg" : ""}${G.cur === c.key ? " cur" : ""}" data-card="${c.key}"
      aria-pressed="${G.sel.has(c.key)}" title="${esc(c.name)}"><img alt="${esc(c.name)}" loading="lazy" decoding="async" ${src ? `src="${esc(src)}"` : `data-need="${c.key}"`}>${badge}
      <span class="pl-card-cls">${c.cls != null ? `<i class="pl-sw" style="background:${colorOf(c.cls)}" aria-hidden="true"></i>` : ""}<span translate="no">${esc(name)}${sure}</span></span></button>`;
  }
  function renderGrid() {
    const G = S.grid; if (!G) return;
    const all = [...G.cards.values()], count = (f) => all.filter(PASS[f]).length;
    $(".pl-filters").innerHTML = Object.entries(FILTERS)
      .filter(([f]) => (G.of === "images" || !["none", "sugg"].includes(f)) && (f === "all" || f === G.filter || count(f) > 0))
      .map(([f, t]) => `<button type="button" data-filter="${f}" aria-pressed="${f === G.filter}">${t}<b>${count(f)}</b></button>`).join("");
    const secs = sections(), total = secs.reduce((n, x) => n + x.keys.length, 0);
    let budget = G.shown; G.visible = [];
    const cls = S.project.classes.map((n, i) => `<option value="${i}">${esc(n)}</option>`).join("");
    const html = secs.map((x) => {
      const keys = x.keys.slice(0, Math.max(0, budget)); budget -= keys.length; G.visible.push(...keys);
      if (!keys.length) return "";
      const head = x.title ? `<h3>${esc(x.title)} <span class="pl-src">${x.keys.length}</span>${x.group != null
        ? ` <button type="button" data-gsel="${x.group}">Select all</button><select data-gname="${x.group}" aria-label="Give ${esc(x.title)} a class">
            <option value="">Give all a class…</option>${cls}</select>` : ""}</h3>` : "";
      return `<section class="pl-gsec">${head}<div class="pl-cards">${keys.map((k) => cardHtml(G.cards.get(k))).join("")}</div></section>`;
    }).join("");
    const grid = $(".pl-grid"), top = grid.scrollTop;
    grid.innerHTML = html ? html + (total > G.shown ? `<button type="button" class="pl-more" data-more="1">Show ${Math.min(800, total - G.shown)} more</button>` : "")
      : `<p class="pl-empty">${!G.loaded ? "Loading…" : G.cards.size ? "Nothing here with this filter." : G.of === "parts" ? "No parts yet: label some first." : "No images."}</p>`;
    grid.scrollTop = top;
    renderGridCount();
    needThumbs();
  }
  function renderGridCount() {
    const G = S.grid;
    $(".pl-gcount").textContent = `${G.cards.size} ${G.of === "images" ? "images" : "parts"}${G.sel.size ? `, ${G.sel.size} selected` : ""}`;
  }
  function paintSel() {
    const G = S.grid;
    el.querySelectorAll(".pl-grid [data-card]").forEach((c) => {
      const k = +c.dataset.card; c.setAttribute("aria-pressed", String(G.sel.has(k))); c.classList.toggle("cur", k === G.cur);
    });
    renderGridCount();
  }
  function gridClick(e) {
    const G = S.grid;
    const f = e.target.closest("[data-filter]"); if (f) { G.filter = f.dataset.filter; G.shown = 400; renderGrid(); return true; }
    if (e.target.closest("[data-more]")) { G.shown += 800; renderGrid(); return true; }
    const gs = e.target.closest("[data-gsel]"); if (gs) { G.sel = new Set(sections().find((x) => x.group === +gs.dataset.gsel)?.keys); paintSel(); return true; }
    const card = e.target.closest("[data-card]"); if (!card) return false;
    const k = +card.dataset.card;
    if (e.shiftKey && G.anchor != null) {
      const a = G.visible.indexOf(G.anchor), b = G.visible.indexOf(k);
      if (a >= 0 && b >= 0) G.visible.slice(Math.min(a, b), Math.max(a, b) + 1).forEach((x) => G.sel.add(x));
    } else if (e.ctrlKey || e.metaKey) { G.sel.has(k) ? G.sel.delete(k) : G.sel.add(k); G.anchor = k; }
    else { G.sel = new Set([k]); G.anchor = k; }
    G.cur = k; paintSel();
    return true;
  }
  function tagSelected(cls) {
    const G = S.grid, keys = G.sel.size ? [...G.sel] : G.cur != null ? [G.cur] : [];
    if (!keys.length) { hint("Select pictures first: click, Shift+click or Ctrl+A"); return; }
    send({ type: "tag", of: G.of, keys, cls });
  }
  function moveCur(step, extend) {
    const G = S.grid; if (!G.visible.length) return;
    let i = G.visible.indexOf(G.cur); i = i < 0 ? 0 : Math.max(0, Math.min(G.visible.length - 1, i + step));
    G.cur = G.visible[i];
    if (extend) G.sel.add(G.cur); else { G.sel = new Set([G.cur]); G.anchor = G.cur; }
    paintSel();
    el.querySelector(`.pl-grid [data-card="${G.cur}"]`)?.scrollIntoView({ block: "nearest" });
  }
  function columns() {
    const c = el.querySelector(".pl-grid .pl-cards");
    return c ? Math.max(1, getComputedStyle(c).gridTemplateColumns.split(" ").length) : 1;
  }
  function gridKey(e) {
    const G = S.grid, k = e.key;
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "a") { G.sel = new Set(G.visible); paintSel(); return true; }
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { actions.undo(); return true; }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    if (/^[0-9]$/.test(k)) setClass(k === "0" ? 9 : +k - 1);
    else if (k === "[" || k === "]") setClass((S.cls + (k === "]" ? 1 : -1) + S.project.classes.length) % S.project.classes.length);
    else if (k === "ArrowRight" || k === "ArrowLeft") moveCur(k === "ArrowRight" ? 1 : -1, e.shiftKey);
    else if (k === "ArrowDown" || k === "ArrowUp") moveCur((k === "ArrowDown" ? 1 : -1) * columns(), e.shiftKey);
    else if ((k === "Delete" || k === "Backspace") && G.of === "images") tagSelectedClear();
    else if (k === " " || k === "Enter") { if (G.cur != null) peek(G.cur); }
    else if (k === "Escape") { if (G.sel.size) { G.sel = new Set(); paintSel(); } else closeGrid(); }
    else if (k === "g") actions.group();
    else if (k === "s" && G.of === "images") actions.suggestTags();
    else if (k === "y" && G.of === "images") actions.acceptTags();
    else return false;
    return true;
  }
  function tagSelectedClear() {
    const G = S.grid, keys = G.sel.size ? [...G.sel] : G.cur != null ? [G.cur] : [];
    if (keys.length) send({ type: "tag", of: G.of, keys, cls: null });
  }
  // Notebooks: thumbnails come as data URLs for the cards in view (the web page loads them by URL)
  let thumbTimer = 0;
  const seen = typeof IntersectionObserver !== "undefined" ? new IntersectionObserver((ents) => {
    for (const en of ents) if (en.isIntersecting && S.grid) { S.grid.want.add(+en.target.dataset.need); seen.unobserve(en.target); }
    clearTimeout(thumbTimer);
    thumbTimer = setTimeout(() => {
      if (!S.grid?.want.size) return;
      const keys = [...S.grid.want].slice(0, 60); keys.forEach((k) => S.grid.want.delete(k));
      send({ type: "thumbs", of: S.grid.of, keys });
    }, 60);
  }, { root: $(".pl-grid"), rootMargin: "400px" }) : null;
  function needThumbs() { if (seen) el.querySelectorAll(".pl-grid img[data-need]").forEach((img) => seen.observe(img)); }
  // A closer look: image projects show the picture (arrows step, numbers give a class); parts open their frame
  function peek(k) {
    const G = S.grid, c = G.cards.get(k); if (!c) return;
    if (G.of === "parts") { closeGrid(); goto(c.item); S.pendingSel = k; return; }
    G.peek = k; G.cur = k; paintSel();
    const dlg = $(".pl-peek"); if (!dlg.open) dlg.showModal();
    $(".pl-peek-t").textContent = c.name;
    const t = G.cards.get(k);
    $(".pl-peek-c").innerHTML = t.cls != null ? `<span class="pl-chip ${isGiven(t) ? "rev" : ""}">${esc(S.project.classes[t.cls])}${isGiven(t) ? "" : " (suggested)"}</span>` : `<span class="pl-src">No class: press 1–9</span>`;
    const img = $(".pl-peek-img");
    img.src = web ? `${model.homeUrl}items/${encodeURIComponent(S.project.name)}/${k}.jpg` : G.thumbs.get(k) || "";
    if (!web) { S.target = k; send({ type: "goto", item: k }); }       // the item message brings the full picture
  }
  function peekStep(step) {
    const G = S.grid, i = G.visible.indexOf(G.peek);
    if (i >= 0 && G.visible[i + step] != null) peek(G.visible[i + step]);
  }
  $(".pl-peek").addEventListener("keydown", (e) => {
    const k = e.key, G = S.grid; if (!G) return;
    if (k === "ArrowRight" || k === "ArrowLeft") peekStep(k === "ArrowRight" ? 1 : -1);
    else if (/^[0-9]$/.test(k)) { S.cls = k === "0" ? 9 : +k - 1; send({ type: "tag", of: G.of, keys: [G.peek], cls: S.cls }); renderSide(); }
    else if (k === "Delete" || k === "Backspace") send({ type: "tag", of: G.of, keys: [G.peek], cls: null });
    else return;
    e.preventDefault(); e.stopPropagation();
  });
  strip.addEventListener("click", (e) => { const r = strip.getBoundingClientRect(); goto(Math.floor((e.clientX - r.left) / r.width * S.statuses.length)); });

  // ---- full screen --------------------------------------------------------------------------
  // The web app and Jupyter/VS Code can use the browser's full screen. Colab's output frames may not (their
  // iframe does not allow it), so there the annotator grows to the height of the window instead; the browser's
  // own full screen (F11) around it gives the rest.
  function setBig(on) {
    S.big = on; el.classList.toggle("big", on);
    const colab = globalThis.google?.colab?.output;
    if (on) {
      const h = Math.max(560, (window.outerHeight || screen.availHeight || 900) - 240);   // minus browser + Colab bars
      el.style.setProperty("--plh", `${h}px`);
      try { colab?.setIframeHeight?.(h + 16, true); } catch {}
      el.scrollIntoView?.({ block: "start" });
      if (!S.bigHint) { S.bigHint = true; toast({ title: "Filled the window height", detail: "For a true full screen, press F11 in your browser as well. Click ⤡ to go back." }); }
    } else {
      el.style.removeProperty("--plh");
      try { colab?.resizeIframeToContent?.(); } catch {}
    }
    renderFull();
  }
  function renderFull() {
    const on = Boolean(document.fullscreenElement) || S.big, b = $(".pl-full");
    b.textContent = on ? "⤡" : "⤢";
    b.setAttribute("aria-label", on ? "Leave full screen" : "Full screen"); b.title = b.getAttribute("aria-label");
  }
  async function toggleFull() {
    if (document.fullscreenElement) { await document.exitFullscreen().catch(() => {}); return; }
    if (S.big) { setBig(false); return; }
    try {
      if (!document.fullscreenEnabled) throw new Error("not allowed here");
      await (web ? document.documentElement : el).requestFullscreen();
    } catch { setBig(true); }
  }
  document.addEventListener("fullscreenchange", () => { renderFull(); drawStrip(); });

  // ---- mouse on the image ----------------------------------------------------------------
  const toImage = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
  function boxAt(x, y) {
    const size = (b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
    if (seg()) { const m = (S.data?.boxes || []).filter((b) => inMask(b, x, y)).sort((a, b) => size(a) - size(b))[0]; if (m) return m.obj; }
    const hits = (S.data?.boxes || []).filter((b) => x >= b.box[0] && x <= b.box[2] && y >= b.box[1] && y <= b.box[3]);
    hits.sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
    return hits[0]?.obj ?? null;
  }
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  // Brush and eraser (outline projects): paint on a copy of the part's mask; on release the page sends the
  // part's whole new mask as a PNG crop (engine: on_paint)
  const painting = () => seg() && (S.tool === "brush" || S.tool === "erase");
  function startPaint(x, y) {
    if (S.tool === "erase" && S.sel == null) { hint("Select an outline first (Ctrl+click it), then erase"); return; }
    const b = S.sel != null ? S.data.boxes.find((bb) => bb.obj === S.sel) : null;
    const t = b?.mask ? tinted(b) : null;
    if (b?.mask && !t) return;                                       // its mask is still loading
    const pc = document.createElement("canvas"); pc.width = cv.width; pc.height = cv.height;
    const px = pc.getContext("2d"), cls = b ? b.cls : S.cls;
    let bounds = [x, y, x, y];
    if (t) {
      const id = px.createImageData(t.w, t.h), [r, g, bl] = rgbOf(cls);
      for (let i = 0; i < t.w * t.h; i++) if (t.inside[i]) { id.data[i * 4] = r; id.data[i * 4 + 1] = g; id.data[i * 4 + 2] = bl; id.data[i * 4 + 3] = 255; }
      px.putImageData(id, b.mask[0], b.mask[1]);
      bounds = [Math.min(x, b.mask[0]), Math.min(y, b.mask[1]), Math.max(x, b.mask[0] + t.w), Math.max(y, b.mask[1] + t.h)];
    }
    px.lineCap = px.lineJoin = "round"; px.lineWidth = S.brush * 2; px.strokeStyle = colorOf(cls);
    px.globalCompositeOperation = S.tool === "erase" ? "destination-out" : "source-over";
    S.paint = { cv: pc, px, obj: b ? b.obj : null, cls, last: [x, y], bounds };
    strokeTo(x, y);
  }
  function strokeTo(x, y) {
    const P = S.paint, [lx, ly] = P.last, r = S.brush + 2;
    P.px.beginPath(); P.px.moveTo(lx, ly); P.px.lineTo(x + 0.01, y); P.px.stroke(); P.last = [x, y];
    P.bounds = [Math.min(P.bounds[0], x - r), Math.min(P.bounds[1], y - r), Math.max(P.bounds[2], x + r), Math.max(P.bounds[3], y + r)];
    draw();
  }
  function finishPaint() {
    const P = S.paint; S.paint = null;
    const x0 = Math.max(0, Math.floor(P.bounds[0])), y0 = Math.max(0, Math.floor(P.bounds[1]));
    const x1 = Math.min(cv.width, Math.ceil(P.bounds[2])), y1 = Math.min(cv.height, Math.ceil(P.bounds[3]));
    let bx0 = Infinity, by0 = Infinity, bx1 = -1, by1 = -1;
    if (x1 > x0 && y1 > y0) {
      const d = P.px.getImageData(x0, y0, x1 - x0, y1 - y0).data, w = x1 - x0;
      for (let i = 3, j = 0; i < d.length; i += 4, j++) {
        if (d[i] > 127) { const xx = j % w, yy = (j / w) | 0; if (xx < bx0) bx0 = xx; if (xx > bx1) bx1 = xx; if (yy < by0) by0 = yy; if (yy > by1) by1 = yy; }
      }
    }
    if (bx1 < 0 && P.obj == null) { draw(); return; }                 // erased nothing into nothing
    const out = document.createElement("canvas");
    let ox = 0, oy = 0;
    if (bx1 < 0) out.width = out.height = 1;                          // all erased: the part is removed
    else {
      out.width = bx1 - bx0 + 1; out.height = by1 - by0 + 1; ox = x0 + bx0; oy = y0 + by0;
      out.getContext("2d").drawImage(P.cv, ox, oy, out.width, out.height, 0, 0, out.width, out.height);
    }
    send({ type: "paint", item: S.item, obj: P.obj ?? undefined, cls: P.cls, x: ox, y: oy, png: out.toDataURL("image/png") });
  }
  let hoverRaf = 0;
  cv.addEventListener("mouseleave", () => { if (S.hover) { S.hover = null; draw(); } });
  cv.addEventListener("mousedown", (e) => {
    if (!S.img) return; const [x, y] = toImage(e); el.focus();
    if (painting() && !(e.ctrlKey || e.metaKey) && e.button === 0) { startPaint(x, y); return; }
    S.drag = { x0: x, y0: y, x1: x, y1: y, moved: false, e };
  });
  cv.addEventListener("mousemove", (e) => {
    if (S.paint) { const [x, y] = toImage(e); strokeTo(x, y); return; }
    if (painting() && !S.drag) { S.hover = toImage(e); if (!hoverRaf) hoverRaf = requestAnimationFrame(() => { hoverRaf = 0; draw(); }); return; }
    if (!S.drag) return; const [x, y] = toImage(e); S.drag.x1 = x; S.drag.y1 = y;
    const r = cv.getBoundingClientRect(), px = cv.width / r.width;
    S.drag.moved ||= Math.hypot(x - S.drag.x0, y - S.drag.y0) > 5 * px; if (S.drag.moved) draw();
  });
  window.addEventListener("mouseup", (e) => {
    if (S.paint) { finishPaint(); return; }
    const d = S.drag; if (!d) return; S.drag = null;
    if (e.ctrlKey || e.metaKey) { S.sel = boxAt(d.x0, d.y0); renderSide(); draw(); return; }
    if (S.tool === "box") {
      if (d.moved) {
        const box = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
        send({ type: "box", item: S.item, cls: S.cls, box, obj: S.sel ?? undefined });
      } else { S.sel = boxAt(d.x0, d.y0); renderSide(); draw(); }
      return;
    }
    if (d.moved) { draw(); return; }
    const refine = S.sel != null && (e.shiftKey || e.altKey || e.button === 2);
    if ((e.altKey || e.button === 2) && S.sel == null) { hint("Select a box first (Ctrl+click), then Alt+click to shrink it"); return; }
    send({ type: "click", item: S.item, x: d.x0, y: d.y0, cls: S.cls, positive: !(e.altKey || e.button === 2),
           obj: refine ? S.sel : undefined });
  });

  // ---- keys --------------------------------------------------------------------------------
  el.addEventListener("keydown", (e) => {
    if (e.target.matches("input,select,textarea") || $(".pl-help-dlg").open || $(".pl-peek").open || center.isOpen) return;
    if (S.grid && gridKey(e)) { e.preventDefault(); return; }
    const k = e.key;
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") actions.undo();
    else if (e.ctrlKey || e.metaKey || e.altKey) return;
    else if (/^[0-9]$/.test(k)) { setClass(k === "0" ? 9 : +k - 1); }
    else if (k === "ArrowRight" || k === "d") { e.shiftKey ? nextTodo() : goto(S.target + 1); }
    else if (k === "ArrowLeft" || k === "a") goto(S.target - 1);
    else if (k === "Enter") actions.review();
    else if (k === "Delete" || k === "Backspace") { if (S.sel != null) { send({ type: "delete", item: S.item, obj: S.sel }); S.sel = null; } }
    else if (k === "Escape") { S.sel = null; S.drag = null; renderSide(); draw(); }
    else if (k === "t") actions.track(); else if (k === "T") actions.trackAll();
    else if (k === "r") actions.trackBack(); else if (k === "R") actions.trackBackAll();
    else if (k === "s") actions.suggest(); else if (k === "f") actions.findAll();
    else if (k === "y") actions.accept(); else if (k === "x") actions.stop();
    else if (k === "c" || k === "b") { S.tool = k === "c" ? "click" : "box"; renderSide(); draw(); }
    else if ((k === "p" || k === "e") && seg()) { S.tool = k === "p" ? "brush" : "erase"; renderSide(); draw(); }
    else if ((k === "," || k === ".") && seg()) { S.brush = Math.max(1, Math.min(120, Math.round(S.brush * (k === "." ? 1.25 : 0.8)))); $(".pl-bsize").value = S.brush; renderSide(); draw(); }
    else if (k === "g" && !classify()) actions.sortParts();
    else if (k === "m") { if (S.sel != null) send({ type: "cycle", item: S.item, obj: S.sel }); }
    else if (k === "n") center.toggle();
    else if (k === "?") actions.help();
    else if (k === "h") helper.open();
    else if (k === "[" || k === "]") setClass((S.cls + (k === "]" ? 1 : -1) + S.project.classes.length) % S.project.classes.length);
    else return;
    e.preventDefault();
  });

  // ---- messages in -------------------------------------------------------------------------
  model.on("msg:custom", (m) => {
    if (m.type === "project") { S.project = m; legend(); renderProject(); renderSide(); renderTop(); if (S.grid) renderGrid(); }
    else if (m.type === "grid" && S.grid?.of === m.of) {
      S.grid.cards = new Map(m.cards.map((c) => [c.key, c])); S.grid.order = m.cards.map((c) => c.key); S.grid.loaded = true; renderGrid();
    }
    else if (m.type === "cards" && S.grid?.of === m.of) { for (const c of m.cards) S.grid.cards.set(c.key, c); renderGrid(); if ($(".pl-peek").open) peek(S.grid.peek); }
    else if (m.type === "groups" && S.grid?.of === m.of) {
      Object.assign(S.grid, { groups: m.groups, unsure: new Set(m.unsure), dups: m.dups, dupOf: new Set(m.dups.flat()) });
      const r = $(".pl-range"), k = $(".pl-k"); r.hidden = m.max_k < 2; k.max = m.max_k; k.value = m.k;
      $(".pl-kv").textContent = `${m.k}${m.k === m.best_k ? " (best fit)" : ""}`;
      renderGrid();
    }
    else if (m.type === "odd" && S.grid?.of === m.of) {
      S.grid.odd = new Map(m.items.map(([k, c, sh]) => [k, [c, sh]])); if (m.items.length) S.grid.filter = "odd"; renderGrid();
    }
    else if (m.type === "thumbs" && S.grid?.of === m.of) {
      for (const [k, src] of Object.entries(m.src)) {
        S.grid.thumbs.set(+k, src);
        el.querySelectorAll(`.pl-grid img[data-need="${k}"]`).forEach((img) => { img.src = src; img.removeAttribute("data-need"); });
      }
    }
    else if (m.type === "select") { S.sel = m.obj; }
    else if (m.type === "status") {
      Object.assign(S, { statuses: m.statuses, flags: m.flags, busy: m.busy, gpu: m.gpu, undo: m.undo });
      if (S.saving) { S.saving = false; S.savedAt = Date.now(); }
      drawStrip(); renderTop(); renderSaved(); renderSide();
    }
    else if (m.type === "item") {
      if (m.item !== S.target) return;                 // an older step; a newer one is on its way
      const img = new Image();
      img.onload = () => {
        if (m.item !== S.target) return;
        if (m.item !== S.item) S.mask = null;
        S.item = m.item; S.data = m; S.img = img;
        if (cv.width !== m.w || cv.height !== m.h) { cv.width = m.w; cv.height = m.h; }
        if (S.sel != null && !m.boxes.some((b) => b.obj === S.sel)) S.sel = null;
        if (S.pendingSel != null) { if (m.boxes.some((b) => b.obj === S.pendingSel)) S.sel = S.pendingSel; S.pendingSel = null; }
        if (S.grid?.peek === m.item && $(".pl-peek").open && !web) $(".pl-peek-img").src = m.src;
        draw(); renderSide(); renderTop(); drawStrip();
      };
      img.src = m.src;
    }
    else if (m.type === "mask") { const img = new Image(); img.onload = () => { S.mask = img; S.maskItem = m.item; S.sel = m.obj; draw(); renderSide(); }; img.src = m.src; }
    else if (m.type === "item_changed") {
      if (S.grid) send({ type: "grid", of: S.grid.of });
      else if (m.items.includes(S.target)) send({ type: "goto", item: S.target });
    }
    else if (m.type === "progress") {
      const f = m.finished ? 0 : m.done / Math.max(1, m.total);
      $(".pl-bar i").style.width = `${f * 100}%`;
      $(".pl-prog").textContent = m.finished ? "" : `${m.task} ${m.done} of ${m.total}${m.rate ? `, ${m.rate} frames/s` : ""}`;
      S.busy = !m.finished; renderTop();
    }
    else if (m.type === "pong") S.pong = Date.now();
    else if (m.type === "notifications") center.set(m.items, m.unread);
    else if (m.type === "notify") {
      center.add(m.item, m.unread);
      if (m.item.project && S.project && m.item.project !== S.project.name) return;   // another tab's event: badge only
      const undoable = m.item.key === "delete";
      toast({ level: m.item.level, title: m.item.title, detail: m.item.detail,
              actionLabel: undoable ? "Undo" : m.item.action?.type === "folder" && web ? "Open folder" : undefined,
              onClick: undoable ? () => send({ type: "undo" }) : () => runAction(m.item.action) });
    }
    else if (m.type === "toast") hint(m.text);
    else if (m.type === "error") toast({ level: "error", title: m.text });
    else if (m.type === "disconnected") { S.lost = true; renderSaved(); const b = $(".pl-banner"); b.hidden = false;
      b.textContent = "Connection to PartLabeler lost. Your work is saved; restart the app (run_windows.bat) and reload this page."; }
  });
  // keep keyboard shortcuts working after clicking buttons, the list or the timeline
  el.addEventListener("mousedown", (e) => { if (!e.target.matches("input,select,option,textarea")) setTimeout(() => el.focus(), 0); });
  new ResizeObserver(drawStrip).observe(strip);
  renderSide();
  el.focus();
  model.send({ type: "ready", item: model.startItem || 0 });
  if (model.startItem) S.target = model.startItem;
  // Notebooks: messages from the kernel's job threads wait until the page asks (Colab drops them otherwise).
  // Poll often while something runs, slowly otherwise; stop when the widget is gone.
  // Each poll is answered with a pong: no pong for 10 s means the kernel is busy with another cell or Colab
  // disconnected. Everything up to the last change is already saved, so the banner says how to get back.
  if (!web) {
    S.pong = Date.now();
    const poll = () => {
      if (!el.isConnected) return;
      model.send({ type: "poll" });
      const quiet = Date.now() - S.pong > 10000, b = $(".pl-banner");
      if (quiet !== Boolean(S.stalled)) {
        S.stalled = quiet; b.hidden = !quiet; b.classList.toggle("info", quiet);
        b.textContent = quiet ? "The notebook is not answering: it is busy running another cell, or Colab disconnected. "
          + "Everything up to your last change is saved. If it disconnected: reconnect, click into Step 6, choose "
          + "Runtime > Run before, then run Step 6 to carry on where you stopped." : "";
      }
      setTimeout(poll, S.busy || document.querySelector(".rv-panel .rv-typing, .rv-send.stop") ? 300 : 1200);
    };
    setTimeout(poll, 500);
  }
}

export default { render };
