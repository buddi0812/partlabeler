// PartLabeler start screen: projects (open, show folder, move to trash), new project (with a file browser),
// Teach & Transfer, the notification center shared with the annotator, and Rivet, the helper robot.
// Talks to ui/host_fastapi.py over plain JSON (/api/...). Long work runs as server jobs we poll.
// Visual identity: the annotation box. Amber corner brackets "lock on" to whatever is in focus.
import { notificationCenter, assistant, webAssistant } from "/ui/canvas.js";

const CSS = `
:root { --alu:#e7ecef; --alu-2:#dde4e8; --paper:#fff; --ink:#1f2a30; --steel:#53636c; --line:#d3dbe0;
  --teal:#0a7c78; --teal-deep:#075e5b; --teal-soft:#e0f0ee; --amber:#f2a900; --amber-ink:#7d5200; --amber-soft:#fdf1d2;
  --ok:#2f9e5b; --lab:#4f8fe0; --bad:#c2412f; --stage:#16232a;
  --display:"Bahnschrift SemiCondensed",Bahnschrift,"DIN Alternate","Barlow Semi Condensed","Segoe UI",system-ui,sans-serif;
  --numeric:Bahnschrift,"DIN Alternate","Segoe UI",system-ui,sans-serif;
  --text:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
  --spring:cubic-bezier(.2,.9,.3,1.25); --out:cubic-bezier(.2,.8,.2,1); }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:72px; }
html, body { overflow-x:clip; }
body { background:var(--alu); color:var(--ink); font:14px/1.5 var(--text); -webkit-font-smoothing:antialiased; }
[hidden] { display:none !important; }
a { color:var(--teal-deep); }
button, input, select, textarea { font:inherit; color:inherit; }
input[type=text], input[type=number], select, textarea { border:1px solid var(--line); border-radius:8px; padding:7px 10px; background:var(--paper);
  transition:border-color .15s, box-shadow .15s; }
input[type=text]:hover, input[type=number]:hover, select:hover, textarea:hover { border-color:#b9c5cc; }
input:focus-visible, select:focus-visible, textarea:focus-visible { outline:none; border-color:var(--teal); box-shadow:0 0 0 3px rgba(10,124,120,.18); }
textarea { min-height:92px; resize:vertical; font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; font-size:12.5px; }
input[type=number] { width:88px; font-variant-numeric:tabular-nums; }
button { border:1px solid var(--line); background:var(--paper); border-radius:8px; padding:7px 14px; cursor:pointer; white-space:nowrap;
  transition:border-color .15s, background .15s, transform .12s, box-shadow .15s; touch-action:manipulation; }
button:hover:not(:disabled) { border-color:var(--teal); }
button:active:not(:disabled) { transform:scale(.97); }
button:disabled { opacity:.5; cursor:default; }
button.primary { background:var(--teal); border-color:var(--teal); color:#fff; font-weight:600; }
button.primary:hover:not(:disabled) { background:var(--teal-deep); border-color:var(--teal-deep); }
button.danger { background:var(--bad); border-color:var(--bad); color:#fff; font-weight:600; }
button:focus-visible, a:focus-visible, summary:focus-visible { outline:2px solid var(--teal); outline-offset:2px; }

/* top bar */
.h-top { position:sticky; top:0; z-index:20; background:rgba(231,236,239,.86); backdrop-filter:saturate(1.3) blur(10px);
  border-bottom:1px solid transparent; transition:border-color .2s, box-shadow .2s; }
.h-top.scrolled { border-bottom-color:var(--line); box-shadow:0 6px 18px rgba(31,42,48,.06); }
.h-top-in { max-width:1240px; margin:0 auto; padding:10px 20px; display:flex; align-items:center; gap:12px; }
.h-logo { display:inline-flex; align-items:center; gap:9px; font:600 19px/1 var(--display); letter-spacing:.01em; color:var(--ink); text-decoration:none; margin-right:auto; }
.h-logo svg { width:26px; height:26px; }
.h-device { display:inline-flex; align-items:center; gap:6px; font-size:12.5px; color:var(--steel); background:var(--paper); border:1px solid var(--line);
  border-radius:99px; padding:3px 11px 3px 8px; max-width:40vw; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h-device i { width:8px; height:8px; border-radius:50%; background:var(--ok); flex:none; box-shadow:0 0 0 3px rgba(47,158,91,.18); }
.h-device.cpu i { background:var(--amber); box-shadow:0 0 0 3px rgba(242,169,0,.2); }

.h-wrap { max-width:1240px; margin:0 auto; padding:8px 20px 120px; display:grid; gap:28px; }

/* hero */
.h-hero { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.08fr); gap:36px; align-items:center; padding:30px 0 8px; }
.h-hero h1 { font:600 clamp(30px, 3.6vw, 44px)/1.06 var(--display); letter-spacing:-.005em; margin:0 0 14px; max-width:15ch; text-wrap:balance; }
.h-hero h1 span { display:block; }
.h-lede { font-size:15.5px; color:#3d4b53; margin:0 0 20px; max-width:52ch; }
.h-actions { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.h-actions button { padding:9px 18px; font-size:14.5px; }
.h-home { color:var(--steel); font-size:12.5px; margin:16px 0 0; overflow-wrap:anywhere; }
.h-stage { position:relative; }
.h-demo { width:100%; height:auto; display:block; }
.h-demo text { font-family:var(--numeric); }
.h-in { animation:h-rise .7s var(--out) both; animation-delay:calc(var(--d, 0) * 90ms); }
@keyframes h-rise { from { opacity:0; transform:translateY(14px); } }

/* sections */
.h-sec-head { display:flex; align-items:baseline; flex-wrap:wrap; gap:6px 14px; margin-bottom:12px; }
.h-sec-head h2 { font:600 22px/1.2 var(--display); margin:0; letter-spacing:.005em; }
.h-sec-head .h-count { color:var(--steel); font:500 15px var(--numeric); }
.h-sec-head .h-sp { flex:1; }
.h-sec-lede { margin:-4px 0 16px; color:var(--steel); max-width:78ch; }
.h-grid { display:grid; grid-template-columns:minmax(0,1.65fr) minmax(320px,1fr); gap:28px; align-items:start; }
.h-panel { background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:20px; position:relative; }
.h-sticky { position:sticky; top:72px; }

/* project tiles */
.h-projects { display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:14px; }
.h-proj { position:relative; background:var(--paper); border:1px solid var(--line); border-radius:12px;
  animation:h-rise .55s var(--out) both; animation-delay:calc(280ms + var(--i, 0) * 55ms);
  transition:transform .25s var(--spring), box-shadow .25s, border-color .2s; }
.h-proj:hover, .h-proj:focus-within { transform:translateY(-3px); border-color:#c2ccd2; box-shadow:0 14px 30px rgba(31,42,48,.10); }
.h-proj-link { display:grid; gap:6px; padding:16px 16px 14px; color:inherit; text-decoration:none; border-radius:12px; min-width:0; }
.h-proj-link:focus-visible { outline:none; }
.h-proj-top { display:flex; align-items:center; gap:9px; padding-right:34px; min-width:0; }
.h-kind { width:30px; height:30px; border-radius:8px; display:grid; place-items:center; background:var(--teal-soft); color:var(--teal-deep); flex:none; }
.h-kind svg { width:17px; height:17px; }
.h-proj-name { font:600 16.5px/1.25 var(--display); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h-proj-meta { color:var(--steel); font-size:12.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h-meter { height:7px; border-radius:4px; background:var(--alu-2); overflow:hidden; display:flex; margin-top:6px; }
.h-meter i { display:block; height:100%; width:var(--w); animation:h-fill 1s var(--out) both; animation-delay:calc(450ms + var(--i, 0) * 55ms); }
.h-meter .rev { background:var(--ok); } .h-meter .lab { background:var(--lab); }
@keyframes h-fill { from { width:0; } }
.h-proj-foot { display:flex; align-items:baseline; gap:6px; font-size:12.5px; color:var(--steel); }
.h-proj-foot b { font:600 15px var(--numeric); color:var(--ink); font-variant-numeric:tabular-nums; }
.h-proj-foot .h-ago { margin-left:auto; }
.h-proj .h-err { font-size:12.5px; }
/* the signature: corner brackets that lock on */
.h-brk { position:absolute; inset:-7px; pointer-events:none; z-index:2; }
.h-brk i { position:absolute; width:16px; height:16px; border:0 solid var(--amber); opacity:0; transition:opacity .2s, transform .35s var(--spring); }
.h-brk i:nth-child(1) { top:0; left:0; border-width:3px 0 0 3px; border-top-left-radius:5px; transform:translate(-7px,-7px); }
.h-brk i:nth-child(2) { top:0; right:0; border-width:3px 3px 0 0; border-top-right-radius:5px; transform:translate(7px,-7px); }
.h-brk i:nth-child(3) { bottom:0; right:0; border-width:0 3px 3px 0; border-bottom-right-radius:5px; transform:translate(7px,7px); }
.h-brk i:nth-child(4) { bottom:0; left:0; border-width:0 0 3px 3px; border-bottom-left-radius:5px; transform:translate(-7px,7px); }
.h-proj:hover .h-brk i, .h-proj:focus-within .h-brk i, .lock > .h-brk i { opacity:1; transform:none; }
.h-panel > .h-brk { inset:-9px; } .h-panel > .h-brk i { width:30px; height:30px; border-width:inherit; }
.h-panel > .h-brk i:nth-child(1) { border-width:4px 0 0 4px; } .h-panel > .h-brk i:nth-child(2) { border-width:4px 4px 0 0; }
.h-panel > .h-brk i:nth-child(3) { border-width:0 4px 4px 0; } .h-panel > .h-brk i:nth-child(4) { border-width:0 0 4px 4px; }
.lock > .h-brk { animation:h-lock 1.6s ease-in-out; }
@keyframes h-lock { 0%, 60% { opacity:1; } 100% { opacity:0; } }
.h-empty { display:grid; justify-items:start; gap:10px; padding:26px; border:1.5px dashed #c3ccd2; border-radius:12px; color:var(--steel); background:rgba(255,255,255,.5); }
.h-empty b { font:600 17px var(--display); color:var(--ink); }
.h-filter { width:min(240px, 100%); }

/* project menu */
.h-menu { position:absolute; top:10px; right:10px; z-index:3; }
.h-menu summary { list-style:none; cursor:pointer; border:1px solid transparent; border-radius:8px; width:30px; height:30px; display:grid; place-items:center;
  font-size:18px; line-height:1; color:var(--steel); transition:background .15s, border-color .15s; }
.h-menu summary::-webkit-details-marker { display:none; }
.h-menu summary:hover, .h-menu[open] summary { border-color:var(--line); background:var(--alu); color:var(--ink); }
.h-menu-list { position:absolute; right:0; top:36px; background:var(--paper); border:1px solid var(--line); border-radius:10px;
  box-shadow:0 14px 34px rgba(31,42,48,.16); display:grid; min-width:200px; padding:5px; transform-origin:top right; animation:h-pop .18s var(--out); }
@keyframes h-pop { from { opacity:0; transform:scale(.92) translateY(-4px); } }
.h-menu-list a, .h-menu-list button { text-align:left; border:0; background:none; padding:8px 11px; border-radius:7px; color:var(--ink); text-decoration:none; font-size:13.5px; }
.h-menu-list a:hover, .h-menu-list button:hover { background:var(--teal-soft); }
.h-menu-list .danger { color:var(--bad); background:none; font-weight:400; }
.h-menu-list .danger:hover { background:#fde8e5; }

/* forms */
.h-form { display:grid; gap:14px; }
.h-panel h2, .h-panel h3 { font:600 19px/1.2 var(--display); margin:0 0 4px; }
.h-panel h3 { font-size:17px; }
.h-field { display:grid; gap:5px; } .h-field > span { font-size:12.5px; font-weight:600; }
.h-field small, .h-sub { color:var(--steel); font-size:12px; margin:0; }
.h-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; } .h-row > input[type=text] { flex:1; min-width:0; }
.h-more summary { cursor:pointer; color:var(--teal-deep); font-weight:600; font-size:13px; list-style:none; display:inline-flex; gap:6px; align-items:center; }
.h-more summary::-webkit-details-marker { display:none; }
.h-more summary::before { content:""; width:7px; height:7px; border:solid currentColor; border-width:0 2px 2px 0; transform:rotate(-45deg); transition:transform .2s; }
.h-more[open] summary::before { transform:rotate(45deg); }
.h-more[open] .h-form { animation:h-rise .3s var(--out); }
.h-seg { position:relative; display:grid; grid-template-columns:1fr 1fr; background:var(--alu); border-radius:10px; padding:3px; width:min(300px, 100%); }
.h-seg-thumb { position:absolute; top:3px; bottom:3px; left:3px; width:calc(50% - 3px); background:var(--paper); border-radius:8px;
  box-shadow:0 1px 3px rgba(31,42,48,.18); transform:translateX(calc(var(--x, 0) * 100%)); transition:transform .3s var(--spring); }
.h-seg button { position:relative; border:0; background:none; padding:6px 10px; border-radius:8px; color:var(--steel); font-weight:600; }
.h-seg button[aria-pressed=true] { color:var(--ink); }
.h-seg button:active:not(:disabled) { transform:none; }

/* jobs */
.h-job { border:1px solid var(--line); border-radius:12px; padding:12px 14px; display:grid; gap:8px; background:var(--paper); animation:h-rise .35s var(--out); }
.h-job .j-text { font-weight:600; }
.h-bar { height:8px; background:var(--alu-2); border-radius:4px; overflow:hidden; }
.h-bar i { display:block; height:100%; width:0; background:var(--teal); border-radius:4px; transition:width .4s var(--out);
  background-image:repeating-linear-gradient(-45deg, rgba(255,255,255,.22) 0 8px, transparent 8px 16px); background-size:22.6px 22.6px; animation:h-stripes .8s linear infinite; }
.h-job.done .h-bar i { animation:none; background-image:none; }
@keyframes h-stripes { to { background-position:22.6px 0; } }
.h-job pre { margin:0; max-height:140px; overflow:auto; font-size:11.5px; color:var(--steel); white-space:pre-wrap; }
.h-check { width:18px; height:18px; color:var(--ok); } .h-check path { stroke-dasharray:24; stroke-dashoffset:24; animation:h-draw .5s .1s var(--out) forwards; }
@keyframes h-draw { to { stroke-dashoffset:0; } }
.h-err { color:var(--bad); font-size:12.5px; white-space:pre-wrap; }
.h-chip { font-size:11.5px; font-weight:600; padding:2px 9px; border-radius:99px; background:var(--alu-2); color:var(--steel); vertical-align:middle; }
.h-chip.ok { background:#dcf1e4; color:#1f7a44; } .h-chip.warn { background:var(--amber-soft); color:var(--amber-ink); }
.h-check-row { display:grid; grid-template-columns:auto 1fr; gap:2px 8px; align-items:center; font-size:13px; font-weight:600; cursor:pointer; }
.h-check-row input { width:16px; height:16px; accent-color:var(--teal); margin:0; }
.h-check-row small { grid-column:2; color:var(--steel); font-weight:400; font-size:12px; }
.h-src { display:flex; gap:8px; align-items:center; min-width:0; } .h-src span { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12.5px; }

/* Teach & Transfer */
.h-rail { list-style:none; margin:4px 0 18px; padding:0; display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; position:relative; }
.h-rail::before { content:""; position:absolute; left:18px; right:calc(25% - 28px); top:17px; height:2px; background:var(--line); }
.h-rail::after { content:""; position:absolute; left:18px; right:calc(25% - 28px); top:17px; height:2px; background:var(--teal);
  transform:scaleX(0); transform-origin:left; transition:transform 1.4s var(--out) .2s; }
.h-rail.in::after { transform:scaleX(1); }
.h-rail li { position:relative; display:grid; gap:4px; align-content:start; }
.h-rail li > i { width:36px; height:36px; border-radius:50%; background:var(--paper); border:2px solid var(--line); display:grid; place-items:center;
  font:600 16px var(--numeric); font-style:normal; color:var(--steel); position:relative; z-index:1; transition:border-color .3s, color .3s, background .3s, transform .4s var(--spring); }
.h-rail.in li > i { border-color:var(--teal); color:var(--teal-deep); background:var(--teal-soft); transition-delay:calc(.25s + var(--i) * .35s); }
.h-rail.in li > i { animation:h-bump .5s var(--spring) both; animation-delay:calc(.25s + var(--i) * .35s); }
@keyframes h-bump { 50% { transform:scale(1.15); } }
.h-rail b { font:600 16px var(--display); margin-top:6px; } .h-rail span { color:var(--steel); font-size:12.5px; }
.h-steps { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:18px; }
.h-runs { display:grid; gap:10px; margin-top:18px; }
.h-runs > details { border:1px solid var(--line); border-radius:12px; padding:12px 14px; background:var(--paper); }
.h-runs summary { cursor:pointer; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.h-report { font-size:12.5px; background:var(--alu); border-radius:10px; padding:10px 12px; max-height:360px; overflow:auto; white-space:pre-wrap;
  font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; margin:10px 0; }
.h-trash summary { cursor:pointer; font:600 17px var(--display); list-style:none; display:flex; gap:10px; align-items:baseline; }
.h-trash summary::-webkit-details-marker { display:none; }
.h-trash[open] summary { margin-bottom:10px; }
.h-trash-list { display:grid; gap:8px; }

/* toasts, dialogs */
.h-toasts { position:fixed; right:108px; bottom:22px; display:grid; gap:8px; z-index:30; width:min(380px, calc(100vw - 140px)); }
.h-toast { background:var(--ink); color:#fff; padding:10px 14px; border-radius:12px; box-shadow:0 12px 28px rgba(31,42,48,.28);
  animation:h-toast-in .45s var(--spring); display:grid; gap:1px; }
.h-toast.out { animation:h-toast-out .25s ease-in forwards; }
.h-toast.error { background:var(--bad); } .h-toast.warning { background:#8a5304; } .h-toast.success { border-left:4px solid #5fd08f; }
.h-toast b { font-weight:600; } .h-toast small { opacity:.85; overflow-wrap:anywhere; }
@keyframes h-toast-in { from { opacity:0; transform:translateX(30px) scale(.96); } }
@keyframes h-toast-out { to { opacity:0; transform:translateX(20px); } }
dialog { border:1px solid var(--line); border-radius:16px; padding:0; width:min(720px, calc(100vw - 32px)); max-height:82vh; color:var(--ink);
  box-shadow:0 30px 70px rgba(31,42,48,.3); }
dialog[open] { animation:h-dlg .28s var(--spring); }
dialog::backdrop { background:rgba(31,42,48,.45); backdrop-filter:blur(2px); }
dialog[open]::backdrop { animation:h-fade .25s ease-out; }
@keyframes h-dlg { from { opacity:0; transform:translateY(14px) scale(.96); } }
@keyframes h-fade { from { opacity:0; } }
.h-confirm { width:min(460px, calc(100vw - 32px)); }
.h-confirm h2 { margin:0; padding:20px 20px 0; font:600 19px var(--display); } .h-confirm p { margin:0; padding:10px 20px 6px; color:#3d4b53; }
.b-head { display:flex; gap:6px; align-items:center; padding:12px; border-bottom:1px solid var(--line); }
.b-head input { flex:1; min-width:0; }
.b-body { overflow:auto; max-height:52vh; padding:6px; }
.b-item { display:flex; gap:9px; align-items:center; width:100%; text-align:left; border:0; border-radius:8px; padding:7px 9px; background:none; }
.b-item:hover, .b-item:focus-visible { background:var(--teal-soft); }
.b-item small { margin-left:auto; color:var(--steel); font-variant-numeric:tabular-nums; }
.b-foot { display:flex; gap:8px; align-items:center; justify-content:flex-end; padding:12px 14px; border-top:1px solid var(--line); }
.b-foot .h-sub { flex:1; }
.h-confirm .b-foot { border-top:0; padding-top:14px; }

/* responsive */
@media (max-width: 1080px) { .h-grid { grid-template-columns:minmax(0,1fr); } .h-sticky { position:static; } }
@media (max-width: 900px) {
  .h-hero { grid-template-columns:minmax(0,1fr); gap:20px; } .h-stage { max-width:620px; }
  .h-rail { grid-template-columns:repeat(2, minmax(0,1fr)); row-gap:18px; } .h-rail::before, .h-rail::after { display:none; }
  .h-steps { grid-template-columns:minmax(0,1fr); } }
@media (max-width: 600px) {
  .h-top-in { padding:8px 16px; gap:8px; } .h-device { display:none; }
  .h-wrap { padding:4px 16px 110px; gap:22px; }
  .h-panel { padding:16px; border-radius:14px; }
  .h-projects { grid-template-columns:minmax(0,1fr); }
  .h-actions button { flex:1 1 auto; }
  .h-toasts { right:16px; left:16px; width:auto; bottom:92px; } }
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior:auto; }
  *, *::before, *::after { animation-duration:.001ms !important; animation-delay:0s !important; transition-duration:.001ms !important; transition-delay:0s !important; } }
`;

const ICON = {
  video: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/></svg>`,
  images: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/></svg>`,
  check: `<svg class="h-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5 10 17l9-10"/></svg>`,
  logo: `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M3 11V3h8M21 3h8v8M29 21v8h-8M11 29H3v-8" fill="none" stroke="#f2a900" stroke-width="3" stroke-linecap="round"/><rect x="9" y="9" width="14" height="14" rx="3" fill="#0a7c78"/></svg>`,
};
const BRK = `<span class="h-brk" aria-hidden="true"><i></i><i></i><i></i><i></i></span>`;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`);
  return data;
};
const size = (n) => (n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${Math.ceil(n / 1e3)} KB`);
const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
const ago = (t) => { const s = Math.round(t - Date.now() / 1000);
  return Math.abs(s) < 60 ? "just now" : Math.abs(s) < 3600 ? rtf.format(Math.round(s / 60), "minute") : Math.abs(s) < 86400 ? rtf.format(Math.round(s / 3600), "hour") : rtf.format(Math.round(s / 86400), "day"); };
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---- hero demo: click a part, box it, track it, flag one frame, confirm, export (10 s loop) ----------
const CELLS = 24, FLAG = 15;
const DEMO = `<svg class="h-demo" viewBox="0 0 520 300" role="img" aria-label="Animation: two parts are clicked and boxed, tracked through the video, checked, confirmed and exported">
  <defs><pattern id="hgrid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="#223540" stroke-width="1"/></pattern>
    <clipPath id="hclip"><rect width="520" height="250" rx="16"/></clipPath></defs>
  <g clip-path="url(#hclip)">
    <rect width="520" height="250" fill="#16232a"/><rect width="520" height="250" fill="url(#hgrid)"/>
    <rect y="214" width="520" height="36" fill="#1d2d35"/><g class="d-rollers" fill="#2b3f48">${Array.from({ length: 14 }, (_, i) => `<circle cx="${i * 40 + 10}" cy="232" r="9"/>`).join("")}</g>
    <g class="d-part">
      <path d="M92 70h176a20 20 0 0 1 20 20v96a20 20 0 0 1-20 20H92a20 20 0 0 1-20-20V90a20 20 0 0 1 20-20z" fill="#8a9ea8" stroke="#c7d3d9" stroke-width="2"/>
      <circle cx="180" cy="138" r="30" fill="#16232a" stroke="#5c707a" stroke-width="4"/><circle cx="180" cy="138" r="16" fill="none" stroke="#3b4e57" stroke-width="3"/>
      ${[[100, 96], [260, 96], [100, 180], [260, 180]].map(([x, y]) => `<g transform="translate(${x} ${y})"><circle r="12" fill="#dfe6e9" stroke="#5c707a" stroke-width="2"/><path d="M6.5 0 3.2 5.6H-3.2L-6.5 0-3.2-5.6H3.2Z" fill="none" stroke="#5c707a" stroke-width="2"/></g>`).join("")}
      <path d="M288 138h22" stroke="#3b4e57" stroke-width="6"/>
      <rect x="308" y="112" width="62" height="52" rx="7" fill="#33444c" stroke="#aebbc1" stroke-width="2"/>
      ${[0, 1, 2, 3].map((i) => `<rect x="${318 + i * 12}" y="124" width="6" height="28" rx="2" fill="#c9a24a"/>`).join("")}
      <g class="d-box d-b1"><rect x="84" y="80" width="32" height="32" rx="2" fill="none" stroke="#f2a900" stroke-width="2.5" pathLength="100"/>
        <g class="d-tag"><rect x="84" y="62" width="34" height="16" rx="2" fill="#f2a900"/><text x="89" y="74" font-size="11" font-weight="700" fill="#1f2a30">bolt</text></g></g>
      <g class="d-box d-b2"><rect x="300" y="104" width="78" height="68" rx="2" fill="none" stroke="#5fd3c6" stroke-width="2.5" pathLength="100"/>
        <g class="d-tag"><rect x="300" y="86" width="66" height="16" rx="2" fill="#5fd3c6"/><text x="305" y="98" font-size="11" font-weight="700" fill="#10302e">connector</text></g></g>
    </g>
    <text class="d-frame" x="18" y="30" font-size="13" fill="#cfe0e6">Frame 1 of 400</text>
    <g class="d-status" opacity="0"><rect x="18" y="40" width="96" height="20" rx="10" fill="#0a7c78"/><text class="d-status-t" x="30" y="54" font-size="11" font-weight="600" fill="#fff">Tracking…</text></g>
    <g class="d-toast" opacity="0"><rect x="300" y="18" width="202" height="42" rx="10" fill="#fff"/><rect x="300" y="18" width="5" height="42" rx="2" fill="#2f9e5b"/>
      <text x="314" y="35" font-size="12" font-weight="700" fill="#1f2a30">Exported YOLO dataset</text><text x="314" y="51" font-size="11" fill="#53636c">120 images, 240 boxes</text></g>
    <g class="d-cursor"><path d="M0 0v19l5-5 4.2 9.2 3.2-1.5-4.2-9.1H15z" fill="#fff" stroke="#1f2a30" stroke-width="1.4" stroke-linejoin="round"/></g>
    <circle class="d-ripple" r="4" fill="none" stroke="#fff" stroke-width="2" opacity="0"/>
  </g>
  <g class="d-strip" transform="translate(0 262)">${Array.from({ length: CELLS }, (_, i) => `<rect x="${i * 21.75}" width="19" height="22" rx="4" fill="#cfd8dd"/>`).join("")}
    <rect class="d-flag" x="${FLAG * 21.75}" y="-5" width="19" height="4" rx="2" fill="#f2a900" opacity="0"/>
    <path class="d-head" d="M0 26l5 6h-10z" fill="#1f2a30"/></g>
</svg>`;

function heroDemo(svg) {
  const q = (s) => svg.querySelector(s), part = q(".d-part"), cursor = q(".d-cursor"), ripple = q(".d-ripple"), frame = q(".d-frame");
  const b1 = q(".d-b1"), b2 = q(".d-b2"), status = q(".d-status"), statusT = q(".d-status-t"), toast = q(".d-toast"), flag = q(".d-flag"), head = q(".d-head");
  const cells = [...svg.querySelectorAll(".d-strip > rect:not(.d-flag)")], rollers = q(".d-rollers");
  const seg = (t, a, b) => Math.max(0, Math.min(1, (t - a) / (b - a)));
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
  const lerp = (a, b, x) => a + (b - a) * x;
  const C = { grey: "#cfd8dd", mine: "#27a79b", tracked: "#4f8fe0", ok: "#2f9e5b" };
  const drawBox = (g, x) => { g.style.opacity = x > 0 ? 1 : 0; g.querySelector("rect").setAttribute("stroke-dasharray", `${x * 100} 100`); };
  let lastFrame = -1;
  function at(t) {
    // cursor path: in from the bottom right, click the bolt, click the connector, leave
    const m1 = ease(seg(t, 0.3, 1.2)), m2 = ease(seg(t, 1.9, 2.7)), out = seg(t, 3.3, 3.7);
    let cx = lerp(480, 104, m1), cy = lerp(240, 100, m1);
    if (t > 1.9) { cx = lerp(104, 336, m2); cy = lerp(100, 140, m2); }
    cursor.setAttribute("transform", `translate(${cx} ${cy})`); cursor.style.opacity = t < 0.3 ? 0 : 1 - out;
    const r1 = seg(t, 1.2, 1.6), r2 = seg(t, 2.7, 3.1), r = t < 2 ? r1 : r2, rx = t < 2 ? 100 : 339, ry = t < 2 ? 96 : 138;
    ripple.setAttribute("cx", rx); ripple.setAttribute("cy", ry); ripple.setAttribute("r", 4 + r * 22); ripple.style.opacity = r > 0 && r < 1 ? 1 - r : 0;
    drawBox(b1, ease(seg(t, 1.3, 1.9))); drawBox(b2, ease(seg(t, 2.8, 3.4)));
    b1.querySelector(".d-tag").style.opacity = seg(t, 1.75, 2.0); b2.querySelector(".d-tag").style.opacity = seg(t, 3.25, 3.5);
    // tracking: the part rides the conveyor, the timeline fills
    const tr = seg(t, 3.8, 7.2), dx = tr * 96, dy = Math.sin(tr * Math.PI * 3) * 3;
    part.setAttribute("transform", `translate(${dx} ${dy})`); rollers.setAttribute("transform", `translate(${-(tr * 96) % 40} 0)`);
    const f = t < 3.8 ? 1 : Math.round(1 + tr * 115);
    if (f !== lastFrame) { frame.textContent = `Frame ${f} of 400`; lastFrame = f; }
    const tracked = Math.floor(tr * (CELLS - 1)), confirmed = Math.floor(seg(t, 7.5, 8.5) * CELLS);
    cells.forEach((c, i) => {
      const col = i < confirmed ? C.ok : i === 0 && t > 1.9 ? C.mine : i > 0 && i <= tracked ? C.tracked : C.grey;
      if (c.getAttribute("fill") !== col) c.setAttribute("fill", col);
    });
    flag.style.opacity = t > 5.6 && confirmed <= FLAG ? 1 : 0;
    const hx = (t < 3.8 ? 0 : t < 7.5 ? tracked : Math.min(CELLS - 1, confirmed)) * 21.75 + 9.5;
    head.setAttribute("transform", `translate(${hx} 0)`);
    status.style.opacity = t > 3.8 && t < 8.6 ? 1 : 0;
    const label = t < 7.2 ? "Tracking…" : "Confirming…";
    if (statusT.textContent !== label) statusT.textContent = label;
    const ts = ease(seg(t, 8.5, 9.0));
    toast.style.opacity = t < 9.6 ? ts : 1 - seg(t, 9.6, 9.95); toast.setAttribute("transform", `translate(${(1 - ts) * 24} 0)`);
    svg.style.opacity = t > 9.7 ? 1 - seg(t, 9.7, 10) * 0.35 : 1;
  }
  if (reduceMotion) { at(9.2); return; }
  let t0 = performance.now(), raf = 0, visible = true;
  const tick = (now) => { at(((now - t0) / 1000) % 10); raf = visible && !document.hidden ? requestAnimationFrame(tick) : 0; };
  const resume = () => { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(tick); };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; resume(); }).observe(svg);
  document.addEventListener("visibilitychange", resume);
  resume();
}

export default function home(root) {
  document.head.appendChild(Object.assign(document.createElement("style"), { textContent: CSS }));
  root.innerHTML = `
  <header class="h-top"><div class="h-top-in">
    <a class="h-logo" href="/">${ICON.logo}<span>PartLabeler</span></a>
    <span class="h-device" hidden><i aria-hidden="true"></i><span></span></span>
    <span class="h-bellmount"></span></div></header>
  <main class="h-wrap">
    <section class="h-hero" aria-labelledby="h-title">
      <div>
        <h1 id="h-title"><span class="h-in" style="--d:0">Box a part once.</span><span class="h-in" style="--d:1">Track it through the whole video.</span></h1>
        <p class="h-lede h-in" style="--d:2">Click a part and SAM 3 outlines it. Tracking carries the boxes through the video, you confirm what's right,
          and the dataset exports to YOLO, COCO, CVAT, Pascal VOC or Label Studio. Everything runs on this computer.</p>
        <div class="h-actions h-in" style="--d:3">
          <button type="button" class="primary" data-go="new">New project</button>
          <button type="button" data-go="teach">Teach &amp; Transfer</button>
          <button type="button" data-go="rivet">Ask Rivet</button></div>
        <p class="h-home h-in" style="--d:4">Loading…</p>
      </div>
      <div class="h-stage h-in" style="--d:2">${DEMO}</div>
    </section>

    <div class="h-grid">
      <section aria-labelledby="h-projects-h">
        <div class="h-sec-head"><h2 id="h-projects-h">Projects</h2><span class="h-count h-pcount"></span><span class="h-sp"></span>
          <input type="text" class="h-filter" name="project-filter" autocomplete="off" aria-label="Filter projects" placeholder="Filter projects…" hidden></div>
        <div class="h-projects"><div class="h-empty">Loading…</div></div>
        <details class="h-panel h-trash" style="margin-top:22px" hidden><summary>Trash <span class="h-count h-trash-count"></span></summary>
          <p class="h-sub" style="margin-bottom:10px">Projects you moved to the trash. Restore brings them back exactly as they were.</p>
          <div class="h-trash-list"></div></details>
      </section>

      <section class="h-panel h-sticky h-newpanel" id="new" aria-labelledby="h-new-h">${BRK}
        <h2 id="h-new-h">New project</h2>
        <p class="h-sub" style="margin-bottom:14px">A video or a folder of images, and the part names to label.</p>
        <form class="h-form h-new" autocomplete="off">
          <label class="h-field"><span>Name</span><input type="text" name="name" required placeholder="e.g. gearbox_line2…"></label>
          <div class="h-field"><span>Source</span>
            <div class="h-seg" role="group" aria-label="Source type"><span class="h-seg-thumb" aria-hidden="true"></span>
              <button type="button" data-kind="video" aria-pressed="true">Video</button><button type="button" data-kind="images" aria-pressed="false">Image folder</button></div>
            <div class="h-row" style="margin-top:4px"><input type="text" name="source" required aria-label="Source path" placeholder="Path to the video file…">
              <button type="button" data-browse="source">Browse…</button></div>
            <small class="h-srchint">Every Nth frame is kept for labeling.</small></div>
          <label class="h-field h-every"><span>Keep every Nth frame</span><input type="number" name="every" min="1" value="5"></label>
          <div class="h-field"><span>Classes</span>
            <textarea name="classes" placeholder="One part name per line, e.g.&#10;bolt&#10;left_bracket&#10;right_bracket…"></textarea>
            <div class="h-row"><button type="button" data-browse="classes">Load from classes.txt / data.yaml…</button></div>
            <small>Names starting left_ / right_ are treated as mirror twins.</small></div>
          <details class="h-more"><summary>More options</summary>
            <div class="h-form" style="margin-top:12px">
              <div class="h-field"><span>Existing YOLO labels to import (optional)</span>
                <div class="h-row"><input type="text" name="labels" aria-label="YOLO labels folder" placeholder="A labels/ folder to review or finish…"><button type="button" data-browse="labels">Browse…</button></div></div>
              <label class="h-field"><span>Parent object (optional)</span><input type="text" name="parent" placeholder="What the parts sit on, e.g. engine block…">
                <small>Suggestions then search inside it. Can be changed later in the annotator.</small></label>
            </div></details>
          <div class="h-row"><button class="primary" type="submit">Create project</button><span class="h-sub h-newmsg"></span></div>
        </form>
        <div class="h-newjob" style="margin-top:12px"></div>
      </section>
    </div>

    <section class="h-panel h-teachsec" id="teach" aria-labelledby="h-teach-h">${BRK}
      <div class="h-sec-head"><h2 id="h-teach-h">Teach &amp; Transfer</h2><span class="h-chip">optional, trains on this computer</span></div>
        <p class="h-sec-lede">Already have one video labeled? Teach learns those labels and proves on held-back frames that it reproduces them.
          Transfer then labels similar videos or image folders in the same format, for you to check in the annotator.</p>
      <ol class="h-rail">
        <li style="--i:0"><i>1</i><b>Teach</b><span>Trains a detector on your labeled video, on this computer.</span></li>
        <li style="--i:1"><i>2</i><b>Prove</b><span>Scores it on frames it never trained on. Target: mAP50 of 0.90.</span></li>
        <li style="--i:2"><i>3</i><b>Transfer</b><span>Labels similar videos or folders with the same classes and file names. Or skip training with a quick preview.</span></li>
        <li style="--i:3"><i>4</i><b>Review</b><span>Opens each result in the annotator to check and export.</span></li></ol>
      <div class="h-teach-off h-err" hidden>Teach &amp; Transfer is not installed: run the installer again without -NoTeach.</div>
      <div class="h-steps">
        <form class="h-form h-teach" autocomplete="off">
          <h3>Teach from a labeled dataset</h3>
          <div class="h-field"><span>Labeled dataset folder</span>
            <div class="h-row"><input type="text" name="dataset" required aria-label="Labeled dataset folder" placeholder="Folder with images/, labels/ and classes.txt…"><button type="button" data-browse="dataset">Browse…</button></div></div>
          <label class="h-field"><span>Run name</span><input type="text" name="name" placeholder="e.g. line2_v1…"></label>
          <div class="h-row" style="align-items:end">
            <label class="h-field"><span>Model size</span><select name="size"><option value="nano">Nano (fastest)</option><option value="small" selected>Small</option><option value="medium">Medium (best)</option></select></label>
            <label class="h-field"><span>Epochs</span><input type="number" name="epochs" min="1" value="30"></label></div>
          <label class="h-field"><span>Parent object (optional)</span><input type="text" name="parent" placeholder="e.g. engine block, circuit board…"></label>
          <div class="h-row"><button class="primary" type="submit">Start teaching</button></div>
          <div class="h-teachjob"></div>
        </form>
        <form class="h-form h-transfer" autocomplete="off">
          <h3>Transfer to similar videos or image folders</h3>
          <label class="h-field"><span>Label with</span><select name="run"></select></label>
          <div class="h-quick" hidden>
            <div class="h-field"><span>Labeled dataset to match</span>
              <div class="h-row"><input type="text" name="qdataset" aria-label="Labeled dataset to match" placeholder="Folder with images/, labels/ and classes.txt…"><button type="button" data-browse="qdataset">Browse…</button></div>
              <small>No training: about 30 of its labeled images become examples, matched in each new frame. A fast preview that finds roughly 70% of parts; check every frame.</small></div>
            <label class="h-field" style="margin-top:10px"><span>Parent object (optional)</span><input type="text" name="qparent" placeholder="e.g. engine block, circuit board…"></label></div>
          <div class="h-field"><span>Label these</span><div class="h-sources" style="display:grid;gap:6px"></div>
            <div class="h-row"><button type="button" data-browse="target-video">Add video…</button><button type="button" data-browse="target-folder">Add image folder…</button></div></div>
          <label class="h-check-row"><input type="checkbox" name="tracks" checked> Check along tracks (videos)
            <small>Lists frames where a part's label flips, a part vanishes for a frame or two, or a box shows up on one frame only. Labels are never changed.</small></label>
          <div class="h-row"><button class="primary" type="submit">Start labeling</button></div>
          <div class="h-transferjob"></div>
        </form>
      </div>
      <div class="h-runs"></div>
    </section>
  </main>
  <dialog class="h-browser" aria-label="Choose a file or folder"><div class="b-head"><button type="button" data-b="up" aria-label="Up one folder" title="Up one folder">↑</button><input type="text" class="b-path" aria-label="Folder path" autocomplete="off" placeholder="Type a path and press Enter…"><button type="button" data-b="places">Places</button></div>
    <div class="b-body"></div>
    <div class="b-foot"><span class="h-sub b-hint"></span><button type="button" data-b="cancel">Cancel</button><button type="button" class="primary" data-b="choose">Use this folder</button></div></dialog>
  <dialog class="h-confirm" aria-labelledby="h-confirm-h"><h2 id="h-confirm-h">Move project to trash?</h2><p class="h-confirm-text"></p>
    <div class="b-foot"><button type="button" data-c="cancel">Cancel</button><button type="button" class="danger" data-c="ok">Move to trash</button></div></dialog>
  <div class="h-toasts" aria-live="polite"></div>`;
  const $ = (s) => root.querySelector(s);
  const newForm = $(".h-new"), teachForm = $(".h-teach"), transferForm = $(".h-transfer");
  let kind = "video", sources = [], projects = [], info = {}, seen = null;

  heroDemo($(".h-demo"));
  const top = $(".h-top");
  const onScroll = () => top.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  new IntersectionObserver((es, obs) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); } }),
    { threshold: 0.4 }).observe($(".h-rail"));

  // "lock on": scroll to a section and flash its brackets
  function lockOn(el, focus) {
    el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    el.classList.remove("lock"); void el.offsetWidth; el.classList.add("lock");
    setTimeout(() => el.classList.remove("lock"), 1700);
    if (focus) setTimeout(() => focus.focus({ preventScroll: true }), reduceMotion ? 0 : 450);
  }

  // ---- projects ----------------------------------------------------------------------------
  function projectTile(p, i) {
    const lab = p.items ? Math.max(0, p.labeled - p.confirmed) / p.items * 100 : 0, rev = p.items ? p.confirmed / p.items * 100 : 0;
    const href = `/p/${encodeURIComponent(p.name)}`, unit = p.kind === "video" ? "frames" : "images";
    return `<article class="h-proj" style="--i:${i}">${BRK}<a class="h-proj-link" href="${href}" title="${esc(p.source)}">
      <span class="h-proj-top"><span class="h-kind">${p.kind === "video" ? ICON.video : ICON.images}</span><b class="h-proj-name" translate="no">${esc(p.name)}</b></span>
      <span class="h-proj-meta">${p.kind === "video" ? "Video" : "Image folder"}, ${p.classes.length} class${p.classes.length === 1 ? "" : "es"}, ${p.items} ${unit}</span>
      ${p.problem ? `<span class="h-err">${esc(p.problem)}</span>` : ""}
      <span class="h-meter" role="img" aria-label="${p.confirmed} confirmed and ${Math.max(0, p.labeled - p.confirmed)} more labeled out of ${p.items}"><i class="rev" style="--w:${rev}%;--i:${i}"></i><i class="lab" style="--w:${lab}%;--i:${i}"></i></span>
      <span class="h-proj-foot"><b>${p.confirmed}</b> of ${p.items} confirmed<span class="h-ago">${ago(p.modified)}</span></span></a>
      <details class="h-menu"><summary aria-label="More actions for ${esc(p.name)}" title="More actions">⋯</summary><div class="h-menu-list">
        <a href="${href}">Open</a>
        <button type="button" data-folder="${esc(p.name)}">Show in folder</button>
        <button type="button" class="danger" data-trash="${esc(p.name)}">Move to trash…</button></div></details></article>`;
  }
  function renderProjects() {
    const q = $(".h-filter").value.trim().toLowerCase();
    const shown = projects.filter((p) => !q || p.name.toLowerCase().includes(q));
    $(".h-pcount").textContent = projects.length ? String(projects.length) : "";
    $(".h-filter").hidden = projects.length <= 6;
    $(".h-projects").innerHTML = shown.length ? shown.map(projectTile).join("") : projects.length
      ? `<div class="h-empty">No project name contains “${esc(q)}”.</div>`
      : `<div class="h-empty"><b>No projects yet</b><span>A project is one video or image folder plus the part names you want boxed.</span>
          <button type="button" class="primary" data-go="new">Create your first project</button></div>`;
  }
  async function loadProjects() {
    const list = await api("/api/projects").catch((e) => ({ error: e.message }));
    if (list.error) { $(".h-projects").innerHTML = `<div class="h-empty"><span class="h-err">${esc(list.error)}</span></div>`; return; }
    projects = list; renderProjects(); loadTrash();
  }
  $(".h-filter").addEventListener("input", renderProjects);

  async function loadTrash() {
    const list = await api("/api/trash").catch(() => []);
    $(".h-trash").hidden = !list.length;
    $(".h-trash-count").textContent = list.length ? String(list.length) : "";
    $(".h-trash-list").innerHTML = list.map((t) => `<div class="h-src"><span translate="no">${esc(t.name)}</span><span class="h-sub">moved ${ago(t.trashed)}</span>
      <button type="button" data-restore="${esc(t.entry)}">Restore</button></div>`).join("");
  }

  function toast(n) {
    const t = document.createElement("div");
    t.className = `h-toast ${n.level || "info"}`; t.setAttribute("role", n.level === "error" ? "alert" : "status");
    t.innerHTML = "<b></b><small></small>"; t.querySelector("b").textContent = n.title; t.querySelector("small").textContent = n.detail || "";
    $(".h-toasts").appendChild(t);
    while ($(".h-toasts").children.length > 4) $(".h-toasts").firstChild.remove();
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 260); }, n.level === "error" ? 10000 : 5000);
  }

  const projectPath = (name) => (info.home ? `${info.home}${info.home.includes("\\") ? "\\" : "/"}${name}` : name);
  async function runAction(a) {
    try {
      if (a.type === "open") location.href = `/p/${encodeURIComponent(a.project)}`;
      else if (a.type === "goto") location.href = `/p/${encodeURIComponent(a.project)}?item=${a.item}`;
      else if (a.type === "folder") await api("/api/open-folder", { path: a.path });
      else if (a.type === "restore") { await api(`/api/trash/${encodeURIComponent(a.entry)}/restore`, {}); loadProjects(); pollNotes(); }
    } catch (err) { toast({ level: "error", title: "That did not work", detail: err.message }); }
  }
  $(".h-projects").addEventListener("click", async (e) => {
    const f = e.target.closest("[data-folder]"), t = e.target.closest("[data-trash]");
    if (f) { e.target.closest("details").open = false; runAction({ type: "folder", path: projectPath(f.dataset.folder) }); }
    if (t) { e.target.closest("details").open = false; confirmTrash(t.dataset.trash); }
  });
  document.addEventListener("click", (e) => {                      // one open ⋯ menu at a time; outside click closes it
    root.querySelectorAll(".h-menu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; });
  });
  $(".h-trash-list").addEventListener("click", (e) => { const r = e.target.closest("[data-restore]"); if (r) runAction({ type: "restore", entry: r.dataset.restore }); });
  function confirmTrash(name) {
    const dlg = $(".h-confirm");
    $(".h-confirm-text").textContent = `"${name}" moves to the Trash with its frames and labels. You can restore it any time from the Trash list below the projects.`;
    dlg.returnValue = ""; dlg.showModal(); dlg.querySelector('[data-c="cancel"]').focus();
    dlg.onclick = async (e) => {
      const c = e.target.closest("[data-c]")?.dataset.c; if (!c) return;
      dlg.close();
      if (c === "ok") {
        try { await api(`/api/projects/${encodeURIComponent(name)}/trash`, {}); loadProjects(); pollNotes(); }
        catch (err) { toast({ level: "error", title: `Could not move ${name} to the trash`, detail: err.message }); }
      }
    };
  }

  // ---- notifications (shared history with every open project) ------------------------------
  const center = notificationCenter({
    mount: $(".h-bellmount"),
    onRead: () => api("/api/notifications/read", {}).catch(() => {}),
    onClear: () => api("/api/notifications/clear", {}).then(pollNotes).catch(() => {}),
    onAction: (a) => runAction(a),
  });
  async function pollNotes() {
    const d = await api("/api/notifications").catch(() => null);
    if (!d) return;
    if (seen) d.items.filter((n) => !seen.has(`${n.id}:${n.count}`)).slice(0, 3).reverse().forEach(toast);
    seen = new Set(d.items.map((n) => `${n.id}:${n.count}`));
    center.set(d.items, d.unread);
  }
  pollNotes();
  setInterval(() => { if (!document.hidden) pollNotes(); }, 3000);

  // ---- Rivet, the helper ---------------------------------------------------------------------
  const helper = assistant({
    mount: document.body, variant: "fab", transport: webAssistant("/"),
    context: () => ({ page: "home", projects: projects.length, teach: info.teach }),
    actions: {
      "new-project": { label: "Go to New project", run: () => lockOn($(".h-newpanel"), newForm.name) },
      teach: { label: "Show Teach & Transfer", run: () => lockOn($(".h-teachsec"), teachForm.dataset) },
      projects: { label: "Show projects", run: () => $("#h-projects-h").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" }) },
      notifications: { label: "Open notifications", run: () => center.open() },
      trash: { label: "Open Trash", run: () => { const t = $(".h-trash"); if (!t.hidden) { t.open = true; t.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" }); } } },
    },
    starters: ["How do I create a project?", "Explain how Teach & Transfer works", "How can I export a dataset?", { q: "What makes PartLabeler different?", local: "pros" }],
    firstTip: 20000,
  });
  root.addEventListener("click", (e) => {
    const go = e.target.closest("[data-go]")?.dataset.go; if (!go) return;
    if (go === "new") lockOn($(".h-newpanel"), newForm.name);
    else if (go === "teach") lockOn($(".h-teachsec"), teachForm.dataset);
    else if (go === "rivet") helper.open();
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input,textarea,select") || document.querySelector("dialog[open]") || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "n") center.toggle();
    else if (e.key === "h" && !helper.isOpen) { e.preventDefault(); helper.open(); }
  });

  // ---- new project -----------------------------------------------------------------------------
  function setKind(k) {
    kind = k;
    newForm.querySelectorAll("[data-kind]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.kind === k));
    $(".h-seg").style.setProperty("--x", k === "video" ? 0 : 1);
    newForm.source.placeholder = k === "video" ? "Path to the video file…" : "Path to the image folder…";
    $(".h-srchint").textContent = k === "video" ? "Every Nth frame is kept for labeling." : "All images in the folder and its sub-folders.";
    $(".h-every").hidden = k !== "video";
  }
  newForm.addEventListener("click", (e) => { const k = e.target.closest("[data-kind]")?.dataset.kind; if (k) setKind(k); });
  newForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = newForm, msg = $(".h-newmsg"), btn = f.querySelector("button[type=submit]");
    msg.textContent = "";
    const body = { name: f.name.value.trim(), [kind]: f.source.value.trim(), every: f.every.value, classes: f.classes.value,
                   labels: f.labels.value.trim(), parent: f.parent.value.trim() };
    btn.disabled = true;
    try {
      const { job } = await api("/api/projects", body);
      follow(job, $(".h-newjob"), (res) => { setTimeout(() => { location.href = `/p/${encodeURIComponent(res.name)}`; }, reduceMotion ? 0 : 700); }, () => { btn.disabled = false; });
    } catch (err) { msg.innerHTML = `<span class="h-err">${esc(err.message)}</span>`; btn.disabled = false; }
  });

  // ---- jobs --------------------------------------------------------------------------------
  function follow(jid, el, onDone, onEnd) {
    el.innerHTML = `<div class="h-job" role="status"><div class="h-row"><b class="j-text">Starting…</b><span style="flex:1"></span><button type="button" class="j-stop">Stop</button></div>
      <span class="h-bar"><i></i></span><pre class="j-log"></pre></div>`;
    el.querySelector(".j-stop").onclick = () => api(`/api/jobs/${jid}/stop`, {});
    const tick = async () => {
      const j = await api(`/api/jobs/${jid}`).catch(() => null);
      if (!j) return setTimeout(tick, 1500);
      el.querySelector(".j-text").textContent = j.text;
      el.querySelector(".h-bar i").style.width = j.total ? `${Math.min(100, j.done / j.total * 100)}%` : "4%";
      el.querySelector(".j-log").textContent = j.log.slice(-12).join("\n");
      if (!j.finished) return setTimeout(tick, 800);
      el.querySelector(".j-stop").remove();
      const box = el.querySelector(".h-job"); box.classList.add("done");
      if (j.error) box.insertAdjacentHTML("beforeend", `<div class="h-err">${esc(j.error)}</div>`);
      else { el.querySelector(".h-bar i").style.width = "100%"; box.querySelector(".h-row").insertAdjacentHTML("afterbegin", ICON.check); onDone?.(j.result); }
      onEnd?.();
      loadProjects(); loadRuns();
    };
    tick();
  }

  // ---- teach & transfer ------------------------------------------------------------------------
  let runs = [];
  async function loadRuns() {
    runs = await api("/api/runs").catch(() => []);
    const sel = transferForm.run, keep = sel.value;
    const ready = runs.filter((r) => r.ready && r.mode !== "quick");
    sel.innerHTML = (ready.length ? `<optgroup label="Teach runs">${ready.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("")}</optgroup>` : "")
      + `<option value="${QUICK}">No training: match a labeled dataset (quick preview)</option>`;
    if (keep) sel.value = keep;
    showQuick();
    $(".h-runs").innerHTML = runs.map((r) => {
      const k = r.report?.knowledge || r.report?.held_out || {};
      const pass = r.report?.passed;
      const outs = r.outputs.map((o) => `<div class="h-src"><span title="${esc(o.path)}">${esc(o.name)}: ${o.summary ? `${o.summary.frames ?? "?"} frames, ${o.summary.boxes ?? "?"} boxes${o.summary.frames_to_check?.length ? `, ${o.summary.frames_to_check.length} to check` : ""}` : "in progress"}</span>
        <button type="button" data-review="${esc(o.path)}">Review in annotator</button></div>`).join("");
      return `<details><summary><b>${esc(r.name)}</b> ${r.mode === "quick" ? `<span class="h-chip">quick, no training</span>` : pass === true ? `<span class="h-chip ok">targets met</span>` : pass === false ? `<span class="h-chip warn">below target</span>` : r.ready ? "" : `<span class="h-chip">not finished</span>`}
        ${k.mAP50 != null ? `<span class="h-sub">held-out mAP50 ${k.mAP50}</span>` : ""}</summary>
        ${r.report_md ? `<div class="h-report">${esc(r.report_md)}</div>` : ""}${outs ? `<h3 style="font:600 15px var(--display);margin:10px 0 6px">Labeled outputs</h3><div style="display:grid;gap:6px">${outs}</div>` : ""}</details>`;
    }).join("");
  }
  teachForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = teachForm;
    try {
      const { job } = await api("/api/teach", { dataset: f.dataset.value.trim(), name: f.name.value.trim(), size: f.size.value, epochs: f.epochs.value, parent: f.parent.value.trim() });
      follow(job, $(".h-teachjob"));
    } catch (err) { $(".h-teachjob").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; }
  });
  function renderSources() {
    $(".h-sources").innerHTML = sources.length ? sources.map((s, i) => `<div class="h-src"><span title="${esc(s)}">${esc(s)}</span><button type="button" data-rm="${i}" aria-label="Remove ${esc(s)}" title="Remove">×</button></div>`).join("")
      : `<span class="h-sub">Nothing added yet.</span>`;
  }
  transferForm.addEventListener("click", (e) => { const i = e.target.closest("[data-rm]")?.dataset.rm; if (i != null) { sources.splice(+i, 1); renderSources(); } });
  const QUICK = "__quick__";
  const showQuick = () => { $(".h-quick").hidden = transferForm.run.value !== QUICK; };
  transferForm.run.addEventListener("change", showQuick);
  transferForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const f = transferForm, quick = f.run.value === QUICK;
      const { job } = quick
        ? await api("/api/quick", { dataset: f.qdataset.value.trim(), parent: f.qparent.value.trim(), sources, tracks: f.tracks.checked })
        : await api("/api/transfer", { run: f.run.value, sources, tracks: f.tracks.checked });
      follow(job, $(".h-transferjob"));
    } catch (err) { $(".h-transferjob").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; }
  });
  $(".h-runs").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-review]"); if (!btn) return;
    btn.disabled = true;
    try {
      const r = await api("/api/review", { output: btn.dataset.review });
      if (r.job) follow(r.job, btn.parentElement.appendChild(document.createElement("div")), (res) => { location.href = `/p/${encodeURIComponent(res.name)}`; });
      else location.href = `/p/${encodeURIComponent(r.name)}`;
    } catch (err) { btn.insertAdjacentHTML("afterend", `<span class="h-err">${esc(err.message)}</span>`); btn.disabled = false; }
  });

  // ---- file browser ----------------------------------------------------------------------------
  const dlg = $(".h-browser");
  let pick = null;                          // {kind, folder: bool, done(path)}
  let cur = { path: "", parent: null };
  async function show(path) {
    try { cur = await api(`/api/browse?path=${encodeURIComponent(path)}&kind=${pick.kind}`); }
    catch (err) { $(".b-hint").innerHTML = `<span class="h-err">${esc(err.message)}</span>`; return; }
    $(".b-path").value = cur.path;
    const dirs = cur.dirs.map((d) => `<button type="button" class="b-item" data-dir="${esc(cur.path ? joinPath(cur.path, d) : d)}">📁 ${esc(d)}</button>`);
    const files = pick.folder ? [] : cur.files.map((f) => `<button type="button" class="b-item" data-file="${esc(joinPath(cur.path, f.name))}">📄 ${esc(f.name)}<small>${size(f.size)}</small></button>`);
    $(".b-body").innerHTML = dirs.concat(files).join("") || `<div class="h-empty">Nothing here${pick.folder ? "" : " that fits"}.</div>`;
    const choose = dlg.querySelector('[data-b="choose"]');
    choose.hidden = !pick.folder || !cur.path;
    $(".b-hint").textContent = pick.folder ? (cur.images_here != null ? `${cur.images_here} images directly in this folder` : "Open the folder, then choose it") : "Pick a file";
  }
  const joinPath = (a, b) => (a.endsWith("/") || a.endsWith("\\") ? a + b : a + (a.includes("\\") ? "\\" : "/") + b);
  function browseFor(which) {
    const spec = {
      source: kind === "video" ? { kind: "video", folder: false, done: (p) => (newForm.source.value = p) } : { kind: "images", folder: true, done: (p) => (newForm.source.value = p) },
      classes: { kind: "classes", folder: false, done: async (p) => { try { newForm.classes.value = (await api(`/api/classes?path=${encodeURIComponent(p)}`)).classes.join("\n"); } catch (err) { $(".h-newmsg").innerHTML = `<span class="h-err">${esc(err.message)}</span>`; } } },
      labels: { kind: "dir", folder: true, done: (p) => (newForm.labels.value = p) },
      dataset: { kind: "dir", folder: true, done: (p) => (teachForm.dataset.value = p) },
      qdataset: { kind: "dir", folder: true, done: (p) => (transferForm.qdataset.value = p) },
      "target-video": { kind: "video", folder: false, done: (p) => { sources.push(p); renderSources(); } },
      "target-folder": { kind: "images", folder: true, done: (p) => { sources.push(p); renderSources(); } },
    }[which];
    pick = spec;
    let last = ""; try { last = localStorage.getItem("pl-last-dir") || ""; } catch {}
    const start = { source: newForm.source.value, labels: newForm.labels.value, dataset: teachForm.dataset.value, qdataset: transferForm.qdataset.value }[which] || last;
    dlg.showModal();
    show(start && !/\.[a-z0-9]{2,4}$/i.test(start) ? start : start.replace(/[\\/][^\\/]*$/, "")).catch(() => show(""));
  }
  root.addEventListener("click", (e) => { const w = e.target.closest("[data-browse]")?.dataset.browse; if (w) browseFor(w); });
  dlg.addEventListener("click", (e) => {
    const t = e.target.closest("[data-dir],[data-file],[data-b]"); if (!t) return;
    if (t.dataset.dir) show(t.dataset.dir);
    else if (t.dataset.file) { finish(t.dataset.file); }
    else if (t.dataset.b === "up") show(cur.parent ?? "");
    else if (t.dataset.b === "places") show("");
    else if (t.dataset.b === "cancel") dlg.close();
    else if (t.dataset.b === "choose") finish(cur.path);
  });
  $(".b-path").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); show($(".b-path").value.trim()); } });
  function finish(path) {
    try { localStorage.setItem("pl-last-dir", pick.folder ? path.replace(/[\\/][^\\/]*$/, "") || path : path.replace(/[\\/][^\\/]*$/, "")); } catch {}
    dlg.close(); pick.done(path);
  }

  // ---- start ---------------------------------------------------------------------------------
  api("/api/info").then((i) => {
    info = i;
    $(".h-home").textContent = `Projects are saved in ${i.home}`;
    const dev = $(".h-device"); dev.hidden = false; dev.title = i.device;
    dev.querySelector("span").textContent = i.device; dev.classList.toggle("cpu", !/cuda|gpu|nvidia|rtx|gtx|tesla/i.test(i.device));
    $(".h-teach-off").hidden = i.teach;
    teachForm.querySelector("button[type=submit]").disabled = !i.teach;
    if (!i.teach) { transferForm.run.value = QUICK; showQuick(); }                // quick transfer needs no Teach extras
  }).catch(() => ($(".h-home").textContent = "Can't reach the PartLabeler server. Restart the app (run_windows.bat) and reload this page."));
  setKind("video"); renderSources(); loadProjects(); loadRuns();
}
