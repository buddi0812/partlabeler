// PartLabeler annotator. One ES module, two hosts: the local FastAPI page and a notebook widget
// (anywidget). `model` only needs send(msg) and on("msg:custom", fn). Protocol: engine/api.py.

const CSS = `
.pl { --bg:#f3f5f7; --panel:#fff; --ink:#15202b; --muted:#5b6874; --line:#d7dde3; --accent:#0a7c78;
  --st0:#e4e9ed; --st1:#b8a8e6; --st2:#6aa6f2; --st3:#27a79b; --st4:#2f9e5b; --flag:#e09a2c; --bad:#c2412f;
  font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg);
  display:grid; grid-template-rows:auto 1fr auto; gap:8px; padding:8px; outline:none; position:relative; }
.pl * { box-sizing:border-box; }
.pl-top { display:flex; flex-wrap:wrap; align-items:center; gap:6px 10px; }
.pl-title { font-weight:600; font-size:14px; margin-right:6px; }
.pl-where { font-variant-numeric:tabular-nums; color:var(--muted); }
.pl-chip { font-size:11px; font-weight:600; padding:2px 8px; border-radius:99px; }
.pl-chip.rev { background:#dcf1e4; color:var(--st4); } .pl-chip.flag { background:#fbecd3; color:#9a5c05; }
.pl-sp { flex:1; }
.pl button { font:inherit; border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:6px;
  padding:4px 10px; cursor:pointer; white-space:nowrap; }
.pl button:hover { border-color:var(--accent); } .pl button:disabled { opacity:.45; cursor:default; }
.pl button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.pl button:focus-visible, .pl select:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.pl select, .pl input[type=number] { font:inherit; border:1px solid var(--line); border-radius:6px; padding:3px 4px; background:var(--panel); }
.pl input[type=number] { width:56px; }
.pl-main { display:grid; grid-template-columns:minmax(0,1fr) 250px; gap:8px; min-height:0; }
.pl-stage { position:relative; background:#0d1217; border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center; min-height:300px; }
.pl-stage canvas { max-width:100%; max-height:74vh; display:block; cursor:crosshair; }
.pl-side { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; display:flex; flex-direction:column; gap:12px; overflow:auto; max-height:80vh; }
.pl-side h4 { margin:0 0 4px; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.pl-tools { display:flex; gap:4px; } .pl-tools button.on { background:var(--ink); color:#fff; border-color:var(--ink); }
.pl-list { display:flex; flex-direction:column; gap:2px; }
.pl-row { display:grid; grid-template-columns:12px 16px minmax(0,1fr) auto; gap:6px; align-items:center; padding:3px 4px; border-radius:5px; cursor:pointer; }
.pl-row:hover { background:var(--bg); } .pl-row.on { background:#e1f0ef; outline:1px solid var(--accent); }
.pl-sw { width:12px; height:12px; border-radius:3px; }
.pl-key { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.pl-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pl-src { font-size:11px; color:var(--muted); }
.pl-x { border:none !important; padding:0 4px !important; color:var(--muted); background:none !important; }
.pl-help { color:var(--muted); font-size:12px; display:grid; grid-template-columns:auto 1fr; gap:2px 8px; }
.pl-help b { color:var(--ink); font-weight:600; }
.pl-bottom { display:grid; gap:4px; }
.pl-strip { width:100%; height:28px; display:block; cursor:pointer; border-radius:4px; background:var(--st0); }
.pl-foot { display:flex; gap:12px; align-items:center; color:var(--muted); font-variant-numeric:tabular-nums; min-height:18px; }
.pl-bar { flex:0 0 160px; height:6px; background:var(--st0); border-radius:3px; overflow:hidden; }
.pl-bar i { display:block; height:100%; background:var(--accent); width:0; }
.pl-legend { display:flex; gap:10px; flex-wrap:wrap; } .pl-legend span { display:inline-flex; gap:4px; align-items:center; }
.pl-home { text-decoration:none; color:var(--ink); font-size:16px; padding:0 4px; }
.pl-group { display:inline-flex; gap:4px; align-items:center; }
.pl-wrap { flex-wrap:wrap; align-items:center; }
.pl-settings summary { cursor:pointer; list-style:none; } .pl-settings summary h4 { display:inline; }
.pl-settings { display:flex; flex-direction:column; gap:6px; }
.pl-settings[open] > *:not(summary) { display:block; margin-top:6px; }
.pl-field { display:flex; flex-direction:column; gap:3px; font-size:12px; }
.pl input[type=text] { font:inherit; border:1px solid var(--line); border-radius:6px; padding:4px 6px; background:var(--panel); }
.pl-toasts { position:absolute; right:12px; bottom:64px; display:flex; flex-direction:column; gap:6px; z-index:5; max-width:380px; }
.pl-toast { background:var(--ink); color:#fff; padding:8px 12px; border-radius:6px; box-shadow:0 4px 14px #0003; }
.pl-toast.err { background:var(--bad); }
@media (max-width: 820px) { .pl-main { grid-template-columns:minmax(0,1fr); } .pl-side { max-height:none; } }
`;

const STATUS = ["No boxes", "Suggestions", "Tracked / imported", "Has your boxes", "Reviewed"];
const colorOf = (i) => `hsl(${(i * 137.508) % 360} 78% 52%)`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function render({ model, el }) {
  if (!document.getElementById("pl-css")) {
    document.head.appendChild(Object.assign(document.createElement("style"), { id: "pl-css", textContent: CSS }));
  }
  el.className = "pl";
  el.tabIndex = 0;
  el.innerHTML = `
    <div class="pl-top">
      <a class="pl-home" href="/" hidden title="All projects">☰</a>
      <span class="pl-title">PartLabeler</span><span class="pl-where"></span><span class="pl-chips"></span>
      <span class="pl-sp"></span>
      <button data-a="prev" title="Previous frame (←)">◀</button><button data-a="next" title="Next frame (→)">▶</button>
      <button data-a="nextTodo" title="Next flagged or unconfirmed frame (Shift+→)">Next to check</button>
      <span class="pl-group pl-video">
        <button data-a="trackBack" title="Track the boxes on this frame backwards (R; Shift+R to the start)">◀ Track</button>
        <input type="number" class="pl-n" min="1" value="20" title="Frames to track">
        <button class="primary" data-a="track" title="Track the boxes on this frame ahead (T; Shift+T to the end)">Track ▶</button>
      </span>
      <button data-a="stop" title="Stop the running job (X)">Stop</button>
      <button data-a="undo" title="Undo (Ctrl+Z)">Undo</button>
      <button class="primary" data-a="review" title="Mark this frame checked and go on (Enter)">Confirm ✓</button>
    </div>
    <div class="pl-main">
      <div class="pl-stage"><canvas class="pl-cv"></canvas></div>
      <aside class="pl-side">
        <div><h4>Tool</h4><div class="pl-tools">
          <button data-tool="click" title="Click a part to outline and box it (C)">Click</button>
          <button data-tool="box" title="Drag to draw a box (B)">Box</button></div></div>
        <div><h4>Class</h4><div class="pl-list pl-classes"></div></div>
        <div><h4>Boxes on this frame</h4><div class="pl-list pl-boxes"></div></div>
        <div><h4>Find parts</h4><div class="pl-tools pl-wrap">
          <button data-b="suggest" title="Suggest boxes that look like parts you labeled elsewhere (S)">Suggest</button>
          <button data-b="findAll" title="More parts like the selected box on this frame (F)">Find similar</button>
          <button data-b="accept" title="Keep all suggestions on this frame (Y)">Accept all</button></div></div>
        <div><h4>Export dataset</h4><div class="pl-tools pl-wrap">
          <select class="pl-fmt" title="Export format"></select>
          <label title="Only frames you confirmed"><input type="checkbox" class="pl-revonly"> confirmed only</label>
          <button data-b="export">Export</button></div></div>
        <details class="pl-settings"><summary><h4>Settings</h4></summary>
          <label class="pl-field">Parent object <input type="text" class="pl-parent" placeholder="optional, e.g. engine block"></label>
          <span class="pl-src">What the parts sit on. Suggestions then search inside it, which helps when the camera or distance changes. Leave empty for the whole image.</span>
          <button data-b="saveSettings">Save</button>
          <span class="pl-src pl-device"></span></details>
        <div><h4>Keys</h4><div class="pl-help">
          <b>Click</b><span>new part (current class)</span><b>M</b><span>next larger outline for it</span>
          <b>Shift / Alt+click</b><span>grow / shrink selected</span>
          <b>Ctrl+click</b><span>select a box</span><b>Drag</b><span>(Box tool) draw; redraws the selected box</span>
          <b>1–9, 0</b><span>class (also re-labels the selected box)</span><b>Del</b><span>delete selected box</span>
          <b>← →</b><span>previous / next frame</span><b>Shift+→</b><span>next frame to check</span>
          <b>Enter</b><span>confirm frame, go on</span><b>Ctrl+Z</b><span>undo</span>
          <b>T / R</b><span>track ahead / back (Shift: to the end / start)</span><b>X</b><span>stop</span>
          <b>S · F · Y</b><span>suggest · find similar · accept all</span></div></div>
      </aside>
    </div>
    <div class="pl-bottom">
      <canvas class="pl-strip" title="Every frame; click to jump"></canvas>
      <div class="pl-foot"><span class="pl-counts"></span><span class="pl-bar"><i></i></span><span class="pl-prog"></span>
        <span class="pl-sp"></span><span class="pl-legend"></span></div>
    </div>
    <div class="pl-toasts"></div>`;
  const $ = (s) => el.querySelector(s);
  const cv = $(".pl-cv"), ctx = cv.getContext("2d"), strip = $(".pl-strip"), sctx = strip.getContext("2d");
  const S = { project: null, statuses: [], flags: [], item: 0, target: 0, data: null, img: null, mask: null,
              maskItem: -1, cls: 0, tool: "click", sel: null, drag: null, busy: false };
  const send = (m) => model.send(m);

  // ---- drawing ---------------------------------------------------------------------------
  function draw() {
    if (!S.img) return;
    ctx.drawImage(S.img, 0, 0);
    if (S.mask && S.maskItem === S.item) ctx.drawImage(S.mask, 0, 0);
    const lw = Math.max(2, cv.width / 700), fs = Math.max(12, cv.width / 95);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    for (const b of S.data?.boxes || []) {
      const [x1, y1, x2, y2] = b.box, col = colorOf(b.cls), on = b.obj === S.sel;
      ctx.setLineDash(b.source === "tracked" ? [lw * 4, lw * 2] : b.source === "suggested" ? [lw, lw * 1.5] : []);
      if (on) { ctx.lineWidth = lw * 3; ctx.strokeStyle = "#fff"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); }
      ctx.lineWidth = on ? lw * 1.8 : lw; ctx.strokeStyle = col; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
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
    $(".pl-classes").innerHTML = cls.map((c, i) => `<div class="pl-row ${i === S.cls ? "on" : ""}" data-cls="${i}">
      <span class="pl-sw" style="background:${colorOf(i)}"></span><span class="pl-key">${i < 10 ? (i + 1) % 10 : ""}</span>
      <span class="pl-name">${esc(c)}</span><span></span></div>`).join("");
    const boxes = S.data?.boxes || [];
    $(".pl-boxes").innerHTML = boxes.length ? boxes.map((b) => `<div class="pl-row ${b.obj === S.sel ? "on" : ""}" data-obj="${b.obj}">
      <span class="pl-sw" style="background:${colorOf(b.cls)}"></span><span class="pl-key">#${b.obj}</span>
      <span class="pl-name">${esc(cls[b.cls] ?? b.cls)} <span class="pl-src">${b.source}${b.score != null && b.source !== "manual" ? " " + b.score.toFixed(2) : ""}</span></span>
      <button class="pl-x" data-del="${b.obj}" title="Delete">×</button></div>`).join("") : `<span class="pl-src">None yet. Click a part.</span>`;
    el.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === S.tool));
  }

  function renderTop() {
    if (!S.project || !S.data) return;
    $(".pl-title").textContent = S.project.name;
    $(".pl-where").textContent = `${S.project.kind === "video" ? "Frame" : "Image"} ${S.item + 1} / ${S.project.count} · ${S.data.name}`;
    const chips = [];
    if (S.data.reviewed) chips.push(`<span class="pl-chip rev">Confirmed</span>`);
    if (S.flags.includes(S.item)) chips.push(`<span class="pl-chip flag">Check: a box changed size, jumped or was lost</span>`);
    $(".pl-chips").innerHTML = chips.join(" ");
    const rev = S.statuses.filter((s) => s === 4).length, lab = S.statuses.filter((s) => s >= 2).length;
    $(".pl-counts").textContent = `${lab} labeled · ${rev} confirmed · ${S.flags.length} to check`;
    const busy = S.busy;
    el.querySelectorAll('[data-a="track"],[data-a="trackBack"],[data-b="suggest"],[data-b="findAll"],[data-b="export"]').forEach((b) => (b.disabled = busy));
    $('[data-a="stop"]').disabled = !busy;
    $('[data-a="undo"]').disabled = busy || !S.undo;
    $(".pl-video").hidden = S.project.kind !== "video";
    $(".pl-device").textContent = [S.project.device, S.gpu && `GPU memory in use: ${S.gpu}`].filter(Boolean).join(" · ");
  }

  const FORMAT_NAMES = { yolo: "YOLO", coco: "COCO", cvat: "CVAT", voc: "Pascal VOC", labelstudio: "Label Studio" };
  function renderProject() {
    const fmt = $(".pl-fmt"), keep = fmt.value;
    fmt.innerHTML = (S.project.formats || ["yolo", "coco"]).map((f) => `<option value="${f}">${FORMAT_NAMES[f] || f}</option>`).join("");
    if (keep) fmt.value = keep;
    $(".pl-parent").value = S.project.parent || "";
    $(".pl-home").hidden = !model.homeUrl;
    if (model.homeUrl) $(".pl-home").href = model.homeUrl;
  }

  function legend() {
    $(".pl-legend").innerHTML = STATUS.map((t, i) => `<span><i class="pl-sw" style="background:var(--st${i})"></i>${t}</span>`).join("")
      + `<span><i class="pl-sw" style="background:var(--flag)"></i>To check</span>`;
  }

  function toast(text, err = false) {
    const t = Object.assign(document.createElement("div"), { className: "pl-toast" + (err ? " err" : ""), textContent: text });
    $(".pl-toasts").appendChild(t); setTimeout(() => t.remove(), err ? 9000 : 4500);
  }

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
    if (todo != null) goto(todo); else toast("Nothing left to check after this frame");
  }
  function setClass(i) {
    if (!S.project || i < 0 || i >= S.project.classes.length) return;
    S.cls = i;
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
    saveSettings: () => send({ type: "settings", parent: $(".pl-parent").value }),
  };
  el.querySelector(".pl-top").addEventListener("click", (e) => { const a = e.target.closest("[data-a]")?.dataset.a; if (a) actions[a](); });
  el.querySelector(".pl-side").addEventListener("click", (e) => {
    const b = e.target.closest("[data-b]")?.dataset.b; if (b) { actions[b](); return; }
    const t = e.target.closest("[data-tool],[data-cls],[data-obj],[data-del]"); if (!t) return;
    if (t.dataset.del) { send({ type: "delete", item: S.item, obj: +t.dataset.del }); if (S.sel === +t.dataset.del) S.sel = null; }
    else if (t.dataset.tool) { S.tool = t.dataset.tool; renderSide(); }
    else if (t.dataset.cls) setClass(+t.dataset.cls);
    else if (t.dataset.obj) { S.sel = +t.dataset.obj; renderSide(); draw(); }
  });
  strip.addEventListener("click", (e) => { const r = strip.getBoundingClientRect(); goto(Math.floor((e.clientX - r.left) / r.width * S.statuses.length)); });

  // ---- mouse on the image ----------------------------------------------------------------
  const toImage = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
  function boxAt(x, y) {
    const hits = (S.data?.boxes || []).filter((b) => x >= b.box[0] && x <= b.box[2] && y >= b.box[1] && y <= b.box[3]);
    hits.sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
    return hits[0]?.obj ?? null;
  }
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("mousedown", (e) => { if (!S.img) return; const [x, y] = toImage(e); S.drag = { x0: x, y0: y, x1: x, y1: y, moved: false, e }; el.focus(); });
  cv.addEventListener("mousemove", (e) => {
    if (!S.drag) return; const [x, y] = toImage(e); S.drag.x1 = x; S.drag.y1 = y;
    const r = cv.getBoundingClientRect(), px = cv.width / r.width;
    S.drag.moved ||= Math.hypot(x - S.drag.x0, y - S.drag.y0) > 5 * px; if (S.drag.moved) draw();
  });
  window.addEventListener("mouseup", (e) => {
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
    if ((e.altKey || e.button === 2) && S.sel == null) { toast("Select a box first (Ctrl+click), then Alt+click to shrink it"); return; }
    send({ type: "click", item: S.item, x: d.x0, y: d.y0, cls: S.cls, positive: !(e.altKey || e.button === 2),
           obj: refine ? S.sel : undefined });
  });

  // ---- keys --------------------------------------------------------------------------------
  el.addEventListener("keydown", (e) => {
    if (e.target.matches("input,select,textarea")) return;
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
    else if (k === "c" || k === "b") { S.tool = k === "c" ? "click" : "box"; renderSide(); }
    else if (k === "m") { if (S.sel != null) send({ type: "cycle", item: S.item, obj: S.sel }); }
    else if (k === "[" || k === "]") setClass((S.cls + (k === "]" ? 1 : -1) + S.project.classes.length) % S.project.classes.length);
    else return;
    e.preventDefault();
  });

  // ---- messages in -------------------------------------------------------------------------
  model.on("msg:custom", (m) => {
    if (m.type === "project") { S.project = m; legend(); renderProject(); renderSide(); renderTop(); }
    else if (m.type === "status") { Object.assign(S, { statuses: m.statuses, flags: m.flags, busy: m.busy, gpu: m.gpu, undo: m.undo }); drawStrip(); renderTop(); }
    else if (m.type === "item") {
      if (m.item !== S.target) return;                 // an older step; a newer one is on its way
      const img = new Image();
      img.onload = () => {
        if (m.item !== S.target) return;
        if (m.item !== S.item) S.mask = null;
        S.item = m.item; S.data = m; S.img = img;
        if (cv.width !== m.w || cv.height !== m.h) { cv.width = m.w; cv.height = m.h; }
        if (S.sel != null && !m.boxes.some((b) => b.obj === S.sel)) S.sel = null;
        draw(); renderSide(); renderTop(); drawStrip();
      };
      img.src = m.src;
    }
    else if (m.type === "mask") { const img = new Image(); img.onload = () => { S.mask = img; S.maskItem = m.item; S.sel = m.obj; draw(); renderSide(); }; img.src = m.src; }
    else if (m.type === "item_changed") { if (m.items.includes(S.target)) send({ type: "goto", item: S.target }); }
    else if (m.type === "progress") {
      const f = m.finished ? 0 : m.done / Math.max(1, m.total);
      $(".pl-bar i").style.width = `${f * 100}%`;
      $(".pl-prog").textContent = m.finished ? "" : `${m.task} ${m.done}/${m.total}${m.rate ? ` · ${m.rate} frames/s` : ""}`;
      S.busy = !m.finished; renderTop();
    }
    else if (m.type === "toast") toast(m.text);
    else if (m.type === "error") toast(m.text, true);
  });
  // keep keyboard shortcuts working after clicking buttons, the list or the timeline
  el.addEventListener("mousedown", (e) => { if (!e.target.matches("input,select,option")) setTimeout(() => el.focus(), 0); });
  new ResizeObserver(drawStrip).observe(strip);
  renderSide();
  el.focus();
  model.send({ type: "ready" });
}

export default { render };
