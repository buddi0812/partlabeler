// PartLabeler start screen: projects, new project (with a file browser), Teach & Transfer.
// Talks to ui/host_fastapi.py over plain JSON (/api/...). Long work runs as server jobs we poll.

const CSS = `
:root { --bg:#f3f5f7; --panel:#fff; --ink:#15202b; --muted:#5b6874; --line:#d7dde3; --accent:#0a7c78; --accent-soft:#e1f0ef;
  --ok:#2f9e5b; --warn:#9a5c05; --bad:#c2412f; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
.h-wrap { max-width:1180px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:18px; }
.h-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 16px; }
.h-head h1 { margin:0; font-size:22px; } .h-head .h-meta { color:var(--muted); font-size:12px; }
.h-cols { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:18px; align-items:start; }
@media (max-width: 900px) { .h-cols { grid-template-columns:minmax(0,1fr); } }
.h-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; display:grid; gap:12px; }
.h-card h2 { margin:0; font-size:15px; } .h-card h3 { margin:0; font-size:13px; color:var(--muted); font-weight:600; }
.h-sub { color:var(--muted); font-size:12.5px; margin:0; }
.h-list { display:grid; gap:8px; }
.h-proj { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:4px 12px; align-items:center; padding:10px 12px;
  border:1px solid var(--line); border-radius:8px; text-decoration:none; color:inherit; }
.h-proj:hover { border-color:var(--accent); background:var(--accent-soft); }
.h-proj b { font-size:14px; } .h-proj .h-sub { grid-column:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h-bar { height:6px; background:#e4e9ed; border-radius:3px; overflow:hidden; display:flex; grid-column:1 / -1; }
.h-bar i { display:block; height:100%; } .h-bar .lab { background:#6aa6f2; } .h-bar .rev { background:var(--ok); }
.h-count { font-variant-numeric:tabular-nums; color:var(--muted); font-size:12px; text-align:right; }
.h-empty { color:var(--muted); padding:18px; text-align:center; border:1px dashed var(--line); border-radius:8px; }
.h-form { display:grid; gap:10px; }
.h-field { display:grid; gap:4px; } .h-field > span { font-size:12px; font-weight:600; }
.h-field small { color:var(--muted); font-size:11.5px; }
.h-row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; } .h-row > input[type=text] { flex:1; min-width:0; }
input[type=text], input[type=number], select, textarea { font:inherit; border:1px solid var(--line); border-radius:6px; padding:6px 8px; background:#fff; color:var(--ink); }
textarea { min-height:84px; resize:vertical; font-family:ui-monospace,Consolas,monospace; font-size:12.5px; }
input[type=number] { width:80px; }
button { font:inherit; border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px; padding:6px 12px; cursor:pointer; white-space:nowrap; }
button:hover { border-color:var(--accent); } button:disabled { opacity:.5; cursor:default; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, a:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.h-seg { display:inline-flex; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
.h-seg button { border:none; border-radius:0; } .h-seg button.on { background:var(--ink); color:#fff; }
.h-job { border:1px solid var(--line); border-radius:8px; padding:10px 12px; display:grid; gap:6px; }
.h-job .h-bar i { background:var(--accent); transition:width .3s; }
.h-job pre { margin:0; max-height:140px; overflow:auto; font-size:11.5px; color:var(--muted); white-space:pre-wrap; }
.h-err { color:var(--bad); font-size:12.5px; white-space:pre-wrap; } .h-ok { color:var(--ok); font-size:12.5px; }
.h-steps { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:18px; }
@media (max-width: 900px) { .h-steps { grid-template-columns:minmax(0,1fr); } }
.h-report { font-size:12.5px; background:var(--bg); border-radius:8px; padding:10px 12px; max-height:360px; overflow:auto; white-space:pre-wrap; font-family:ui-monospace,Consolas,monospace; }
.h-chip { font-size:11px; font-weight:600; padding:1px 8px; border-radius:99px; background:#e4e9ed; color:var(--muted); }
.h-chip.ok { background:#dcf1e4; color:var(--ok); } .h-chip.warn { background:#fbecd3; color:var(--warn); }
.h-src { display:flex; gap:6px; align-items:center; } .h-src span { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12.5px; }
dialog { border:1px solid var(--line); border-radius:10px; padding:0; width:min(720px, calc(100vw - 32px)); max-height:80vh; }
dialog::backdrop { background:#15202b66; }
.b-head { display:flex; gap:6px; align-items:center; padding:12px; border-bottom:1px solid var(--line); }
.b-head input { flex:1; min-width:0; }
.b-body { overflow:auto; max-height:52vh; padding:6px; }
.b-item { display:flex; gap:8px; align-items:center; width:100%; text-align:left; border:none; border-radius:6px; padding:6px 8px; background:none; }
.b-item:hover, .b-item:focus-visible { background:var(--accent-soft); }
.b-item small { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
.b-foot { display:flex; gap:8px; align-items:center; padding:12px; border-top:1px solid var(--line); }
.b-foot .h-sub { flex:1; }
`;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`);
  return data;
};
const size = (n) => (n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${Math.ceil(n / 1e3)} KB`);
const ago = (t) => { const s = Date.now() / 1000 - t; return s < 3600 ? `${Math.max(1, Math.round(s / 60))} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`; };

export default function home(root) {
  document.head.appendChild(Object.assign(document.createElement("style"), { textContent: CSS }));
  root.innerHTML = `
  <div class="h-wrap">
    <header class="h-head"><h1>PartLabeler</h1><span class="h-meta h-info">…</span></header>
    <div class="h-cols">
      <section class="h-card"><h2>Projects</h2><div class="h-list h-projects"><div class="h-empty">Loading…</div></div></section>
      <section class="h-card">
        <h2>New project</h2>
        <form class="h-form h-new" autocomplete="off">
          <label class="h-field"><span>Name</span><input type="text" name="name" required placeholder="e.g. gearbox_line2"></label>
          <div class="h-field"><span>Source</span>
            <div class="h-seg" role="radiogroup"><button type="button" data-kind="video" class="on">Video</button><button type="button" data-kind="images">Image folder</button></div>
            <div class="h-row"><input type="text" name="source" required placeholder="path to the video file">
              <button type="button" data-browse="source">Browse…</button></div>
            <small class="h-srchint">Every Nth frame is kept for labeling.</small></div>
          <label class="h-field h-every"><span>Keep every Nth frame</span><input type="number" name="every" min="1" value="5"></label>
          <div class="h-field"><span>Classes</span>
            <textarea name="classes" placeholder="one part name per line, e.g.&#10;bolt&#10;left_bracket&#10;right_bracket"></textarea>
            <div class="h-row"><button type="button" data-browse="classes">Load from classes.txt / data.yaml…</button></div>
            <small>Names starting left_ / right_ are treated as mirror twins.</small></div>
          <details><summary class="h-sub">More options</summary>
            <div class="h-form" style="margin-top:8px">
              <div class="h-field"><span>Existing YOLO labels to import (optional)</span>
                <div class="h-row"><input type="text" name="labels" placeholder="a labels/ folder, to review or finish"><button type="button" data-browse="labels">Browse…</button></div></div>
              <label class="h-field"><span>Parent object (optional)</span><input type="text" name="parent" placeholder="what the parts sit on, e.g. engine block">
                <small>Suggestions then search inside it. Can be changed later in the annotator.</small></label>
            </div></details>
          <div class="h-row"><button class="primary" type="submit">Create project</button><span class="h-sub h-newmsg"></span></div>
        </form>
        <div class="h-newjob"></div>
      </section>
    </div>
    <section class="h-card">
      <h2>Teach &amp; Transfer <span class="h-chip">automatic</span></h2>
      <p class="h-sub">Already have one video labeled? Teach learns those labels (it trains a detector on this machine, which takes time), proves on held-out frames that it reproduces them, then Transfer labels other similar videos or image folders in the same format. Check the results in the annotator before using them.</p>
      <div class="h-teach-off h-err" hidden>Teach &amp; Transfer is not installed: run the installer again without -NoTeach.</div>
      <div class="h-steps">
        <form class="h-form h-teach" autocomplete="off">
          <h3>1 · Teach from a labeled dataset</h3>
          <div class="h-field"><span>Labeled dataset folder</span>
            <div class="h-row"><input type="text" name="dataset" required placeholder="folder with images/, labels/ and classes.txt"><button type="button" data-browse="dataset">Browse…</button></div></div>
          <label class="h-field"><span>Run name</span><input type="text" name="name" placeholder="e.g. line2_v1"></label>
          <div class="h-row">
            <label class="h-field"><span>Model size</span><select name="size"><option value="nano">Nano (fastest)</option><option value="small" selected>Small</option><option value="medium">Medium (best)</option></select></label>
            <label class="h-field"><span>Epochs</span><input type="number" name="epochs" min="1" value="30"></label></div>
          <label class="h-field"><span>Parent object (optional)</span><input type="text" name="parent" placeholder="e.g. car, engine block, circuit board"></label>
          <div class="h-row"><button class="primary" type="submit">Start teaching</button></div>
          <div class="h-teachjob"></div>
        </form>
        <form class="h-form h-transfer" autocomplete="off">
          <h3>2 · Transfer to similar videos or image folders</h3>
          <label class="h-field"><span>Teach run</span><select name="run"></select></label>
          <div class="h-field"><span>Label these</span><div class="h-list h-sources"></div>
            <div class="h-row"><button type="button" data-browse="target-video">Add video…</button><button type="button" data-browse="target-folder">Add image folder…</button></div></div>
          <div class="h-row"><button class="primary" type="submit">Start labeling</button></div>
          <div class="h-transferjob"></div>
        </form>
      </div>
      <div class="h-runs"></div>
    </section>
  </div>
  <dialog class="h-browser"><div class="b-head"><button type="button" data-b="up" title="Up one folder">↑</button><input type="text" class="b-path" placeholder="type a path and press Enter"><button type="button" data-b="places">Places</button></div>
    <div class="b-body"></div>
    <div class="b-foot"><span class="h-sub b-hint"></span><button type="button" data-b="cancel">Cancel</button><button type="button" class="primary" data-b="choose">Use this folder</button></div></dialog>`;
  const $ = (s) => root.querySelector(s);
  const newForm = $(".h-new"), teachForm = $(".h-teach"), transferForm = $(".h-transfer");
  let kind = "video", sources = [];

  // ---- projects ----------------------------------------------------------------------------
  async function loadProjects() {
    const list = await api("/api/projects").catch((e) => ({ error: e.message }));
    const el = $(".h-projects");
    if (list.error) { el.innerHTML = `<div class="h-err">${esc(list.error)}</div>`; return; }
    el.innerHTML = list.length ? list.map((p) => {
      const lab = p.items ? (p.labeled - p.confirmed) / p.items * 100 : 0, rev = p.items ? p.confirmed / p.items * 100 : 0;
      return `<a class="h-proj" href="/p/${encodeURIComponent(p.name)}">
        <b>${esc(p.name)}</b><span class="h-count">${p.confirmed} / ${p.items} confirmed</span>
        <span class="h-sub">${p.kind === "video" ? "Video" : "Images"} · ${p.classes.length} classes · ${esc(p.source)}</span>
        <span class="h-count">${ago(p.modified)}</span>
        ${p.problem ? `<span class="h-err" style="grid-column:1/-1">${esc(p.problem)}</span>` : ""}
        <span class="h-bar"><i class="rev" style="width:${rev}%"></i><i class="lab" style="width:${lab}%"></i></span></a>`;
    }).join("") : `<div class="h-empty">No projects yet. Create one on the right.</div>`;
  }

  function setKind(k) {
    kind = k;
    newForm.querySelectorAll("[data-kind]").forEach((b) => b.classList.toggle("on", b.dataset.kind === k));
    newForm.source.placeholder = k === "video" ? "path to the video file" : "path to the image folder";
    $(".h-srchint").textContent = k === "video" ? "Every Nth frame is kept for labeling." : "All images in the folder and its sub-folders.";
    $(".h-every").hidden = k !== "video";
  }
  newForm.addEventListener("click", (e) => { const k = e.target.closest("[data-kind]")?.dataset.kind; if (k) setKind(k); });
  newForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = newForm, msg = $(".h-newmsg");
    msg.textContent = "";
    const body = { name: f.name.value.trim(), [kind]: f.source.value.trim(), every: f.every.value, classes: f.classes.value,
                   labels: f.labels.value.trim(), parent: f.parent.value.trim() };
    try {
      const { job } = await api("/api/projects", body);
      follow(job, $(".h-newjob"), (res) => { location.href = `/p/${encodeURIComponent(res.name)}`; });
    } catch (err) { msg.innerHTML = `<span class="h-err">${esc(err.message)}</span>`; }
  });

  // ---- jobs --------------------------------------------------------------------------------
  function follow(jid, el, onDone) {
    el.innerHTML = `<div class="h-job"><div class="h-row"><b class="j-text">Starting…</b><span class="h-sp" style="flex:1"></span><button type="button" class="j-stop">Stop</button></div>
      <span class="h-bar"><i style="width:0"></i></span><pre class="j-log"></pre></div>`;
    el.querySelector(".j-stop").onclick = () => api(`/api/jobs/${jid}/stop`, {});
    const tick = async () => {
      const j = await api(`/api/jobs/${jid}`).catch(() => null);
      if (!j) return setTimeout(tick, 1500);
      el.querySelector(".j-text").textContent = j.text;
      el.querySelector(".h-bar i").style.width = j.total ? `${Math.min(100, j.done / j.total * 100)}%` : "0";
      el.querySelector(".j-log").textContent = j.log.slice(-12).join("\n");
      if (!j.finished) return setTimeout(tick, 800);
      el.querySelector(".j-stop").remove();
      if (j.error) el.querySelector(".h-job").insertAdjacentHTML("beforeend", `<div class="h-err">${esc(j.error)}</div>`);
      else { el.querySelector(".h-bar i").style.width = "100%"; onDone?.(j.result); }
      loadProjects(); loadRuns();
    };
    tick();
  }

  // ---- teach & transfer ------------------------------------------------------------------------
  let runs = [];
  async function loadRuns() {
    runs = await api("/api/runs").catch(() => []);
    const sel = transferForm.run, keep = sel.value;
    const ready = runs.filter((r) => r.ready);
    sel.innerHTML = ready.length ? ready.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("") : `<option value="">No finished runs yet</option>`;
    if (keep) sel.value = keep;
    $(".h-runs").innerHTML = runs.map((r) => {
      const k = r.report?.knowledge || r.report?.held_out || {};
      const pass = r.report?.passed;
      const outs = r.outputs.map((o) => `<div class="h-src"><span title="${esc(o.path)}">${esc(o.name)} · ${o.summary ? `${o.summary.frames ?? "?"} frames, ${o.summary.boxes ?? "?"} boxes${o.summary.frames_to_check?.length ? `, ${o.summary.frames_to_check.length} to check` : ""}` : "in progress"}</span>
        <button type="button" data-review="${esc(o.path)}">Review in annotator</button></div>`).join("");
      return `<details class="h-job"><summary><b>${esc(r.name)}</b> ${pass === true ? `<span class="h-chip ok">targets met</span>` : pass === false ? `<span class="h-chip warn">below target</span>` : r.ready ? "" : `<span class="h-chip">not finished</span>`}
        ${k.mAP50 != null ? `<span class="h-sub">held-out mAP50 ${k.mAP50}</span>` : ""}</summary>
        ${r.report_md ? `<div class="h-report">${esc(r.report_md)}</div>` : ""}${outs ? `<h3>Labeled outputs</h3>${outs}` : ""}</details>`;
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
    $(".h-sources").innerHTML = sources.length ? sources.map((s, i) => `<div class="h-src"><span title="${esc(s)}">${esc(s)}</span><button type="button" data-rm="${i}" title="Remove">×</button></div>`).join("")
      : `<span class="h-sub">Nothing added yet.</span>`;
  }
  transferForm.addEventListener("click", (e) => { const i = e.target.dataset.rm; if (i != null) { sources.splice(+i, 1); renderSources(); } });
  transferForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const { job } = await api("/api/transfer", { run: transferForm.run.value, sources });
      follow(job, $(".h-transferjob"));
    } catch (err) { $(".h-transferjob").innerHTML = `<div class="h-err">${esc(err.message)}</div>`; }
  });
  $(".h-runs").addEventListener("click", async (e) => {
    const out = e.target.dataset.review; if (!out) return;
    e.target.disabled = true;
    try {
      const r = await api("/api/review", { output: out });
      if (r.job) follow(r.job, e.target.parentElement.appendChild(document.createElement("div")), (res) => { location.href = `/p/${encodeURIComponent(res.name)}`; });
      else location.href = `/p/${encodeURIComponent(r.name)}`;
    } catch (err) { e.target.insertAdjacentHTML("afterend", `<span class="h-err">${esc(err.message)}</span>`); e.target.disabled = false; }
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
      "target-video": { kind: "video", folder: false, done: (p) => { sources.push(p); renderSources(); } },
      "target-folder": { kind: "images", folder: true, done: (p) => { sources.push(p); renderSources(); } },
    }[which];
    pick = spec;
    const start = { source: newForm.source.value, labels: newForm.labels.value, dataset: teachForm.dataset.value }[which] || localStorage.getItem("pl-last-dir") || "";
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
    $(".h-info").textContent = `${i.device} · projects in ${i.home}`;
    $(".h-teach-off").hidden = i.teach;
    teachForm.querySelector("button[type=submit]").disabled = transferForm.querySelector("button[type=submit]").disabled = !i.teach;
  }).catch(() => ($(".h-info").textContent = "server not reachable"));
  setKind("video"); renderSources(); loadProjects(); loadRuns();
}
