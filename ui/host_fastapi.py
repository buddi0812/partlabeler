"""Local web host: a start screen (projects, new project, Teach & Transfer) and the annotator.

    python -m engine.cli app [--home projects] [--port 8765]      # or double-click run_windows.bat
    python -m ui.host_fastapi --project projects/my_run            # open one project directly

Everything lives in the home folder: one sub-folder per project, `_teach/<run>` per Teach run.
The server listens on 127.0.0.1 only; the file browser shows this computer's folders to this computer.
"""
import argparse
import asyncio
import json
import os
import string
import threading
import time
import traceback
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from engine.api import Session
from engine.project import IMAGE_EXTS, Project, read_classes

UI_DIR = Path(__file__).parent
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
CLASS_EXTS = {".txt", ".yaml", ".yml"}
HEAD = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect x='2' y='3' width='12' height='10' rx='2' fill='none' stroke='%230a7c78' stroke-width='2'/></svg>">"""
ANNOTATOR = HEAD + """<title>{name} · PartLabeler</title>
<body style="margin:0;background:#f3f5f7"><div id="app"></div>
<script type="module">
import canvas from "/ui/canvas.js";
const handlers = [], queue = [];
const ws = new WebSocket(`ws://${{location.host}}/ws/{name_js}`);
const model = {{
  homeUrl: "/",
  send: (m) => (ws.readyState === 1 ? ws.send(JSON.stringify(m)) : queue.push(m)),
  on: (evt, fn) => {{ if (evt === "msg:custom") handlers.push(fn); }},
}};
ws.onopen = () => queue.splice(0).forEach((m) => ws.send(JSON.stringify(m)));
ws.onmessage = (e) => {{ const m = JSON.parse(e.data); handlers.forEach((fn) => fn(m)); }};
ws.onclose = () => document.title = "PartLabeler (disconnected: restart the app and reload)";
canvas.render({{ model, el: document.getElementById("app") }});
</script>"""
HOME = HEAD + """<title>PartLabeler</title>
<body style="margin:0"><div id="home"></div>
<script type="module">import home from "/ui/home.js"; home(document.getElementById("home"));</script>"""

state: dict = {"home": Path("projects"), "sessions": {}, "clients": {}, "jobs": {}}


# ---- projects and sessions --------------------------------------------------------------
def project_dir(name: str) -> Path:
    folder = (state["home"] / name).resolve()
    if folder.parent != state["home"].resolve() or not (folder / "project.json").exists():
        raise HTTPException(404, f"no project called {name!r}")
    return folder


def session(name: str) -> Session:
    if name not in state["sessions"]:
        p = Project(project_dir(name))
        state["clients"].setdefault(name, set())
        state["sessions"][name] = Session(p, lambda m, n=name: broadcast(n, m),
                                          image_src=lambda item, n=name: f"/items/{n}/{item}.jpg")
    return state["sessions"][name]


def broadcast(name: str, msg: dict) -> None:
    """Thread-safe: background jobs call this; each open tab has its own outgoing queue."""
    loop = state["loop"]
    for q in list(state["clients"].get(name, ())):
        loop.call_soon_threadsafe(q.put_nowait, msg)


def summary(folder: Path) -> dict:
    meta = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    out = {"name": folder.name, "kind": meta["kind"], "source": meta["source"], "classes": meta["classes"],
           "parent": meta.get("parent"), "modified": (folder / "labels.sqlite").stat().st_mtime
           if (folder / "labels.sqlite").exists() else (folder / "project.json").stat().st_mtime}
    try:
        p = state["sessions"][folder.name].p if folder.name in state["sessions"] else Project(folder)
        st = p.statuses()
        out.update(items=len(st), labeled=sum(s >= 2 for s in st), confirmed=sum(s == 4 for s in st))
        if folder.name not in state["sessions"]:
            p.db.close()
    except Exception as e:                                   # a moved source folder should not hide the project
        out.update(items=0, labeled=0, confirmed=0, problem=f"{type(e).__name__}: {e}")
    return out


# ---- background jobs for the start screen ----------------------------------------------
def start_job(kind: str, fn) -> str:
    """Run fn(job) in a thread; job is a dict the page polls: done/total/text/log/result/error."""
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "done": 0, "total": 0, "text": "Starting…", "log": [], "finished": False,
           "error": None, "result": None, "stop": False, "started": time.time()}
    state["jobs"][jid] = job

    def progress(done, total, text=""):
        job.update(done=done, total=total)
        if text and text != job["text"]:
            job["text"] = text
            job["log"].append(f"{time.strftime('%H:%M:%S')}  {text}")
            del job["log"][:-200]

    def run():
        try:
            job["result"] = fn(progress, lambda: job["stop"])
        except Exception as e:
            traceback.print_exc()
            job["error"] = f"{type(e).__name__}: {e}"
        finally:
            job["finished"] = True

    threading.Thread(target=run, daemon=True).start()
    return jid


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "stop"}


def classes_from(body: dict) -> list[str]:
    if body.get("classes_file"):
        return read_classes(Path(body["classes_file"]))
    names = [c.strip() for c in str(body.get("classes", "")).replace(",", "\n").splitlines() if c.strip()]
    if not names:
        raise HTTPException(400, "Give the class names (one per line) or a classes.txt / data.yaml file")
    return names


def safe_name(name: str) -> str:
    name = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in name).strip(" .")
    if not name:
        raise HTTPException(400, "Give the project a name")
    return name


# ---- app --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["loop"] = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@app.get("/", response_class=HTMLResponse)
def home_page() -> str:
    return HOME


@app.get("/p/{name}", response_class=HTMLResponse)
def annotator_page(name: str) -> str:
    project_dir(name)
    return ANNOTATOR.format(name=name.replace("<", "&lt;"), name_js=json.dumps(name)[1:-1])


@app.get("/items/{name}/{item}.jpg")
def item_image(name: str, item: int):
    items = session(name).p.items
    if not 0 <= item < len(items):
        raise HTTPException(404)
    return FileResponse(items[item]["path"])


@app.websocket("/ws/{name}")
async def ws(sock: WebSocket, name: str):
    await sock.accept()
    try:
        sess = await asyncio.to_thread(session, name)
    except HTTPException as e:
        await sock.send_json({"type": "error", "text": e.detail})
        await sock.close()
        return
    q: asyncio.Queue = asyncio.Queue()
    state["clients"][name].add(q)

    async def pump():
        while True:
            await sock.send_json(await q.get())

    sender = asyncio.create_task(pump())
    try:
        while True:
            msg = await sock.receive_json()
            await asyncio.to_thread(sess.handle, msg)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        state["clients"][name].discard(q)


@app.get("/api/info")
def info() -> dict:
    from engine import hw
    return {"home": str(state["home"].resolve()), "device": hw.describe(), "gpu": hw.memory(),
            "teach": _has_teach()}


def _has_teach() -> bool:
    try:
        import engine.teach  # noqa: F401
        import engine.transfer  # noqa: F401
        return True
    except Exception:
        return False


@app.get("/api/projects")
def list_projects() -> list:
    home = state["home"]
    found = [f for f in home.iterdir() if (f / "project.json").exists()] if home.exists() else []
    return sorted((summary(f) for f in found), key=lambda s: -s["modified"])


@app.post("/api/projects")
def create_project(body: dict = Body(...)) -> dict:
    name = safe_name(body.get("name", ""))
    folder = state["home"] / name
    if (folder / "project.json").exists():
        raise HTTPException(400, f"A project called {name!r} already exists")
    classes = classes_from(body)
    video, images = body.get("video") or None, body.get("images") or None
    if not (video or images):
        raise HTTPException(400, "Pick a video file or an image folder")
    src = Path(video or images)
    if not src.exists():
        raise HTTPException(400, f"Not found: {src}")
    labels, parent, every = body.get("labels") or None, (body.get("parent") or "").strip() or None, int(body.get("every") or 5)

    def job(progress, should_stop):
        p = Project.create(folder, classes, video=video, images=images, every=every,
                           progress=lambda i, n, t: progress(i, n, t))
        if parent:
            p.set_meta(parent=parent)
        res = {"name": name, "items": len(p.items)}
        if labels:
            progress(0, 0, "Importing labels")
            res["imported"] = p.import_yolo(labels)
        progress(1, 1, f"Created {name}: {len(p.items)} {'frames' if video else 'images'}")
        p.db.close()
        return res

    return {"job": start_job("create", job)}


@app.get("/api/jobs")
def list_jobs() -> list:
    return [public(j) for j in sorted(state["jobs"].values(), key=lambda j: -j["started"])]


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    if jid not in state["jobs"]:
        raise HTTPException(404)
    return public(state["jobs"][jid])


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str) -> dict:
    if jid in state["jobs"]:
        state["jobs"][jid]["stop"] = True
    return {"ok": True}


@app.get("/api/classes")
def classes_file(path: str) -> dict:
    try:
        return {"classes": read_classes(Path(path))}
    except Exception as e:
        raise HTTPException(400, f"Could not read classes from {path}: {e}")


@app.get("/api/browse")
def browse(path: str = "", kind: str = "any") -> dict:
    """Folders and matching files under `path`; with no path, the drives and handy starting points."""
    if not path:
        places = [str(Path.home()), str(state["home"].resolve()), str(UI_DIR.parent)]
        if os.name == "nt":                                   # listing drives must not touch them: slow network drives
            places += list(os.listdrives()) if hasattr(os, "listdrives") else [
                f"{d}:\\" for i, d in enumerate(string.ascii_uppercase) if __import__("ctypes").windll.kernel32.GetLogicalDrives() >> i & 1]
        else:
            places += ["/"] + [p for p in ("/content", "/content/drive/MyDrive", "/mnt", "/media") if os.path.isdir(p)]
        return {"path": "", "parent": None, "dirs": list(dict.fromkeys(places)), "files": []}
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"Not a folder: {folder}")
    exts = {"video": VIDEO_EXTS, "classes": CLASS_EXTS, "images": IMAGE_EXTS, "any": None, "dir": set()}.get(kind)
    dirs, files = [], []
    try:
        for e in sorted(folder.iterdir(), key=lambda e: e.name.lower()):
            if e.name.startswith((".", "$")):
                continue
            try:
                if e.is_dir():
                    dirs.append(e.name)
                elif exts is None or e.suffix.lower() in exts:
                    files.append({"name": e.name, "size": e.stat().st_size})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"No permission to read {folder}")
    n_images = sum(1 for f in files if Path(f["name"]).suffix.lower() in IMAGE_EXTS) if kind == "images" else None
    return {"path": str(folder.resolve()), "parent": str(folder.resolve().parent) if folder.resolve().parent != folder.resolve() else "",
            "dirs": dirs, "files": files[:2000], "images_here": n_images}


# ---- Teach & Transfer ---------------------------------------------------------------------
def runs_dir() -> Path:
    return state["home"] / "_teach"


@app.get("/api/runs")
def list_runs() -> list:
    out = []
    for d in sorted(runs_dir().glob("*"), key=lambda d: -d.stat().st_mtime) if runs_dir().exists() else []:
        if not d.is_dir():
            continue
        run = {"name": d.name, "path": str(d.resolve()), "ready": (d / "settings.json").exists()}
        if (d / "report.json").exists():
            try:
                run["report"] = json.loads((d / "report.json").read_text(encoding="utf-8"))
            except ValueError:
                pass
        if (d / "report.md").exists():
            run["report_md"] = (d / "report.md").read_text(encoding="utf-8")
        outs = d / "labels"
        run["outputs"] = [{"name": o.name, "path": str(o.resolve()),
                           "summary": json.loads((o / "summary.json").read_text()) if (o / "summary.json").exists() else None}
                          for o in sorted(outs.iterdir())] if outs.exists() else []
        out.append(run)
    return out


@app.post("/api/teach")
def teach(body: dict = Body(...)) -> dict:
    from engine.teach import teach as run_teach
    name = safe_name(body.get("name") or f"run_{time.strftime('%Y%m%d_%H%M')}")
    dataset = Path(body.get("dataset", ""))
    if not dataset.is_dir():
        raise HTTPException(400, "Pick the labeled dataset folder (with images/ and labels/)")
    kw = {k: body[k] for k in ("size", "epochs", "resolution") if body.get(k)}
    parent = (body.get("parent") or "").strip() or None

    def job(progress, should_stop):
        return run_teach(dataset, runs_dir() / name, parent=parent, progress=progress, should_stop=should_stop,
                         **{k: (int(v) if k != "size" else v) for k, v in kw.items()})

    return {"job": start_job("teach", job), "run": name}


@app.post("/api/transfer")
def transfer(body: dict = Body(...)) -> dict:
    from engine.transfer import transfer as run_transfer
    run = runs_dir() / safe_name(body.get("run", ""))
    if not (run / "settings.json").exists():
        raise HTTPException(400, "Pick a finished Teach run")
    sources = [s for s in body.get("sources", []) if s]
    if not sources:
        raise HTTPException(400, "Add at least one video or image folder to label")
    every = int(body["every"]) if body.get("every") else None

    def job(progress, should_stop):
        return run_transfer(run, sources, run / "labels", every=every, progress=progress, should_stop=should_stop)

    return {"job": start_job("transfer", job)}


@app.post("/api/review")
def review(body: dict = Body(...)) -> dict:
    """Open transferred labels for checking: a project over the source with the labels imported."""
    out = Path(body.get("output", ""))
    settings = json.loads((out.parent.parent / "settings.json").read_text(encoding="utf-8"))
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else {}
    source = summ.get("source") or body.get("source")
    if not source:
        raise HTTPException(400, "This output does not record its source video; create the project by hand")
    name = safe_name(body.get("name") or f"review_{out.name}")
    folder = state["home"] / name
    if (folder / "project.json").exists():
        return {"name": name}
    classes = settings["classes"]
    every = summ.get("every") or settings.get("every") or 5
    is_video = Path(source).is_file()

    def job(progress, should_stop):
        p = Project.create(folder, classes, video=source if is_video else None, images=None if is_video else source,
                           every=every, progress=progress)
        res = p.import_yolo(out / "labels")
        p.db.close()
        progress(1, 1, f"Imported {res['boxes']} boxes into {name}")
        return {"name": name, **res}

    return {"job": start_job("review", job), "name": name}


# ---- entry points -------------------------------------------------------------------------
def serve(home: Path = Path("projects"), port: int = 8765, open_browser: bool = True, project: str | None = None) -> None:
    state["home"] = Path(home)
    state["home"].mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}/" + (f"p/{project}" if project else "")
    print(f"PartLabeler: {url}   (projects in {state['home'].resolve()}; Ctrl+C to quit)", flush=True)
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=os.environ.get("PARTLABELER_HOME", "projects"), help="folder holding the projects")
    ap.add_argument("--project", help="open this project folder directly (its parent becomes the home folder)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.project:
        folder = Path(args.project).resolve()
        if not (folder / "project.json").exists():
            raise SystemExit(f"{folder} is not a project; create one on the start screen or with `partlabeler new`")
        serve(folder.parent, args.port, not args.no_browser, folder.name)
    else:
        serve(Path(args.home), args.port, not args.no_browser)


if __name__ == "__main__":
    main()
