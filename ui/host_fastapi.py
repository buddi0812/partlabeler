"""Local web host: a start screen (projects, new project, Teach & Transfer) and the annotator.

    python -m engine.cli app [--home projects] [--port 8765]      # or double-click run_windows.bat
    python -m ui.host_fastapi --project projects/my_run            # open one project directly

Everything lives in the home folder: one sub-folder per project, `_teach/<run>` per Teach run,
`_trash/` for projects moved to the trash, `.partlabeler/notifications.json` for the notification history.
Each account (engine/accounts.py) may pick its own home folder in Settings; --home is the default one.
The server listens on 127.0.0.1 only; the file browser shows this computer's folders to this computer.
Every page and API call needs a signed-in account, except /login, /api/auth/* and the UI files.
"""
import argparse
import asyncio
import contextvars
import json
import os
import shutil
import string
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from urllib.parse import quote
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from engine import assistant
from engine.accounts import Accounts
from engine.api import Session
from engine.notify import Notifications
from engine.project import IMAGE_EXTS, Project, read_classes

UI_DIR = Path(__file__).parent
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
CLASS_EXTS = {".txt", ".yaml", ".yml"}
HEAD = """<!doctype html><html lang="en"%%THEME%%><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#ffffff">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect x='2' y='3' width='12' height='10' rx='2' fill='none' stroke='%230a7c78' stroke-width='2'/></svg>">
<link rel="stylesheet" href="/ui/theme.css">%%PREFS%%"""
ANNOTATOR = HEAD + """<title>{name} · PartLabeler</title>
<body style="margin:0;background:var(--c-bg,#eef1f4)"><div id="app"></div>
<script type="module">
import canvas from "/ui/canvas.js";
const handlers = [], queue = [];
const ws = new WebSocket(`ws://${{location.host}}/ws/{name_js}`);
const model = {{
  homeUrl: "/",
  job: {job},
  startItem: Math.max(0, parseInt(new URLSearchParams(location.search).get("item") || "0", 10) || 0),
  send: (m) => (ws.readyState === 1 ? ws.send(JSON.stringify(m)) : queue.push(m)),
  on: (evt, fn) => {{ if (evt === "msg:custom") handlers.push(fn); }},
}};
ws.onopen = () => queue.splice(0).forEach((m) => ws.send(JSON.stringify(m)));
ws.onmessage = (e) => {{ const m = JSON.parse(e.data); handlers.forEach((fn) => fn(m)); }};
ws.onclose = () => {{ document.title = "Disconnected · PartLabeler"; handlers.forEach((fn) => fn({{ type: "disconnected" }})); }};
canvas.render({{ model, el: document.getElementById("app") }});
</script>"""
HOME = HEAD + """<title>PartLabeler</title>
<body style="margin:0"><div id="home"></div>
<script type="module">import home from "/ui/home.js"; home(document.getElementById("home"));</script>"""
PAGE = HEAD + """<title>{title} · PartLabeler</title>
<body style="margin:0"><div id="page"></div>
<script type="module">import {{ page }} from "/ui/pages.js"; page(document.getElementById("page"), {args});</script>"""

state: dict = {"home": Path("projects"), "sessions": {}, "clients": {}, "jobs": {}, "update_check": None, "restart": None,
               "notes": {}, "accounts": None}
UPDATE_CHECK_S = 6 * 3600                                    # ask GitHub at most this often
COOKIE = "pl_session"
USER: contextvars.ContextVar = contextvars.ContextVar("user", default=None)   # who is signed in, per request


# ---- accounts ---------------------------------------------------------------------------------
def accounts() -> Accounts:
    if state["accounts"] is None:
        state["accounts"] = Accounts()
    return state["accounts"]


def home() -> Path:
    """The projects folder of whoever is signed in (Settings), else the app's default."""
    custom = accounts().prefs(USER.get())["projects"] if USER.get() else ""
    return Path(custom) if custom else state["home"]


def themed(html: str) -> str:
    """A page for the signed-in person: their theme on <html>, their preferences for the scripts."""
    user = USER.get()
    prefs = accounts().prefs(user)
    attrs = f' data-theme="{prefs["theme"]}" data-accent="{prefs["accent"]}"' + ("" if prefs["motion"] else ' data-motion="off"')
    data = json.dumps({"prefs": {k: v for k, v in prefs.items() if k != "projects"},
                       "user": accounts().profile(user) if user else None}).replace("<", "\\u003c")
    return html.replace("%%THEME%%", attrs, 1).replace("%%PREFS%%", f"<script>window.PL = {data};</script>", 1)


# ---- projects and sessions --------------------------------------------------------------
def project_dir(name: str) -> Path:
    folder = (home() / name).resolve()
    if folder.parent != home().resolve() or not (folder / "project.json").exists():
        raise HTTPException(404, f"no project called {name!r}")
    return folder


def session(name: str) -> Session:
    """The project's session, one per project folder, shared by every tab (and account) that opens it."""
    key = str(project_dir(name))
    if key not in state["sessions"]:
        p = Project(Path(key))
        state["clients"].setdefault(key, set())
        state["sessions"][key] = Session(p, lambda m, k=key: broadcast(k, m),
                                         image_src=lambda item, n=name: f"/items/{n}/{item}.jpg", notes=notes())
        state["sessions"][key].thumb_src = lambda of, key, n=name: f"/thumbs/{quote(n)}/{of}/{key}.jpg"
    return state["sessions"][key]


def notes() -> Notifications:
    """The notification history of the current projects folder, shared by the start screen and every tab."""
    h = home().resolve()
    if h not in state["notes"]:
        state["notes"][h] = Notifications(h / ".partlabeler" / "notifications.json", on_change=lambda m, h=h: broadcast_home(h, m))
    return state["notes"][h]


def note(level: str, title: str, detail: str = "", project: str | None = None, action: dict | None = None) -> dict:
    return notes().add(level, title, detail, project=project, action=action)


def open_action(name: str) -> dict:
    return {"type": "open", "project": name, "label": "Open project"}


def project(name: str) -> Project:
    """The project, as the open annotator holds it (so both see every change), or freshly loaded."""
    return session(name).p


def broadcast_home(h: Path, msg: dict) -> None:
    """To the tabs of every project in the projects folder `h`."""
    for key in list(state["clients"]):
        if Path(key).parent == h:
            broadcast(key, msg)


def broadcast(key: str, msg: dict) -> None:
    """To the open tabs of one project (key: its folder). Thread-safe: background jobs call this."""
    loop = state.get("loop")
    if loop is None:
        return
    for q in list(state["clients"].get(key, ())):
        loop.call_soon_threadsafe(q.put_nowait, msg)


def summary(folder: Path) -> dict:
    meta = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    tasks = meta.get("sources") if "sources" in meta else [{"name": Path(meta["source"]).stem, "kind": meta["kind"]}]
    out = {"name": folder.name, "kind": meta.get("kind") or (tasks[0]["kind"] if tasks else "video"),
           "task": meta.get("task", "detect"), "source": meta.get("source", ""), "created": meta.get("created"),
           "classes": meta["classes"], "colors": meta.get("colors") or [],
           "tasks": [{"name": t["name"], "kind": t["kind"]} for t in tasks],
           "preview": f"/previews/{quote(folder.name)}/first.jpg" if tasks else None,
           "parent": meta.get("parent"), "modified": (folder / "labels.sqlite").stat().st_mtime
           if (folder / "labels.sqlite").exists() else (folder / "project.json").stat().st_mtime}
    try:
        key = str(folder.resolve())
        p = state["sessions"][key].p if key in state["sessions"] else Project(folder)
        st = p.statuses()
        out.update(items=len(st), labeled=sum(s >= 2 for s in st), confirmed=sum(s == 4 for s in st))
        if key not in state["sessions"]:
            p.db.close()
    except Exception as e:                                   # a moved source folder should not hide the project
        out.update(items=0, labeled=0, confirmed=0, problem=f"{type(e).__name__}: {e}")
    return out


# ---- background jobs for the start screen ----------------------------------------------
def start_job(kind: str, fn, on_done=None, fail_title: str | None = None) -> str:
    """Run fn(progress, should_stop) in a thread; the page polls the job dict (done/total/text/log/result/error).
    on_done(result) records the success notification; a failure is recorded as `fail_title`."""
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "done": 0, "total": 0, "text": "Starting…", "log": [], "finished": False,
           "error": None, "result": None, "stop": False, "started": time.time(), "user": USER.get()}
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
            if on_done:
                on_done(job["result"])
        except Exception as e:
            traceback.print_exc()
            job["error"] = f"{type(e).__name__}: {e}"
            note("error", fail_title or f"{kind.capitalize()} failed", job["error"])
        finally:
            job["finished"] = True

    threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True).start()   # same account inside
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


@app.middleware("http")
async def revalidate_ui(request, call_next):
    """UI files change with every update: browsers must check (cheap 304s) instead of running a stale copy."""
    response = await call_next(request)
    if request.url.path.startswith("/ui/") or request.url.path == "/" or request.url.path.startswith("/p/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


OPEN = ("/login", "/api/auth/", "/ui/")                  # reachable before signing in


@app.middleware("http")
async def signed_in(request, call_next):
    """Pages send people to /login until they sign in; API calls answer 401. Sets USER for the request."""
    user = accounts().user_for(request.cookies.get(COOKIE))
    path = request.url.path
    if user is None and not path.startswith(OPEN):
        if request.method == "GET" and not path.startswith(("/api/", "/items/", "/thumbs/", "/previews/")):
            back = path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(f"/login?next={quote(back)}", status_code=303)
        return JSONResponse({"detail": "Sign in first (open the start page)"}, status_code=401)
    token = USER.set(user)
    try:
        return await call_next(request)
    finally:
        USER.reset(token)


def _signed_in_as(response: Response, user: str, keep: bool) -> dict:
    response.set_cookie(COOKIE, accounts().sign_in(user, keep), max_age=30 * 86400 if keep else None,
                        httponly=True, samesite="strict", path="/")
    return accounts().profile(user)


@app.get("/login", response_class=HTMLResponse)
def login_page(next: str = "/"):
    if USER.get():
        return RedirectResponse(next if next.startswith("/") and not next.startswith("//") else "/", status_code=303)
    return _page("Sign in", view="login", first=not accounts().users())


@app.post("/api/auth/login")
def login(response: Response, body: dict = Body(...)) -> dict:
    # ponytail: no attempt limit; the server only listens on 127.0.0.1, where the files themselves are readable
    user = accounts().verify(body.get("username", ""), body.get("password", ""))
    if not user:
        raise HTTPException(401, "Wrong username or password")
    return _signed_in_as(response, user, bool(body.get("keep")))


@app.post("/api/auth/signup")
def signup(response: Response, body: dict = Body(...)) -> dict:
    try:
        user = accounts().create(body.get("username", ""), body.get("password", ""), body.get("name", ""))["username"]
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _signed_in_as(response, user, bool(body.get("keep")))


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict:
    accounts().sign_out(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/settings", response_class=HTMLResponse)
def settings_page() -> str:
    return _page("Settings", view="settings")


@app.get("/api/me")
def me() -> dict:
    user = USER.get()
    return {"user": accounts().profile(user), "prefs": accounts().prefs(user), "home": str(home().resolve()),
            "default_home": str(state["home"].resolve())}


@app.patch("/api/me")
def update_me(body: dict = Body(...)) -> dict:
    return accounts().rename(USER.get(), body.get("name", ""))


@app.post("/api/me/prefs")
def update_prefs(body: dict = Body(...)) -> dict:
    """Change preferences (not the projects folder: that has its own call, which can move the projects)."""
    if "projects" in body:
        raise HTTPException(400, "Change the projects folder with /api/me/projects-folder")
    try:
        return accounts().set_prefs(USER.get(), body)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/me/password")
def change_password(request: Request, response: Response, body: dict = Body(...)) -> dict:
    user = USER.get()
    if not accounts().verify(user, body.get("old", "")):
        raise HTTPException(400, "The current password is not right")
    try:
        accounts().set_password(user, body.get("new", ""))           # signs out everywhere…
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _signed_in_as(response, user, True)                        # …except here


@app.post("/api/me/delete")
def delete_me(response: Response, body: dict = Body(...)) -> dict:
    """Remove the account (after its password); the projects stay where they are."""
    user = USER.get()
    if not accounts().verify(user, body.get("password", "")):
        raise HTTPException(400, "The password is not right")
    accounts().delete(user)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.post("/api/me/projects-folder")
def set_projects_folder(body: dict = Body(...)) -> dict:
    """Use another projects folder ("" = the app's default). With move, everything in the current folder moves
    there first (a background job): projects, trash, backups, Teach runs."""
    user, old = USER.get(), home().resolve()
    raw = str(body.get("path", "")).strip().strip('"')
    new = Path(raw).expanduser() if raw else state["home"]
    if not new.is_absolute() and raw:
        raise HTTPException(400, "Give the folder's full path, e.g. D:\\PartLabeler\\projects")
    new = new.resolve()
    value = "" if new == state["home"].resolve() else str(new)
    if new == old:
        return {"home": str(old), "job": None}
    try:
        new.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"Could not use {new}: {e}")
    if not body.get("move"):
        accounts().set_prefs(user, {"projects": value})
        return {"home": str(new), "job": None}
    if old in new.parents or new in old.parents:
        raise HTTPException(400, "One folder is inside the other: pick a separate folder to move the projects to")
    if any(not j["finished"] for j in state["jobs"].values()) or any(
            s.running for k, s in state["sessions"].items() if Path(k).parent == old):
        raise HTTPException(409, "A job is running (tracking, export, Teach…). Let it finish or stop it, then move the projects.")
    for k in [k for k in state["sessions"] if Path(k).parent == old]:      # let go of the files before moving them
        state["sessions"].pop(k).p.db.close()

    def job(progress, should_stop):
        accounts().set_prefs(user, {"projects": value})
        entries, moved, skipped = sorted(old.iterdir()) if old.exists() else [], [], []
        for k, e in enumerate(entries):
            progress(k, len(entries), f"Moving {e.name}")
            if (new / e.name).exists():
                skipped.append(e.name)
                continue
            shutil.move(str(e), str(new / e.name))
            moved.append(e.name)
        return {"home": str(new), "moved": moved, "skipped": skipped}

    def done(res):
        n = len(res['moved'])
        note("success", f"Moved {n} item{'' if n == 1 else 's'} to {res['home']}",
             f"Already there, so left in {old}: {', '.join(res['skipped'])}" if res["skipped"] else "The projects folder is now there.",
             action={"type": "folder", "path": res["home"], "label": "Open folder"})

    return {"home": str(new), "job": start_job("move", job, done, f"Could not move everything out of {old}; the rest is still there")}


@app.get("/", response_class=HTMLResponse)
def home_page() -> str:
    return themed(HOME)


def _page(title: str, **args) -> str:
    return themed(PAGE.format(title=title.replace("<", "&lt;"), args=json.dumps(args).replace("</", "<\\/")))


@app.get("/p/{name}", response_class=HTMLResponse)
def old_link(name: str, item: int | None = None):
    """Links from before tasks and jobs: the job that holds `item`, else the project page."""
    from fastapi.responses import RedirectResponse
    project_dir(name)
    if item is not None:
        p = project(name)
        j = next((j for j in p.jobs() if j["start"] <= item < j["end"]), None)
        if j:
            return RedirectResponse(f"/projects/{quote(name)}/tasks/{j['task']}/jobs/{j['id']}?item={item}")
    return RedirectResponse(f"/projects/{quote(name)}")


@app.get("/projects/{name}", response_class=HTMLResponse)
def project_page(name: str) -> str:
    project_dir(name)
    return _page(name, view="project", project=name)


@app.get("/projects/{name}/tasks/create", response_class=HTMLResponse)
def create_task_page(name: str) -> str:
    project_dir(name)
    return _page(f"New task · {name}", view="create-task", project=name)


@app.get("/projects/{name}/tasks/{tid}", response_class=HTMLResponse)
def task_page(name: str, tid: int) -> str:
    project_dir(name)
    return _page(f"Task {tid} · {name}", view="task", project=name, task=tid)


@app.get("/projects/{name}/tasks/{tid}/jobs/{jid}", response_class=HTMLResponse)
def job_page(name: str, tid: int, jid: int) -> str:
    project_dir(name)
    return themed(ANNOTATOR.format(name=name.replace("<", "&lt;"), name_js=json.dumps(name)[1:-1], job=int(jid)))



@app.get("/tasks", response_class=HTMLResponse)
def tasks_page() -> str:
    return _page("Tasks", view="tasks")


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page() -> str:
    return _page("Jobs", view="jobs")


def task_summary(p: Project, t: dict, jobs: list[dict], statuses: list[int]) -> dict:
    a, b = p.ranges.get(t["id"], (0, 0))
    part = statuses[a:b]
    updated = max([j["updated"] for j in jobs if j.get("updated")] or [t.get("created") or ""]) or None
    return {"id": t["id"], "name": t["name"], "subset": t.get("subset", ""), "kind": t["kind"], "source": t["source"],
            "created": t.get("created"), "updated": updated, "info": t.get("info") or {}, "frames": b - a,
            "start_item": a, "every": t.get("every", 1), "start": t.get("start"), "stop": t.get("stop"),
            "quality": t.get("quality", 95), "format": t.get("format", "jpg"), "sorting": t.get("sorting"),
            "segment_size": t.get("segment_size", 0), "labeled": sum(x >= 2 for x in part),
            "confirmed": sum(x == 4 for x in part), "jobs": jobs, **Project.task_status(jobs),
            "preview": f"/previews/{quote(p.meta['name'])}/{t['id']}.jpg"}


def project_detail(name: str) -> dict:
    p = project(name)
    p.fill_info()
    jobs, st = p.jobs(), p.statuses()
    tasks = [task_summary(p, t, [j for j in jobs if j["task"] == t["id"]], st) for t in p.sources]
    return {"name": name, "task": p.task, "created": p.meta.get("created"), "labels": p.labels(),
            "formats": list(p.formats), "frame_format": p.meta.get("frame_format", "jpg"), "tasks": tasks,
            "subsets": sorted({t["subset"] for t in tasks if t["subset"]}), "items": len(p.items)}


def refresh(name: str, reload: bool = False) -> None:
    """Tell open annotator tabs: tasks, jobs or labels changed (reload: item numbers moved)."""
    key = str(project_dir(name))
    if key in state["sessions"]:
        s_ = state["sessions"][key]
        if reload:
            state["sessions"].pop(key)
            broadcast(key, {"type": "reload"})
        else:
            broadcast(key, s_.project_msg())
            broadcast(key, s_.status_msg())


@app.get("/api/projects/{name}")
def get_project(name: str) -> dict:
    project_dir(name)
    return project_detail(name)


@app.patch("/api/projects/{name}")
def patch_project(name: str, body: dict = Body(...)) -> dict:
    """The label constructor: {"labels": [{"name", "color", "from": old index or null}]}."""
    project_dir(name)
    p = project(name)
    if "labels" in body:
        try:
            res = p.set_labels(body["labels"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        if res["removed"]:
            note("info", f"Deleted labels in {name}", f"{res['removed']} annotations of the deleted labels went with them", name)
        refresh(name, reload=bool(res["removed"]))
    return project_detail(name)


@app.post("/api/projects/{name}/tasks")
def create_tasks(name: str, body: dict = Body(...)) -> dict:
    """CVAT's "Create a new task" (one path) or "Create multi tasks" (several: one task each, named after the
    file unless a name template with {{file_name}} / {{index}} is given)."""
    project_dir(name)
    paths = [str(x).strip().strip('"') for x in (body.get("sources") or []) if str(x).strip()]
    if not paths:
        raise HTTPException(400, "Select a video or an image folder")
    missing = [x for x in paths if not Path(x).exists()]
    if missing:
        raise HTTPException(400, f"Not found: {', '.join(missing)}")
    template = (body.get("name") or "").strip()
    opts = {"subset": body.get("subset") or "", "every": int(body.get("every") or 5),
            "start": int(body["start"]) if str(body.get("start", "")).strip() else None,
            "stop": int(body["stop"]) if str(body.get("stop", "")).strip() else None,
            "quality": int(body.get("quality") or 95), "lossless": bool(body.get("lossless")),
            "sorting": body.get("sorting") or "lexicographical", "segment_size": int(body.get("segment_size") or 0)}

    def job(progress, should_stop):
        p, made = project(name), []
        for i, src in enumerate(paths):
            if should_stop():
                break
            nm = (template.replace("{{file_name}}", Path(src).stem).replace("{{index}}", str(i + 1))
                  if template and (len(paths) == 1 or "{{" in template) else None)
            progress(i, len(paths), f"Task {i + 1} of {len(paths)}: {Path(src).name}")
            t = p.add_source(video=src if Path(src).is_file() else None, images=None if Path(src).is_file() else src,
                             progress=lambda d, n, text: progress(d, n, f"{Path(src).name}: {text}"), name=nm, **opts)
            made.append(t["id"])
        refresh(name)
        return {"project": name, "tasks": made, "first_job": p.jobs(made[0])[0]["id"] if made and p.jobs(made[0]) else None}

    def done(res):
        note("success", f"Created {len(res['tasks'])} task{'s' if len(res['tasks']) != 1 else ''} in {name}", "", name, open_action(name))

    return {"job": start_job("create task", job, done, f"Could not create the task in {name}")}


@app.get("/api/projects/{name}/tasks/{tid}")
def get_task(name: str, tid: int) -> dict:
    project_dir(name)
    d = project_detail(name)
    t = next((t for t in d["tasks"] if t["id"] == tid), None)
    if t is None:
        raise HTTPException(404, f"no task {tid} in {name}")
    return {**t, "project": name, "project_task": d["task"], "labels": d["labels"], "formats": d["formats"],
            "subsets": d["subsets"]}


@app.patch("/api/projects/{name}/tasks/{tid}")
def patch_task(name: str, tid: int, body: dict = Body(...)) -> dict:
    project_dir(name)
    try:
        project(name).update_source(tid, name=body.get("name"), subset=body.get("subset"))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e).strip("'"))
    refresh(name)
    return get_task(name, tid)


@app.delete("/api/projects/{name}/tasks/{tid}")
def delete_task(name: str, tid: int) -> dict:
    project_dir(name)
    try:
        res = project(name).remove_source(tid)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e).strip("'"))
    refresh(name, reload=True)
    note("info", f"Deleted task {res['task']['name']} from {name}", f"{res['items']} frames or images, {res['labels']} labels", name)
    return {"ok": True, **{k: v for k, v in res.items() if k != "task"}}


@app.patch("/api/projects/{name}/jobs/{jid}")
def patch_job(name: str, jid: int, body: dict = Body(...)) -> dict:
    project_dir(name)
    try:
        j = project(name).set_job(jid, stage=body.get("stage"), state=body.get("state"))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e).strip("'"))
    refresh(name)
    return j


@app.post("/api/projects/{name}/annotations")
def upload_annotations(name: str, body: dict = Body(...)) -> dict:
    """CVAT's "Upload annotations" into a task or a job: a YOLO labels folder, Replace or Append."""
    project_dir(name)
    folder = Path(str(body.get("labels", "")).strip().strip('"'))
    if not folder.is_dir():
        raise HTTPException(400, f"Not a folder: {folder}")
    p = project(name)
    if body.get("job"):
        j = p.job(int(body["job"]))
        items = range(j["start"], j["end"])
    else:
        items = range(*p.ranges.get(int(body.get("task", -1)), (0, 0)))
    res = p.import_yolo(folder, items=items, replace=body.get("mode") == "replace")
    refresh(name)
    note("success", f"Uploaded annotations to {name}", f"{res['boxes']} labels from {res['files_matched']} files"
         + (f"; {res['files_unmatched']} files matched no frame" if res["files_unmatched"] else ""), name)
    return res


@app.post("/api/projects/{name}/export")
def export_dataset(name: str, body: dict = Body(...)) -> dict:
    """CVAT's "Export dataset" for the project, a task or a job: format, save images, custom name."""
    project_dir(name)
    fmt = body.get("format") or "yolo"
    tasks, jobs = body.get("tasks") or None, body.get("jobs") or None
    what = f"job_{jobs[0]}" if jobs else (f"task_{tasks[0]}" if tasks and len(tasks) == 1 else "project")
    custom = safe_name(body["name"]) if (body.get("name") or "").strip() else None

    def job(progress, should_stop):
        p = project(name)
        folder = custom or f"{what}_{name}_dataset_{time.strftime('%Y_%m_%d_%H_%M_%S')}_{fmt}"
        out = p.folder / "exports" / folder
        progress(0, 1, "Exporting")
        res = p.export(fmt, out, reviewed_only=bool(body.get("reviewed_only")), tasks=tasks, jobs=jobs,
                       save_images=body.get("save_images", True) is not False)
        return {**res, "folder": str(out)}

    def done(res):
        note("success", f"Exported {name} as {fmt.upper()}", f"{res['images']} images. {res['folder']}", name,
             {"type": "folder", "path": res["folder"], "label": "Open folder"})

    return {"job": start_job("export", job, done, f"Could not export {name}")}


@app.post("/api/projects/{name}/backup")
def backup(name: str) -> dict:
    from engine.project import backup_project
    project_dir(name)

    def job(progress, should_stop):
        progress(0, 1, "Packing the project")
        out = home() / "_backups" / f"project_{name}_backup_{time.strftime('%Y_%m_%d_%H_%M_%S')}.zip"
        return backup_project(home() / name, out)

    def done(res):
        note("success", f"Backed up {name}", f"{res['items']} frames or images, {res['size'] / 2**20:.1f} MB. {res['file']}", name,
             {"type": "folder", "path": str(Path(res["file"]).parent), "label": "Open folder"})

    return {"job": start_job("backup", job, done, f"Could not back up {name}")}


@app.post("/api/projects/{name}/tasks/{tid}/backup")
def backup_task(name: str, tid: int) -> dict:
    """CVAT's "Backup task": one task with its frames, labels, confirmed frames and jobs, as a zip."""
    from engine.project import backup_project
    folder = project_dir(name)
    t = next((t for t in project(name).sources if t["id"] == tid), None)
    if t is None:
        raise HTTPException(404, f"no task {tid} in {name}")

    def job(progress, should_stop):
        progress(0, 1, f"Packing task {t['name']}")
        out = home() / "_backups" / f"task_{name}_{t['name']}_backup_{time.strftime('%Y_%m_%d_%H_%M_%S')}.zip"
        return backup_project(folder, out, tasks=[tid])

    def done(res):
        note("success", f"Backed up task {t['name']}", f"{res['items']} frames or images, {res['size'] / 2**20:.1f} MB. {res['file']}", name,
             {"type": "folder", "path": str(Path(res["file"]).parent), "label": "Open folder"})

    return {"job": start_job("backup", job, done, f"Could not back up task {t['name']}")}


@app.post("/api/projects/{name}/tasks/import")
def import_tasks(name: str, body: dict = Body(...)) -> dict:
    """Add the task(s) of a backup zip to this project, with their progress (a task or a project backup)."""
    project_dir(name)
    path = Path(str(body.get("path", "")).strip().strip('"'))
    if not path.is_file():
        raise HTTPException(400, f"Not a file: {path}")

    def job(progress, should_stop):
        progress(0, 1, "Unpacking the backup")
        res = project(name).add_tasks_from(path)
        refresh(name)
        return res

    def done(res):
        extra = f" New labels: {', '.join(res['labels_added'])}." if res["labels_added"] else ""
        note("success", f"Added {', '.join(res['tasks'])} to {name}", f"{res['items']} frames or images with their labels.{extra}",
             name, open_action(name))

    return {"job": start_job("import", job, done, f"Could not add the backup to {name}")}


@app.post("/api/backups/restore")
def restore(body: dict = Body(...)) -> dict:
    """CVAT's "Create from backup": a project backup zip becomes a new project."""
    from engine.project import restore_project
    path = Path(str(body.get("path", "")).strip().strip('"'))
    if not path.is_file():
        raise HTTPException(400, f"Not a file: {path}")

    def job(progress, should_stop):
        progress(0, 1, "Unpacking the backup")
        dest = restore_project(path, home(), (body.get("name") or "").strip() or None)
        return {"name": dest.name}

    def done(res):
        note("success", f"Created {res['name']} from a backup", str(path), res["name"], open_action(res["name"]))

    return {"job": start_job("restore", job, done, "Could not restore the backup")}


@app.get("/api/tasks")
def all_tasks() -> list:
    """Every task of every project (the Tasks page)."""
    out = []
    for s_ in list_projects():
        if s_.get("problem"):
            continue
        d = project_detail(s_["name"])
        out += [{**t, "project": s_["name"], "project_task": d["task"]} for t in d["tasks"]]
    return sorted(out, key=lambda t: t.get("updated") or "", reverse=True)


@app.get("/api/annotation-jobs")
def all_jobs() -> list:
    """Every job of every project (the Jobs page)."""
    return [{**j, "project": t["project"], "task_kind": t["kind"], "preview": t["preview"]}
            for t in all_tasks() for j in t["jobs"]]


@app.get("/previews/{name}/{which}.jpg")
def preview(name: str, which: str):
    """A task's first frame (which = task id) or the project's (which = first), as a thumbnail."""
    project_dir(name)
    p = project(name)
    if not p.items:
        raise HTTPException(404)
    k = 0 if which == "first" else p.ranges.get(int(which) if which.isdigit() else -1, (None, None))[0]
    if k is None or k >= len(p.items):
        raise HTTPException(404)
    return Response(session(name).thumb_bytes("images", k), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/items/{name}/{item}.jpg")
def item_image(name: str, item: int):
    items = session(name).p.items
    if not 0 <= item < len(items):
        raise HTTPException(404)
    return FileResponse(items[item]["path"])


@app.get("/thumbs/{name}/{of}/{key}.jpg")
def thumb(name: str, of: str, key: int):
    """Grid thumbnails: whole images (of=images) or parts (of=parts)."""
    if of not in ("images", "parts"):
        raise HTTPException(404)
    try:
        data = session(name).thumb_bytes(of, key)
    except (KeyError, IndexError, OSError):
        raise HTTPException(404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.websocket("/ws/{name}")
async def ws(sock: WebSocket, name: str):
    await sock.accept()
    user = accounts().user_for(sock.cookies.get(COOKIE))
    if user is None:
        await sock.send_json({"type": "error", "text": "Signed out: open the start page and sign in again"})
        await sock.close()
        return
    USER.set(user)                                            # this connection's task only
    try:
        sess = await asyncio.to_thread(session, name)
    except HTTPException as e:
        await sock.send_json({"type": "error", "text": e.detail})
        await sock.close()
        return
    key = str(sess.p.folder)
    q: asyncio.Queue = asyncio.Queue()
    state["clients"][key].add(q)

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
        state["clients"][key].discard(q)


@app.get("/api/info")
def info() -> dict:
    from engine import hw, update
    try:
        version = (update.current()["sha"] or "")[:7]
    except Exception:
        version = ""
    return {"home": str(home().resolve()), "device": hw.describe(), "gpu": hw.memory(),
            "teach": _has_teach(), "version": version, "restart": state["restart"]}


# ---- updates (engine/update.py) --------------------------------------------------------------
@app.get("/api/update")
def update_check(force: bool = False) -> dict:
    """Is a newer version on GitHub? Asked at most every 6 hours (PARTLABELER_NO_UPDATE_CHECK=1: never unless forced)."""
    from engine import update
    if os.environ.get("PARTLABELER_NO_UPDATE_CHECK") and not force:
        return {"available": False, "off": True}
    cached = state["update_check"]
    if cached and not force and time.time() - cached[0] < UPDATE_CHECK_S:
        return cached[1]
    try:
        res = update.check()
    except Exception as e:                                   # offline, GitHub down, rate limit: say so, try later
        res = {"available": False, "error": f"Could not reach GitHub ({type(e).__name__})"}
    state["update_check"] = (time.time(), res)
    return res


@app.post("/api/update")
def update_now() -> dict:
    """Update the app files now; Python packages, if they changed, at the next start. Needs a restart after."""
    from engine import update
    if any(not j["finished"] for j in state["jobs"].values()) or any(s.running for s in state["sessions"].values()):
        raise HTTPException(409, "A job is running (tracking, Teach, Transfer…). Let it finish or stop it, then update.")

    def job(progress, should_stop):
        return update.update(home(), packages_now=False, log=lambda text: progress(0, 0, text))

    def done(res):
        state["restart"] = res
        state["update_check"] = None
        if res["from"] == res["to"]:
            note("info", "PartLabeler is up to date", f"Version {res['to']}")
            return
        note("success", f"Updated PartLabeler to version {res['to']}",
             "Restart to finish: close the PartLabeler window and start it again (run_windows.bat)."
             + (f" Your labels were backed up to {res['backup']}." if res["backup"] else ""))

    return {"job": start_job("update", job, done, "Could not update PartLabeler")}


def _has_teach() -> bool:
    try:
        import engine.teach  # noqa: F401
        import engine.transfer  # noqa: F401
        return True
    except Exception:
        return False


@app.get("/api/projects")
def list_projects() -> list:
    h = home()
    found = [f for f in h.iterdir() if (f / "project.json").exists() and not f.name.startswith((".", "_trash"))] \
        if h.exists() else []
    return sorted((summary(f) for f in found), key=lambda s: -s["modified"])


@app.post("/api/projects")
def create_project(body: dict = Body(...)) -> dict:
    """Create a project (CVAT: name + labels). With no video or folder it is made at once, empty, and tasks are
    added on its page; with one, it is made in the background with that first task."""
    name = safe_name(body.get("name", ""))
    folder = home() / name
    if (folder / "project.json").exists():
        raise HTTPException(400, f"A project called {name!r} already exists")
    colors = None
    if isinstance(body.get("labels"), list):                # the label constructor: [{"name", "color"}]
        classes = [" ".join(str(l.get("name", "")).split()) for l in body["labels"] if str(l.get("name", "")).strip()]
        colors = [l.get("color") for l in body["labels"] if str(l.get("name", "")).strip()]
        if not classes and body.get("task") != "classify":
            raise HTTPException(400, "Add at least one label")
    else:
        classes = [] if body.get("task") == "classify" and not (body.get("classes") or "").strip() and not body.get("classes_file") \
            else classes_from(body)                          # image classes may be named later, while sorting
    video, images = body.get("video") or None, body.get("images") or None
    if not (video or images):
        task = body.get("task") or "detect"
        if task not in ("detect", "segment", "classify"):
            raise HTTPException(400, f"Unknown project type {task!r}")
        try:
            Project.create(folder, classes, task=task, colors=colors,
                           frame_format="webp" if body.get("frame_format") == "webp" else "jpg").db.close()
        except ValueError as e:
            raise HTTPException(400, str(e))
        note("success", f"Created project {name}", "Add its first task on the project page", name, open_action(name))
        return {"name": name}
    src = Path(video or images)
    if not src.exists():
        raise HTTPException(400, f"Not found: {src}")
    labels, parent, every = body.get("labels") or None, (body.get("parent") or "").strip() or None, int(body.get("every") or 5)
    task, frame_format = body.get("task") or "detect", body.get("frame_format") or "jpg"
    if task not in ("detect", "segment", "classify"):
        raise HTTPException(400, f"Unknown label type {task!r}")
    if frame_format not in ("jpg", "webp"):
        raise HTTPException(400, f"Unknown frame format {frame_format!r}")

    def job(progress, should_stop):
        p = Project.create(folder, classes, video=video, images=images, every=every,
                           progress=lambda i, n, t: progress(i, n, t), task=task, frame_format=frame_format)
        if parent:
            p.set_meta(parent=parent)
        res = {"name": name, "items": len(p.items)}
        if labels:
            progress(0, 0, "Importing labels")
            res["imported"] = p.import_yolo(labels)
        progress(1, 1, f"Created {name}: {len(p.items)} {'frames' if video else 'images'}")
        p.db.close()
        return res

    def done(res):
        detail = f"{res['items']} {'frames' if video else 'images'}"
        if res.get("imported"):
            detail += f", {res['imported']['boxes']} boxes imported"
        note("success", f"Created project {name}", detail, name, open_action(name))

    return {"job": start_job("create", job, done, f"Could not create project {name}")}


@app.get("/api/jobs")
def list_jobs() -> list:
    return [public(j) for j in sorted(state["jobs"].values(), key=lambda j: -j["started"]) if j.get("user") == USER.get()]


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
        places = [str(Path.home()), str(home().resolve()), str(UI_DIR.parent)]
        if os.name == "nt":                                   # listing drives must not touch them: slow network drives
            places += list(os.listdrives()) if hasattr(os, "listdrives") else [
                f"{d}:\\" for i, d in enumerate(string.ascii_uppercase) if __import__("ctypes").windll.kernel32.GetLogicalDrives() >> i & 1]
        else:
            places += ["/"] + [p for p in ("/content", "/content/drive/MyDrive", "/mnt", "/media") if os.path.isdir(p)]
        return {"path": "", "parent": None, "dirs": list(dict.fromkeys(places)), "files": []}
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"Not a folder: {folder}")
    exts = {"video": VIDEO_EXTS, "classes": CLASS_EXTS, "images": IMAGE_EXTS, "any": None, "dir": set(),
            "backup": {".zip"}}.get(kind)
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
    return home() / "_teach"


@app.get("/api/runs")
def list_runs() -> list:
    out = []
    for d in sorted(runs_dir().glob("*"), key=lambda d: -d.stat().st_mtime) if runs_dir().exists() else []:
        if not d.is_dir():
            continue
        run = {"name": d.name, "path": str(d.resolve()), "ready": (d / "settings.json").exists()}
        try:
            run["mode"] = json.loads((d / "settings.json").read_text(encoding="utf-8")).get("mode", "trained")
        except (OSError, ValueError):
            run["mode"] = "trained"
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

    def done(res):
        res = res or {}
        if res.get("status") == "stopped":
            note("warning", f"Teach run {name} stopped", "Start it again to resume from the last epoch")
            return
        held = (res.get("held_out") or {}).get("mAP50")
        note("success" if res.get("passed", True) else "warning", f"Teach run {name} finished",
             (f"Held-out mAP50 {held:.2f}. " if isinstance(held, (int, float)) else "") + "Next: Transfer it to similar videos.",
             action={"type": "folder", "path": str((runs_dir() / name).resolve()), "label": "Open folder"})

    return {"job": start_job("teach", job, done, f"Teach run {name} failed"), "run": name}


@app.post("/api/transfer")
def transfer(body: dict = Body(...)) -> dict:
    from engine.transfer import transfer as run_transfer
    run = runs_dir() / safe_name(body.get("run", ""))
    if not (run / "settings.json").exists() or "checkpoint" not in json.loads((run / "settings.json").read_text(encoding="utf-8")):
        raise HTTPException(400, "Pick a finished Teach run")
    sources = [s for s in body.get("sources", []) if s]
    if not sources:
        raise HTTPException(400, "Add at least one video or image folder to label")
    every = int(body["every"]) if body.get("every") else None
    tracks = body.get("tracks", True) is not False

    def job(progress, should_stop):
        return run_transfer(run, sources, run / "labels", every=every, tracks=tracks, progress=progress, should_stop=should_stop)

    def done(res):
        n = len((res or {}).get("sources", {})) or len(sources)
        note("success", f"Transfer finished: {n} source{'s' if n != 1 else ''} labeled",
             "Review them in the annotator from the Teach & Transfer panel",
             action={"type": "folder", "path": str((run / "labels").resolve()), "label": "Open folder"})

    return {"job": start_job("transfer", job, done, "Transfer failed")}


@app.post("/api/quick")
def quick(body: dict = Body(...)) -> dict:
    """Quick transfer, no training: match a labeled dataset's examples in other videos or folders."""
    from engine.quick import quick_transfer
    dataset = Path(body.get("dataset", ""))
    if not (dataset / "images").is_dir():
        raise HTTPException(400, "Pick the labeled dataset folder (with images/ and labels/)")
    sources = [s for s in body.get("sources", []) if s]
    if not sources:
        raise HTTPException(400, "Add at least one video or image folder to label")
    name = safe_name(body.get("name") or f"quick_{dataset.name}")
    parent = (body.get("parent") or "").strip() or None
    tracks = body.get("tracks", True) is not False

    def job(progress, should_stop):
        return quick_transfer(dataset, sources, runs_dir() / name, parent=parent, tracks=tracks,
                              progress=progress, should_stop=should_stop)

    def done(res):
        n = len((res or {}).get("sources", {})) or len(sources)
        note("success", f"Quick transfer finished: {n} source{'s' if n != 1 else ''} labeled (no training)",
             "A preview: check every frame in the annotator (Review in annotator)",
             action={"type": "folder", "path": str((runs_dir() / name / "labels").resolve()), "label": "Open folder"})

    return {"job": start_job("quick", job, done, "Quick transfer failed"), "run": name}


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
    folder = home() / name
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

    def done(res):
        note("success", f"Created review project {name}", f"{res['boxes']} boxes imported to check", name, open_action(name))

    return {"job": start_job("review", job, done, f"Could not create review project {name}"), "name": name}


# ---- notifications, trash, folders ----------------------------------------------------------
@app.get("/api/notifications")
def get_notifications() -> dict:
    return notes().snapshot()


@app.post("/api/notifications/read")
def read_notifications() -> dict:
    notes().mark_read()
    return {"ok": True}


@app.post("/api/notifications/clear")
def clear_notifications() -> dict:
    notes().clear()
    return {"ok": True}


def trash_dir() -> Path:
    return home() / "_trash"


@app.post("/api/projects/{name}/trash")
def trash_project(name: str) -> dict:
    """Move a project to <home>/_trash (recoverable from the notification or the Trash list)."""
    folder = project_dir(name)
    sess = state["sessions"].pop(str(folder), None)
    if sess:
        sess.p.db.close()
    entry = f"{name}__{time.strftime('%Y%m%d_%H%M%S')}"
    trash_dir().mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(folder), str(trash_dir() / entry))
    except OSError as e:
        raise HTTPException(409, f"Could not move {name} to the trash ({e}). Close other programs using it and try again.")
    note("warning", f"Moved {name} to the trash", "Restore it from here or from Trash on the start screen",
         action={"type": "restore", "entry": entry, "label": "Restore"})
    return {"entry": entry}


@app.get("/api/trash")
def list_trash() -> list:
    d = trash_dir()
    items = [e for e in d.iterdir() if (e / "project.json").exists()] if d.exists() else []
    return [{"entry": e.name, "name": e.name.rsplit("__", 1)[0], "trashed": e.stat().st_mtime}
            for e in sorted(items, key=lambda e: -e.stat().st_mtime)]


@app.post("/api/trash/{entry}/restore")
def restore_project(entry: str) -> dict:
    src = (trash_dir() / entry).resolve()
    if src.parent != trash_dir().resolve() or not (src / "project.json").exists():
        raise HTTPException(404, "That project is no longer in the trash")
    name = entry.rsplit("__", 1)[0]
    target, k = home() / name, 2
    while target.exists():
        target, k = home() / f"{name}_{k}", k + 1
    shutil.move(str(src), str(target))
    note("success", f"Restored {target.name}", "", target.name, open_action(target.name))
    return {"name": target.name}


@app.post("/api/open-folder")
def open_folder(body: dict = Body(...)) -> dict:
    """Show a folder in the file manager; only folders inside the projects folder."""
    path = Path(body.get("path", "")).resolve()
    h = home().resolve()
    if not path.is_dir() or (path != h and h not in path.parents):
        raise HTTPException(400, "Only folders inside the projects folder can be opened")
    if os.name == "nt":
        os.startfile(str(path))                                     # noqa: S606 (local desktop app)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    return {"ok": True}


# ---- Rivet, the helper (engine/assistant.py) --------------------------------------------------
@app.get("/api/assistant")
def assistant_info() -> dict:
    return assistant.info()


@app.post("/api/assistant/key")
def assistant_key(body: dict = Body(...)) -> dict:
    """Save (or with an empty key, remove) the Gemini key in ~/.partlabeler; the key never comes back out."""
    try:
        assistant.save_key(body.get("key", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not check the key with Gemini ({type(e).__name__}). Check the internet connection.")
    return assistant.info()


@app.post("/api/assistant/chat")
def assistant_chat(body: dict = Body(...)):
    """One answer, streamed as JSON lines: {"model"} or {"offline"}, {"text"}..., {"done"} (or {"error"})."""
    events = assistant.reply(body.get("messages"), body.get("context"), body.get("actions"))
    return StreamingResponse((json.dumps(e) + "\n" for e in events), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---- entry points -------------------------------------------------------------------------
def serve(home: Path = Path("projects"), port: int = 8765, open_browser: bool = True, project: str | None = None) -> None:
    state["home"] = Path(home)
    state["home"].mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}/" + (f"p/{project}" if project else "")
    print(f"PartLabeler: {url}   (projects in {state['home'].resolve()}; Ctrl+C to quit)", flush=True)
    if not accounts().users():
        print("First start: the page asks you to create your account.", flush=True)
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
