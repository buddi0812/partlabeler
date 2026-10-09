"""The integration API (v1): what a test suite or a production system needs to use PartLabeler as a service, at
stable paths under /api/v1, JSON in and out. docs/INTEGRATION.md is the contract; /docs shows the schemas.

    GET  /api/v1/health                                   up and which version (no token: for deploy probes)
    POST /api/v1/annotate                                 one image + object names -> parts (stateless)
    POST /api/v1/projects                                 create {name, type, labels}
    GET  /api/v1/projects/{name}                          the project, its tasks and progress
    POST /api/v1/projects/{name}/uploads                  a video or images (multipart) -> a new task (a job)
    POST /api/v1/projects/{name}/annotate-text            SAM 3 by text on a frame / rest of a task / task / project
    GET  /api/v1/projects/{name}/status                   busy, frames per status, frames to check
    GET  /api/v1/projects/{name}/annotations              parts as JSON (task, item or items)
    POST /api/v1/projects/{name}/export                   a dataset export (a job)
    GET  /api/v1/projects/{name}/exports/{job}.zip        download a finished export
    GET  /api/v1/jobs/{job}    POST /api/v1/jobs/{job}/stop
    GET  /api/v1/tools         POST /api/v1/tools/{name}  every other tool (the same as /api/tools and MCP)
    GET  /api/v1/models                                   exports for hawk-api's deploy picker - see below

Every call but health needs an API token (Settings → API access, or `partlabeler account token NAME`) as
`Authorization: Bearer <token>`. Network mode (`partlabeler app --host 0.0.0.0`): requests from other machines, or
through a proxy (X-Forwarded-For), reach only /api/v1, never the pages, sign-in or file browser, and the tools that
read or write paths on this computer (add_tasks, import_backup, frames_to_video) are refused there: send files with
uploads instead.

/api/v1/models is the one exception to all of the above: a same-box service (hawk-api, hawk-ai-internal §12.3)
calls it with header `X-Hawk-Token` matching app.json "models_token" instead of an account token, and it refuses
a remote caller even with the right token (its "onnx_path" is an absolute path on this computer).
"""
import asyncio
import base64
import io
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from PIL import Image
from pydantic import BaseModel, Field

from ui import mcp

H = None                                                  # the host module (ui/host_fastapi.py), set by mount()
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
LOCAL_ONLY_TOOLS = {"add_tasks", "import_backup", "frames_to_video"}      # they take paths on this computer
# Team server: what other machines may not do (it acts on this computer's desktop or setup, or makes accounts).
TEAM_LOCAL_ONLY = ("/api/auth/signup", "/api/open-folder", "/api/assistant/key", "/api/me/projects-folder")
MAX_IMAGE = 40 * 2**20                                    # bytes, one image sent to /annotate
router = APIRouter(prefix="/api/v1", tags=["integration v1"])


def remote(request: Request) -> bool:
    """From another machine, or relayed by a proxy (whose own address is this computer)."""
    client = request.client.host if request.client else ""
    return client not in LOOPBACK or any(h in request.headers for h in ("x-forwarded-for", "forwarded"))


def tool(name: str, args: dict, request: Request | None = None):
    """Run a tool as the caller; its errors as HTTP statuses (404 unknown, 409 busy, 400 the rest)."""
    if request is not None and remote(request) and name in LOCAL_ONLY_TOOLS:
        raise HTTPException(403, f"{name} reads paths on the server: from another machine send files to "
                                 "/api/v1/projects/{name}/uploads")
    try:
        out, img = mcp.call(name, {k: v for k, v in args.items() if v is not None})
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(404, mcp.error_text(e))
    except RuntimeError as e:
        raise HTTPException(409, mcp.error_text(e))
    except Exception as e:
        raise HTTPException(400, mcp.error_text(e))
    if img is not None:
        out["image"] = "data:image/jpeg;base64," + mcp._content({}, img)[1]["data"]
    return out


# ---- request bodies (they also document the API in /docs) ----------------------------------------------------
class ProjectIn(BaseModel):
    name: str
    type: str = Field("detect", description="detect (boxes), segment (outlines) or classify")
    labels: list = Field(default_factory=list, description="Label names, or {name, color}")


class AnnotateTextIn(BaseModel):
    objects: list[str] = Field(..., description="One phrase per kind of object: 'car', 'truck cabin', 'wheel'")
    labels: list[str] | None = Field(None, description="Label per phrase, same order (default: the phrase)")
    scope: str = Field("frame", description="frame | rest (to the end of its task) | task | project")
    item: int | None = Field(None, description="Frame number across the project; or task + frame")
    task: int | None = None
    frame: int | None = Field(None, description="Frame within the task, from 1")
    threshold: float = Field(0.5, ge=0, le=1, description="SAM 3 sureness")
    include_confirmed: bool = False
    wait_s: float = Field(120, ge=0, description="Seconds to wait; the job keeps running after (poll status)")


class ExportIn(BaseModel):
    format: str = Field(..., description="yolo, coco, cvat, voc, labelstudio (classify: folders, csv)")
    tasks: list[int] | None = None
    confirmed_only: bool = False
    save_images: bool = True
    wait_s: float = Field(0, ge=0)


# ---- endpoints -------------------------------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    from engine import update
    try:
        version = (update.current()["sha"] or "")[:7]
    except Exception:
        version = ""
    return {"status": "ok", "api": "v1", "version": version}


@router.get("/models")
def models_for_deploy(request: Request) -> dict:
    """Finished exports, for hawk-api's deploy picker (hawk-ai-internal §12.3) - the one /api/v1 route that
    answers to a service key (app.json "models_token" as header X-Hawk-Token) instead of an account token, and
    stays local-only even with the right key: onnx_path is an absolute path on this computer."""
    import hmac
    from engine.accounts import app_config
    token = app_config().get("models_token")
    if not token or not hmac.compare_digest(request.headers.get("x-hawk-token", ""), token):
        raise HTTPException(403, "Missing or wrong X-Hawk-Token")
    if remote(request):
        raise HTTPException(403, "This route is local-only (127.0.0.1), even with a valid token")
    return {"models": H.models_for_deploy()}


def _image(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE:
        raise HTTPException(413, f"image over {MAX_IMAGE // 2**20} MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image (send JPEG, PNG, WebP…)")


def detect(img: Image.Image, objects, threshold: float = 0.5, outlines: bool = False) -> dict:
    """Parts of each named kind of object on one image: [{label, box, score, outline?}], surest first per kind."""
    from engine import api, textlabel
    from engine import masks as M
    prompts = textlabel.parse(objects)
    if not prompts:
        raise HTTPException(400, "give the objects to find, e.g. car, truck, wheel")
    t0 = time.perf_counter()
    finder = api.model("concept")
    with api._LOCKS["concept"]:
        hits = finder.detect_texts(img, [t for t, _ in prompts], threshold, masks=outlines)
    parts = []
    for text, label in prompts:
        for box, score, mask in sorted(hits.get(text, []), key=lambda h: -h[1]):
            d = {"label": label, "box": [round(float(v), 1) for v in box], "score": round(float(score), 4)}
            if outlines:
                poly = M.single_polygon(mask) if mask is not None and mask.any() else None
                d["outline"] = [] if poly is None else [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
            parts.append(d)
    return {"width": img.width, "height": img.height, "parts": parts, "ms": round(1000 * (time.perf_counter() - t0))}


@router.post("/annotate")
async def annotate(request: Request):
    """One image and the objects to find → parts. JSON {image: base64 or data URL, objects: [...] or "car, wheel",
    threshold?, outlines?} or multipart (file, objects, threshold, outlines). Nothing is stored."""
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        upload = form.get("file") or form.get("image")
        if upload is None or isinstance(upload, str):
            raise HTTPException(400, "send the image as the 'file' field")
        data, objects = await upload.read(), form.getlist("objects")
        objects = objects[0] if len(objects) == 1 else objects
        threshold, outlines = float(form.get("threshold") or 0.5), str(form.get("outlines", "")).lower() in ("1", "true", "yes")
    else:
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "send JSON {image, objects} or a multipart form")
        raw = str(body.get("image") or "")
        try:
            data = base64.b64decode(raw.split(",", 1)[1] if raw.startswith("data:") else raw, validate=True)
        except Exception:
            raise HTTPException(400, "image must be base64 (or a data URL)")
        objects, threshold, outlines = body.get("objects"), float(body.get("threshold") or 0.5), bool(body.get("outlines"))
    img = _image(data)
    return await asyncio.to_thread(detect, img, objects, threshold, outlines)


@router.post("/projects")
def create_project(body: ProjectIn, request: Request):
    return tool("create_project", body.model_dump(), request)


@router.get("/projects/{name}")
def get_project(name: str, request: Request):
    return tool("get_project", {"project": name}, request)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}


@router.post("/projects/{name}/uploads")
async def upload(name: str, request: Request):
    """Multipart: one video (`file`) or images (`files`, any number) become one new task; optional `name`,
    `every` (keep every Nth video frame), `subset`. Answers the add_tasks job (poll /api/v1/jobs/{job})."""
    from engine.project import IMAGE_EXTS
    H.get_project(name)                                   # 404 before any file is written
    form = await request.form()
    files = [f for f in form.getlist("file") + form.getlist("files") if not isinstance(f, str)]
    if not files:
        raise HTTPException(400, "send a video as 'file', or images as 'files'")
    kinds = {Path(f.filename or "").suffix.lower() for f in files}
    if kinds <= VIDEO_EXTS and len(files) == 1:
        video = True
    elif kinds <= IMAGE_EXTS:
        video = False
    else:
        raise HTTPException(400, "one video (" + ", ".join(sorted(VIDEO_EXTS)) + ") or images ("
                            + ", ".join(sorted(IMAGE_EXTS)) + ")")
    folder = H.home() / "_uploads" / re.sub(r"[^\w.-]", "_", name) / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for f in files:
            safe = re.sub(r"[^\w.-]", "_", Path(f.filename or "upload").name)[-120:] or "upload"
            with open(folder / safe, "wb") as out:
                while chunk := await f.read(1 << 20):
                    out.write(chunk)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    source = str(next(folder.iterdir())) if video else str(folder)
    args = {"project": name, "sources": [source], "wait_s": 0,
            **{k: form.get(k) for k in ("name", "subset") if form.get(k)},
            **({"every": int(form.get("every"))} if form.get("every") else {})}
    return await asyncio.to_thread(tool, "add_tasks", args)              # the server's own path: always allowed


@router.post("/projects/{name}/annotate-text")
async def annotate_text(name: str, body: AnnotateTextIn, request: Request):
    return await asyncio.to_thread(tool, "annotate_text", {"project": name, **body.model_dump()}, request)


@router.get("/projects/{name}/status")
def status(name: str, request: Request, task: int | None = None):
    return tool("status", {"project": name, "task": task}, request)


@router.get("/projects/{name}/annotations")
def annotations(name: str, request: Request, task: int | None = None, item: int | None = None,
                items: str | None = None, outlines: bool = False, only_labeled: bool = True):
    """Parts per frame: a whole task (up to 500 frames per call), one item, or items=1,2,3."""
    some = [int(x) for x in items.split(",") if x.strip()] if items else None
    return tool("get_annotations", {"project": name, "task": task, "item": item, "items": some, "outlines": outlines,
                                    "only_labeled": only_labeled}, request)


@router.post("/projects/{name}/export")
async def export(name: str, body: ExportIn, request: Request):
    return await asyncio.to_thread(tool, "export", {"project": name, **body.model_dump()}, request)


@router.get("/projects/{name}/exports/{job}.zip")
def download_export(name: str, job: str):
    """A finished export of this project as one zip."""
    j = H.state["jobs"].get(job)
    if not j or j.get("kind") != "export" or not j.get("finished") or j.get("error") or not j.get("result"):
        raise HTTPException(404, "no finished export with that job id")
    res = j["result"]
    src = Path(res.get("folder") or Path(res["file"]).parent).resolve()
    proj = (H.home() / name).resolve()
    if proj not in src.parents and src != proj:
        raise HTTPException(404, "that export belongs to another project")
    tmp = Path(tempfile.mkdtemp(prefix="pl_export_"))
    zipped = shutil.make_archive(str(tmp / src.name), "zip", src)
    return FileResponse(zipped, filename=f"{name}_{src.name}.zip", media_type="application/zip",
                        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))   # the zip goes once sent


@router.get("/jobs/{job}")
def get_job(job: str):
    if job not in H.state["jobs"]:
        raise HTTPException(404, "no such job")
    return mcp._job(job)


@router.post("/jobs/{job}/stop")
def stop_job(job: str):
    if job not in H.state["jobs"]:
        raise HTTPException(404, "no such job")
    H.state["jobs"][job]["stop"] = True
    return {"ok": True}


@router.get("/tools")
def list_tools(request: Request):
    rem = remote(request)
    return {"guide": mcp.GUIDE, "tools": [{"name": n, "description": d, "input": s}
                                          for n, (_, d, s) in mcp.TOOLS.items() if not (rem and n in LOCAL_ONLY_TOOLS)]}


@router.post("/tools/{name}")
async def run_tool(name: str, request: Request, body: dict | None = None):
    return await asyncio.to_thread(tool, name, body or {}, request)


def mount(app, host) -> None:
    """Add /api/v1 and the network guard to the app (the signed_in middleware checks the token first)."""
    global H
    H = host
    app.include_router(router)

    @app.middleware("http")
    async def network_guard(request: Request, call_next):
        """Other machines (or anything relayed by a proxy) reach only the integration API; on a team server (serve
        team=True) the whole app, signed in, except what acts on this computer itself or makes accounts."""
        if remote(request):
            path = request.url.path
            if H.state.get("team"):
                if path.startswith(TEAM_LOCAL_ONLY) or (path == "/api/update" and request.method != "GET"):
                    return JSONResponse({"detail": "Only on the server itself: ask the person who runs PartLabeler "
                                                   "there (accounts, updates, folders and settings of that computer)"},
                                        status_code=403)
            elif not path.startswith("/api/v1/"):
                return JSONResponse({"detail": "From another machine only the integration API (/api/v1) is open"},
                                    status_code=403)
        return await call_next(request)
