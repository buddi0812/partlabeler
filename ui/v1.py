"""The integration API (v1): what a test suite or a production system needs to use PartLabeler as a service, at
stable paths under /api/v1, JSON in and out. docs/INTEGRATION.md is the contract; /docs shows the schemas.

    GET  /api/v1/health                                   up and which version (no token: for deploy probes)
    POST /api/v1/annotate                                 one image + object names -> parts (stateless)
    POST /api/v1/projects                                 create {name, type, labels}
    POST /api/v1/projects/import                          a project handed over: images + seed boxes, one per source
    POST /api/v1/projects/{name}/train  GET /api/v1/train/queue  GET /api/v1/train/runs/{run}   the training queue
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
a remote caller even with the right token (its "onnx_path" is an absolute path on this computer). POST
/api/v1/projects/import takes the same service key (local only) as well as an account token; with the key, the
project goes into the app's default projects folder.
"""
import asyncio
import base64
import io
import json
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
from typing import Literal

from pydantic import BaseModel, Field

from ui import mcp, trainq

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


def service_key(request: Request) -> None:
    """hawk-api's service key: header X-Hawk-Token matching app.json "models_token", from this computer only."""
    import hmac
    from engine.accounts import app_config
    token = app_config().get("models_token")
    if not token or not hmac.compare_digest(request.headers.get("x-hawk-token", ""), token):
        raise HTTPException(403, "Missing or wrong X-Hawk-Token")
    if remote(request):
        raise HTTPException(403, "This route is local-only (127.0.0.1), even with a valid token")


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
    service_key(request)
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


class ImportImage(BaseModel):
    name: str
    jpeg_b64: str
    cls: str | None = Field(None, alias="class", description="classify: the image's class (one of `classes`)")


class ImportBox(BaseModel):
    image: str = Field(..., description="The name of one of the images")
    cls: str = Field(..., alias="class")
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    w: float = Field(..., gt=0, le=1)
    h: float = Field(..., gt=0, le=1)


class ImportVideo(BaseModel):
    name: str = Field(..., min_length=1, max_length=120, description="The recording's id: added once per project")
    task: str | None = Field(None, max_length=120, description="The task's name (default: `name`)")
    path: str = Field(..., description="The recording's file on this computer, under hawk_media_dir's recorded/ or golden/")
    every: int = Field(1, ge=1, description="Keep every Nth frame (Hawk-AI asks for 3 per second of the recording)")
    crop: list[float] | None = Field(None, description="[x, y, w, h] shares of the frame: the camera's work area")
    camera_id: str | None = None


HAWK_VIDEO_DIRS = ("recorded", "golden")                 # Hawk-AI's recorded cycles and studio runs, nothing else


def hawk_video(path: str) -> Path:
    """A recording Hawk-AI hands over: a video file in its media folder's recorded/ or golden/ (app.json
    "hawk_media_dir", written by its installer). Anything else is refused, before anything is stored."""
    from engine.accounts import app_config
    root = app_config().get("hawk_media_dir")
    p = Path(path).resolve()
    if root and p.is_file() and p.suffix.lower() in VIDEO_EXTS and \
            any(p.is_relative_to((Path(root) / d).resolve()) for d in HAWK_VIDEO_DIRS):
        return p
    raise HTTPException(400, f"{Path(path).name!r} is not one of this box's recordings")


class ImportSource(BaseModel):
    station: str
    demo_version: int


class ImportIn(BaseModel):
    name: str
    type: Literal["detect", "segment", "classify"] = "detect"
    source: ImportSource
    classes: list[str] = Field(..., min_length=1)
    images: list[ImportImage] = Field(default_factory=list, max_length=2000)
    videos: list[ImportVideo] = Field(default_factory=list, max_length=50, description="Recordings, each a video task (Track works there)")
    boxes: list[ImportBox] = Field(default_factory=list, description="Seed labels: x, y, w, h shares of the image")


@router.post("/projects/import")
def import_project(body: ImportIn, request: Request):
    """A labeling project handed over from another system (Hawk-AI's SOP studio, hawk-ai-internal app-015-plan (b)):
    its images become one new task, its boxes parts to check ("imported"). The same source (station + demo
    version) always lands in the same project: a second call adds the images it hasn't seen, never a second project."""
    if H.USER.get() is None:                             # no account: hawk-api's service key, into the default folder
        service_key(request)
    key = {**body.source.model_dump(), "type": body.type}                  # one project per station + version + type
    same = lambda src: isinstance(src, dict) and {"type": "detect", **src} == key      # noqa: E731 (older: detect)
    name = next((f.parent.name for f in sorted(H.home().glob("*/project.json"))
                 if same(json.loads(f.read_text(encoding="utf-8")).get("import_source"))), None)
    known = H.project(name).classes if name else []
    ok = lambda c: c in body.classes or c in known      # noqa: E731
    if body.type == "classify" and body.boxes:
        raise HTTPException(400, "a classify project takes each image's `class`, not boxes")
    if not all(ok(b.cls) for b in body.boxes) or not all(im.cls is None or ok(im.cls) for im in body.images):
        raise HTTPException(400, "every box's and image's class must be one of `classes`")
    from engine.recordings import valid_crop
    vseen = set(H.project(name).meta.get("import_videos", [])) if name else set()
    videos = [(v, hawk_video(v.path)) for v in body.videos if v.name not in vseen]   # checked before anything is stored
    if any(v.crop is not None and not valid_crop(v.crop) for v, _ in videos):
        raise HTTPException(400, "a video's crop must be [x, y, w, h] shares inside the frame")
    seen = set(H.project(name).meta.get("import_images", [])) if name else set()
    stems, new = set(), []                                # checked before anything is stored
    for im in body.images:
        stem = re.sub(r"[^\w.-]", "_", Path(im.name).stem)[-100:] or "image"
        if im.name in seen or stem in stems:
            continue
        try:
            data = base64.b64decode(im.jpeg_b64.split(",", 1)[-1], validate=True)
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise HTTPException(400, f"image {im.name!r} is not a base64 JPEG")
        stems.add(stem)
        new.append((im.name, stem, data, im.cls))
    if name is None:
        base = re.sub(r'[\\/:*?"<>|]', "_", body.name).strip(" .") or "import"
        name = next(n for n in (base if i == 1 else f"{base} {i}" for i in range(1, 1000)) if not (H.home() / n).exists())
        tool("create_project", {"name": name, "type": body.type, "labels": body.classes}, request)
    p = H.project(name)
    p.set_meta(import_source=key)
    missing = [c for c in dict.fromkeys(body.classes) if c not in p.classes]
    if missing:
        tool("set_labels", {"project": name, "labels": p.classes + missing}, request)
    out = {"project_id": name, "url": f"{str(request.base_url).rstrip('/')}/p/{name}", "images_added": len(new), "boxes_added": 0,
           "videos_added": 0}
    for v, path in videos:                                # one task each, through the app's own add-task path
        opts = {"sources": [{"path": str(path), "camera_id": v.camera_id, "crop": v.crop}], "name": v.task or v.name,
                "every": v.every, "frames": "jpg" if v.crop else None}   # a crop needs stored frames
        job = tool("get_job", {"job": H.create_tasks(name, opts)["job"], "wait_s": 600})
        if not job["finished"] or job["error"]:
            raise HTTPException(500, f"adding {v.name} did not finish: {job['error'] or job['text']}")
        vseen.add(v.name)
        H.project(name).set_meta(import_videos=sorted(vseen))
        out["videos_added"] += 1
    if not new:
        return out
    folder = H.home() / "_uploads" / re.sub(r"[^\w.-]", "_", name) / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    labels = folder.parent / (folder.name + "_labels")
    folder.mkdir(parents=True)
    labels.mkdir()
    lines: dict[str, list[str]] = {}
    by_name = {n: s for n, s, *_ in new}
    for b in body.boxes:
        if b.image in by_name:
            w, h = min(b.w, 1 - b.x), min(b.h, 1 - b.y)
            lines.setdefault(by_name[b.image], []).append(f"{p.classes.index(b.cls)} {b.x + w / 2:.6f} {b.y + h / 2:.6f} {w:.6f} {h:.6f}")
    for _, stem, data, _ in new:
        (folder / f"{stem}.jpg").write_bytes(data)
    for stem, ls in lines.items():
        (labels / f"{stem}.txt").write_text("\n".join(ls) + "\n", encoding="utf-8")
    task = f"{body.source.station} v{body.source.demo_version}"
    job = tool("add_tasks", {"project": name, "sources": [str(folder)], "name": task, "wait_s": 600})
    if not job["finished"] or job["error"]:
        raise HTTPException(500, f"adding the images did not finish: {job['error'] or job['text']}")
    p = H.project(name)
    tid = max(t["id"] for t in p.sources)
    items = range(*p.ranges[tid])
    out["boxes_added"] = p.import_yolo(labels, items=items)["boxes"]
    shutil.rmtree(labels, ignore_errors=True)
    for c in {c for *_, c in new if c}:                   # classify: each image's class, as a class to check
        p.set_tags([k for k in (p._match(stem, items) for _, stem, _, c2 in new if c2 == c) if k is not None],
                   p.classes.index(c), source="imported")
    out["classes_added"] = sum(1 for *_, c in new if c)
    p.set_meta(import_images=sorted(seen | {n for n, *_ in new}))
    H.refresh(name)
    return {**out, "task": tid}


class TrainIn(BaseModel):
    confirmed_only: bool = Field(True, description="Teach learns from the confirmed frames only (the one choice)")
    size: str | None = Field(None, description="Default: small (detect) or seg-small (segment)")
    epochs: int | None = Field(None, ge=1)
    resolution: int | None = Field(None, ge=320)
    augmentation: Literal["fixed", "worn", "off"] = Field("fixed", description="How the camera is mounted: fixed (today's "
                                                          "strong set), worn (more turning, tilt and motion blur), off")
    compare_to: str | None = Field(None, description="An earlier run of this project (its export id, e.g. the model a "
                                   "station runs now): the report compares both on the same check frames")


@router.post("/projects/{name}/train")
def train(name: str, body: TrainIn, request: Request):
    """Send a project's confirmed frames to the training queue (ui/trainq.py): Teach, then the export for production.
    One run at a time; GET /api/v1/train/runs/{run} follows it."""
    if H.USER.get() is None:
        service_key(request)
    if not body.confirmed_only:
        raise HTTPException(400, "Teach learns from confirmed frames only")
    H.get_project(name)                                  # 404
    p = H.project(name)
    size = "classify" if p.task == "classify" else body.size or ("seg-small" if p.task == "segment" else "small")
    from engine.detector import SIZES, check_resolution
    if size != "classify" and size not in SIZES:
        raise HTTPException(400, f"size: one of {', '.join(SIZES)}")
    if body.resolution and size != "classify":
        try:
            check_resolution(body.resolution, size)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
    frames = p.statuses().count(4)
    if not frames:
        raise HTTPException(400, "no confirmed frames yet: confirm the frames whose labels are right, then send it")
    from engine.detector import MOUNTS
    r = trainq.enqueue(name, H.USER.get(), size, body.epochs, body.resolution, frames=frames, aug=MOUNTS[body.augmentation],
                         compare_to=body.compare_to)
    return {"run": r["run"], "position": r["position"], "state": r["state"], "minutes": r["minutes"]}


class PrelabelIn(BaseModel):
    run: str = Field(description="A finished run of this project (its export id): its model labels the frames")
    tasks: list[str] | None = Field(None, description="Task names; default: the tasks with no confirmed frame yet "
                                    "(new recordings)")


PRELABEL_GB = 1.5                                        # graphics card memory a model needs to label frames


@router.post("/projects/{name}/prelabel")
def prelabel(name: str, body: PrelabelIn, request: Request):
    """Pre-label a project's new recordings with one of its finished runs (Hawk-AI's "Pre-label with the last model",
    2026-10-24): the model's parts, to check, on every frame nobody labeled or confirmed (the label_tasks job, one
    undo step). 422: the project has parts the model never learned (label those by hand); 409: something runs in
    the project, or the graphics card has no room now."""
    if H.USER.get() is None:
        service_key(request)
    H.get_project(name)                                  # 404
    run = H.runs_dir() / H.safe_name(body.run)
    try:
        st = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(404, f"no finished run {body.run!r}") from None
    if "checkpoint" not in st:
        raise HTTPException(400, f"{body.run} has no model that labels parts")
    sess = H.session(name)
    p = sess.p
    if new := [c for c in p.classes if c not in st["classes"]]:
        raise HTTPException(422, f"the model never learned {', '.join(new)}: label those by hand")
    from engine import hw
    free = hw.free_memory_gb()
    if free is not None and free < PRELABEL_GB:
        raise HTTPException(409, "the graphics card has no room now; try again when it is free")
    tids = [t["id"] for t in p.sources if (t["name"] in body.tasks if body.tasks is not None
                                           else not any(p.is_reviewed(k) for k in range(*p.ranges[t["id"]])))]
    if not tids:
        return {"tasks": 0, "state": "nothing"}
    try:
        sess.call({"type": "label_tasks", "run": str(run), "tasks": tids}, wait=0)
    except RuntimeError as e:                            # another job runs in the project
        raise HTTPException(409, str(e)) from None
    return {"tasks": len(tids), "state": "started"}


@router.get("/projects/{name}/train/estimate")
def train_estimate(name: str, request: Request, size: str | None = None, epochs: int | None = None,
                   resolution: int | None = None):
    """Before sending it: how long a run of this project should take (minutes as [low, high]) and the graphics
    card memory it needs (need_gb) against what is free now (free_gb)."""
    if H.USER.get() is None:
        service_key(request)
    H.get_project(name)
    p = H.project(name)
    size = size or ("classify" if p.task == "classify" else "seg-small" if p.task == "segment" else "small")
    est = trainq.estimate(p.statuses().count(4), size, epochs, resolution)
    _, need, free = trainq.room(size, est["resolution"])
    return {**est, "need_gb": need, "free_gb": free}


@router.get("/train/queue")
def train_queue(request: Request):
    if H.USER.get() is None:
        service_key(request)
    return {"runs": trainq.queue()}


@router.post("/train/runs/{run}/cancel")
def train_cancel(run: str, request: Request):
    """Cancel a run: queued or waiting ends now; running stops at Teach's next safe point. Ends as "cancelled"."""
    if H.USER.get() is None:
        service_key(request)
    r = trainq.cancel(run)
    if r is None:
        raise HTTPException(404, f"No training run {run}")
    return r


@router.get("/train/runs/{run}")
def train_run(run: str, request: Request):
    if H.USER.get() is None:
        service_key(request)
    r = trainq.get(run)
    if r is None:
        raise HTTPException(404, f"No training run {run}")
    return r


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
    H = trainq.H = host
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
