# PartLabeler integration API (v1)

This is the contract for using PartLabeler from other software: a test framework, a data pipeline or a production
system. It also covers how to design the production side so the two work together smoothly: deployment, auth, jobs,
errors, retries, capacity and testing.

- Base path: `/api/v1` (for example `http://127.0.0.1:8765/api/v1`). JSON in and out, UTF-8.
- Ready-made client: [`client/partlabeler_client.py`](../client/partlabeler_client.py), one file with only the Python
  standard library. Copy it into your code base.
- Machine-readable schemas: `/openapi.json` and `/docs` on the server (signed in, from the server itself).
- Everything in the annotator (tracking, Suggest, Teach & Transfer, …) is also reachable as a *tool*
  (`/api/v1/tools`), the same tools AI apps use over MCP (see the user guide, "Use PartLabeler from AI apps and
  scripts").

## 1. What it is for

| You want to… | Use |
|---|---|
| Get boxes (and outlines) of named objects on one image now, store nothing | `POST /annotate` |
| Build a dataset from videos or images with a person reviewing it | `POST /projects` → `POST /projects/{name}/uploads` → `POST /projects/{name}/annotate-text` → people check and confirm in the app → `POST /projects/{name}/export` → `GET …/exports/{job}.zip` |
| Read the current labels of a project (to train, evaluate or audit) | `GET /projects/{name}/annotations` |
| Watch progress or wait for work | `GET /projects/{name}/status`, `GET /jobs/{job}` |
| Anything else the app can do | `GET /tools`, `POST /tools/{name}` |

Annotation needs no training. Objects are found by **SAM 3** from short phrases ("car", "truck cabin", "wheel",
"license plate", "forklift", …). One or two plain words per kind of object work best. Results are *proposals*: in a
project they are stored as parts **to check**, and nothing counts as confirmed until a person confirms the frame.

## 2. Architecture: two ways to deploy

```
 A. Sidecar (recommended)                       B. Network mode
 ┌──────────── one machine ─────────────┐       ┌── your service ──┐   HTTPS   ┌──── GPU machine ────────────────┐
 │ your service ──HTTP 127.0.0.1:8765──▶│       │  client + token  │ ────────▶ │ reverse proxy (TLS, limits)      │
 │               PartLabeler (GPU)       │       └──────────────────┘           │   └▶ PartLabeler 127.0.0.1:8765  │
 └───────────────────────────────────────┘                                      │      or --host 0.0.0.0           │
                                                                                └──────────────────────────────────┘
```

**A. Sidecar on the same machine.** Run PartLabeler next to your service and call `http://127.0.0.1:8765/api/v1`.
Nothing is exposed to the network, files can also be passed by path (`add_tasks` tool), and latency is lowest.

**B. Network mode.** Start the app with `partlabeler app --host 0.0.0.0` (or `python -m ui.host_fastapi --host
0.0.0.0 --no-browser`). Requests from other machines, and any request carrying `X-Forwarded-For` or `Forwarded`
(that is, relayed by a proxy), are restricted:

- only `/api/v1/...` is reachable. The app's pages, sign-in, file browser and its other routes answer **403**;
- every call except `/api/v1/health` needs a token;
- tools that read or write paths on the server (`add_tasks`, `import_backup`, `frames_to_video`) are refused (**403**)
  and hidden from `/api/v1/tools`. Send files with `POST /projects/{name}/uploads` instead.

Plain HTTP carries the token in clear text. On any shared network put a TLS reverse proxy in front: the app on
`127.0.0.1`, the proxy on the network. Example (Caddy):

```
annotate.example.internal {
    reverse_proxy 127.0.0.1:8765
    request_body { max_size 4GB }      # video uploads
}
```

Also firewall the port to the machines that need it.

**What runs where.** One PartLabeler process owns the GPU and one projects folder. Models load on first use and stay
loaded. Work on a project runs one job at a time (see §6). Scale out with more instances, each with its own GPU and
projects folder, and route each project to one instance. Two instances must never share a projects folder.

## 3. Authentication

- Every call except `GET /api/v1/health` needs `Authorization: Bearer <token>`.
- A token acts as one account and works in that account's projects folder.
- Create a token in the app under **Settings → API access**, or on the server:
  `partlabeler account token <username> --label "line-3 pipeline"`. It is printed once; only a fingerprint is stored.
- One token per service and environment. Keep it in your secret store, never in code or images.
- Rotate by creating the new token, deploying it, then **revoking** the old one in Settings. Revocation is immediate.
- Missing or wrong token: **401**. Wrong place (another machine on a non-API route, or a path tool): **403**.
- `GET /api/v1/models` is a different, narrower door, not an integration route: a service key (`X-Hawk-Token`
  matching app.json `"models_token"`) instead of a Bearer token, and local-only (127.0.0.1) even with the right
  key. It is how a same-box production system picks a PartLabeler export to deploy, not something your own
  integration should call.

## 4. Conventions

| Thing | Meaning |
|---|---|
| `box` | `[x1, y1, x2, y2]` in pixels of the full-size frame, top-left origin |
| `outline` | `[[x, y], …]` polygon in the same pixels (when asked for and available) |
| `score` | SAM 3 sureness 0–1; `null` for parts a person drew |
| `threshold` | lowest sureness kept (default 0.5); lower finds more, raise for fewer false finds |
| `item` | frame number across the whole project, from 0 |
| `task` + `frame` | a task id and the frame inside it, **from 1** (`item` and `task`+`frame` are interchangeable) |
| `source` | how a stored part was made: `manual` (a person), `tracked`, `imported` (annotate by text, imports, Transfer), `suggested` (not accepted yet) |
| `status` | per frame: `no parts`, `suggestions only`, `tracked or imported`, `has your parts`, `confirmed` |
| labels | names; each phrase of annotate-by-text becomes a label of its own name unless you map it (`labels`) |

## 5. Endpoints

All examples use `PL=http://127.0.0.1:8765/api/v1` and `-H "Authorization: Bearer $TOKEN"` (left out below for
brevity). The Python lines use the client: `pl = PartLabeler("http://127.0.0.1:8765", token=TOKEN)`.

### `GET /health`: up and which version (no token)

```json
{"status": "ok", "api": "v1", "version": "574c94c"}
```
Use it for liveness and readiness probes. `version` is the app's build; log it with your results.

### `POST /annotate`: one image → parts (stateless)

JSON: `{"image": "<base64 or data URL>", "objects": ["car", "wheel"] | "car, wheel", "threshold": 0.5, "outlines": false}`,
or multipart with `file`, `objects`, `threshold`, `outlines`. Max 40 MB per image (**413** above).

```bash
curl -s -X POST $PL/annotate -F file=@frame.jpg -F "objects=car, truck cabin, wheel" -F outlines=true
```
```json
{"width": 1920, "height": 1080, "ms": 1028,
 "parts": [{"label": "car", "box": [652.0, 442.0, 1264.0, 848.0], "score": 0.9609, "outline": [[655.0, 447.0], …]},
           {"label": "wheel", "box": [700.5, 760.0, 790.0, 845.5], "score": 0.81}]}
```
```python
parts = pl.annotate("frame.jpg", ["car", "wheel"])["parts"]
```
Parts are grouped by phrase in the order given, surest first. Nothing is stored. Use it for online pre-annotation,
quick checks and evaluations.

### `POST /projects`: create a project

```json
{"name": "yard-cams", "type": "detect", "labels": ["truck", "truck cabin", "wheel"]}
```
`type`: `detect` (boxes), `segment` (outlines; exports polygons/masks) or `classify` (one class per image). It
answers `{"name": "yard-cams"}`. It needs at least one label; a name that already exists: **400**.

### `POST /projects/import`: a project handed over with its images and seed boxes
For a system that already has the frames and some boxes (Hawk-AI's SOP studio: "Make a labeling project").
`{name, source: {station, demo_version}, classes: [...], images: [{name, jpeg_b64}], boxes: [{image, class, x, y,
w, h}]}` (box shares 0-1 of its image, top-left corner) → `{project_id, url, images_added, boxes_added, task}`.
The images become one new task; the boxes become parts to check ("imported"), never confirmed. Idempotent per
`source`: the same station and demo version always land in the same project (its name is only used the first
time); a second call adds the images it hasn't seen (by `name`) as another task and any new classes, and never
makes a second project. 400: a box whose class isn't in `classes`, or an image that isn't a base64 JPEG (nothing
is stored then). Waits until the task is added (up to 10 min).
`type` (default detect): detect | segment | classify, the project's type. One project per station + demo version
+ type. classify takes no boxes: each image may carry its own `class` (one of `classes`), stored as a class to
check; the answer then also says `classes_added`.
Auth: an account's API token as every other call (the project goes into that account's projects folder), or -
for hawk-api on the same box - header `X-Hawk-Token` matching app.json "models_token" with no account token,
from this computer only (403 from another machine or with a wrong key). With the service key the project goes
into the app's default projects folder (the service's `--home`, else app.json "home"): the one every account on
that PartLabeler sees, unless the account picked its own folder in Settings.

### `POST /projects/{name}/train`, `GET /train/queue`, `GET /train/runs/{run}`: the training queue
Sends a project's confirmed frames to Teach and then exports the model for production, one run at a time in the
order sent. Body `{confirmed_only: true, size?, epochs?, resolution?}` (size default: small, or seg-small for an
outline project; a classify project trains its own classifier, below; no confirmed frames: 400) -> `{run, position, state}`.
A run: `{run, project, state, position, progress 0-1, text, passed, knowledge_score, export_id, error, need_gb,
free_gb, queued_at, started_at, finished_at}`; `state` is queued | waiting_for_room | running | done | failed.
- `waiting_for_room`: the graphics card hasn't the memory the run needs (`need_gb`, Teach's measured peak at batch 1
  plus 1 GB of headroom) free (`free_gb`). It starts by itself, checked every 10 s, as soon as there is: a caller may
  simply wait, or make room (e.g. stop another model) when it sees the run queued.
- `passed` and `knowledge_score` (0-100, Teach's own: held-out mAP50 x 100) come from Teach's proof; `export_id` is
  the id `GET /models` lists. The queue survives a restart (a run the restart cut off is failed with that reason).
- Every run ends `done`, `failed` or `cancelled`: a caller that paused something for it can always restore it.
  `POST /train/runs/{run}/cancel`: a queued or waiting run ends at once, a running one at Teach's next safe point.
- `GET /projects/{name}/train/estimate?size&epochs&resolution` before sending: `{frames, size, epochs, resolution,
  minutes: [low, high], based_on, need_gb, free_gb}`. The minutes come from this computer's own finished runs of the
  same size once it has some, else from a typical 12 GB card; the queued run carries the same `minutes`.
- A classify project (engine/classifier.py): a head on frozen DINOv3 small features of its confirmed images with their
  class, a fifth of each class held out to prove it (passed: balanced accuracy and every class's recall >= 0.90;
  knowledge_score: balanced accuracy x 100), exported as one ONNX model, input `[1, 3, 224, 224]` RGB 0-1 (the part's
  crop cut as the training images were, stretched; normalisation inside), output `probs [1, classes]`. Needs two
  classes with two or more confirmed images each. Minutes, not hours; about 1.5 GB of graphics card memory.
- `GET /models` also says each export's `task`: detect, segment or classify.
- The batch size follows the free graphics card memory when the run starts (Teach's batch_plan, measured peaks):
  with the whole 12 GB free, a small model at 640 px trains at batch 4.
Auth: an account token, or hawk-api's `X-Hawk-Token` from this computer as for `/projects/import`.

### `GET /projects/{name}`: the project, its tasks and progress

It answers `name`, `type`, `labels`, `formats` (exports available), `frames`, and `tasks`: `[{id, name, kind,
frames, start_item, every, labeled, confirmed, video: {width, height, fps, …}, jobs: […]}]`.

### `POST /projects/{name}/uploads`: a video or images → a new task (a job)

Multipart: one video as `file` (mp4, mov, avi, mkv, m4v, webm, mpg, mpeg, wmv), or any number of images as `files`;
optional `name` (task name), `every` (keep every Nth video frame, default 5) and `subset` (Train, Validation, Test).

```bash
curl -s -X POST $PL/projects/yard-cams/uploads -F file=@gate1.mp4 -F every=5 -F name=gate1
```
```json
{"job": "a1b2c3d4", "kind": "add_tasks", "finished": false, "text": "…", "done": 0, "total": 1, "error": null, "result": null}
```
Poll `GET /jobs/{job}` until `finished` (`pl.wait(job["job"])`). Uploaded files are kept on the server under
`<projects folder>/_uploads/<project>/` because a video task reads its frames from the video. Delete them only together
with the task.

### `POST /projects/{name}/annotate-text`: find objects by text on a frame, the rest of a task, a task or the project

```json
{"objects": ["truck", "truck cabin", "wheel"], "labels": null, "scope": "task", "task": 1, "frame": 1,
 "threshold": 0.5, "include_confirmed": false, "wait_s": 0}
```

| Field | Meaning |
|---|---|
| `objects` | one phrase per kind of object |
| `labels` | optional label per phrase, same order (e.g. map "lorry" to `truck`); default: the phrase. New labels are added to the project |
| `scope` | `frame` (just that frame), `rest` (that frame to the end of its task), `task` (the whole task), `project` (every frame) |
| where | `item`, or `task` + `frame`; `task` alone means its first frame; `project` scope needs neither |
| `include_confirmed` | also annotate frames a person confirmed (default: they are left alone) |
| `wait_s` | how long the call waits (default 120 s). The work continues after the call returns |

```json
{"item": 0, "task": 1, "task_name": "gate1", "frame": 1, "name": "gate1_f000000.jpg", "scope": "task",
 "labels": ["truck", "truck cabin", "wheel"], "messages": ["Annotated 412 parts by text on 120 frames. …"], "busy": false}
```
`busy: true` means it is still running: poll `GET /projects/{name}/status` until `busy` is false
(`pl.wait_idle(name)`), and read `messages` there (`last_error`) or the parts via annotations. Properties you can rely
on:

- a part already boxed (same label, overlapping) is never added twice, so **re-running is safe** (idempotent in effect);
- the same object keeps its part `id` from frame to frame within a task;
- one call is one undo step in the app;
- parts are stored with `source: "imported"` and the frames count as *to check* until a person confirms them.

### `GET /projects/{name}/status`: busy and progress

```json
{"busy": false, "last_error": null, "undo_steps": 3,
 "tasks": [{"id": 1, "name": "gate1", "frames": 120,
            "by_status": {"no parts": 0, "suggestions only": 0, "tracked or imported": 118, "has your parts": 0, "confirmed": 2},
            "to_check": [17, 18, 64]}]}
```
`?task=1` limits it to one task.

### `GET /projects/{name}/annotations`: parts as JSON

Query: `task=1` (a whole task, up to 500 frames per call), or `item=12`, or `items=3,4,5`; `outlines=true`;
`only_labeled=false` to include frames without parts.

```json
{"frames": [{"item": 0, "task": 1, "task_name": "gate1", "frame": 1, "name": "gate1_f000000.jpg",
             "status": "tracked or imported", "confirmed": false,
             "parts": [{"id": 1, "label": "truck", "box": [102.0, 88.5, 911.0, 640.0], "source": "imported", "score": 0.93}]}],
 "truncated": false}
```
For tasks over 500 frames, page with `items=` using `start_item` and `frames` from `GET /projects/{name}`. Train only
on `confirmed: true` frames unless you accept unreviewed labels.

### `POST /projects/{name}/export` and `GET /projects/{name}/exports/{job}.zip`

```json
{"format": "coco", "tasks": [1], "confirmed_only": true, "save_images": true, "wait_s": 0}
```
`format` is one of the project's `formats`: `yolo`, `coco`, `cvat`, `voc`, `labelstudio` (classification: `folders`,
`csv`). It answers a job. When it is finished, download `GET /projects/{name}/exports/{job}.zip`
(`pl.export(name, "coco", path="out.zip")` does both). Every export is a new folder on the server.

### `GET /jobs/{job}` and `POST /jobs/{job}/stop`

```json
{"job": "a1b2c3d4", "kind": "export", "finished": true, "text": "…", "done": 120, "total": 120,
 "error": null, "result": {"folder": "…", …}}
```
`/stop` asks a server job (uploads, export, backup) to stop. Project work (annotate-text, tracking, Suggest) stops
with `POST /tools/stop {"project": name}`.

### `GET /tools` and `POST /tools/{name}`

`GET /tools` lists every tool with a description and its JSON-schema input. `POST /tools/{name}` runs one with a
JSON body, for example `POST /tools/confirm_frames {"project": "yard-cams", "items": [4, 5]}` or
`POST /tools/track {"project": "yard-cams", "task": 1, "frame": 1, "frames": 50}`. Tool errors use the status codes
below.

### `GET /models`: exports for a deploy picker (not for integrations - see §3 below)

```json
{"models": [{"id": "run_1", "name": "yard-cams", "created": "2026-10-09T18:04:11+05:30", "classes": ["bolt"],
             "crop": {"cam1": [0.1, 0.1, 0.5, 0.5]}, "parent": null, "resolution": 640, "threshold": 0.42,
             "passed": true, "knowledge_score": 94.2, "frames": {"train": 180, "held_out": 45},
             "parity": {"frames": 20, "boxes": 31, "missing": 0, "worst_px": 1.1},
             "onnx_path": "D:\\labeling\\_teach\\run_1\\export\\model.onnx"}]}
```
Newest first. This route does not take a Bearer token at all (see §3) - it exists for a same-box production
system (hawk-ai-internal §12.3) to list PartLabeler's exports for a deploy picker, never for a pipeline.

## 6. Jobs, concurrency and long work

There are two kinds of long work:

1. **Server jobs** (uploads, export, backup): answered with a `job` id; poll `GET /jobs/{job}` (every 1–2 s is plenty).
2. **Project work** (annotate-text, tracking, Suggest, Teach & Transfer): one at a time per project. The call waits up
   to `wait_s`, then returns with `busy: true` while the work continues; poll `GET /projects/{name}/status`.

A second piece of project work while one runs answers **409** ("Busy with another job…"). Wait for `busy: false`
and send it again, or stop the running one. Different projects can run in parallel, but they share the GPU: SAM 3
calls take turns on one model lock, so throughput does not grow with more parallel calls on one GPU.

A person can work in the app on the same project at the same time. Their edits and yours are both live; to avoid
surprises, run bulk annotate-text before people start checking a task, and never on frames they already confirmed
(the default).

## 7. Errors

| Status | Meaning | What to do |
|---|---|---|
| 400 | Bad input (unknown scope, empty objects, not an image, bad label) | Fix the request; `detail` says what |
| 401 | No token or a wrong/revoked token | Check the secret; do not retry |
| 403 | Not allowed from this machine (network mode) | Use `/api/v1` and uploads, or run as a sidecar |
| 404 | Unknown project, task, frame, job or tool | Do not retry |
| 409 | The project is busy with another job, or the job this call started failed | Read `detail`: if busy, wait for `busy: false` and retry |
| 413 | Image over 40 MB | Resize or re-encode |
| 422 | Body does not match the schema (wrong type, missing field) | Fix the request |
| 5xx / no connection | Server down, restarting or overloaded | Retry with backoff (below) |

Bodies are `{"detail": "…"}` in plain words. Log `detail` together with the call.

## 8. Retries, idempotency and timeouts

- **Safe to retry:** every `GET`, `POST /annotate` (stateless) and `POST …/annotate-text` (it never adds a part
  twice). The client retries GETs and `/annotate` on connection errors and 502/503/504 with backoff 1, 2, 4 s.
- **Not idempotent:** `POST /projects` (409/400 the second time, which is harmless), `POST …/uploads` (a retry makes
  a second task: before retrying, check `GET /projects/{name}` for a task with your `name`), `POST …/export` (a new
  folder each time).
- **Timeouts:** `/annotate` usually answers in about 1 s per image on a mid-range GPU (more for the first call; see
  §9). Set the client timeout to about 30 s for it and to the upload size ÷ your bandwidth plus a margin for uploads.
  For project work, call with `wait_s: 0` and poll rather than holding a connection open for minutes.
- **Polling:** 1–2 s for jobs, 2–5 s for project status. Stop polling after your own deadline and call stop.

## 9. Performance and capacity

Measured on one RTX 3060 (12 GB), 1920×1080 frames, SAM 3 in bfloat16:

| What | Time |
|---|---|
| First call after start (loads SAM 3) | ~6 s extra (plus a one-time model download on a fresh machine) |
| `POST /annotate`, 3–4 object names, boxes | ~0.8–1.0 s |
| … with outlines | ~1.2 s |
| annotate-text per frame | about the same per frame; the image is encoded once for all names |

- **Warm up** after each deploy: call `POST /annotate` once with a small image and wait for success before taking
  traffic (readiness = health ok **and** one warm-up annotate).
- The cost per frame grows a little with each object name; the image encoding dominates. Ask for all names in one call
  rather than one call per name.
- The model stays on the GPU. Do not run other GPU work (training, other models) on the same card while
  PartLabeler serves; if you must, expect slower calls rather than errors.
- For more throughput, run more instances (one GPU each) behind your own router, with each project pinned to one
  instance.

## 10. Designing the production side

**A thin client layer.** Wrap the client in your own module: one place for the base URL, token, timeouts, retries,
logging (method, path, status, `detail`, duration, server `version`) and metrics. Treat PartLabeler as a dependency
that can be down: health-check it, and degrade (skip pre-annotation, queue the work) rather than fail your own
requests.

**Pattern 1: online pre-annotation.** Your service gets an image and calls `POST /annotate` with the object names it
cares about, filters by `score`, and uses the boxes, for example to crop, count or route. Keep a per-call deadline
and a fallback path. Nothing is stored, so there is nothing to clean up.

**Pattern 2: dataset factory with human review.**
1. `POST /projects` once per dataset (detect or segment, with your label list).
2. Per batch: `POST /uploads` (video or images, a meaningful task `name` such as `site-date-camera`), wait for the job.
3. `POST /annotate-text` with `scope: "task"`, `wait_s: 0`; wait until not busy.
4. People open the task in the app, fix what is wrong and confirm frames (the frames *to check* guide them).
5. Your pipeline polls `GET /status` (or the `confirmed` counts in `GET /projects/{name}`) and, when a task is fully
   confirmed, exports with `confirmed_only: true` and downloads the zip into your training store.
6. Keep the export zip and the `version` from `/health` together as the dataset's provenance.

**Pattern 3: automated tests.** See §11.

**Naming.** Project names are folder names: use `[A-Za-z0-9_-]`. Task names should be unique within a project so
retries and lookups by name work.

**Data lifecycle.** Uploads and exports stay on the server. Plan a retention policy: back up a finished project
(`POST /tools/backup`), copy the export zips you need, then remove the project in the app (Trash) or with your own
housekeeping on the projects folder while the app is stopped.

**Privacy.** Images are processed locally on the PartLabeler machine; nothing is sent to other services.
In network mode, use TLS and restrict who can reach the port.

**Versioning.** `/api/v1` changes only by addition (new endpoints, new optional fields, new fields in answers).
Ignore unknown fields in answers. A breaking change would come as `/api/v2`, with v1 kept alongside for a transition.
The tools behind `/tools` follow the app and can gain inputs. Pin the app version you test against (`version` in
`/health`) and run your contract tests (§11) before upgrading production.

## 11. Testing your integration

**Contract tests against a throwaway instance.** Start the app on a free port with its own config and projects
folders, create an account and token, and run your tests through your client layer:

```python
# conftest.py
import os, socket, subprocess, sys, time, tempfile, pathlib, urllib.request
import pytest

PARTLABELER = pathlib.Path("/opt/partlabeler")            # where PartLabeler is installed

@pytest.fixture(scope="session")
def partlabeler():
    tmp = pathlib.Path(tempfile.mkdtemp())
    env = {**os.environ, "PARTLABELER_CONFIG": str(tmp / "cfg"), "PARTLABELER_HOME": str(tmp / "projects")}
    py = PARTLABELER / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    token = subprocess.check_output([py, "-c",                       # a test account and token in the temp config
        "from engine.accounts import Accounts; a = Accounts(); a.create('ci', 'ci-password-123');"
        "print(a.create_token('ci', 'tests'))"], cwd=PARTLABELER, env=env, text=True).strip()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    proc = subprocess.Popen([py, "-m", "ui.host_fastapi", "--port", str(port), "--no-browser"],
                            cwd=PARTLABELER, env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/api/v1/health", timeout=1); break
        except OSError:
            time.sleep(0.5)
    yield url, token
    proc.terminate()
```
```python
# test_contract.py
from partlabeler_client import PartLabeler, PartLabelerError

def test_round_trip(partlabeler, tmp_path):
    url, token = partlabeler
    pl = PartLabeler(url, token=token)
    pl.create_project("ct", labels=["car"])
    pl.wait(pl.upload("ct", "fixtures/short.mp4", every=10, name="clip")["job"])
    pl.annotate_text("ct", ["car", "wheel"], scope="task", task=1, wait_s=0)
    assert pl.wait_idle("ct")["last_error"] is None
    frames = pl.annotations("ct", task=1)["frames"]
    assert all(p["source"] == "imported" for f in frames for p in f["parts"])
    assert pl.export("ct", "coco", path=tmp_path / "ct.zip")["path"]
```

- Without a GPU, SAM 3 runs on the CPU, many times slower: for CI keep fixtures short (a few frames) and timeouts long.
- Assert on structure and invariants (fields, labels, no duplicates on re-run, 409 when busy, 401 on a bad token),
  not on exact boxes, which change with model versions.
- Keep one small set of your own images with expected objects, and track a simple recall/precision number per app
  version, so a model change that hurts your use case shows up before production.
- PartLabeler's own tests of this API are in `tests/test_v1.py`, including the client against a live server and the
  network-mode guard.

## 12. Deployment checklist

- [ ] GPU machine with the app installed (`install_windows.bat` or `pip install -e .`); models downloaded once.
- [ ] A dedicated account for the service, a token per environment in your secret store; revoke unused tokens.
- [ ] Sidecar on `127.0.0.1`, or network mode behind a TLS proxy with a firewall rule and an upload size limit.
- [ ] Readiness = `/health` ok **and** one warm-up `/annotate`.
- [ ] Client timeouts and retries as in §8; uploads checked by task name before retrying.
- [ ] Project work polled with `wait_s: 0`; 409 handled as "busy, try later".
- [ ] Only `confirmed` frames used for training unless you decided otherwise.
- [ ] Retention for uploads, exports and finished projects; backups of what you keep.
- [ ] Contract tests (§11) run against the exact app version before every upgrade; `version` logged with results.
- [ ] No other GPU workloads on the card while serving, or capacity measured with them.
