"""PartLabeler for programs: every feature as a tool, over MCP (Model Context Protocol) at /mcp for AI apps (Claude
Code, opencode, Gemini CLI, Cursor…) and as plain JSON at /api/tools/<name> for scripts (GET /api/tools lists them).

Both need an API token (Settings → API access, or `partlabeler account token NAME`) sent as
`Authorization: Bearer <token>`; the program then acts as that account, in its projects folder. Tools run the same
code as the annotator (engine/api.py `Session.call`), so changes show up live in open tabs and Ctrl+Z undoes them.
MCP here is the streamable-HTTP transport answering with plain JSON (no server-sent events, no session state), which
current clients support. Coordinates are always pixels of the full-size frame.
"""
import asyncio
import base64
import colorsys
import io
import json
import time
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, ImageFont

from engine import masks as M

H = None                                                  # the host module (ui/host_fastapi.py), set by mount()
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
STATUS = ["no parts", "suggestions only", "tracked or imported", "has your parts", "confirmed"]
TYPES = {"detect": "object detection (boxes)", "segment": "segmentation (outlines)", "classify": "classification"}
GUIDE = """PartLabeler annotates videos and image folders. A project has a type (detect: boxes, segment: outlines,
classify: one class per image), labels, and tasks (one video or image folder each, split into jobs). Frames are
numbered across the whole project ("item"); get_project gives each task's start_item and frame count, and tools also
accept task + frame (from 1). Coordinates are pixels of the full-size frame (get_frame reports width, height and the
scale of the image it returns). Typical flow: get_project → get_frame → add_part (click points or a box; SAM 3 draws
the outline) or find_parts (words) → track (carries parts through the next frames) → status (frames to check) →
fix → confirm_frames → export. Suggestions (source "suggested") are proposals until accept_suggestions. Every change
is visible live to people in the app; undo reverts the last change. Confirming a frame means a person checked it:
only confirm when the user asked you to."""


# ---- tool registry ----------------------------------------------------------------------------------
TOOLS: dict = {}


def tool(description: str, required=(), **props):
    def deco(fn):
        TOOLS[fn.__name__] = (fn, description, {"type": "object", "properties": props, "required": list(required)})
        return fn
    return deco


def S(d, **k): return {"type": "string", "description": d, **k}
def I(d, **k): return {"type": "integer", "description": d, **k}
def B(d, **k): return {"type": "boolean", "description": d, **k}
def A(d, items, **k): return {"type": "array", "description": d, "items": items, **k}


PROJECT = S("Project name (list_projects)")
ITEM = I("Frame number across the project (0-based); or give task + frame")
TASK = I("Task id (get_project)")
FRAME = I("Frame within the task, from 1 (with task)")
LABEL = S("Label name (get_project labels)")
PART = I("Part id (get_frame / get_annotations)")
BOX = A("[x1, y1, x2, y2] in full-size frame pixels", {"type": "number"}, minItems=4, maxItems=4)
WAIT = I("Seconds to wait for the result (the job keeps running after; see status / get_job)", minimum=0)
AT = {"project": PROJECT, "item": ITEM, "task": TASK, "frame": FRAME}


def call(name: str, args: dict):
    """Run a tool: (result, image or None). Raises ValueError / KeyError / RuntimeError / HTTPException on errors."""
    if name not in TOOLS:
        raise KeyError(f"no tool {name!r}")
    fn, _, schema = TOOLS[name]
    missing = [k for k in schema["required"] if args.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    out = fn(args)
    return out, out.pop("_image", None) if isinstance(out, dict) else None


def error_text(e: Exception) -> str:
    return str(e.detail) if isinstance(e, HTTPException) else (e.args[0] if e.args and isinstance(e, KeyError) else str(e)) or type(e).__name__


# ---- helpers ------------------------------------------------------------------------------------------
def _sess(args):
    return H.session(str(args["project"]))


def _item(p, args) -> int:
    if args.get("item") is not None:
        item = int(args["item"])
    elif args.get("task") is not None and args.get("frame") is not None:
        if int(args["task"]) not in p.ranges:
            raise ValueError(f"no task {args['task']}; tasks: {', '.join(str(t['id']) for t in p.sources)}")
        a, b = p.ranges[int(args["task"])]
        item = a + int(args["frame"]) - 1
        if not a <= item < b:
            raise ValueError(f"task {args['task']} has frames 1–{b - a}")
    else:
        raise ValueError("give item, or task and frame")
    if not 0 <= item < len(p.items):
        raise ValueError(f"item must be 0–{len(p.items) - 1}")
    return item


def _cls(p, label) -> int:
    names = p.classes
    if label in names:
        return names.index(label)
    low = [n.lower() for n in names]
    if str(label).lower() in low:
        return low.index(str(label).lower())
    raise ValueError(f"no label {label!r}; labels: {', '.join(names) or '(none yet: set_labels)'}")


def _where(p, item) -> dict:
    t = p.task_of(item)
    return {"item": item, "task": t["id"], "task_name": t["name"], "frame": item - p.ranges[t["id"]][0] + 1,
            "name": p.items[item]["name"]}


def _parts(p, item, outlines=False) -> list[dict]:
    rles = p.masks(item) if outlines and p.task == "segment" else {}
    out = []
    for b in p.boxes(item):
        d = {"id": b["obj"], "label": p.classes[b["cls"]] if 0 <= b["cls"] < len(p.classes) else b["cls"],
             "box": [round(v, 1) for v in b["box"]], "source": b["source"],
             "score": None if b["score"] is None else round(b["score"], 3)}
        if b["obj"] in rles:
            poly = M.single_polygon(M.decode(rles[b["obj"]]))
            d["outline"] = [] if poly is None else [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
        out.append(d)
    return out


def _new_parts(p, item, before: set) -> list[dict]:
    return [d for d in _parts(p, item) if d["id"] not in before]


def _color(p, cls: int):
    colors = p.meta.get("colors") or []
    c = colors[cls] if cls < len(colors) and colors[cls] else None
    if c and len(c) == 7:
        return tuple(int(c[k:k + 2], 16) for k in (1, 3, 5))
    r, g, b = colorsys.hls_to_rgb((cls * 137.508) % 360 / 360, 0.52, 0.78)
    return int(r * 255), int(g * 255), int(b * 255)


def _render(p, item, overlay=True, crop=None, max_side=1280):
    img = p.image(item).convert("RGB")
    if overlay:
        rles = p.masks(item) if p.task == "segment" else {}
        tint = Image.new("RGBA", img.size, (0, 0, 0, 0))
        for b in p.boxes(item):
            if b["obj"] in rles:
                m = Image.fromarray(M.decode(rles[b["obj"]]).astype("uint8") * 110)
                tint.paste((*_color(p, b["cls"]), 110), mask=m)
        img = Image.alpha_composite(img.convert("RGBA"), tint).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=max(14, img.width // 90))
        for b in p.boxes(item):
            col = _color(p, b["cls"])
            x1, y1, x2, y2 = b["box"]
            draw.rectangle([x1, y1, x2, y2], outline=col, width=max(2, img.width // 640))
            name = p.classes[b["cls"]] if 0 <= b["cls"] < len(p.classes) else str(b["cls"])
            text = f"#{b['obj']} {name}{' ?' if b['source'] == 'suggested' else ''}"
            tw = draw.textlength(text, font=font)
            ty = max(0, y1 - font.size - 4)
            draw.rectangle([x1, ty, x1 + tw + 6, ty + font.size + 4], fill=col)
            draw.text((x1 + 3, ty + 1), text, fill=(0, 0, 0), font=font)
    ox = oy = 0
    if crop:
        x1, y1, x2, y2 = [int(round(v)) for v in crop]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            raise ValueError("crop is empty")
        img, ox, oy = img.crop((x1, y1, x2, y2)), x1, y1
    scale = min(1.0, max(64, int(max_side)) / max(img.size))
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    return img, scale, (ox, oy)


def _wait(jid: str, wait_s: float) -> dict:
    t0 = time.time()
    while not H.state["jobs"][jid]["finished"] and time.time() - t0 < wait_s:
        time.sleep(0.3)
    return _job(jid)


def _job(jid: str) -> dict:
    j = H.get_job(jid)
    return {"job": jid, "kind": j["kind"], "finished": j["finished"], "text": j["text"], "done": j["done"],
            "total": j["total"], "error": j["error"], "result": j["result"]}


def _run(sess, msg, wait_s=120) -> list[str]:
    return sess.call(msg, wait=float(wait_s))


# ---- projects ------------------------------------------------------------------------------------------
@tool("List the projects in your projects folder with their type, labels and progress.")
def list_projects(args):
    return {"projects": [{"name": s["name"], "type": s["task"], "labels": s["classes"], "tasks": len(s["tasks"]),
                          "frames": s.get("items", 0), "labeled": s.get("labeled", 0), "confirmed": s.get("confirmed", 0),
                          **({"problem": s["problem"]} if s.get("problem") else {})} for s in H.list_projects()]}


@tool("A project: type, labels, export formats, and its tasks (frame ranges, video details, progress, jobs).",
      ["project"], project=PROJECT)
def get_project(args):
    d = H.get_project(str(args["project"]))
    keep = ("width", "height", "fps", "duration", "codec", "kept")
    return {"name": d["name"], "type": d["task"], "type_name": TYPES.get(d["task"]), "labels": d["labels"],
            "formats": d["formats"], "frames": d["items"], "subsets": d["subsets"],
            "tasks": [{"id": t["id"], "name": t["name"], "kind": t["kind"], "subset": t["subset"], "frames": t["frames"],
                       "start_item": t["start_item"], "every": t["every"], "status": t["status"],
                       "jobs_done": t["done"], "jobs_total": t["total"], "labeled": t["labeled"], "confirmed": t["confirmed"],
                       "video": {k: t["info"].get(k) for k in keep if t["info"].get(k) is not None},
                       "jobs": [{"id": j["id"], "first_item": j["start"], "frames": j["frames"], "stage": j["stage"],
                                 "state": j["state"], "confirmed": j["confirmed"]} for j in t["jobs"]]} for t in d["tasks"]]}


@tool("Create a project (empty; add videos or image folders with add_tasks). type: detect, segment or classify.",
      ["name", "type"], name=S("Project name"), type=S("detect | segment | classify", enum=list(TYPES)),
      labels=A("Label names, or {name, color '#rrggbb'}", {"anyOf": [{"type": "string"}, {"type": "object"}]}))
def create_project(args):
    labels = [x if isinstance(x, dict) else {"name": x} for x in args.get("labels") or []]
    return H.create_project({"name": args["name"], "task": args["type"], "labels": labels})


@tool("Replace the project's label list. Labels are matched by name (or `was` for a rename); labels left out are "
      "deleted WITH their annotations, which needs allow_delete.", ["project", "labels"], project=PROJECT,
      labels=A("New list: {name, color?, was?} or names", {"anyOf": [{"type": "string"}, {"type": "object"}]}),
      allow_delete=B("Allow deleting labels that are left out, and their annotations"))
def set_labels(args):
    p = _sess(args).p
    new = [x if isinstance(x, dict) else {"name": x} for x in args["labels"]]
    body = [{"name": x["name"], "color": x.get("color"),
             "from": p.classes.index(x.get("was") or x["name"]) if (x.get("was") or x["name"]) in p.classes else None}
            for x in new]
    gone = [c for i, c in enumerate(p.classes) if i not in {b["from"] for b in body}]
    if gone and not args.get("allow_delete"):
        raise ValueError(f"these labels and all their annotations would be deleted: {', '.join(gone)}; "
                         "pass allow_delete=true to do it")
    return {"labels": H.patch_project(str(args["project"]), {"labels": body})["labels"]}


@tool("Move a project to the trash (restorable in the app).", ["project", "confirm"], project=PROJECT,
      confirm=B("Must be true"))
def trash_project(args):
    if args.get("confirm") is not True:
        raise ValueError("pass confirm=true to move the project to the trash")
    return H.trash_project(str(args["project"]))


# ---- tasks and jobs -------------------------------------------------------------------------------------------
@tool("Add tasks: each video file or image folder path becomes one task (every Nth video frame is stored). "
      "Runs as a job; returns it (wait with wait_s or get_job).", ["project", "sources"], project=PROJECT,
      sources=A("Paths of videos or image folders on this computer", {"type": "string"}),
      name=S("Task name (several sources: may use {{file_name}} and {{index}})"), subset=S("Train, Validation, Test…"),
      every=I("Keep every Nth video frame (default 5)", minimum=1),
      frames=S("How video frames are kept: video (read from the video, no copies; default) | jpg | webp", enum=["video", "jpg", "webp"]),
      quality=I("JPEG quality 5–100 (default 95)"), start=I("First video frame"), stop=I("Last video frame"),
      segment_size=I("Frames per job (default: one job per task)"), sorting=S("Image folders: lexicographical | natural"),
      wait_s=WAIT)
def add_tasks(args):
    body = {k: args[k] for k in ("sources", "name", "subset", "every", "frames", "quality", "start", "stop",
                                 "segment_size", "sorting") if args.get(k) is not None}
    jid = H.create_tasks(str(args["project"]), body)["job"]
    return _wait(jid, float(args.get("wait_s") or 0))


@tool("Rename a task or set its subset.", ["project", "task"], project=PROJECT, task=TASK, name=S("New name"),
      subset=S("Train, Validation, Test… ('' for none)"))
def update_task(args):
    body = {k: args[k] for k in ("name", "subset") if args.get(k) is not None}
    t = H.patch_task(str(args["project"]), int(args["task"]), body)
    return {"id": t["id"], "name": t["name"], "subset": t["subset"]}


@tool("Delete a task with its stored frames and all its labels (cannot be undone).", ["project", "task", "confirm"],
      project=PROJECT, task=TASK, confirm=B("Must be true"))
def delete_task(args):
    if args.get("confirm") is not True:
        raise ValueError("pass confirm=true to delete the task and its labels")
    return H.delete_task(str(args["project"]), int(args["task"]))


@tool("Set a job's stage (annotation, validation, acceptance) and/or state (new, in progress, rejected, completed).",
      ["project", "job"], project=PROJECT, job=I("Job id"), stage=S("annotation | validation | acceptance"),
      state=S("new | in progress | rejected | completed"))
def set_job(args):
    body = {k: args[k] for k in ("stage", "state") if args.get(k)}
    j = H.patch_job(str(args["project"]), int(args["job"]), body)
    return {k: j[k] for k in ("id", "stage", "state")}


@tool("A background job's progress and result (add_tasks, export, backup, import_backup).", ["job"],
      job=S("Job id"), wait_s=WAIT)
def get_job(args):
    return _wait(str(args["job"]), float(args.get("wait_s") or 0))


# ---- frames and annotations ------------------------------------------------------------------------------------
@tool("See a frame: the image (parts drawn with #id and label, suggestions marked ?) and its parts as JSON. "
      "Use crop to zoom into small parts; coordinates stay full-frame pixels.", ["project"], **AT,
      overlay=B("Draw the parts (default true)"), crop=BOX, max_side=I("Longest side of the returned image (default 1280)"))
def get_frame(args):
    p = _sess(args).p
    item = _item(p, args)
    img, scale, (ox, oy) = _render(p, item, args.get("overlay", True) is not False, args.get("crop"),
                                   args.get("max_side") or 1280)
    w, h = p.image_size(item)
    return {**_where(p, item), "width": w, "height": h, "image_scale": round(scale, 4), "image_origin": [ox, oy],
            "note": "image pixel (u, v) = frame pixel (origin + (u, v) / image_scale)",
            "confirmed": p.is_reviewed(item), "status": STATUS[p.statuses()[item]], "parts": _parts(p, item),
            **({"image_class": p.classes[t["cls"]]} if p.task == "classify" and (t := p.tags().get(item)) else {}),
            "_image": img}


@tool("Annotations as JSON for one frame, a list of frames, or a whole task (up to 500 frames per call).",
      ["project"], project=PROJECT, item=ITEM, items=A("Frame numbers", {"type": "integer"}), task=TASK,
      frame=FRAME, outlines=B("Include each part's outline polygon (segmentation projects)"),
      only_labeled=B("Skip frames without parts (default true)"))
def get_annotations(args):
    p = _sess(args).p
    if args.get("items"):
        items = [int(i) for i in args["items"]]
    elif args.get("item") is not None or args.get("frame") is not None:
        items = [_item(p, args)]
    elif args.get("task") is not None:
        a, b = p.ranges[int(args["task"])]
        items = list(range(a, b))
    else:
        raise ValueError("give item, items, or task")
    st, tags = p.statuses(), p.tags() if p.task == "classify" else {}
    out = []
    for i in items[:500]:
        if not 0 <= i < len(p.items):
            raise ValueError(f"no frame {i}")
        if args.get("only_labeled", True) is not False and st[i] == 0:
            continue
        d = {**_where(p, i), "status": STATUS[st[i]], "confirmed": st[i] == 4, "parts": _parts(p, i, args.get("outlines"))}
        if i in tags:
            d["image_class"] = p.classes[tags[i]["cls"]]
        out.append(d)
    return {"frames": out, "truncated": len(items) > 500}


@tool("Add a part: click points on it (SAM 3 outlines it; [x, y, 1] = on the part, [x, y, 0] = not the part) and/or "
      "a box around it. In segmentation projects the outline is stored; the box follows it.", ["project", "label"],
      **AT, label=LABEL, points=A("[[x, y, 1|0], …] in frame pixels", {"type": "array", "items": {"type": "number"}}),
      box=BOX, size=I("First click only: 1 = smallest outline SAM offers (default), 2 or 3 = larger", minimum=1, maximum=3))
def add_part(args):
    sess = _sess(args)
    p = sess.p
    item, cls = _item(p, args), _cls(p, args["label"])
    pts, box = args.get("points") or [], args.get("box")
    if not pts and not box:
        raise ValueError("give points or a box")
    before, said, obj = {b["obj"] for b in p.boxes(item)}, [], None
    if box:
        said += _run(sess, {"type": "box", "item": item, "cls": cls, "box": [float(v) for v in box]})
        obj = next((d["id"] for d in _new_parts(p, item, before)), None)
    for k, pt in enumerate(pts):
        x, y, pos = float(pt[0]), float(pt[1]), (int(pt[2]) if len(pt) > 2 else 1) == 1
        said += _run(sess, {"type": "click", "item": item, "x": x, "y": y, "positive": pos, "cls": cls,
                            **({"obj": obj} if obj is not None else {})})
        if obj is None:
            obj = next((d["id"] for d in _new_parts(p, item, before)), None)
            for _ in range(int(args.get("size") or 1) - 1):
                if obj is not None:
                    said += _run(sess, {"type": "cycle", "item": item, "obj": obj})
    part = next((d for d in _parts(p, item) if d["id"] == obj), None)
    if part is None:
        raise ValueError("no part was found there" + (f": {'; '.join(said)}" if said else ""))
    return {**_where(p, item), "part": part, "messages": said}


@tool("Change a part: its label (applies to this part on every frame it was tracked to), redraw its box, or add "
      "points ([x, y, 1] add area, [x, y, 0] remove area).", ["project", "part"], **AT, part=PART, label=LABEL,
      box=BOX, points=A("[[x, y, 1|0], …]", {"type": "array", "items": {"type": "number"}}))
def edit_part(args):
    sess = _sess(args)
    p = sess.p
    item, obj, said = _item(p, args), int(args["part"]), []
    if obj not in {b["obj"] for b in p.boxes(item)}:
        raise ValueError(f"no part {obj} on this frame")
    if args.get("label") is not None:
        said += _run(sess, {"type": "set_class", "obj": obj, "cls": _cls(p, args["label"]), "item": item})
    if args.get("box"):
        said += _run(sess, {"type": "box", "item": item, "cls": 0, "obj": obj, "box": [float(v) for v in args["box"]]})
    for pt in args.get("points") or []:
        said += _run(sess, {"type": "click", "item": item, "obj": obj, "x": float(pt[0]), "y": float(pt[1]),
                            "positive": (int(pt[2]) if len(pt) > 2 else 1) == 1, "cls": 0})
    return {**_where(p, item), "part": next((d for d in _parts(p, item) if d["id"] == obj), None), "messages": said}


@tool("Delete parts from a frame.", ["project", "parts"], **AT, parts=A("Part ids", {"type": "integer"}))
def delete_parts(args):
    sess = _sess(args)
    item, said = _item(sess.p, args), []
    for obj in args["parts"]:
        said += _run(sess, {"type": "delete", "item": item, "obj": int(obj)})
    return {**_where(sess.p, item), "parts": _parts(sess.p, item), "messages": said}


@tool("Find parts described in words on a frame (SAM 3 text search, e.g. 'round amber lamp'). They are added as "
      "suggestions of `label` until accept_suggestions.", ["project", "text", "label"], **AT,
      text=S("What to look for, in a few words"), label=LABEL, threshold={"type": "number", "description": "0–1, default 0.5"},
      wait_s=WAIT)
def find_parts(args):
    sess = _sess(args)
    p = sess.p
    item = _item(p, args)
    before = {b["obj"] for b in p.boxes(item)}
    said = _run(sess, {"type": "find_text", "item": item, "text": args["text"], "cls": _cls(p, args["label"]),
                       "threshold": float(args.get("threshold") or 0.5)}, args.get("wait_s", 120))
    return {**_where(p, item), "found": _new_parts(p, item, before), "messages": said, "busy": sess.running}


@tool("Find more parts like one part on the same frame (SAM 3 example search); added as suggestions.",
      ["project", "part"], **AT, part=PART, threshold={"type": "number", "description": "0–1, default 0.3"}, wait_s=WAIT)
def find_similar(args):
    sess = _sess(args)
    p = sess.p
    item = _item(p, args)
    before = {b["obj"] for b in p.boxes(item)}
    said = _run(sess, {"type": "find_all", "item": item, "obj": int(args["part"]),
                       "threshold": float(args.get("threshold") or 0.3)}, args.get("wait_s", 120))
    return {**_where(p, item), "found": _new_parts(p, item, before), "messages": said, "busy": sess.running}


@tool("Suggest parts on a frame from the parts people labeled on other frames (DINOv3 matching); added as suggestions.",
      ["project"], **AT, wait_s=WAIT)
def suggest(args):
    sess = _sess(args)
    p = sess.p
    item = _item(p, args)
    said = _run(sess, {"type": "suggest", "item": item}, args.get("wait_s", 120))
    return {**_where(p, item), "suggestions": [d for d in _parts(p, item) if d["source"] == "suggested"],
            "messages": said, "busy": sess.running}


@tool("Turn a frame's suggestions into parts.", ["project"], **AT)
def accept_suggestions(args):
    sess = _sess(args)
    item = _item(sess.p, args)
    said = _run(sess, {"type": "accept", "item": item})
    return {**_where(sess.p, item), "parts": _parts(sess.p, item), "messages": said}


@tool("Track the frame's parts through the next (or previous) frames of its task with SAM 3; frames people "
      "confirmed are not changed. Then check the frames that status lists as to_check.", ["project"], **AT,
      frames=I("How many frames (default 20)", minimum=1), to_end=B("Track to the end (or start) of the task"),
      direction=S("forward | backward", enum=["forward", "backward"]),
      redo=B("Re-track: on unconfirmed frames reached, replace every tracked part (parts gone from this frame go there too)"),
      wait_s=WAIT)
def track(args):
    sess = _sess(args)
    p = sess.p
    item = _item(p, args)
    said = _run(sess, {"type": "track", "item": item, "count": -1 if args.get("to_end") else int(args.get("frames") or 20),
                       "direction": -1 if args.get("direction") == "backward" else 1, "redo": bool(args.get("redo"))},
                args.get("wait_s", 120))
    return {**_where(p, item), "finished": not sess.running, "messages": said,
            "to_check": [i for i in p.flags() if p.task_of(i)["id"] == p.task_of(item)["id"]][:100]}


@tool("Remove parts of some labels (or all) from the frames after (or before) a frame in its task: the next N frames "
      "or to the task's end. Confirmed frames only with include_confirmed. Undoable.", ["project"], **AT,
      labels=A("Labels to remove (default: every label)", {"type": "string"}), frames=I("How many frames (default: to the end)", minimum=1),
      direction=S("forward | backward", enum=["forward", "backward"]), include_confirmed=B("Also frames people confirmed"))
def clear_parts(args):
    sess = _sess(args)
    p = sess.p
    item = _item(p, args)
    classes = [_cls(p, x) for x in args["labels"]] if args.get("labels") else None
    said = _run(sess, {"type": "clear", "item": item, "classes": classes, "count": int(args.get("frames") or -1),
                       "direction": -1 if args.get("direction") == "backward" else 1,
                       "confirmed": bool(args.get("include_confirmed"))})
    return {**_where(p, item), "messages": said}


@tool("Mark frames as confirmed (checked by a person) or not. Confirm only when the user asked you to.",
      ["project", "items"], project=PROJECT, items=A("Frame numbers", {"type": "integer"}),
      confirmed=B("true (default) or false"))
def confirm_frames(args):
    sess = _sess(args)
    value, said = args.get("confirmed", True) is not False, []
    for i in args["items"]:
        _item(sess.p, {"item": i})
        said += _run(sess, {"type": "review", "item": int(i), "value": value})
    return {"frames": len(args["items"]), "confirmed": value, "messages": said[-3:]}


@tool("Set the class of whole images (classification projects); label null removes it.", ["project", "items"],
      project=PROJECT, items=A("Frame/image numbers", {"type": "integer"}), label={"type": ["string", "null"],
                                                                                    "description": "Label, or null"})
def set_image_class(args):
    sess = _sess(args)
    cls = None if args.get("label") is None else _cls(sess.p, args["label"])
    said = _run(sess, {"type": "tag", "of": "images", "keys": [int(i) for i in args["items"]], "cls": cls})
    return {"images": len(args["items"]), "label": args.get("label"), "messages": said}


@tool("Undo the last change(s) in the project (the same history as Ctrl+Z in the app).", ["project"],
      project=PROJECT, steps=I("How many (default 1)", minimum=1))
def undo(args):
    sess = _sess(args)
    said = []
    for _ in range(int(args.get("steps") or 1)):
        said += _run(sess, {"type": "undo"})
    return {"messages": said}


@tool("Stop the project's running job (tracking, suggestions…).", ["project"], project=PROJECT)
def stop(args):
    sess = _sess(args)
    sess.call({"type": "stop"})
    return {"stopping": sess.running}


@tool("Progress: frames per status for each task, frames to check (tracked parts that jumped, changed size or got "
      "lost), whether a job is running, undo steps.", ["project"], project=PROJECT, task=TASK)
def status(args):
    sess = _sess(args)
    p = sess.p
    st, flags = p.statuses(), set(p.flags())
    tasks = []
    for t in p.sources:
        if args.get("task") is not None and t["id"] != int(args["task"]):
            continue
        a, b = p.ranges.get(t["id"], (0, 0))
        tasks.append({"id": t["id"], "name": t["name"], "frames": b - a,
                      "by_status": {s: sum(x == k for x in st[a:b]) for k, s in enumerate(STATUS)},
                      "to_check": sorted(i for i in flags if a <= i < b)[:200]})
    return {"busy": sess.running, "last_error": sess.error, "undo_steps": len(sess.history), "tasks": tasks}


# ---- export, backup ----------------------------------------------------------------------------------------------
@tool("Export the dataset (whole project, some tasks or jobs) into the project's exports folder. Formats: see "
      "get_project. Runs as a job; waits up to wait_s (default 300).", ["project", "format"], project=PROJECT,
      format=S("yolo | coco | cvat | voc | labelstudio | folders | csv"), tasks=A("Task ids", {"type": "integer"}),
      jobs=A("Job ids", {"type": "integer"}), confirmed_only=B("Only confirmed frames"),
      save_images=B("Include the images (default true)"), name=S("Folder name"), wait_s=WAIT)
def export(args):
    body = {"format": args["format"], "tasks": args.get("tasks"), "jobs": args.get("jobs"),
            "reviewed_only": bool(args.get("confirmed_only")), "save_images": args.get("save_images", True) is not False,
            "name": args.get("name")}
    return _wait(H.export_dataset(str(args["project"]), body)["job"], float(args.get("wait_s") or 300))


@tool("Read a project's video frames straight from its videos instead of stored image files (same frames and labels, "
      "a fraction of the space; the old files move to the _old_frames folder).", ["project"], project=PROJECT, wait_s=WAIT)
def frames_to_video(args):
    return _wait(H.frames_to_video(str(args["project"]))["job"], float(args.get("wait_s") or 300))


@tool("Back up a project, or one task, as a zip with frames, labels and progress (into the _backups folder).",
      ["project"], project=PROJECT, task=TASK, wait_s=WAIT)
def backup(args):
    name = str(args["project"])
    jid = (H.backup_task(name, int(args["task"])) if args.get("task") is not None else H.backup(name))["job"]
    return _wait(jid, float(args.get("wait_s") or 300))


@tool("Open a backup zip as a new project, or add its tasks (with their progress) to an existing project.", ["path"],
      path=S("Path of the .zip"), into_project=S("Existing project to add the tasks to (default: a new project)"),
      wait_s=WAIT)
def import_backup(args):
    body = {"path": args["path"]}
    jid = (H.import_tasks(args["into_project"], body) if args.get("into_project") else H.restore(body))["job"]
    return _wait(jid, float(args.get("wait_s") or 300))


# ---- transports -------------------------------------------------------------------------------------------------
def _content(result, image) -> list[dict]:
    blocks = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str)}]
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=88)
        blocks.append({"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mimeType": "image/jpeg"})
    return blocks


def _rpc(m: dict) -> dict | None:
    mid, method, params = m.get("id"), m.get("method"), m.get("params") or {}
    if mid is None:                                        # a notification (e.g. notifications/initialized)
        return None
    if method == "initialize":
        want = params.get("protocolVersion")
        res = {"protocolVersion": want if want in PROTOCOLS else PROTOCOLS[1],
               "capabilities": {"tools": {"listChanged": False}},
               "serverInfo": {"name": "partlabeler", "title": "PartLabeler", "version": "1"}, "instructions": GUIDE}
    elif method == "ping":
        res = {}
    elif method == "tools/list":
        res = {"tools": [{"name": n, "description": d, "inputSchema": s} for n, (_, d, s) in TOOLS.items()]}
    elif method == "tools/call":
        try:
            out, img = call(params.get("name", ""), params.get("arguments") or {})
            res = {"content": _content(out, img), "isError": False}
        except Exception as e:                            # the model sees the reason and can retry
            res = {"content": [{"type": "text", "text": error_text(e)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def mount(app, host) -> None:
    """Add /mcp and /api/tools to the app (the signed_in middleware has already checked the token)."""
    global H
    H = host

    def local_origin(request: Request) -> bool:           # a web page elsewhere must not drive the tool
        origin = request.headers.get("origin")
        return not origin or urlparse(origin).hostname in ("localhost", "127.0.0.1", "::1", "[::1]")

    @app.post("/mcp")
    async def mcp(request: Request):
        if not local_origin(request):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}, 400)
        msgs = body if isinstance(body, list) else [body]
        out = [r for r in [await asyncio.to_thread(_rpc, m) for m in msgs] if r is not None]
        if not out:
            return Response(status_code=202)
        return JSONResponse(out if isinstance(body, list) else out[0])

    @app.get("/mcp")
    def mcp_stream():                                      # no server-initiated stream on this server
        return Response(status_code=405, headers={"Allow": "POST"})

    @app.get("/api/tools")
    def list_tools() -> dict:
        return {"guide": GUIDE, "tools": [{"name": n, "description": d, "input": s} for n, (_, d, s) in TOOLS.items()]}

    @app.post("/api/tools/{name}")
    def run_tool(name: str, request: Request, body: dict | None = None):
        if not local_origin(request):
            raise HTTPException(403, "origin not allowed")
        try:
            out, img = call(name, body or {})
        except HTTPException:
            raise
        except KeyError as e:
            raise HTTPException(404 if name not in TOOLS else 400, error_text(e))
        except Exception as e:
            raise HTTPException(400, error_text(e))
        if img is not None:
            out["image"] = "data:image/jpeg;base64," + _content({}, img)[1]["data"]
        return out
