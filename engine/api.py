"""One open project, shared by every UI host (local web page, notebook widget).

Hosts pass each incoming message to `Session.handle`; everything going back to the screen goes
through the `send` callback, including progress from background jobs (tracking, suggestions,
export). Protocol:

In:  ready | goto {item} | box {item, cls, box, obj?} | click {item, x, y, positive, cls, obj?}
     cycle {item, obj} | delete {item, obj} | set_class {obj, cls, item?} | review {item, value}
     accept {item} | undo | track {item, count, direction} | stop | suggest {item}
     find_all {item, obj} | settings {parent, task?} | export {format, reviewed_only}
     paint {item, obj?, cls, x, y, png} | outline {item, all?}                     (outline projects)
     add_class {name, of?, keys?} | grid {of} | thumbs {of, keys} | sort {of, k?} | tag {of, keys, cls} | accept_tags {keys}
     suggest_tags | odd {of}                       (the grid: of = "images" in class projects, "parts" otherwise)
     notifications_read | notifications_clear | assistant_info | assistant {id, messages, context, actions}
     assistant_key {key}
Out: project {...} | item {...} | mask {item, obj, src} | status {statuses, flags, busy, gpu, undo}
     item_changed {items} | progress {task, done, total} | toast {text} | error {text}
     grid {of, cards} | cards {of, cards} | thumbs {of, src} | groups {of, k, groups, unsure, dups, ...}
     odd {of, items} | notify {item, unread} | notifications {items, unread} | assistant_info {...}
     assistant {id, text|done|...}

Outline projects (task "segment") keep a mask per part: clicks, boxes (SAM 3 outlines the part inside), tracking,
suggestions and Find similar all store one, the brush and eraser edit it, and item messages carry each part's
mask as a small PNG crop. Image-class projects (task "classify") are labeled on a grid of thumbnails, with smart
sorting from engine/sort.py; box and outline projects use the same grid to sort their parts ("Sort parts").

Events worth finding later (confirmed, deleted, tracked, exported, errors...) go through `notify`, which
records them in the notification history (engine/notify.py) and shows them as a toast; `toast` is only
for passing hints.

Models load on first use and are shared by every open project (one copy in GPU memory).
"""
import base64
import io
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from engine import masks as M
from engine.notify import Notifications

MASK_RGBA = (10, 124, 120, 110)
PAIR = re.compile(r"^(left|right)_(.+)$")
HISTORY = 50                                              # undo steps kept per project
MODEL_NAMES = {"seg": "outline", "track": "tracking", "suggest": "matching", "concept": "find-similar",
               "embed": "sorting"}
UNDO_WORDS = {"box": "drawing a box", "outline": "outlining a part", "delete": "deleting a box",
              "class change": "a class change", "confirm": "confirming a frame", "accept": "accepting suggestions",
              "tracking": "tracking", "suggestions": "suggestions", "find similar": "find similar",
              "paint": "painting an outline", "outline boxes": "outlining boxes", "classes": "a class change"}
FORMAT_NAMES = {"yolo": "YOLO", "coco": "COCO", "cvat": "CVAT", "voc": "Pascal VOC", "labelstudio": "Label Studio",
                "folders": "class folders", "csv": "CSV"}
THUMB = 224                                               # px, longest side of grid thumbnails

_MODELS: dict = {}
_LOCKS = {k: threading.Lock() for k in ("seg", "track", "suggest", "concept", "embed", "load")}


def model(name: str, announce=None):
    """The shared model `name` (seg | track | suggest | concept | embed), loaded on first use."""
    with _LOCKS["load"]:
        if name not in _MODELS:
            if announce:
                announce(name)
            if name == "seg":
                from engine.segmenter import Segmenter
                _MODELS[name] = Segmenter()
            elif name == "track":
                from engine.tracker import Tracker
                _MODELS[name] = Tracker()
            elif name == "concept":
                from engine.parent import ConceptFinder
                _MODELS[name] = ConceptFinder()
            elif name == "embed":                         # whole-image / crop features for sorting
                from engine import hw
                from engine.embed import Embedder
                _MODELS[name] = Embedder("base" if hw.device() == "cuda" else "small")
            else:
                from engine.suggest import Suggester
                _MODELS[name] = Suggester(finder=lambda: model("concept"))
        return _MODELS[name]


def release_models() -> None:
    """Drop the loaded models (each loads again on its next use), e.g. to give the GPU to a training run in
    another process, as the Colab notebook does before Teach and Transfer."""
    from engine import hw
    with _LOCKS["load"]:
        _MODELS.clear()
    hw.free_gpu_memory()


def data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": 90} if fmt == "JPEG" else {"compress_level": 1}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def mask_png(mask: np.ndarray) -> str:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = MASK_RGBA
    return data_url(Image.fromarray(rgba, "RGBA"), "PNG")


class Session:
    def __init__(self, project, send, image_src=None, notes: Notifications | None = None):
        """`notes`: a shared history (the web host keeps one per projects folder); default: this session's own."""
        self.p, self.send = project, send
        self.notes = notes or Notifications(on_change=send)
        self.image_src = image_src or (lambda item: data_url(project.image(item)))
        self._locks = _LOCKS
        self.points = {}                                  # (item, obj) -> [[x, y, label], ...]
        self.outlines = {}                                # (item, obj) -> ([(mask, score), ...] small->large, index)
        self.history = []                                 # [(label, snapshot)] for undo
        self.job, self.running, self.stop_flag = None, False, False
        self.thumb_src = None                              # web host: (of, key) -> URL; notebooks ask with `thumbs`
        self.sorting = {}                                  # of -> {"keys", "vectors", "groups"}
        self._thumbs, self._frames, self._frame_lock = {}, {}, threading.Lock()

    def _model(self, name):
        return model(name, lambda n: self.send({"type": "toast",
                                                "text": f"Loading the {MODEL_NAMES[n]} model (first use only)…"}))

    def notify(self, level: str, title: str, detail: str = "", action=None, key=None, plural=None) -> dict:
        return self.notes.add(level, title, detail, project=self.p.meta["name"], action=action, key=key, plural=plural)

    def _goto(self, item: int) -> dict:
        return {"type": "goto", "project": self.p.meta["name"], "item": item, "label": "Go to frame"}

    def _frame(self, item: int) -> str:
        return f"{'frame' if self.p.meta['kind'] == 'video' else 'image'} {item + 1}"

    def _remember(self, items, label: str) -> None:
        self.history.append((label, self.p.snapshot(items)))
        del self.history[:-HISTORY]

    # ---- messages out --------------------------------------------------------------------
    def project_msg(self):
        from engine import hw
        return {"type": "project", "name": self.p.meta["name"], "kind": self.p.meta["kind"], "task": self.p.task,
                "classes": self.p.classes, "count": len(self.p.items),
                "names": [it["name"] for it in self.p.items], "parent": self.p.meta.get("parent") or "",
                "device": hw.describe(), "formats": list(self.p.formats), "task": self.p.task}

    def status_msg(self):
        from engine import hw
        return {"type": "status", "statuses": self.p.statuses(), "flags": self.p.flags(), "busy": self.running,
                "gpu": hw.memory(), "undo": len(self.history)}

    def item_msg(self, item: int):
        it = self.p.items[item]
        with Image.open(it["path"]) as im:
            w, h = im.size
        boxes = self.p.boxes(item)
        if self.p.task == "segment":
            rles = self.p.masks(item)
            for b in boxes:
                if b["obj"] in rles:
                    b["mask"] = M.crop_png(M.decode(rles[b["obj"]]))
        msg = {"type": "item", "item": item, "name": it["name"], "src": self.image_src(item), "w": w, "h": h,
               "boxes": boxes, "reviewed": self.p.is_reviewed(item)}
        if self.p.task == "classify":
            msg["tag"] = self.p.tags().get(item)
        return msg

    def _size(self, item: int):
        with Image.open(self.p.items[item]["path"]) as im:
            return im.size

    def refresh(self, item: int):
        self.send(self.item_msg(item))
        self.send(self.status_msg())

    # ---- messages in ---------------------------------------------------------------------
    def handle(self, msg: dict) -> None:
        try:
            getattr(self, "on_" + msg.get("type", ""), self.on_unknown)(msg)
        except Exception as e:                            # report, keep the session alive
            traceback.print_exc()
            self.notify("error", "Something went wrong", f"{type(e).__name__}: {e}")

    def on_unknown(self, msg):
        self.notify("error", "The app sent a message the server does not know", repr(msg.get("type")))

    def on_ready(self, msg):
        self.send(self.project_msg())
        self.send(self.status_msg())
        self.send(self.item_msg(max(0, min(msg.get("item", 0), len(self.p.items) - 1))))
        self.send({"type": "notifications", **self.notes.snapshot()})

    def on_notifications_read(self, msg):
        self.notes.mark_read()

    def on_notifications_clear(self, msg):
        self.notes.clear()

    # Rivet, the helper (engine/assistant.py). The web page asks the server over HTTP; notebooks ask here.
    def on_assistant_info(self, msg):
        from engine import assistant
        self.send({"type": "assistant_info", **assistant.info()})

    def on_assistant(self, msg):
        from engine import assistant

        def run():
            try:
                for ev in assistant.reply(msg.get("messages"), msg.get("context"), msg.get("actions")):
                    self.send({"type": "assistant", "id": msg.get("id"), **ev})
            except Exception as e:
                self.send({"type": "assistant", "id": msg.get("id"), "error": f"{type(e).__name__}: {e}"})
        threading.Thread(target=run, daemon=True).start()

    def on_assistant_key(self, msg):
        from engine import assistant
        try:
            assistant.save_key(msg.get("key", ""))
            self.send({"type": "assistant_info", **assistant.info()})
        except Exception as e:
            self.send({"type": "assistant_info", **assistant.info(), "key_error": str(e)})

    def on_goto(self, msg):
        self.send(self.item_msg(msg["item"]))

    def on_box(self, msg):
        """A drawn box. Outline projects: SAM 3 outlines the part inside it."""
        obj = msg.get("obj") or self.p.new_obj()
        cls = msg["cls"] if msg.get("obj") is None else next(
            (b["cls"] for b in self.p.boxes(msg["item"]) if b["obj"] == obj), msg["cls"])
        mask = self._outline_boxes(msg["item"], [msg["box"]])[0] if self.outlines_on else None
        if mask is not None and not mask.any():
            self.send({"type": "toast", "text": "No part found inside that box; draw it around the whole part"})
            return
        self._remember([msg["item"]], "box")
        self.p.put(msg["item"], obj, cls, msg["box"], "manual", mask=mask)
        self.points.pop((msg["item"], obj), None)
        if mask is not None:
            self.send({"type": "select", "item": msg["item"], "obj": obj})
        self.refresh(msg["item"])

    def on_paint(self, msg):
        """Brush or eraser on an outline: the page sends the part's whole edited mask as a PNG crop at (x, y).
        An empty mask deletes the part; no obj starts a new part of class `cls`."""
        item = msg["item"]
        W, H = self._size(item)
        mask = M.from_png(msg["png"], msg["x"], msg["y"], H, W)
        existing = {b["obj"]: b for b in self.p.boxes(item)}
        obj = msg.get("obj") if msg.get("obj") in existing else None
        self._remember([item], "paint")
        if not mask.any():
            if obj is not None:
                self.p.delete(item, obj)
            self.refresh(item)
            return
        obj = obj or self.p.new_obj()
        cls = existing[obj]["cls"] if obj in existing else msg["cls"]
        self.p.put(item, obj, cls, M.box_of(mask), "manual", mask=mask)
        self.points.pop((item, obj), None)
        self.send({"type": "select", "item": item, "obj": obj})
        self.refresh(item)

    def on_outline(self, msg):
        """Outline every box that has no outline yet: on this item, or with all=True on every item (imported
        boxes, boxes drawn before the project was switched to outlines, Teach & Transfer results)."""
        if not self.outlines_on:
            self.send({"type": "toast", "text": "Switch the project to outlines first (Settings, Label type)"})
            return
        todo = {}
        rows = self.p.db.execute("SELECT item, obj, x1, y1, x2, y2 FROM boxes WHERE rle IS NULL AND source != 'suggested'"
                                 + ("" if msg.get("all") else " AND item=?"), () if msg.get("all") else (msg["item"],))
        for item, obj, *box in rows:
            todo.setdefault(item, []).append((obj, box))
        if not todo:
            self.send({"type": "toast", "text": "Every box already has an outline"})
            return
        self._remember(list(todo), "outline boxes")

        def job():
            done = n = 0
            for item, objs in sorted(todo.items()):
                if self.stop_flag:
                    break
                masks = self._outline_boxes(item, [b for _, b in objs])
                rows = {b["obj"]: b for b in self.p.boxes(item)}
                for (obj, _), mask in zip(objs, masks):
                    if obj in rows and mask.any():
                        b = rows[obj]
                        self.p.put(item, obj, b["cls"], b["box"], b["source"], b["score"], mask=mask)
                        n += 1
                done += 1
                self.send({"type": "progress", "task": "Outlining boxes", "done": done, "total": len(todo)})
            self.notify("success", f"Outlined {n} box{'es' if n != 1 else ''} on {done} "
                                   f"{'frame' if self.p.meta['kind'] == 'video' else 'image'}{'s' if done != 1 else ''}",
                        "Check the outlines; the brush (P) and eraser (E) fix the edges")
            self.send({"type": "item_changed", "items": list(todo)})

        self._run("Outlining boxes", job)

    def _seg_on(self, item: int):
        """The outline model with `item`'s image encoded (call with the "seg" lock held)."""
        seg = self._model("seg")
        key = (str(self.p.folder), item)
        if getattr(seg, "image_key", None) != key:
            seg.set_image(self.p.image(item))
            seg.image_key = key
        return seg

    def _outline_boxes(self, item: int, boxes) -> list:
        """SAM 3 outlines of the parts inside these boxes on one item (box prompts), cleaned of specks and pinholes."""
        with self._locks["seg"]:
            seg = self._seg_on(item)
            return [M.clean(seg.segment(box=[float(v) for v in b])[0]) for b in boxes]

    @property
    def outlines_on(self) -> bool:
        return self.p.task == "segment"

    def on_click(self, msg):
        """First click on a new part: SAM's three nested outlines are kept small -> large and the
        smallest one is used (you are labeling parts, not whole objects); M steps larger.
        Shift/Alt clicks add points to the selected part and use SAM's single best outline."""
        item = msg["item"]
        existing = {b["obj"]: b for b in self.p.boxes(item)}
        obj = msg.get("obj") if msg.get("obj") in existing else self.p.new_obj()
        pts = self.points.setdefault((item, obj), [])
        pts.append([float(msg["x"]), float(msg["y"]), 1 if msg.get("positive", True) else 0])
        prompt_box = existing[obj]["box"] if obj in existing and len(pts) == 1 and msg.get("obj") else None
        with self._locks["seg"]:
            seg = self._seg_on(item)
            points, labels = [p[:2] for p in pts], [p[2] for p in pts]
            if len(pts) == 1 and prompt_box is None:
                outs = sorted(seg.candidates(points, labels), key=lambda ms: int(ms[0].sum()))
                pick = next((k for k, (m, s) in enumerate(outs) if m.sum() >= 16), len(outs) - 1)
            else:
                outs, pick = [seg.segment(points, labels, box=prompt_box)], 0
        self.outlines[(item, obj)] = (outs, pick)
        self._apply_outline(item, obj, msg["cls"])

    def on_cycle(self, msg):
        """M: switch the selected part to SAM's next larger outline for its last click."""
        key = (msg["item"], msg["obj"])
        if key not in self.outlines or len(self.outlines[key][0]) < 2:
            self.send({"type": "toast", "text": "No other outline sizes for this box; click the part again"})
            return
        outs, pick = self.outlines[key]
        self.outlines[key] = (outs, (pick + 1) % len(outs))
        self._apply_outline(msg["item"], msg["obj"], None)

    def _apply_outline(self, item, obj, cls):
        from engine.tracker import mask_box
        outs, pick = self.outlines[(item, obj)]
        mask, score = outs[pick]
        box = mask_box(mask)
        if box is None:
            self.send({"type": "toast", "text": "Nothing found there; try clicking nearer the part's centre"})
            return
        cls = next((b["cls"] for b in self.p.boxes(item) if b["obj"] == obj), cls)
        self._remember([item], "outline")
        if self.outlines_on:
            self.p.put(item, obj, cls, box, "manual", score, mask=M.clean(mask))
            self.send({"type": "select", "item": item, "obj": obj})
        else:
            self.p.put(item, obj, cls, box, "manual", score)
            self.send({"type": "mask", "item": item, "obj": obj, "src": mask_png(mask)})
        if len(outs) > 1:
            self.send({"type": "toast", "text": f"Outline {pick + 1} of {len(outs)} (small to large). M: next size"})
        self.refresh(item)

    def on_delete(self, msg):
        gone = next((b for b in self.p.boxes(msg["item"]) if b["obj"] == msg["obj"]), None)
        self._remember([msg["item"]], "delete")
        self.p.delete(msg["item"], msg["obj"])
        if gone:
            self.notify("info", f"Deleted a {self.p.classes[gone['cls']]} box on {self._frame(msg['item'])}",
                        "Ctrl+Z brings it back", self._goto(msg["item"]), key="delete", plural="Deleted {n} boxes")
        self.points.pop((msg["item"], msg["obj"]), None)
        self.refresh(msg["item"])

    def on_set_class(self, msg):
        self._remember(self.p.items_with(msg["obj"]), "class change")
        self.p.set_class(msg["obj"], msg["cls"])
        if "item" in msg:
            self.refresh(msg["item"])

    def on_review(self, msg):
        self._remember([msg["item"]], "confirm")
        self.p.set_reviewed(msg["item"], msg.get("value", True))
        if msg.get("value", True):
            self.notify("success", f"Confirmed {self._frame(msg['item'])}", "Saved to the project",
                        self._goto(msg["item"]), key="confirm", plural="Confirmed {n} frames")
        else:
            self.notify("info", f"Unconfirmed {self._frame(msg['item'])}", "", self._goto(msg["item"]))
        self.refresh(msg["item"])

    def on_accept(self, msg):
        n = sum(b["source"] == "suggested" for b in self.p.boxes(msg["item"]))
        self._remember([msg["item"]], "accept")
        self.p.accept(msg["item"])
        if n:
            self.notify("success", f"Accepted {n} suggestion{'s' if n != 1 else ''} on {self._frame(msg['item'])}",
                        "", self._goto(msg["item"]))
        self.refresh(msg["item"])

    def on_stop(self, msg):
        self.stop_flag = True

    def on_undo(self, msg):
        if self.running:
            self.send({"type": "toast", "text": "A job is running; press Stop (X) before undoing"})
            return
        if not self.history:
            self.send({"type": "toast", "text": "Nothing to undo"})
            return
        label, snap = self.history.pop()
        items = self.p.restore(snap)
        self.points = {k: v for k, v in self.points.items() if k[0] not in snap["rows"]}
        self.outlines = {k: v for k, v in self.outlines.items() if k[0] not in snap["rows"]}
        what = UNDO_WORDS.get(label, label)
        self.notify("info", f"Undid {what}" + (f" on {len(items)} frames" if len(items) > 1 else ""), "",
                    self._goto(min(items)) if items else None)
        self.send({"type": "item_changed", "items": items})
        self.send(self.status_msg())

    def on_add_class(self, msg):
        """A new class at the end of the list (existing labels keep their numbers); with keys, those cards get it."""
        name = " ".join(str(msg.get("name", "")).split())
        if not name:
            return
        if name in self.p.classes:
            self.send({"type": "toast", "text": f"There is already a class called {name}"})
        else:
            self.p.set_meta(classes=[*self.p.classes, name])
            self.send(self.project_msg())
            self.notify("success", f"Added class {name}", f"Number {len(self.p.classes)} in the list")
        if msg.get("keys"):
            self.on_tag({"of": msg.get("of", "images"), "keys": msg["keys"], "cls": self.p.classes.index(name)})

    def on_settings(self, msg):
        parent = (msg.get("parent") or "").strip() or None
        self.p.set_meta(parent=parent)
        task = msg.get("task")
        switched = task in ("detect", "segment") and self.p.task in ("detect", "segment") and task != self.p.task
        if switched:
            self.p.set_meta(task=task)
        self.send(self.project_msg())
        detail = f"Parent object: {parent}" if parent else "Parent object: none (suggestions search the whole image)"
        if switched:
            detail += (". Label type: outlines. Boxes without an outline: use Outline boxes" if task == "segment"
                       else ". Label type: boxes (outlines are kept)")
        self.notify("success", "Settings saved", detail)
        if switched:
            self.send({"type": "item_changed", "items": list(range(len(self.p.items)))})

    # ---- background jobs -----------------------------------------------------------------
    def _run(self, name, fn):
        if self.running:
            self.send({"type": "toast", "text": "Busy with another job; press Stop first"})
            return
        self.stop_flag, self.running = False, True
        self.send({"type": "progress", "task": name, "done": 0, "total": 1})

        def wrapped():
            try:
                fn()
            except Exception as e:
                traceback.print_exc()
                hint = " Close other programs that use the GPU and try again." if "out of memory" in str(e) else ""
                self.notify("error", f"{name} failed", f"{type(e).__name__}: {e}.{hint}")
            finally:
                self.running = False
                self.send({"type": "progress", "task": name, "done": 1, "total": 1, "finished": True})
                self.send(self.status_msg())

        self.job = threading.Thread(target=wrapped, daemon=True)
        self.job.start()

    def on_track(self, msg):
        start, count = msg["item"], msg.get("count", 20)
        direction = -1 if (msg.get("direction") or 1) < 0 else 1
        count = len(self.p.items) if count in (None, -1) else count
        total = min(count, start if direction < 0 else len(self.p.items) - start - 1)
        if total <= 0:
            self.send({"type": "toast", "text": f"No frames {'before' if direction < 0 else 'after'} this one"})
            return
        span = range(start - total, start) if direction < 0 else range(start + 1, start + total + 1)
        self._remember(span, "tracking")

        def job():
            t0, last_push = time.perf_counter(), 0.0

            def on_item(item, res):
                nonlocal last_push
                self.p.put_tracked(item, res)
                done = abs(item - start)
                if time.perf_counter() - last_push > 1.0 or done == total:
                    last_push = time.perf_counter()
                    self.send({"type": "progress", "task": "Tracking", "done": done, "total": total,
                               "rate": round(done / (time.perf_counter() - t0), 2)})
                    self.send(self.status_msg())

            with self._locks["track"]:
                n = self._model("track").track(self.p, start, count, on_item, lambda: self.stop_flag, direction,
                                               masks=self.outlines_on)
            self.notify("success", f"Tracked {n} frames {'back' if direction < 0 else 'ahead'}",
                        f"From {self._frame(start)} in {time.perf_counter() - t0:.0f} s. Frames marked 'to check' may need a look.",
                        self._goto(start))
            self.send({"type": "item_changed", "items": list(span)})

        self._run("Tracking", job)

    def on_suggest(self, msg):
        item = msg["item"]

        def job():
            with self._locks["suggest"]:
                found = self._model("suggest").suggest(self.p, item)
            self._remember([item], "suggestions")
            for b in self.p.boxes(item):
                if b["source"] == "suggested":
                    self.p.delete(item, b["obj"])
            obj = self.p.new_obj()
            masks = self._outline_boxes(item, [box for _, box, _ in found]) if self.outlines_on else [None] * len(found)
            for k, ((cls, box, score), mask) in enumerate(zip(found, masks)):
                self.p.put(item, obj + k, cls, box, "suggested", score, mask=mask if mask is not None and mask.any() else None)
            self.notify("info", f"{len(found)} suggestion{'s' if len(found) != 1 else ''} on {self._frame(item)}",
                        "Y accepts all; select one and press Delete to drop it", self._goto(item))
            self.refresh(item)

        self._run("Suggesting", job)

    def on_find_all(self, msg):
        """F: more parts like the selected box in this image (SAM 3 exemplar search), as suggestions."""
        item, obj = msg["item"], msg.get("obj")
        ex = next((b for b in self.p.boxes(item) if b["obj"] == obj), None)
        if ex is None:
            self.send({"type": "toast", "text": "Select a box first (Ctrl+click it), then press F"})
            return

        def job():
            from engine.parent import overlap
            with self._locks["concept"]:
                hits = self._model("concept").find_like(self.p.image(item), ex["box"],
                                                        threshold=msg.get("threshold", 0.3))
            have = [b["box"] for b in self.p.boxes(item)]
            new = [(box, sc) for box, sc in hits if all(overlap(box, h) < 0.5 for h in have)]
            self._remember([item], "find similar")
            obj0 = self.p.new_obj()
            masks = self._outline_boxes(item, [box for box, _ in new]) if self.outlines_on else [None] * len(new)
            for k, ((box, sc), mask) in enumerate(zip(new, masks)):
                self.p.put(item, obj0 + k, self._side(ex["cls"], box), box, "suggested", sc,
                           mask=mask if mask is not None and mask.any() else None)
            self.notify("info", f"Found {len(new)} more like this on {self._frame(item)}",
                        "Y accepts all; select one and press Delete to drop it" if new else
                        "Try a lower threshold or another example", self._goto(item))
            self.refresh(item)

        self._run("Finding similar", job)

    # ---- the grid: image classes, and sorting parts --------------------------------------------
    # Cards are images (of="images", class projects) or parts (of="parts": one card per tracked part, shown by
    # its largest box drawn by a person, else its largest box). Keys are item numbers or part (obj) numbers.
    def _parts(self) -> dict:
        """{obj: (item, box, cls, source)}: each part's representative box."""
        best = {}
        for obj, item, x1, y1, x2, y2, cls, source in self.p.db.execute(
                "SELECT obj, item, x1, y1, x2, y2, cls, source FROM boxes WHERE source != 'suggested'"):
            rank = (source == "manual", (x2 - x1) * (y2 - y1))
            if obj not in best or rank > best[obj][0]:
                best[obj] = (rank, (item, (x1, y1, x2, y2), cls, source))
        return {o: v for o, (_, v) in best.items()}

    def _cards(self, of: str, keys=None) -> list[dict]:
        if of == "images":
            tags = self.p.tags()
            keys = range(len(self.p.items)) if keys is None else keys
            cards = [{"key": k, "name": self.p.items[k]["name"], **({"cls": tags[k]["cls"], "source": tags[k]["source"],
                      "score": tags[k]["score"]} if k in tags else {"cls": None})} for k in keys]
        else:
            parts = self._parts()
            keys = sorted(parts) if keys is None else [k for k in keys if k in parts]
            unit = "frame" if self.p.meta["kind"] == "video" else "image"
            cards = [{"key": o, "name": f"#{o}, {unit} {parts[o][0] + 1}", "item": parts[o][0], "cls": parts[o][2],
                      "source": parts[o][3]} for o in keys]
        if self.thumb_src:
            for c in cards:
                c["src"] = self.thumb_src(of, c["key"])
        return cards

    def thumb_image(self, of: str, key: int) -> Image.Image:
        """A grid thumbnail: the image, or a square around the part with some context."""
        if of == "images":
            with Image.open(self.p.items[key]["path"]) as im:
                im.draft("RGB", (THUMB * 2, THUMB * 2))           # JPEG: decode at a fraction of full size
                img = im.convert("RGB")
        else:
            item, box, *_ = self._parts()[key]
            img = self._crop(self._frame_image(item), box)
        img.thumbnail((THUMB, THUMB))
        return img

    def _frame_image(self, item: int) -> Image.Image:
        """A few recent frames kept decoded: cutting out parts visits each frame several times in a row."""
        with self._frame_lock:                            # page thumbnails and grid jobs share it
            if item not in self._frames:
                self._frames[item] = self.p.image(item)
                while len(self._frames) > 8:
                    self._frames.pop(next(iter(self._frames)))
            return self._frames[item]

    @staticmethod
    def _crop(img: Image.Image, box) -> Image.Image:
        x1, y1, x2, y2 = box
        side, cx, cy = max(x2 - x1, y2 - y1) * 1.2, (x1 + x2) / 2, (y1 + y2) / 2
        return img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))

    def thumb_bytes(self, of: str, key: int) -> bytes:
        k = (of, key, None if of == "images" else self._parts().get(key, (None, None))[1])
        if k not in self._thumbs:
            buf = io.BytesIO()
            self.thumb_image(of, key).save(buf, "JPEG", quality=82)
            self._thumbs[k] = buf.getvalue()
            while len(self._thumbs) > 3000:
                self._thumbs.pop(next(iter(self._thumbs)))
        return self._thumbs[k]

    def on_grid(self, msg):
        of = msg.get("of", "images")
        self.send({"type": "grid", "of": of, "cards": self._cards(of)})

    def on_thumbs(self, msg):
        """Notebooks: thumbnails for the cards in view, as data URLs."""
        of = msg.get("of", "images")
        src = {}
        for key in msg.get("keys", [])[:120]:
            try:
                src[key] = "data:image/jpeg;base64," + base64.b64encode(self.thumb_bytes(of, key)).decode()
            except (KeyError, IndexError, OSError):
                pass
        self.send({"type": "thumbs", "of": of, "src": src})

    def _vectors(self, of: str, progress=None):
        """(keys, features) for every card, computed once and kept in the project's cache/ folder; parts are
        recomputed when their box changes."""
        import hashlib
        if of == "images":
            keys = list(range(len(self.p.items)))
            sig = [f"{it['name']}|{Path(it['path']).stat().st_mtime_ns}" for it in self.p.items]
        else:
            parts = self._parts()
            keys = sorted(parts)
            sig = [f"{o}|{parts[o][0]}|{','.join(f'{v:.0f}' for v in parts[o][1])}" for o in keys]
        ids = [hashlib.sha1(x.encode()).hexdigest()[:16] for x in sig]
        cache = self.p.folder / "cache" / f"vectors_{of}.npz"
        known = {}
        if cache.exists():
            with np.load(cache) as z:
                known = dict(zip(z["ids"].tolist(), z["v"]))
        todo = [k for k, i in enumerate(ids) if i not in known]
        if of == "parts":
            todo.sort(key=lambda k: parts[keys[k]][0])          # frame by frame: each frame is read once
        if todo:
            with self._locks["embed"]:
                emb = self._model("embed")
                for s in range(0, len(todo), 64):
                    if self.stop_flag:
                        raise InterruptedError("stopped")
                    batch = todo[s:s + 64]
                    imgs = []
                    for k in batch:
                        if of == "images":
                            with Image.open(self.p.items[keys[k]]["path"]) as im:
                                im.draft("RGB", (448, 448))
                                imgs.append(im.convert("RGB"))
                        else:
                            item, box, *_ = parts[keys[k]]
                            imgs.append(self._crop(self._frame_image(item), box))
                    for k, v in zip(batch, emb.vectors(imgs, pool="avg")):
                        known[ids[k]] = v
                    if progress:
                        progress(min(s + 64, len(todo)), len(todo))
            cache.parent.mkdir(exist_ok=True)
            np.savez(cache, ids=np.array(list(known)), v=np.stack(list(known.values())))
        return keys, np.stack([known[i] for i in ids]) if ids else np.zeros((0, 1), np.float32)

    def on_sort(self, msg):
        """Group the cards by look (engine/sort.py). With `k`, re-cut the last grouping (the page's slider)."""
        of = msg.get("of", "images")
        st = self.sorting.get(of)
        if msg.get("k") and st and st.get("groups"):
            self._send_groups(of, msg["k"])
            return

        def job():
            from engine.sort import Groups, near_duplicates
            t0 = time.perf_counter()
            keys, v = self._vectors(of, lambda d, n: self.send({"type": "progress", "task": "Reading images"
                                                                if of == "images" else "Reading parts", "done": d, "total": n}))
            if len(keys) < 3:
                self.send({"type": "toast", "text": "Too few to group; add more images or parts first"})
                return
            self.send({"type": "progress", "task": "Grouping", "done": 0, "total": 1})
            groups = Groups(v)
            dups = near_duplicates(v)
            self.sorting[of] = {"keys": keys, "vectors": v, "groups": groups,
                                "dups": [[keys[i] for i in d] for d in dups]}
            self._send_groups(of, msg.get("k"))
            n = self.sorting[of]
            self.notify("info", f"Grouped {len(keys)} {'images' if of == 'images' else 'parts'} into {n['k']} groups",
                        f"{time.perf_counter() - t0:.0f} s. Fewer or more groups: the Groups slider under Sort"
                        + (f". {len(dups)} sets of near-duplicates" if dups else ""))

        self._run("Grouping", job)

    def _send_groups(self, of: str, k=None):
        st = self.sorting[of]
        cut = st["groups"].cut(k)
        st["k"] = cut["k"]
        keys = st["keys"]
        self.send({"type": "groups", "of": of, "k": cut["k"], "best_k": cut["best_k"], "max_k": cut["max_k"],
                   "groups": [[keys[i] for i in g] for g in cut["groups"]], "unsure": [keys[i] for i in cut["unsure"]],
                   "dups": st["dups"]})

    def on_tag(self, msg):
        """Give these cards class `cls` (None clears; images only): images get the class, parts are re-labeled
        on every frame they appear in."""
        of, keys, cls = msg.get("of", "images"), [int(k) for k in msg.get("keys", [])], msg.get("cls")
        if not keys:
            return
        if of == "images":
            self._remember(keys, "classes")
            self.p.set_tags(keys, cls)
        else:
            if cls is None:
                return
            items = sorted({i for o in keys for i in self.p.items_with(o)})
            self._remember(items, "classes")
            for o in keys:
                self.p.set_class(o, int(cls))
        self.send({"type": "cards", "of": of, "cards": self._cards(of, keys)})
        self.send(self.status_msg())

    def on_accept_tags(self, msg):
        keys = [int(k) for k in msg.get("keys") or [k for k, t in self.p.tags().items() if t["source"] == "suggested"]]
        self._remember(keys, "accept")
        n = self.p.accept_tags(keys)
        self.send({"type": "cards", "of": "images", "cards": self._cards("images", keys)})
        self.send(self.status_msg())
        if n:
            self.notify("success", f"Accepted {n} suggested class{'es' if n != 1 else ''}", "")

    def on_suggest_tags(self, msg):
        """Suggest a class for every image nobody has classed yet, from the images already classed."""
        def job():
            from engine.sort import predict
            keys, v = self._vectors("images", lambda d, n: self.send({"type": "progress", "task": "Reading images",
                                                                       "done": d, "total": n}))
            tags = self.p.tags()
            labeled = {k: t["cls"] for k, t in tags.items() if t["source"] != "suggested"}
            got = predict(v, labeled)
            if got is None:
                self.send({"type": "toast", "text": "Give at least one image of two different classes first"})
                return
            cls, score, fits = got
            todo = [k for k in keys if k not in labeled and fits[k]]
            unsure = sum(1 for k in keys if k not in labeled and not fits[k])
            stale = [k for k in keys if k not in labeled and not fits[k] and k in tags]
            self._remember(todo + stale, "suggestions")
            self.p.set_tags(stale, None)                          # earlier guesses that no longer fit
            by_cls = {}
            for k in todo:
                by_cls.setdefault(int(cls[k]), []).append(k)
            for c, ks in by_cls.items():
                self.p.set_tags(ks, c, "suggested", [score[k] for k in ks])
            self.send({"type": "cards", "of": "images", "cards": self._cards("images", todo + stale)})
            self.notify("info", f"Suggested classes for {len(todo)} image{'s' if len(todo) != 1 else ''}",
                        "Dotted cards are suggestions: check them (the least sure are first under Suggested), then "
                        "Accept (Y)." + (f" {unsure} look unlike every class so far and were left without one: name "
                        "some of them, then suggest again." if unsure else ""), None)

        self._run("Suggesting classes", job)

    def on_odd(self, msg):
        """Cards whose class disagrees with their look-alikes: likely labeling mistakes."""
        of = msg.get("of", "images")

        def job():
            from engine.sort import odd_ones
            keys, v = self._vectors(of, lambda d, n: self.send({"type": "progress", "task": "Reading", "done": d, "total": n}))
            if of == "images":
                labeled = {i: t["cls"] for i, t in self.p.tags().items() if t["source"] != "suggested" and i < len(keys)}
                pos = {k: i for i, k in enumerate(keys)}
                idx = {pos[k]: c for k, c in labeled.items()}
            else:
                parts = self._parts()
                idx = {i: parts[k][2] for i, k in enumerate(keys)}
            odd = [[keys[i], c, round(sh, 2)] for i, c, sh in odd_ones(v, idx)]
            self.send({"type": "odd", "of": of, "items": odd})
            self.notify("info" if odd else "success", f"{len(odd)} possible labeling mistake{'s' if len(odd) != 1 else ''}"
                        if odd else "No labeling mistakes found",
                        "Each looks like the class shown on it; check them under To check" if odd else
                        "Every labeled card looks like its class")

        self._run("Checking labels", job)

    def _side(self, cls: int, box) -> int:
        """For left_/right_ twins: the twin whose confirmed boxes sit closer horizontally to `box`."""
        m = PAIR.match(self.p.classes[cls])
        twin = f"{'right' if m and m.group(1) == 'left' else 'left'}_{m.group(2)}" if m else None
        if twin not in self.p.classes:
            return cls
        other = self.p.classes.index(twin)
        mean = lambda c: self.p.db.execute("SELECT AVG((x1 + x2) / 2) FROM boxes WHERE cls=? AND source='manual'",
                                           (c,)).fetchone()[0]
        a, b = mean(cls), mean(other)
        if a is None or b is None:
            return cls
        cx = (box[0] + box[2]) / 2
        return cls if abs(cx - a) <= abs(cx - b) else other

    def on_export(self, msg):
        fmt, reviewed_only = msg.get("format", "yolo"), msg.get("reviewed_only", False)

        def job():
            out = self.p.folder / "exports" / f"{fmt}_{time.strftime('%Y%m%d_%H%M%S')}"
            res = self.p.export(fmt, out, reviewed_only)
            what = {"detect": "boxes", "segment": "outlines"}.get(self.p.task)
            detail = f"{res['images']} images" + (f", {res['boxes']} {what}" if what else "")
            if res.get("no_outline"):
                detail += f" ({res['no_outline']} boxes without an outline left out: use Outline boxes)"
            self.notify("success", f"Exported {FORMAT_NAMES.get(fmt, fmt)} dataset", f"{detail}. {out}",
                        {"type": "folder", "path": str(out), "label": "Open folder"})

        self._run("Exporting", job)
