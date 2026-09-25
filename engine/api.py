"""One open project, shared by every UI host (local web page, notebook widget).

Hosts pass each incoming message to `Session.handle`; everything going back to the screen goes
through the `send` callback, including progress from background jobs (tracking, suggestions,
export). Protocol:

In:  ready | goto {item} | box {item, cls, box, obj?} | click {item, x, y, positive, cls, obj?}
     cycle {item, obj} | delete {item, obj} | set_class {obj, cls, item?} | review {item, value}
     accept {item} | undo | track {item, count, direction} | stop | suggest {item}
     find_all {item, obj} | settings {parent} | export {format, reviewed_only}
     notifications_read | notifications_clear | assistant_info | assistant {id, messages, context, actions}
     assistant_key {key}
Out: project {...} | item {...} | mask {item, obj, src} | status {statuses, flags, busy, gpu, undo}
     item_changed {items} | progress {task, done, total} | toast {text} | error {text}
     notify {item, unread} | notifications {items, unread} | assistant_info {...} | assistant {id, text|done|...}

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

import numpy as np
from PIL import Image

from engine.notify import Notifications

MASK_RGBA = (10, 124, 120, 110)
PAIR = re.compile(r"^(left|right)_(.+)$")
HISTORY = 50                                              # undo steps kept per project
MODEL_NAMES = {"seg": "outline", "track": "tracking", "suggest": "matching", "concept": "find-similar"}
UNDO_WORDS = {"box": "drawing a box", "outline": "outlining a part", "delete": "deleting a box",
              "class change": "a class change", "confirm": "confirming a frame", "accept": "accepting suggestions",
              "tracking": "tracking", "suggestions": "suggestions", "find similar": "find similar"}
FORMAT_NAMES = {"yolo": "YOLO", "coco": "COCO", "cvat": "CVAT", "voc": "Pascal VOC", "labelstudio": "Label Studio"}

_MODELS: dict = {}
_LOCKS = {k: threading.Lock() for k in ("seg", "track", "suggest", "concept", "load")}


def model(name: str, announce=None):
    """The shared model `name` (seg | track | suggest | concept), loaded on first use."""
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
        return {"type": "project", "name": self.p.meta["name"], "kind": self.p.meta["kind"],
                "classes": self.p.classes, "count": len(self.p.items),
                "names": [it["name"] for it in self.p.items], "parent": self.p.meta.get("parent") or "",
                "device": hw.describe(), "formats": list(self.p.FORMATS)}

    def status_msg(self):
        from engine import hw
        return {"type": "status", "statuses": self.p.statuses(), "flags": self.p.flags(), "busy": self.running,
                "gpu": hw.memory(), "undo": len(self.history)}

    def item_msg(self, item: int):
        it = self.p.items[item]
        with Image.open(it["path"]) as im:
            w, h = im.size
        return {"type": "item", "item": item, "name": it["name"], "src": self.image_src(item), "w": w, "h": h,
                "boxes": self.p.boxes(item), "reviewed": self.p.is_reviewed(item)}

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
        obj = msg.get("obj") or self.p.new_obj()
        cls = msg["cls"] if msg.get("obj") is None else next(
            (b["cls"] for b in self.p.boxes(msg["item"]) if b["obj"] == obj), msg["cls"])
        self._remember([msg["item"]], "box")
        self.p.put(msg["item"], obj, cls, msg["box"], "manual")
        self.points.pop((msg["item"], obj), None)
        self.refresh(msg["item"])

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
            seg = self._model("seg")
            key = (str(self.p.folder), item)
            if getattr(seg, "image_key", None) != key:
                seg.set_image(self.p.image(item))
                seg.image_key = key
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

    def on_settings(self, msg):
        parent = (msg.get("parent") or "").strip() or None
        self.p.set_meta(parent=parent)
        self.send(self.project_msg())
        self.notify("success", "Settings saved",
                    f"Parent object: {parent}" if parent else "Parent object: none (suggestions search the whole image)")

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
                n = self._model("track").track(self.p, start, count, on_item, lambda: self.stop_flag, direction)
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
            for k, (cls, box, score) in enumerate(found):
                self.p.put(item, obj + k, cls, box, "suggested", score)
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
            for k, (box, sc) in enumerate(new):
                self.p.put(item, obj0 + k, self._side(ex["cls"], box), box, "suggested", sc)
            self.notify("info", f"Found {len(new)} more like this on {self._frame(item)}",
                        "Y accepts all; select one and press Delete to drop it" if new else
                        "Try a lower threshold or another example", self._goto(item))
            self.refresh(item)

        self._run("Finding similar", job)

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
            self.notify("success", f"Exported {FORMAT_NAMES.get(fmt, fmt)} dataset",
                        f"{res['images']} images, {res['boxes']} boxes. {out}",
                        {"type": "folder", "path": str(out), "label": "Open folder"})

        self._run("Exporting", job)
