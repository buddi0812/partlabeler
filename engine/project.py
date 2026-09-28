"""A PartLabeler project: one or more tasks (videos or image folders), a class list, and every label.

Folder layout:
    project.json     task (label type), classes, and `sources`: the tasks, each {"id", "name", "kind" ("video" |
                     "images"), "source", "every", "frames" (its frame folder), "format", "info" (resolution, fps...)}
    frames/          the first video's sampled frames (f000000.jpg, f000005.jpg, ...); frames/t1/, frames/t2/...
                     for the videos added later. Image folders are read where they are.
    labels.sqlite    boxes, outlines, image classes and review status; every change is committed immediately

Items (frames and images) are numbered across all tasks, in the order the tasks were added, so labels, undo,
suggestions and sorting work over the whole project; tracking stays inside one task.

As in CVAT: labels have colours; a task has a subset (train, validation, test or any word; exports put each
subset in its own folders when there are several) and is split into jobs of `segment_size` items (0 = one job),
each with a stage (annotation, validation, acceptance) and a state (new, in progress, rejected, completed; a new
job counts as in progress once something in it changes). `touched` records when each item last changed, for the
"last updated" of tasks and jobs. Frames are stored as
JPEG at quality 95 with optimized coding ("jpg", visually lossless, the default) or as lossless WebP ("webp",
pixel-exact, about twice the size; S12 in spikes/REPORT.md). Exports hard-link the stored frames instead of
copying them, so exporting takes almost no extra disk space and the pixels stay identical.
Projects made before tasks keep their layout: one task described by the top-level kind/source/every.

The task says what is labeled: "detect" (boxes around parts), "segment" (outlines: a mask per part, its box
is the mask's box) or "classify" (one class per image). Boxes are stored in image pixels (x1, y1, x2, y2),
masks as COCO RLE (engine/masks.py), image classes in `tags`. `source` records where a label came from:
manual (drawn or clicked by a person), tracked, imported, suggested (not yet accepted).
"""
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import av
import numpy as np
from PIL import Image

from engine import masks as M

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TASKS = ("detect", "segment", "classify")
TASK_NAMES = {"detect": "an object detection", "segment": "a segmentation", "classify": "a classification"}
FRAME_FORMATS = ("video", "jpg", "webp")   # frames read from the task's video, or stored as JPEG / lossless WebP
BOX_FORMATS = ("yolo", "coco", "cvat", "voc", "labelstudio")
CLASSIFY_FORMATS = ("folders", "yolo", "csv")
FRAME_RE = re.compile(r"_f(\d+)$")
JOB_STAGES = ("annotation", "validation", "acceptance")
JOB_STATES = ("new", "in progress", "rejected", "completed")
SORTING = ("lexicographical", "natural")
TABLES = ("boxes", "reviewed", "tags", "touched")         # every table with a row per item
OFFSET = 1_000_000_000                                    # parks item numbers while renumbering


def read_classes(path: Path) -> list[str]:
    """Class names from classes.txt (one per line) or an ultralytics data.yaml (names: map or list)."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        m = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
        return [re.sub(r"^\s*\d+:\s*", "", l).strip().strip("'\"") for l in
                text.split("names:", 1)[1].splitlines() if re.match(r"^\s+\d+:", l)]
    return [l.strip() for l in text.splitlines() if l.strip()]


def mirrored_pairs(classes) -> bool:
    """True when class names come in left_/right_ pairs: mirroring an image would swap their meaning,
    so horizontal-flip augmentation must stay off."""
    names = set(classes)
    return any(n.startswith("left_") and "right_" + n[5:] in names for n in names)


class _Rows(list):
    """A statement's rows, already fetched (iterate, fetchone, fetchall, rowcount)."""
    def __init__(self, rows, rowcount):
        super().__init__(rows)
        self.rowcount = rowcount

    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return list(self)


class SharedDB:
    """The project's one SQLite connection, used by the app's request thread and background jobs (tracking writes
    frames while a person confirms others). Python's sqlite3 connection is not safe for that: two threads running the
    same statement share one cached prepared statement ("InterfaceError: bad parameter or other API misuse"). So every
    statement runs under one lock and returns its rows fetched, and a `with db:` transaction holds the lock to its end."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn, self.lock = conn, threading.RLock()

    def execute(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            return _Rows(cur.fetchall(), cur.rowcount)

    def executemany(self, sql, rows):
        with self.lock:
            return self.conn.executemany(sql, rows).rowcount

    def executescript(self, script):
        with self.lock:
            self.conn.executescript(script)

    def commit(self):
        with self.lock:
            self.conn.commit()

    def backup(self, target, **kw):
        with self.lock:
            self.conn.backup(target, **kw)

    def close(self):
        with self.lock:
            self.conn.close()

    def __enter__(self):
        self.lock.acquire()
        try:
            return self.conn.__enter__()
        except BaseException:
            self.lock.release()
            raise

    def __exit__(self, *exc):
        try:
            return self.conn.__exit__(*exc)
        finally:
            self.lock.release()


class Project:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "project.json").read_text(encoding="utf-8"))
        self.db = SharedDB(sqlite3.connect(self.folder / "labels.sqlite", check_same_thread=False))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS boxes (item INTEGER, obj INTEGER, cls INTEGER, x1 REAL, y1 REAL,
                x2 REAL, y2 REAL, source TEXT, score REAL, PRIMARY KEY (item, obj));
            CREATE TABLE IF NOT EXISTS reviewed (item INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS tags (item INTEGER PRIMARY KEY, cls INTEGER, source TEXT, score REAL);
            CREATE TABLE IF NOT EXISTS touched (item INTEGER PRIMARY KEY, t REAL);
        """)
        if "rle" not in {r[1] for r in self.db.execute("PRAGMA table_info(boxes)")}:
            self.db.execute("ALTER TABLE boxes ADD COLUMN rle TEXT")      # projects made before outlines
            self.db.commit()
        self._index()
        self._ensure_jobs()

    # ---- creation -------------------------------------------------------------------------
    @classmethod
    def create(cls, folder, classes: list[str], video=None, images=None, every: int = 5, progress=None,
               task: str = "detect", frame_format: str = "jpg", colors=None, **task_options):
        """A new project, with a first task when a video or an image folder is given (add more with add_source,
        as CVAT's "Create a new task"). `colors`: one hex colour per class (None: the default palette)."""
        folder = Path(folder)
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose one of {', '.join(TASKS)}")
        if frame_format not in FRAME_FORMATS:
            raise ValueError(f"unknown frame format {frame_format!r}; choose one of {', '.join(FRAME_FORMATS)}")
        if (folder / "project.json").exists():
            raise FileExistsError(f"{folder} already holds a project; open it instead")
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"name": folder.name, "task": task, "classes": list(classes), "parent": None,
                "frame_format": frame_format, "sources": [], "created": _now(), "next_job": 1}
        if colors:
            meta["colors"] = list(colors)
        (folder / "project.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        p = cls(folder)
        if video or images:
            p.add_source(video=video, images=images, every=every, progress=progress, **task_options)
        return p

    @property
    def sources(self) -> list[dict]:
        """The tasks: videos and image folders (see the module notes)."""
        if "sources" in self.meta:
            return self.meta["sources"]
        if "source" not in self.meta:
            return []
        src = Path(self.meta["source"])                   # a project made before tasks
        return [{"id": 0, "name": src.stem if self.meta["kind"] == "video" else src.name, "kind": self.meta["kind"],
                 "source": self.meta["source"], "every": self.meta.get("every", 1), "frames": "frames", "format": "jpg",
                 "info": self.meta.get("info")}]

    def add_source(self, video=None, images=None, every: int = 5, progress=None, name: str | None = None,
                   subset: str = "", start: int | None = None, stop: int | None = None, quality: int = 95,
                   lossless: bool | None = None, sorting: str = "lexicographical", segment_size: int = 0,
                   frames: str | None = None) -> dict:
        """Add a task (CVAT's "Create a new task"): a video, of which every `every`-th frame from `start` to `stop`
        is kept (`frames`: "video" = read from the video itself, no image files; "jpg" of `quality`; "webp"
        lossless; default: the project's choice), or an image folder read in place,
        in `sorting` order. The task is split into jobs of `segment_size` items (0 = one job). Returns the task;
        items already in the project keep their numbers."""
        if not (video or images) or (video and images):
            raise ValueError("give a video or an image folder")
        path = Path(video or images).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        tasks = [dict(t) for t in self.sources]
        tid = max((t["id"] for t in tasks), default=0) + 1      # as CVAT: numbered from 1 (older projects began at 0)
        base = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or (path.stem if video else path.name))) or "task"
        taken, name = {t["name"] for t in tasks}, None
        for k in range(1, 1000):
            name = base if k == 1 else f"{base}_{k}"
            if name not in taken:
                break
        fmt = frames or (("webp" if lossless else "jpg") if lossless is not None else self.meta.get("frame_format", "jpg"))
        if fmt not in FRAME_FORMATS:
            raise ValueError(f"unknown frame storage {fmt!r}; choose one of {', '.join(FRAME_FORMATS)}")
        if sorting not in SORTING:
            raise ValueError(f"unknown sorting {sorting!r}; choose one of {', '.join(SORTING)}")
        t = {"id": tid, "name": name, "kind": "video" if video else "images", "source": str(path),
             "every": max(1, int(every)) if video else 1, "frames": "frames" if tid == 0 else f"frames/t{tid}",
             "format": fmt, "quality": int(quality), "subset": (subset or "").strip(), "created": _now(),
             "segment_size": max(0, int(segment_size or 0))}
        if video:
            t.update(start=start, stop=stop)
            t["info"] = probe_video(path)
            kept, stored = (index_video(path, self.folder / t["frames"], t["every"], progress, start=start, stop=stop)
                            if fmt == "video" else
                            extract_frames(path, self.folder / t["frames"], t["every"], fmt, progress,
                                           start=start, stop=stop, quality=int(quality)))
            t["info"].update(kept=kept, stored=stored)
        else:
            t["sorting"] = sorting
            t["info"] = probe_images(path)
        tasks.append(t)
        first = tasks[0]
        self.set_meta(sources=tasks, kind=first["kind"], source=first["source"], every=first["every"])
        self._index()
        self._ensure_jobs()
        return next(x for x in self.sources if x["id"] == tid)

    def update_source(self, tid: int, name: str | None = None, subset: str | None = None) -> dict:
        """Rename a task and/or set its subset (train, validation, test or any word; "" for none)."""
        tasks = [dict(t) for t in self.sources]
        t = next((x for x in tasks if x["id"] == tid), None)
        if t is None:
            raise KeyError(f"no task {tid}")
        if name is not None:
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name.strip())
            if not name:
                raise ValueError("give the task a name")
            if any(x["name"] == name and x["id"] != tid for x in tasks):
                raise ValueError(f"there is already a task called {name}")
            t["name"] = name
        if subset is not None:
            t["subset"] = subset.strip()
        first = tasks[0]
        self.set_meta(sources=tasks, kind=first["kind"], source=first["source"], every=first["every"])
        self._index()
        return t

    def remove_source(self, tid: int) -> dict:
        """Delete a task: its labels and stored frames (an image folder stays where it is). The items after it
        move down to keep the numbering continuous. Returns {"task", "items", "labels"} removed."""
        tasks = [dict(t) for t in self.sources]
        t = next((x for x in tasks if x["id"] == tid), None)
        if t is None:
            raise KeyError(f"no task {tid}")
        if len(tasks) == 1:
            raise ValueError("a project keeps at least one task: delete the project instead")
        a, b = self.ranges.get(tid, (0, 0))
        n = b - a
        with self.db:
            labels = self.db.execute("SELECT COUNT(*) FROM boxes WHERE item >= ? AND item < ?", (a, b)).fetchone()[0]
            labels += self.db.execute("SELECT COUNT(*) FROM tags WHERE item >= ? AND item < ?", (a, b)).fetchone()[0]
            for table in TABLES:
                self.db.execute(f"DELETE FROM {table} WHERE item >= ? AND item < ?", (a, b))
                # two steps, so no row ever lands on a number another row still has
                self.db.execute(f"UPDATE {table} SET item = item + ? WHERE item >= ?", (OFFSET, b))
                self.db.execute(f"UPDATE {table} SET item = item - ? - ? WHERE item >= ?", (OFFSET, n, OFFSET))
        if t["kind"] == "video":
            d = self.folder / t["frames"]
            for f in [*d.glob("f*.*"), *d.glob("video.*"), d / "index.json"]:   # only this task's: frames/ also holds t1/...
                if f.is_file() and (f.stem[1:].isdigit() or f.stem in ("video", "index")):
                    close_video(f)
                    f.unlink()
            if d != self.folder / "frames" and d.exists() and not any(d.iterdir()):
                d.rmdir()
        tasks = [x for x in tasks if x["id"] != tid]
        first = tasks[0]
        self.set_meta(sources=tasks, kind=first["kind"], source=first["source"], every=first["every"])
        self._index()
        return {"task": t, "items": n, "labels": labels}

    def fill_info(self) -> bool:
        """Video details for tasks that have none (projects made before tasks): read once from the source and
        the stored frames, then saved. Returns True when something was added."""
        tasks, changed = [dict(t) for t in self.sources], False
        for t in tasks:
            if t.get("info") is not None:
                continue
            try:
                if t["kind"] == "video":
                    t["info"] = probe_video(Path(t["source"])) if Path(t["source"]).is_file() else {}
                    frames = [it for it in self.items if it["task"] == t["id"]]
                    stored = (frames[0]["video"].stat().st_size if frames and frames[0].get("video")
                              else sum(Path(it["path"]).stat().st_size for it in frames))
                    t["info"].update(kept=len(frames), stored=stored)
                else:
                    t["info"] = probe_images(Path(t["source"]))
                changed = True
            except (OSError, av.error.FFmpegError, ValueError, IndexError):
                t["info"] = {}                                # a moved or unreadable source: details unknown
                changed = True
        if changed:
            first = tasks[0]
            self.set_meta(sources=tasks, kind=first["kind"], source=first["source"], every=first["every"])
        return changed

    def add_tasks_from(self, zip_path) -> dict:
        """Add the tasks of a backup zip (one task's, or a whole project's) with their frames, labels, confirmed
        frames and jobs, so unfinished work carries on here. Labels are matched by name; names this project lacks
        are added. Returns {"tasks": [names], "items", "labels_added": [names]}."""
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix=".import_", dir=self.folder))    # same drive: files move, not copy
        src = None
        try:
            src = Project(restore_project(zip_path, tmp))
            if src.task != self.task:
                raise ValueError(f"That backup is {TASK_NAMES[src.task]} project and this one is {TASK_NAMES[self.task]}: "
                                 "open it as a project of its own instead (Create from backup)")
            classes, have_colors = list(self.classes), bool(self.meta.get("colors") or src.meta.get("colors"))
            colors = (list(self.meta.get("colors") or []) + [None] * len(classes))[:len(classes)]
            src_colors, cmap, added = src.meta.get("colors") or [], {}, []
            for i, name in enumerate(src.classes):
                if name not in classes:
                    classes.append(name)
                    colors.append(src_colors[i] if i < len(src_colors) else None)
                    added.append(name)
                cmap[i] = classes.index(name)
            tasks, names = [dict(t) for t in self.sources], []
            tid = max((t["id"] for t in tasks), default=0) + 1
            nxt, base, obj0, count = int(self.meta.get("next_job", 1)), len(self.items), self.new_obj(), 0
            rows = {table: [] for table in TABLES}
            cols = {table: [r[1] for r in src.db.execute(f"PRAGMA table_info({table})")] for table in TABLES}
            for t in src.sources:
                a, b = src.ranges.get(t["id"], (0, 0))
                nt = dict(t, id=tid, created=t.get("created") or _now())
                taken = {x["name"] for x in tasks}
                nt["name"] = next(n for n in (t["name"] if k == 1 else f"{t['name']}_{k}" for k in range(1, 10000)) if n not in taken)
                if t["kind"] == "video":
                    dest = self.folder / "frames" / f"t{tid}"
                    dest.mkdir(parents=True, exist_ok=True)
                    for f in (src.folder / t["frames"]).iterdir():    # frames, or the video and its index
                        if f.is_file():
                            f.replace(dest / f.name)
                    nt["frames"] = f"frames/t{tid}"
                else:
                    dest = self.folder / "images" / f"t{tid}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Path(t["source"]).replace(dest)
                    nt["source"] = str(dest)
                nt["jobs"] = []
                for j in t.get("jobs") or []:
                    nt["jobs"].append(dict(j, id=nxt))
                    nxt += 1
                for table in TABLES:
                    for r in src.db.execute(f"SELECT {', '.join(cols[table])} FROM {table} WHERE item >= ? AND item < ?", (a, b)):
                        r = dict(zip(cols[table], r))
                        r["item"] += base - a
                        if r.get("cls") is not None:
                            r["cls"] = cmap.get(r["cls"], r["cls"])
                        if table == "boxes":
                            r["obj"] += obj0
                        rows[table].append(r)
                base += b - a
                count += b - a
                tasks.append(nt)
                names.append(nt["name"])
                tid += 1
            with self.db:
                for table, rs in rows.items():
                    for r in rs:
                        self.db.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(r)}) VALUES ({', '.join('?' * len(r))})",
                                        list(r.values()))
            first = tasks[0]
            meta = dict(sources=tasks, classes=classes, next_job=nxt, kind=first["kind"], source=first["source"], every=first["every"])
            if have_colors:
                meta["colors"] = colors
            self.set_meta(**meta)
            self._index()
            self._ensure_jobs()
            return {"tasks": names, "items": count, "labels_added": added}
        finally:
            if src is not None:
                src.db.close()
            shutil.rmtree(tmp, ignore_errors=True)

    def frames_to_video(self, tid: int, old_to: Path) -> dict:
        """Switch a video task from stored frame files to frames read from its video: the same frames with the same
        numbers, so its labels stay. Pixels are checked first (lossless frames must match exactly, JPEG ones
        closely); the old files move to `old_to`/t<id> (delete that folder when happy)."""
        tasks = [dict(t) for t in self.sources]
        t = next((x for x in tasks if x["id"] == tid), None)
        if t is None or t["kind"] != "video" or t.get("format") == "video":
            raise ValueError(f"task {tid} is not a video task with stored frame files")
        src = Path(t["source"])
        if not src.is_file():
            raise FileNotFoundError(f"the task's video is not there any more: {src}")
        d = self.folder / t["frames"]
        old = {int(p.stem[1:]): p for p in d.glob("f*.*") if p.is_file() and p.stem[1:].isdigit()}
        dst = d / ("video" + src.suffix.lower())
        place_image(src, dst, link=True)
        found = dict(zip(*_frame_times(dst, lambda i: i in old, None)))
        if set(old) - set(found):
            raise ValueError(f"{len(set(old) - set(found))} stored frames are not in the video; nothing changed")
        frames = sorted(old)
        reader = VideoFrames(dst)
        try:
            for i in frames[:: max(1, len(frames) // 5)][:6]:
                a = np.asarray(reader.frame(found[i]), np.int16)
                with Image.open(old[i]) as im:
                    b = np.asarray(im.convert("RGB"), np.int16)
                diff = np.abs(a - b).mean() if a.shape == b.shape else 255
                if diff > (0 if old[i].suffix.lower() == ".webp" else 3):
                    raise ValueError(f"frame {i} decodes differently from the stored one ({diff:.1f}); nothing changed")
        finally:
            reader.close()
        (d / "index.json").write_text(json.dumps({"video": dst.name, "frames": frames, "pts": [found[i] for i in frames]}),
                                      encoding="utf-8")
        t["format"] = "video"
        t["info"] = {**(t.get("info") or {}), "kept": len(frames), "stored": dst.stat().st_size}
        first = tasks[0]
        self.set_meta(sources=tasks, kind=first["kind"], source=first["source"], every=first["every"])
        self._index()                                     # from here on frames come from the video...
        freed = sum(p.stat().st_size for p in old.values())
        keep = Path(old_to) / f"t{tid}"
        keep.mkdir(parents=True, exist_ok=True)
        for p in old.values():                            # ...so the old files can go (to a folder, to delete later)
            shutil.move(str(p), str(keep / p.name))
        return {"task": t["name"], "frames": len(frames), "freed": freed, "old": str(keep)}

    def _index(self) -> None:
        self.items = self._list_items()
        self.ranges = {}                                  # task id -> (first item, one past the last)
        for k, it in enumerate(self.items):
            a, _ = self.ranges.get(it["task"], (k, k))
            self.ranges[it["task"]] = (a, k + 1)

    def _list_items(self) -> list[dict]:
        items, first = [], min((t["id"] for t in self.sources), default=0)
        for t in self.sources:
            if t["kind"] == "video" and t.get("format") == "video":       # frames read from the video
                d = self.folder / t["frames"]
                idx = json.loads((d / "index.json").read_text(encoding="utf-8")) if (d / "index.json").exists() else \
                    {"video": "", "frames": [], "pts": []}
                items += [{"name": f"{t['name']}_f{i:06d}", "path": None, "video": d / idx["video"], "pts": pts,
                           "frame": i, "task": t["id"]} for i, pts in zip(idx["frames"], idx["pts"])]
            elif t["kind"] == "video":
                files = sorted(p for p in (self.folder / t["frames"]).glob("f*.*")
                               if p.suffix.lower() in (".jpg", ".webp") and p.stem[1:].isdigit())
                items += [{"name": f"{t['name']}_{p.stem}", "path": p, "frame": int(p.stem[1:]), "task": t["id"]}
                          for p in files]
            else:
                root = Path(t["source"])
                prefix = "" if t["id"] == first else f"{t['name']}/"   # names stay unique across folders
                files = sorted((p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS),
                               key=_natural if t.get("sorting") == "natural" else None) if root.is_dir() else []
                items += [{"name": prefix + str(p.relative_to(root).with_suffix("")).replace("\\", "/"), "path": p,
                           "frame": None, "task": t["id"]} for p in files]
        return items

    # ---- jobs (CVAT: a task's frames in segments, each with a stage and a state) ---------------
    def _ensure_jobs(self) -> None:
        """Every task gets its jobs (projects made before jobs: one job per task)."""
        if not any("jobs" not in t for t in self.sources):
            return
        tasks, nxt = [dict(t) for t in self.sources], int(self.meta.get("next_job", 1))
        for t in tasks:
            if "jobs" in t:
                continue
            a, b = self.ranges.get(t["id"], (0, 0))
            size = int(t.get("segment_size") or 0)
            spans = [(0, max(0, b - a - 1))] if not size or size >= b - a else \
                [(s, min(b - a, s + size) - 1) for s in range(0, b - a, size)]
            t["jobs"] = []
            for s0, s1 in spans:
                t["jobs"].append({"id": nxt, "start": s0, "stop": s1, "stage": "annotation", "state": "new",
                                  "created": t.get("created") or _now(), "state_set": 0})
                nxt += 1
            t.setdefault("created", self.meta.get("created") or _now())
        first = tasks[0]
        self.set_meta(sources=tasks, next_job=nxt, kind=first["kind"], source=first["source"], every=first["every"])

    def jobs(self, task_id: int | None = None) -> list[dict]:
        """Jobs with their item ranges [start, end) across the project, effective state, progress and last change."""
        touched = dict(self.db.execute("SELECT item, t FROM touched"))
        st = self.statuses()
        out = []
        for t in self.sources:
            if task_id is not None and t["id"] != task_id:
                continue
            a, b = self.ranges.get(t["id"], (0, 0))
            js = t.get("jobs") or []
            for k, j in enumerate(js):
                s0 = min(b, a + j["start"])
                e = b if k == len(js) - 1 else min(b, a + j["stop"] + 1)   # the last job reaches the task's end
                last = max((touched.get(i, 0) for i in range(s0, e)), default=0)
                state = j.get("state", "new")
                if state == "new" and last > j.get("state_set", 0):      # as CVAT: work started, so in progress
                    state = "in progress"
                part = st[s0:e]
                out.append({"id": j["id"], "task": t["id"], "task_name": t["name"], "subset": t.get("subset", ""),
                            "start": s0, "end": e, "frames": e - s0, "stage": j.get("stage", "annotation"),
                            "state": state, "created": j.get("created"), "updated": _iso(last) if last else None,
                            "labeled": sum(x >= 2 for x in part), "confirmed": sum(x == 4 for x in part)})
        return out

    def job(self, jid: int) -> dict:
        found = [j for j in self.jobs() if j["id"] == jid]
        if not found:
            raise KeyError(f"no job {jid}")
        return found[0]

    def set_job(self, jid: int, stage: str | None = None, state: str | None = None) -> dict:
        """Change a job's stage and/or state. As in CVAT, a new stage without a state starts it as "new"."""
        if stage is not None and stage not in JOB_STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        if state is not None and state not in JOB_STATES:
            raise ValueError(f"unknown state {state!r}")
        tasks = [dict(t, jobs=[dict(j) for j in t.get("jobs", [])]) for t in self.sources]
        for t in tasks:
            for j in t["jobs"]:
                if j["id"] == jid:
                    if stage is not None:
                        j["stage"] = stage
                        j["state"] = state or "new"
                    elif state is not None:
                        j["state"] = state
                    j["state_set"] = time.time()
                    self.set_meta(sources=tasks)
                    return self.job(jid)
        raise KeyError(f"no job {jid}")

    @staticmethod
    def task_status(jobs: list[dict]) -> dict:
        """CVAT's task status and progress from its jobs: done = acceptance and completed, on review =
        validation stage, annotating = the rest."""
        done = sum(j["stage"] == "acceptance" and j["state"] == "completed" for j in jobs)
        review = sum(j["stage"] == "validation" for j in jobs)
        status = "annotation" if any(j["stage"] == "annotation" for j in jobs) else \
            "validation" if any(j["stage"] == "validation" for j in jobs) else "completed"
        return {"status": status, "done": done, "review": review, "annotating": len(jobs) - done - review, "total": len(jobs)}

    def _touch(self, items) -> None:
        now = time.time()
        self.db.executemany("INSERT OR REPLACE INTO touched VALUES (?, ?)", [(int(i), now) for i in set(items)])

    # ---- labels (CVAT's label constructor: names and colours) -------------------------------------
    def labels(self) -> list[dict]:
        colors = self.meta.get("colors") or []
        return [{"name": n, "color": colors[i] if i < len(colors) else None} for i, n in enumerate(self.classes)]

    def set_labels(self, new: list[dict]) -> dict:
        """Replace the labels: [{"name", "color", "from": old index or None (a new label)}]. Labels left out are
        deleted with their annotations; the others keep theirs (renamed, recoloured or moved)."""
        names = [" ".join(str(l.get("name", "")).split()) for l in new]
        if any(not n for n in names):
            raise ValueError("every label needs a name")
        if len(set(names)) != len(names):
            raise ValueError("label names must differ")
        for l in new:
            if l.get("color") and not re.fullmatch(r"#[0-9a-fA-F]{6}", l["color"]):
                raise ValueError(f"not a colour: {l['color']!r} (use #rrggbb)")
        old = len(self.classes)
        moved = {int(l["from"]): i for i, l in enumerate(new) if l.get("from") is not None and 0 <= int(l["from"]) < old}
        gone = [i for i in range(old) if i not in moved]
        with self.db:
            removed = 0
            for table in ("boxes", "tags"):
                if gone:
                    q = ",".join("?" * len(gone))
                    removed += self.db.execute(f"DELETE FROM {table} WHERE cls IN ({q})", gone).rowcount
                if moved and any(k != v for k, v in moved.items()):
                    case = " ".join(f"WHEN {k} THEN {v}" for k, v in moved.items())
                    self.db.execute(f"UPDATE {table} SET cls = CASE cls {case} END")
        self.set_meta(classes=names, colors=[l.get("color") for l in new])
        return {"labels": self.labels(), "removed": removed}

    def task_of(self, item: int) -> dict:
        tid = self.items[item]["task"]
        return next(t for t in self.sources if t["id"] == tid)

    # ---- reading --------------------------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return self.meta["classes"]

    @property
    def task(self) -> str:
        return self.meta.get("task", "detect")

    FORMATS = BOX_FORMATS

    @property
    def formats(self) -> tuple:
        """Export formats for this project's task."""
        return CLASSIFY_FORMATS if self.task == "classify" else BOX_FORMATS

    def image(self, item: int) -> Image.Image:
        it = self.items[item]
        if it.get("video"):
            return _reader(it["video"]).frame(it["pts"])
        return Image.open(it["path"]).convert("RGB")

    def image_size(self, item: int) -> tuple[int, int]:
        it = self.items[item]
        if it.get("video"):
            info = self.task_of(item).get("info") or {}
            return (info["width"], info["height"]) if info.get("width") else self.image(item).size
        with Image.open(it["path"]) as im:
            return im.size

    def thumbnail(self, item: int, side: int) -> Image.Image:
        """A small copy of the image (JPEG files decode at a fraction of full size)."""
        it = self.items[item]
        if it.get("video"):
            img = self.image(item)
        else:
            with Image.open(it["path"]) as im:
                im.draft("RGB", (side * 2, side * 2))
                img = im.convert("RGB")
        img.thumbnail((side, side))
        return img

    def file_of(self, item: int) -> Path:
        """The item's image file; for frames read from a video, the name it gets when exported (not on disk)."""
        it = self.items[item]
        return Path(it["path"]) if it["path"] else it["video"].parent / f"f{it['frame']:06d}.jpg"

    def place(self, item: int, dst: Path) -> None:
        """Put the item's image at dst: a hard link to a stored frame, a copy of a picture, or a video frame saved
        as JPEG (quality 95)."""
        it = self.items[item]
        if it.get("video"):
            self.image(item).save(dst, "JPEG", quality=95, optimize=True)
        else:
            path = Path(it["path"])
            place_image(path, dst, link=self.folder in path.parents)

    def boxes(self, item: int) -> list[dict]:
        rows = self.db.execute("SELECT obj, cls, x1, y1, x2, y2, source, score FROM boxes WHERE item=? ORDER BY obj",
                               (item,)).fetchall()
        return [{"obj": o, "cls": c, "box": [x1, y1, x2, y2], "source": s, "score": sc}
                for o, c, x1, y1, x2, y2, s, sc in rows]

    def is_reviewed(self, item: int) -> bool:
        return self.db.execute("SELECT 1 FROM reviewed WHERE item=?", (item,)).fetchone() is not None

    def masks(self, item: int) -> dict:
        """{obj: RLE text} for this item's objects that have an outline."""
        return dict(self.db.execute("SELECT obj, rle FROM boxes WHERE item=? AND rle IS NOT NULL", (item,)))

    def mask(self, item: int, obj: int):
        row = self.db.execute("SELECT rle FROM boxes WHERE item=? AND obj=?", (item, obj)).fetchone()
        return M.decode(row[0]) if row and row[0] else None

    def statuses(self) -> list[int]:
        """Per item: 0 empty, 1 suggestions only, 2 tracked/imported, 3 has manual boxes, 4 reviewed.
        Image classes: 0 none, 1 suggested, 2 imported, 4 given by a person."""
        out = [0] * len(self.items)
        rank = {"suggested": 1, "tracked": 2, "imported": 2, "manual": 3}
        if self.task == "classify":
            for item, source in self.db.execute("SELECT item, source FROM tags"):
                if item < len(out):
                    out[item] = 4 if source == "manual" else rank.get(source, 2)
            return out
        for item, source in self.db.execute("SELECT item, source FROM boxes"):
            out[item] = max(out[item], rank.get(source, 2))
        for (item,) in self.db.execute("SELECT item FROM reviewed"):
            out[item] = 4
        return out

    def flags(self, size: float = 1.5, moved: float = 0.3) -> list[int]:
        """Unreviewed items worth a look. A tracked box is suspect when its area differs from the
        person's nearest box of that part by more than `size` x, or its centre moved more than `moved`
        x its diagonal since the previous item; a part lost for a while is suspect where it went.
        (S4c, 1,794 tracked boxes: these catch 57% of wrong boxes and 2 of 3 flags are real; the
        tracker's own score adds nothing on top, and Laya zero-shot did worse than chance.)"""
        rows = self.db.execute("SELECT obj, item, x1, y1, x2, y2, source FROM boxes "
                               "WHERE source != 'suggested' ORDER BY obj, item").fetchall()
        reviewed = {r[0] for r in self.db.execute("SELECT item FROM reviewed")}
        anchors: dict[int, list] = {}
        for obj, item, x1, y1, x2, y2, source in rows:
            if source == "manual":
                anchors.setdefault(obj, []).append((item, (x2 - x1) * (y2 - y1)))
        flagged, prev = set(), None
        for obj, item, x1, y1, x2, y2, source in rows:
            if source in ("tracked", "imported"):
                a = (x2 - x1) * (y2 - y1)
                if obj in anchors:
                    ref = min(anchors[obj], key=lambda ia: abs(ia[0] - item))[1]
                    if not 1 / size <= (a + 1) / (ref + 1) <= size:
                        flagged.add(item)
                if prev and prev[0] == obj:
                    _, pitem, px1, py1, px2, py2 = prev
                    if item == pitem + 1:
                        diag = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5 or 1.0
                        shift = (((x1 + x2) - (px1 + px2)) ** 2 + ((y1 + y2) - (py1 + py2)) ** 2) ** 0.5 / 2
                        if shift > moved * diag:
                            flagged.add(item)
                    elif item > pitem + 1:
                        flagged.add(pitem + 1)                  # part was lost for a while
            prev = (obj, item, x1, y1, x2, y2)
        return sorted(flagged - reviewed)

    def anchors(self, obj: int) -> list[int]:
        """Items where a person vouched for this part: drew it, or confirmed the frame it was tracked onto."""
        return [r[0] for r in self.db.execute(
            "SELECT item FROM boxes WHERE obj=? AND (source='manual' OR item IN (SELECT item FROM reviewed)) ORDER BY item",
            (obj,))]

    # ---- writing --------------------------------------------------------------------------
    def new_obj(self) -> int:
        return (self.db.execute("SELECT MAX(obj) FROM boxes").fetchone()[0] or 0) + 1

    def put(self, item: int, obj: int, cls: int, box, source: str = "manual", score=None, mask=None) -> None:
        """With a mask (outline projects), the box is the mask's box. Without one, any old mask is dropped."""
        rle = None
        if mask is not None:
            box = M.box_of(mask) or box
            rle = M.encode(mask)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO boxes VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (item, obj, cls, *map(float, box), source, score, rle))
            self._touch([item])

    def delete(self, item: int, obj: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM boxes WHERE item=? AND obj=?", (item, obj))
            self._touch([item])

    def set_class(self, obj: int, cls: int, item: int | None = None) -> None:
        """A part's class on every item, or on one item only (a part that changes look: its state there)."""
        with self.db:
            if item is None:
                self.db.execute("UPDATE boxes SET cls=? WHERE obj=?", (cls, obj))
                self._touch(self.items_with(obj))
            else:
                self.db.execute("UPDATE boxes SET cls=? WHERE obj=? AND item=?", (cls, obj, item))
                self._touch([item])

    def same_part(self, a: int, b: int) -> bool:
        """Classes a and b are states of one part (project "states")."""
        names = {self.classes[a], self.classes[b]}
        return a != b and any(names <= set(g) for g in self.meta.get("states") or [])

    def set_reviewed(self, item: int, value: bool = True) -> None:
        with self.db:
            self._touch([item])
            if value:
                self.db.execute("INSERT OR IGNORE INTO reviewed VALUES (?)", (item,))
                self.db.execute("UPDATE boxes SET source='manual' WHERE item=? AND source='suggested'", (item,))
            else:
                self.db.execute("DELETE FROM reviewed WHERE item=?", (item,))

    def accept(self, item: int) -> None:
        with self.db:
            self._touch([item])
            self.db.execute("UPDATE boxes SET source='manual' WHERE item=? AND source='suggested'", (item,))

    def put_tracked(self, item: int, results: dict, replace: bool = False) -> None:
        """Replace this item's tracked boxes for the given objects; manual/reviewed boxes are kept.
        results: {obj: (cls, box or None, score[, mask])}; a mask replaces the box with its own.
        replace: also drop this item's other tracked boxes (re-tracking: parts gone from the start frame go here too)."""
        with self.db:
            if self.is_reviewed(item):
                return
            self._touch([item])
            if replace:
                self.db.execute(f"DELETE FROM boxes WHERE item=? AND source='tracked' AND obj NOT IN ({','.join('?' * len(results))})",
                                (item, *results))
            for obj, (cls, box, score, *rest) in results.items():
                mask = rest[0] if rest else None
                rle = None
                if mask is not None and box is not None:
                    box, rle = M.box_of(mask) or box, M.encode(mask)
                row = self.db.execute("SELECT source FROM boxes WHERE item=? AND obj=?", (item, obj)).fetchone()
                if row and row[0] == "manual":
                    continue
                if box is None:
                    self.db.execute("DELETE FROM boxes WHERE item=? AND obj=? AND source='tracked'", (item, obj))
                else:
                    self.db.execute("INSERT OR REPLACE INTO boxes VALUES (?,?,?,?,?,?,?,?,?,?)",
                                    (item, obj, cls, *map(float, box), "tracked", score, rle))

    def swap_classes(self, items, src: int, dst: int, both: bool = False) -> tuple[int, list[int]]:
        """Parts of class `src` on these items become `dst`; with `both`, parts of `dst` become `src` at the same time.
        Returns (parts changed, items that changed)."""
        if both:
            sql, args = "UPDATE boxes SET cls = CASE cls WHEN ? THEN ? ELSE ? END WHERE item=? AND cls IN (?, ?)", (src, dst, src)
        else:
            sql, args = "UPDATE boxes SET cls = ? WHERE item=? AND cls = ?", (dst,)
        n, changed = 0, []
        with self.db:
            for k in items:
                done = self.db.execute(sql, (*args, k, src, dst) if both else (*args, k, src)).rowcount
                if done:
                    n += done
                    changed.append(k)
            self._touch(changed)
        return n, changed

    def clear_parts(self, items, classes=None) -> tuple[int, list[int]]:
        """Delete the parts of `classes` (class indices; None = every class) on these items.
        Returns (parts deleted, items that changed)."""
        n, changed = 0, []
        where = "" if classes is None else f" AND cls IN ({','.join('?' * len(classes))})"
        with self.db:
            for k in items:
                gone = self.db.execute(f"DELETE FROM boxes WHERE item=?{where}", (k, *(classes or []))).rowcount
                if gone:
                    n += gone
                    changed.append(k)
            self._touch(changed)
        return n, changed

    # ---- image classes (classify projects) ---------------------------------------------------
    def tags(self) -> dict:
        """{item: {"cls", "source", "score"}} for every image with a class (given or suggested)."""
        return {i: {"cls": c, "source": s, "score": sc} for i, c, s, sc in self.db.execute("SELECT * FROM tags")}

    def set_tags(self, items, cls, source: str = "manual", scores=None) -> None:
        """Give these images class `cls` (None clears it). Suggestions never replace a person's or imported class."""
        with self.db:
            self._touch(items)
            for k, item in enumerate(items):
                if cls is None:
                    self.db.execute("DELETE FROM tags WHERE item=?", (item,))
                    continue
                if source == "suggested":
                    row = self.db.execute("SELECT source FROM tags WHERE item=?", (item,)).fetchone()
                    if row and row[0] != "suggested":
                        continue
                self.db.execute("INSERT OR REPLACE INTO tags VALUES (?,?,?,?)",
                                (item, int(cls), source, None if scores is None else float(scores[k])))

    def accept_tags(self, items) -> int:
        with self.db:
            self._touch(items)
            return sum(self.db.execute("UPDATE tags SET source='manual' WHERE item=? AND source='suggested'",
                                       (i,)).rowcount for i in items)

    # ---- import / export ------------------------------------------------------------------
    def _match(self, stem: str, scope=None):
        by_name = {it["name"]: k for k, it in enumerate(self.items) if scope is None or k in scope}
        for name in (stem, stem.replace("__", "/")):          # exports flatten sub-folders with "__"
            if name in by_name:
                return by_name[name]
        m = FRAME_RE.search(stem)
        tasks = {self.items[k]["task"] for k in scope} if scope is not None else {t["id"] for t in self.sources}
        if m and stem[:m.start()] in {t["name"] for t in self.sources if t["id"] not in tasks}:
            return None                                   # a file of another task in this project
        if m and len(tasks) == 1 and self.task_of(next(iter(scope)) if scope else 0)["kind"] == "video":
            by_frame = {it["frame"]: k for k, it in enumerate(self.items)                  # one video: the frame
                        if it["task"] in tasks and (scope is None or k in scope)}           # number is enough
            return by_frame.get(int(m.group(1)))
        return None

    def import_yolo(self, labels_dir, source: str = "imported", items=None, replace: bool = False) -> dict:
        """Load YOLO label files, matched to items by name, else by the _fNNNNNN frame number. Lines may be
        boxes (cls cx cy w h) or polygons (cls x1 y1 x2 y2 ...); polygons become outlines in outline projects.
        `items`: only these (CVAT's "Upload annotations" into a task or job); `replace`: their labels go first
        (CVAT's Replace; otherwise Append)."""
        scope = set(items) if items is not None else None
        if replace:
            with self.db:
                for k in (scope if scope is not None else range(len(self.items))):
                    self.db.execute("DELETE FROM boxes WHERE item=?", (k,))
                    self.db.execute("DELETE FROM reviewed WHERE item=?", (k,))
        matched = unmatched = boxes = 0
        obj = self.new_obj()
        for f in sorted(Path(labels_dir).glob("*.txt")):
            item = self._match(f.stem, scope)
            if item is None:
                unmatched += 1
                continue
            W, H = self.image_size(item)
            for line in f.read_text().splitlines():
                v = line.split()
                if len(v) < 5:
                    continue
                if len(v) >= 7 and len(v) % 2 == 1:                     # a polygon
                    pts = np.array(v[1:], float).reshape(-1, 2) * (W, H)
                    mask = M.draw([(pts, [])], (H, W)) if self.task == "segment" else None
                    box = (*pts.min(0), *pts.max(0))
                else:
                    cx, cy, w, h = float(v[1]) * W, float(v[2]) * H, float(v[3]) * W, float(v[4]) * H
                    box, mask = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), None
                self.put(item, obj, int(v[0]), box, source, mask=mask)
                obj += 1; boxes += 1
            matched += 1
        return {"files_matched": matched, "files_unmatched": unmatched, "boxes": boxes}

    def _export_items(self, reviewed_only: bool) -> list[int]:
        """Items that get a label file: reviewed ones (an empty file = checked, nothing there) and,
        unless reviewed_only, any item with boxes. Unlabeled items are left out, as in YOLO. Only the items
        chosen for this export (self.export_scope; None = all)."""
        st, keep = self.statuses(), self.export_scope
        return [k for k, s in enumerate(st) if (s == 4 or (not reviewed_only and s >= 2)) and (keep is None or k in keep)]

    export_scope = None
    save_images = True

    def _groups(self, items) -> dict:
        """{subset folder: items}: "" when everything falls in one subset (flat, as a single task exports), else
        one entry per task subset, with CVAT's "default" for tasks without one."""
        by = {}
        for k in items:
            by.setdefault(self.task_of(k).get("subset") or "default", []).append(k)
        return {"": list(items)} if len(by) <= 1 else by

    def _export_rows(self, items):
        """(item, file stem, image path, W, H, [(cls, obj, x1, y1, x2, y2, mask)]) with boxes clipped to the
        image; suggestions nobody accepted are left out. Outline projects: `mask` is the part's mask, and parts
        without an outline are left out (counted in self.no_outline)."""
        seg = self.task == "segment"
        for k in items:
            it = self.items[k]
            W, H = self.image_size(k)
            rles = self.masks(k) if seg else {}
            boxes = []
            for b in self.boxes(k):
                if b["source"] == "suggested":
                    continue
                mask = None
                if seg:
                    if b["obj"] not in rles:
                        self.no_outline += 1
                        continue
                    mask = M.decode(rles[b["obj"]])
                    if mask.shape != (H, W) or not mask.any():
                        continue
                x1, y1 = max(0.0, b["box"][0]), max(0.0, b["box"][1])
                x2, y2 = min(float(W), b["box"][2]), min(float(H), b["box"][3])
                if x2 > x1 and y2 > y1:
                    boxes.append((b["cls"], b["obj"], x1, y1, x2, y2, mask))
            yield k, it["name"].replace("/", "__"), self.file_of(k), W, H, boxes

    def _copy_image(self, k: int, folder: Path, stem: str) -> str:
        name = stem + self.file_of(k).suffix.lower()
        if self.save_images:
            folder.mkdir(parents=True, exist_ok=True)
            self.place(k, folder / name)
        return name

    def export_yolo(self, out_dir, groups) -> dict:
        """Ultralytics YOLO: images/ and labels/ (one subset), or images/<subset>/ and labels/<subset>/ with the
        subsets named in data.yaml (train / val / test when the subsets are called so)."""
        out = Path(out_dir)
        n_img = n_boxes = 0
        for sub, items in groups.items():
            img_dir, lab_dir = out / "images" / sub, out / "labels" / sub
            lab_dir.mkdir(parents=True, exist_ok=True)
            for k, stem, path, W, H, boxes in self._export_rows(items):
                self._copy_image(k, img_dir, stem)
                lines = []
                for c, _, x1, y1, x2, y2, mask in boxes:
                    if mask is None:
                        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                    elif (poly := M.single_polygon(mask)) is not None:
                        lines.append(f"{c} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly))
                (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_boxes += len(lines)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        # No "path:": Ultralytics and RF-DETR then use this file's own folder ("path: ." meant the folder the
        # training was started from, so training from anywhere else found no images).
        split = _splits(list(groups))
        paths = "".join(f"{k}: {v}\n" for k, v in split.items())
        (out / "data.yaml").write_text(paths + f"\nnc: {len(self.classes)}\nnames:\n" +
                                       "".join(f"  {i}: {n}\n" for i, n in enumerate(self.classes)), encoding="utf-8")
        return self._result(n_img, n_boxes, folder=str(out))

    def _result(self, images: int, boxes: int, **where) -> dict:
        res = {"images": images, "boxes": boxes, **where}
        if self.task == "segment":
            res["no_outline"] = self.no_outline
        return res

    def export_coco(self, out_dir, groups) -> dict:
        """COCO: annotations.json (+ images/) for one subset, annotations/instances_<subset>.json (+ images/<subset>/)
        for several, as CVAT writes them. Without saved images, file_name is the image's full path."""
        out = Path(out_dir)
        n_img = n_ann = 0
        for sub, items in groups.items():
            images, annotations = [], []
            img_dir = out / "images" / sub
            for k, stem, path, W, H, boxes in self._export_rows(items):
                name = self._copy_image(k, img_dir, stem)
                images.append({"id": k + 1, "file_name": f"{name}" if self.save_images else str(path), "width": W, "height": H})
                for c, obj, x1, y1, x2, y2, mask in boxes:
                    ann = {"id": len(annotations) + 1, "image_id": k + 1, "category_id": c + 1,
                           "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "track_id": obj}
                    if mask is not None:
                        ann.update(segmentation=M.coco_segmentation(mask), area=int(mask.sum()))
                    annotations.append(ann)
            coco = {"images": images, "annotations": annotations,
                    "categories": [{"id": i + 1, "name": n} for i, n in enumerate(self.classes)]}
            f = out / "annotations.json" if not sub else out / "annotations" / f"instances_{sub}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(coco), encoding="utf-8")
            n_img += len(images)
            n_ann += len(annotations)
        return self._result(n_img, n_ann, folder=str(out),
                            **({"file": str(out / "annotations.json")} if list(groups) == [""] else {}))

    def export_cvat(self, out_dir, groups) -> dict:
        """CVAT for images 1.1: annotations.xml next to images/ (import both into a CVAT task or project); with
        several subsets each image carries its subset."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        parts, n_img, n_box = [], 0, 0
        rows = [(sub, r) for sub, items in groups.items() for r in self._export_rows(items)]
        for sub, (k, stem, path, W, H, boxes) in rows:
            name = self._copy_image(k, out / "images", stem)
            subset = f' subset={quoteattr(sub)}' if sub else ""
            parts.append(f'  <image id="{n_img}" name={quoteattr(name)}{subset} width="{W}" height="{H}">')
            for c, _, x1, y1, x2, y2, mask in boxes:
                label = quoteattr(self.classes[c])
                shape = M.polygons(mask) if mask is not None else None
                if mask is None:
                    parts.append(f'    <box label={label} occluded="0" source="manual" '
                                 f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')
                elif len(shape) == 1 and not shape[0][1]:              # one piece, no holes: a polygon
                    pts = ";".join(f"{x:.2f},{y:.2f}" for x, y in shape[0][0])
                    parts.append(f'    <polygon label={label} occluded="0" source="manual" points="{pts}" z_order="0"></polygon>')
                else:                                                  # pieces or holes: a mask
                    runs, left, top, w, h = M.cvat_rle(mask)
                    parts.append(f'    <mask label={label} occluded="0" source="manual" rle="{", ".join(map(str, runs))}" '
                                 f'left="{left}" top="{top}" width="{w}" height="{h}" z_order="0"></mask>')
            parts.append("  </image>")
            n_img += 1
            n_box += len(boxes)
        kind = "any" if self.task == "segment" else "rectangle"
        labels = "".join(f"<label><name>{escape(n)}</name><type>{kind}</type><attributes></attributes></label>"
                         for n in self.classes)
        xml = ('<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n'
               f"  <meta><task><name>{escape(self.meta['name'])}</name><size>{n_img}</size>"
               f"<labels>{labels}</labels></task></meta>\n" + "\n".join(parts) + "\n</annotations>\n")
        (out / "annotations.xml").write_text(xml, encoding="utf-8")
        return self._result(n_img, n_box, folder=str(out))

    def export_voc(self, out_dir, groups) -> dict:
        """Pascal VOC: Annotations/*.xml, JPEGImages/, ImageSets/Main/default.txt, labelmap.txt. Outline projects
        add SegmentationClass/ (pixel = class index + 1) and SegmentationObject/ (pixel = part number) palette PNGs,
        0 = background; larger parts are painted first so small ones stay on top."""
        out = Path(out_dir)
        seg = self.task == "segment"
        (out / "Annotations").mkdir(parents=True, exist_ok=True)
        (out / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
        if seg:
            for d in ("SegmentationClass", "SegmentationObject", "ImageSets/Segmentation"):
                (out / d).mkdir(parents=True, exist_ok=True)
        stems, n_box, lists = [], 0, {}
        rows = [(sub, r) for sub, items in groups.items() for r in self._export_rows(items)]
        for sub, (k, stem, path, W, H, boxes) in rows:
            lists.setdefault(sub or "default", []).append(stem)
            name = self._copy_image(k, out / "JPEGImages", stem)
            objs = "".join(f"<object><name>{escape(self.classes[c])}</name><pose>Unspecified</pose>"
                           f"<truncated>0</truncated><difficult>0</difficult><bndbox><xmin>{x1:.0f}</xmin>"
                           f"<ymin>{y1:.0f}</ymin><xmax>{x2:.0f}</xmax><ymax>{y2:.0f}</ymax></bndbox></object>"
                           for c, _, x1, y1, x2, y2, _ in boxes)
            if seg:
                cls_img, obj_img = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
                for n, (c, *_, mask) in sorted(enumerate(boxes, 1), key=lambda nb: -nb[1][-1].sum()):
                    cls_img[mask], obj_img[mask] = c + 1, min(n, 254)
                for arr, d in ((cls_img, "SegmentationClass"), (obj_img, "SegmentationObject")):
                    im = Image.fromarray(arr, "P")
                    im.putpalette(M.voc_palette())
                    im.save(out / d / f"{stem}.png")
            (out / "Annotations" / f"{stem}.xml").write_text(
                f"<annotation><folder>JPEGImages</folder><filename>{escape(name)}</filename><size><width>{W}</width>"
                f"<height>{H}</height><depth>3</depth></size>{objs}</annotation>\n", encoding="utf-8")
            stems.append(stem)
            n_box += len(boxes)
        for sub, names in (lists or {"default": []}).items():             # one list per subset, as CVAT
            (out / "ImageSets" / "Main" / f"{sub}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
            if seg:
                (out / "ImageSets" / "Segmentation" / f"{sub}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
            pal = M.voc_palette()
            rgb = lambda i: ",".join(map(str, pal[3 * i:3 * i + 3]))
            (out / "labelmap.txt").write_text("# label:color_rgb:parts:actions\nbackground:0,0,0::\n" + "".join(
                f"{n}:{rgb(i + 1)}::\n" for i, n in enumerate(self.classes)), encoding="utf-8")
        else:
            (out / "labelmap.txt").write_text("".join(f"{n}:::\n" for n in self.classes), encoding="utf-8")
        return self._result(len(stems), n_box, folder=str(out))

    def export_labelstudio(self, out_dir, groups) -> dict:
        """Label Studio: tasks.json (rectangle labels, percent coordinates), labeling_config.xml, images/; each
        task's data names its subset when there are several."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tasks, n_box = [], 0
        rows = [(sub, r) for sub, items in groups.items() for r in self._export_rows(items)]
        for sub, (k, stem, path, W, H, boxes) in rows:
            name = self._copy_image(k, out / "images", stem)
            result = []
            for c, obj, x1, y1, x2, y2, mask in boxes:
                base = {"from_name": "label", "to_name": "image", "original_width": W, "original_height": H}
                if mask is None:
                    result.append({"id": f"{k}_{obj}", "type": "rectanglelabels", **base,
                                   "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                                             "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [self.classes[c]]}})
                    continue
                for n, (outer, _) in enumerate(M.polygons(mask)):     # one polygon per piece (holes are not kept)
                    result.append({"id": f"{k}_{obj}_{n}", "type": "polygonlabels", **base,
                                   "value": {"points": [[x / W * 100, y / H * 100] for x, y in outer],
                                             "polygonlabels": [self.classes[c]]}})
            tasks.append({"data": {"image": f"images/{name}" if self.save_images else str(path), **({"subset": sub} if sub else {})},
                          "annotations": [{"result": result}]})
            n_box += len(boxes)
        (out / "tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
        tag = "PolygonLabels" if self.task == "segment" else "RectangleLabels"
        config = (f'<View>\n  <Image name="image" value="$image"/>\n  <{tag} name="label" toName="image">\n'
                  + "".join(f"    <Label value={quoteattr(n)}/>\n" for n in self.classes)
                  + f"  </{tag}>\n</View>\n")
        (out / "labeling_config.xml").write_text(config, encoding="utf-8")
        return self._result(len(tasks), n_box, folder=str(out))

    # ---- image class exports -----------------------------------------------------------------
    def _tag_rows(self, items):
        """(item, file name, image path, class name) per image with a class, of `items`; suggestions nobody
        accepted are left out (still guesses)."""
        tags, keep = self.tags(), set(items)
        for item, t in sorted(tags.items()):
            if t["source"] == "suggested" or item >= len(self.items) or item not in keep:
                continue
            it = self.items[item]
            path = self.file_of(item)
            yield item, it["name"].replace("/", "__") + path.suffix.lower(), path, self.classes[t["cls"]]

    @staticmethod
    def _safe_dir(name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .") or "_"

    def export_folders(self, out_dir, groups) -> dict:
        """One folder per class holding its images (ImageFolder layout), under a folder per subset when there
        are several, plus classes.txt."""
        out = Path(out_dir)
        n = 0
        for sub, items in groups.items():
          for item, name, path, cls in self._tag_rows(items):
            d = out / sub / self._safe_dir(cls)
            d.mkdir(parents=True, exist_ok=True)
            self.place(item, d / name)
            n += 1
        out.mkdir(parents=True, exist_ok=True)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        return {"images": n, "folder": str(out)}

    def export_yolo_classify(self, out_dir, groups, val_every: int = 5, block: int = 10) -> dict:
        """Ultralytics classification layout: train/<class>/ and val/<class>/ (test/ too). With subsets, the tasks'
        subsets decide; otherwise items go to val in whole blocks of `block` neighbours (every `val_every`-th block),
        so near-identical neighbouring frames or burst shots do not end up on both sides."""
        out = Path(out_dir)
        n = {"train": 0, "val": 0}
        named = {v.split("/")[-1]: k for k, v in _splits([g for g in groups if g]).items()} if list(groups) != [""] else {}
        for sub, items in groups.items():
          for item, name, path, cls in self._tag_rows(items):
            split = named.get(sub) or ("val" if (item // block) % val_every == val_every - 1 else "train")
            n.setdefault(split, 0)
            d = out / split / self._safe_dir(cls)
            d.mkdir(parents=True, exist_ok=True)
            self.place(item, d / name)
            n[split] += 1
        out.mkdir(parents=True, exist_ok=True)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        return {"images": sum(n.values()), **n, "folder": str(out)}

    def export_csv(self, out_dir, groups) -> dict:
        """labels.csv (path of the original image, class, subset when there are several), plus classes.txt; no
        images are copied."""
        import csv
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        many = list(groups) != [""]
        rows = [(str(path), cls, *([sub] if many else [])) for sub, items in groups.items()
                for _, _, path, cls in self._tag_rows(items)]
        with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "class", *(["subset"] if many else [])])
            w.writerows(rows)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        return {"images": len(rows), "folder": str(out)}

    def export(self, fmt: str, out_dir, reviewed_only: bool = False, tasks=None, jobs=None,
               save_images: bool = True) -> dict:
        """Write the dataset in one of self.formats into out_dir, as CVAT's "Export dataset": the whole project,
        only `tasks` (ids or names) or only `jobs` (ids). Tasks' subsets become folders when there are several.
        `save_images` False writes the labels only (images are referred to by their full path where needed)."""
        if fmt not in self.formats:
            raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(self.formats)}")
        scope = None
        if tasks:
            ids = {t["id"]: t["id"] for t in self.sources} | {t["name"]: t["id"] for t in self.sources}
            missing = [x for x in tasks if x not in ids]
            if missing:
                raise ValueError(f"no task called {', '.join(map(str, missing))}; tasks: {', '.join(t['name'] for t in self.sources)}")
            scope = {k for tid in {ids[x] for x in tasks} for k in range(*self.ranges.get(tid, (0, 0)))}
        if jobs:
            by_id = {j["id"]: j for j in self.jobs()}
            missing = [x for x in jobs if int(x) not in by_id]
            if missing:
                raise ValueError(f"no job {', '.join(map(str, missing))}")
            scope = (scope or set()) | {k for x in jobs for k in range(by_id[int(x)]["start"], by_id[int(x)]["end"])}
        self.export_scope, self.save_images, self.no_outline = scope, bool(save_images), 0
        try:
            if self.task == "classify":
                items = sorted(k for k in self.tags() if scope is None or k in scope)
                return getattr(self, "export_yolo_classify" if fmt == "yolo" else f"export_{fmt}")(out_dir, self._groups(items))
            return getattr(self, f"export_{fmt}")(out_dir, self._groups(self._export_items(reviewed_only)))
        finally:
            self.export_scope, self.save_images = None, True

    # ---- settings and undo ----------------------------------------------------------------
    def set_meta(self, **values) -> None:
        self.meta.update(values)
        (self.folder / "project.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    def items_with(self, obj: int) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT item FROM boxes WHERE obj=? ORDER BY item", (obj,))]

    def snapshot(self, items) -> dict:
        """Boxes and review state of these items, for undo."""
        items = sorted(set(items))
        return {"rows": {k: self.db.execute("SELECT * FROM boxes WHERE item=?", (k,)).fetchall() for k in items},
                "tags": {k: self.db.execute("SELECT * FROM tags WHERE item=?", (k,)).fetchall() for k in items},
                "reviewed": {k: self.is_reviewed(k) for k in items}}

    def restore(self, snap: dict) -> list[int]:
        """Put items back exactly as `snapshot` saw them; returns the items touched."""
        with self.db:
            self._touch(snap["rows"])
            for k, rows in snap["rows"].items():
                self.db.execute("DELETE FROM boxes WHERE item=?", (k,))
                self.db.executemany("INSERT INTO boxes VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                self.db.execute("DELETE FROM tags WHERE item=?", (k,))
                self.db.executemany("INSERT INTO tags VALUES (?,?,?,?)", snap.get("tags", {}).get(k, []))
                if snap["reviewed"][k]:
                    self.db.execute("INSERT OR IGNORE INTO reviewed VALUES (?)", (k,))
                else:
                    self.db.execute("DELETE FROM reviewed WHERE item=?", (k,))
        return list(snap["rows"])

    def mirrored_pairs(self) -> bool:
        return mirrored_pairs(self.classes)


# ---- tasks: frames, video details, linked export images -------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def _natural(p: Path):
    """Natural order: img2 before img10."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(p))]


def _splits(subsets) -> dict:
    """data.yaml keys for subset folders: train / val / test by name (validation and valid count as val); a lone
    or unrecognised subset trains and validates; val falls back to train."""
    if subsets == [""] or not subsets:
        return {"train": "images", "val": "images"}
    key = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    out = {}
    for sub in subsets:
        k = key.get(sub.lower())
        if k and k not in out:
            out[k] = f"images/{sub}"
    rest = [sub for sub in subsets if key.get(sub.lower()) is None]
    if "train" not in out:
        out["train"] = f"images/{rest[0] if rest else subsets[0]}"
    out.setdefault("val", out["train"])
    return {k: out[k] for k in ("train", "val", "test") if k in out}

def _save_frame(img: Image.Image, path: Path, fmt: str, quality: int = 95) -> None:
    if fmt == "webp":                                     # lossless: the pixels as decoded, about 2x JPEG's size
        img.save(path, "WEBP", lossless=True, quality=50, method=1)
    else:                                                 # 95: visually lossless; optimize: smaller, same pixels
        img.save(path, "JPEG", quality=max(5, min(100, int(quality))), optimize=True)


class VideoFrames:
    """Frames read straight from a video by presentation time: seek to the keyframe before one and decode up to it;
    reading ahead (the next kept frames, tracking, export) goes on with the open decoder instead of seeking again.
    The pixels are exactly those of a full decode."""
    AHEAD_S = 2.0

    def __init__(self, path):
        self.path, self.lock, self.c = Path(path), threading.Lock(), None

    def close(self) -> None:
        with self.lock:
            if self.c is not None:
                self.c.close()
                self.c = None

    def frame(self, pts: int) -> Image.Image:
        with self.lock:
            if getattr(self, "held", (None,))[0] == pts:          # the frame just read, asked for again
                return self.held[1].copy()
            for fresh in (False, True):
                if self.c is None or fresh:
                    if self.c is not None:
                        self.c.close()
                    self.c = av.open(str(self.path))
                    self.s = self.c.streams.video[0]
                    self.s.thread_type = "AUTO"
                    self.frames = self.last = None
                if self.frames is None or self.last is None or not self.last < pts <= self.last + self.AHEAD_S / self.s.time_base:
                    self.c.seek(pts, stream=self.s, backward=True, any_frame=False)
                    self.frames = self.c.decode(self.s)
                for fr in self.frames:
                    self.last = fr.pts
                    if fr.pts == pts:
                        self.held = (pts, fr.to_image())
                        return self.held[1].copy()
                    if fr.pts is not None and fr.pts > pts:
                        break
                self.frames = None
            raise ValueError(f"{self.path.name}: no frame at time {pts}")


_READERS: dict = {}
_READERS_LOCK = threading.Lock()


def _reader(path: Path) -> VideoFrames:
    """The open reader of a video (a few stay open, most recently used first)."""
    with _READERS_LOCK:
        r = _READERS.pop(str(path), None) or VideoFrames(path)
        _READERS[str(path)] = r
        while len(_READERS) > 8:
            _READERS.pop(next(iter(_READERS))).close()
        return r


def close_video(path) -> None:
    """Let go of a video file (Windows cannot delete or move a file that is open)."""
    with _READERS_LOCK:
        r = _READERS.pop(str(path), None)
    if r is not None:
        r.close()


def _frame_times(video: Path, keep, stop, progress=None) -> tuple[list[int], list[int]]:
    """(frame numbers, presentation times) of the decoded frames for which keep(number) is true, up to `stop`."""
    frames, pts = [], []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        total = s.frames or 0
        for i, fr in enumerate(c.decode(s)):
            if stop is not None and i > stop:
                break
            if keep(i):
                if fr.pts is None:
                    raise ValueError(f"{video.name} has frames without timestamps: store its frames as files instead")
                frames.append(i)
                pts.append(fr.pts)
            if progress and i % 200 == 0:
                progress(i, total, "Indexing the video's frames")
    return frames, pts


def index_video(video: Path, out: Path, every: int, progress=None, start: int | None = None,
                stop: int | None = None) -> tuple[int, int]:
    """Keep a task's frames in its video (no image files): the video is hard-linked into `out` (copied across drives,
    so the project keeps working if the original moves) and index.json lists the kept frames, every `every`-th from
    `start` to `stop`, with their presentation times. Returns (frames, bytes of the video)."""
    out.mkdir(parents=True, exist_ok=True)
    dst = out / ("video" + Path(video).suffix.lower())
    place_image(Path(video), dst, link=True)
    first = start or 0
    frames, pts = _frame_times(dst, lambda i: i >= first and (i - first) % every == 0, stop, progress)
    (out / "index.json").write_text(json.dumps({"video": dst.name, "frames": frames, "pts": pts}), encoding="utf-8")
    return len(frames), dst.stat().st_size


def extract_frames(video: Path, out: Path, every: int, fmt: str = "jpg", progress=None, start: int | None = None,
                   stop: int | None = None, quality: int = 95) -> tuple[int, int]:
    """Store every `every`-th frame from `start` to `stop` (inclusive; None = the ends) in `out`
    (f<frame number>.<fmt>, JPEG of `quality`); returns (frames, bytes). Frames are encoded on every core while
    the next ones decode."""
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir(parents=True, exist_ok=True)
    ext = ".webp" if fmt == "webp" else ".jpg"
    kept, pending = 0, []
    workers = os.cpu_count() or 4                          # encoding releases the GIL: every core helps (S12)
    with av.open(str(video)) as c, ThreadPoolExecutor(max_workers=workers) as pool:
        total = c.streams.video[0].frames or 0
        first = start or 0
        for i, fr in enumerate(c.decode(video=0)):
            if stop is not None and i > stop:
                break
            if i >= first and (i - first) % every == 0:
                pending.append(pool.submit(_save_frame, fr.to_image(), out / f"f{i:06d}{ext}", fmt, quality))
                kept += 1
                while len(pending) > 2 * workers:         # bounded: decoded frames are large
                    pending.pop(0).result()
            if progress and i % 50 == 0:
                progress(i, total, "Extracting frames" + (" (lossless)" if fmt == "webp" else ""))
        for f in pending:
            f.result()
    return kept, sum(p.stat().st_size for p in out.glob(f"f*{ext}"))


def probe_video(path: Path) -> dict:
    """Resolution, frame rate, length, frame count, codec, bit rate and file size of a video."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else None
        dur = float(c.duration) / 1_000_000 if c.duration else (float(s.duration * s.time_base) if s.duration else None)
        frames = s.frames or (round(dur * fps) if dur and fps else None)
        return {"width": s.codec_context.width, "height": s.codec_context.height, "fps": round(fps, 3) if fps else None,
                "duration": round(dur, 2) if dur else None, "frames": frames, "codec": s.codec_context.name,
                "bitrate": c.bit_rate or None, "size": Path(path).stat().st_size}


def probe_images(root: Path, sample: int = 50) -> dict:
    """Image count, the most common resolution (of up to `sample` images) and total size of a folder."""
    from collections import Counter
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    sizes = Counter()
    for f in files[:: max(1, len(files) // sample)][:sample]:
        try:
            with Image.open(f) as im:
                sizes[im.size] += 1
        except OSError:
            pass
    (w, h), _ = sizes.most_common(1)[0] if sizes else ((None, None), 0)
    return {"images": len(files), "width": w, "height": h, "mixed_sizes": len(sizes) > 1,
            "size": sum(f.stat().st_size for f in files)}


def place_image(src: Path, dst: Path, link: bool) -> None:
    """Put an image into an export: a hard link to the project's own stored frame (no extra space, identical
    pixels; falls back to a copy across drives), a copy of anything else (never a link to a person's own file)."""
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy(src, dst)


# ---- backup and restore (CVAT's "Backup project" / "Create from backup") ------------------------------
def backup_project(folder, out_zip, tasks=None) -> dict:
    """Zip a project: project.json, a consistent copy of labels.sqlite, the stored frames, and the pictures of
    image-folder tasks (they normally stay where they are, so a backup is complete on another computer).
    Pictures are stored without re-compression. Exports and caches are left out.
    With `tasks` (task ids): only those tasks, with their labels, confirmed frames and jobs (CVAT's "Backup task").
    Such a zip opens as a project of its own (restore_project) or joins another one (Project.add_tasks_from)."""
    import zipfile
    folder, out_zip = Path(folder), Path(out_zip)
    p = Project(folder)
    keep = [t for t in p.sources if tasks is None or t["id"] in set(tasks)]
    if tasks is not None and not keep:
        p.db.close()
        raise KeyError(f"no task {', '.join(map(str, tasks))}")
    meta = dict(p.meta)
    if tasks is not None:
        meta.update(sources=keep, kind=keep[0]["kind"], source=keep[0]["source"], every=keep[0]["every"])
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        z.writestr("project.json", json.dumps(meta, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        tmp = out_zip.with_suffix(".sqlite.tmp")
        dst = sqlite3.connect(tmp)
        with dst:
            p.db.backup(dst)
            if tasks is not None:                         # only these tasks' rows, numbered as the smaller project will be
                spans, new = [], 0
                for t in keep:
                    a, b = p.ranges.get(t["id"], (0, 0))
                    spans.append((a, b, new - a))
                    new += b - a
                for table in TABLES:
                    rows = dst.execute(f"SELECT rowid, item FROM {table}").fetchall()
                    gone = [(r,) for r, i in rows if not any(a <= i < b for a, b, _ in spans)]
                    dst.executemany(f"DELETE FROM {table} WHERE rowid = ?", gone)
                    for a, b, shift in spans:             # via a high offset, so no row lands on a number still in use
                        dst.execute(f"UPDATE {table} SET item = item + ? WHERE item >= ? AND item < ?", (OFFSET + shift, a, b))
                    dst.execute(f"UPDATE {table} SET item = item - ? WHERE item >= ?", (OFFSET, OFFSET))
        dst.close()
        z.write(tmp, "labels.sqlite", compress_type=zipfile.ZIP_DEFLATED)
        tmp.unlink()
        for t in keep:
            a, b = p.ranges.get(t["id"], (0, 0))
            if t.get("format") == "video" and t["kind"] == "video":     # frames read from the video: zip the video
                d = folder / t["frames"]
                for f in [*d.glob("video.*"), d / "index.json"]:
                    if f.is_file():
                        z.write(f, f.relative_to(folder).as_posix())
                n += b - a
                continue
            for k in range(a, b):
                path = Path(p.items[k]["path"])
                arc = (path.relative_to(folder).as_posix() if folder in path.parents
                       else f"images/t{t['id']}/{path.relative_to(Path(t['source'])).as_posix()}")
                z.write(path, arc)
                n += 1
    p.db.close()
    return {"file": str(out_zip), "items": n, "size": out_zip.stat().st_size, "tasks": [t["name"] for t in keep]}


def restore_project(zip_path, home, name: str | None = None) -> Path:
    """Unpack a backup into `home` as a new project (a free name, from the backup's unless given); image-folder
    tasks then read their pictures from inside the project."""
    import zipfile
    home = Path(home)
    with zipfile.ZipFile(zip_path) as z:
        meta = json.loads(z.read("project.json"))
        base = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in (name or meta.get("name") or "restored")).strip(" .") or "restored"
        dest = home / base
        k = 2
        while dest.exists():
            dest, k = home / f"{base}_{k}", k + 1
        for m in z.namelist():                            # refuse paths that would leave the project folder
            if m.startswith(("/", "\\")) or ".." in Path(m).parts:
                raise ValueError(f"unsafe path in backup: {m}")
        z.extractall(dest)
    for t in meta.get("sources", []):
        if t["kind"] == "images":
            t["source"] = str(dest / "images" / f"t{t['id']}")
    if meta.get("sources"):
        meta["source"] = meta["sources"][0]["source"]
    meta["name"] = dest.name
    (dest / "project.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return dest
