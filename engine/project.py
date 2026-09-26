"""A PartLabeler project: one video or image folder, its class list, and every label.

Folder layout:
    project.json     kind ("video" | "images"), task, source path, sampling step, classes
    frames/          video only: the sampled frames as JPEG (f000000.jpg, f000005.jpg, ...)
    labels.sqlite    boxes, outlines, image classes and review status; every change is committed immediately

The task says what is labeled: "detect" (boxes around parts), "segment" (outlines: a mask per part, its box
is the mask's box) or "classify" (one class per image). Boxes are stored in image pixels (x1, y1, x2, y2),
masks as COCO RLE (engine/masks.py), image classes in `tags`. `source` records where a label came from:
manual (drawn or clicked by a person), tracked, imported, suggested (not yet accepted).
"""
import json
import re
import shutil
import sqlite3
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import av
import numpy as np
from PIL import Image

from engine import masks as M

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TASKS = ("detect", "segment", "classify")
BOX_FORMATS = ("yolo", "coco", "cvat", "voc", "labelstudio")
CLASSIFY_FORMATS = ("folders", "yolo", "csv")
FRAME_RE = re.compile(r"_f(\d+)$")


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


class Project:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "project.json").read_text(encoding="utf-8"))
        self.db = sqlite3.connect(self.folder / "labels.sqlite", check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS boxes (item INTEGER, obj INTEGER, cls INTEGER, x1 REAL, y1 REAL,
                x2 REAL, y2 REAL, source TEXT, score REAL, PRIMARY KEY (item, obj));
            CREATE TABLE IF NOT EXISTS reviewed (item INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS tags (item INTEGER PRIMARY KEY, cls INTEGER, source TEXT, score REAL);
        """)
        if "rle" not in {r[1] for r in self.db.execute("PRAGMA table_info(boxes)")}:
            self.db.execute("ALTER TABLE boxes ADD COLUMN rle TEXT")      # projects made before outlines
            self.db.commit()
        self.items = self._list_items()

    # ---- creation -------------------------------------------------------------------------
    @classmethod
    def create(cls, folder, classes: list[str], video=None, images=None, every: int = 5, progress=None,
               task: str = "detect"):
        folder = Path(folder)
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose one of {', '.join(TASKS)}")
        if (folder / "project.json").exists():
            raise FileExistsError(f"{folder} already holds a project; open it instead")
        folder.mkdir(parents=True, exist_ok=True)
        if video:
            meta = {"kind": "video", "source": str(Path(video).resolve()), "every": every}
            (folder / "frames").mkdir(exist_ok=True)
            with av.open(str(video)) as c:
                total = c.streams.video[0].frames or 0
                for i, fr in enumerate(c.decode(video=0)):
                    if i % every == 0:
                        fr.to_image().save(folder / "frames" / f"f{i:06d}.jpg", quality=95)
                    if progress and i % 50 == 0:
                        progress(i, total, "Extracting frames")
        elif images:
            meta = {"kind": "images", "source": str(Path(images).resolve())}
        else:
            raise ValueError("give a video or an image folder")
        meta.update(name=folder.name, task=task, classes=list(classes), parent=None)
        (folder / "project.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return cls(folder)

    def _list_items(self) -> list[dict]:
        if self.meta["kind"] == "video":
            stem = Path(self.meta["source"]).stem
            return [{"name": f"{stem}_{p.stem}", "path": p, "frame": int(p.stem[1:])}
                    for p in sorted((self.folder / "frames").glob("f*.jpg"))]
        root = Path(self.meta["source"])
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        return [{"name": str(p.relative_to(root).with_suffix("")).replace("\\", "/"), "path": p, "frame": None}
                for p in files]

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
        return Image.open(self.items[item]["path"]).convert("RGB")

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
            if source == "tracked":
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
        return [r[0] for r in self.db.execute(
            "SELECT item FROM boxes WHERE obj=? AND source='manual' ORDER BY item", (obj,))]

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

    def delete(self, item: int, obj: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM boxes WHERE item=? AND obj=?", (item, obj))

    def set_class(self, obj: int, cls: int) -> None:
        with self.db:
            self.db.execute("UPDATE boxes SET cls=? WHERE obj=?", (cls, obj))

    def set_reviewed(self, item: int, value: bool = True) -> None:
        with self.db:
            if value:
                self.db.execute("INSERT OR IGNORE INTO reviewed VALUES (?)", (item,))
                self.db.execute("UPDATE boxes SET source='manual' WHERE item=? AND source='suggested'", (item,))
            else:
                self.db.execute("DELETE FROM reviewed WHERE item=?", (item,))

    def accept(self, item: int) -> None:
        with self.db:
            self.db.execute("UPDATE boxes SET source='manual' WHERE item=? AND source='suggested'", (item,))

    def put_tracked(self, item: int, results: dict) -> None:
        """Replace this item's tracked boxes for the given objects; manual/reviewed boxes are kept.
        results: {obj: (cls, box or None, score[, mask])}; a mask replaces the box with its own."""
        with self.db:
            if self.is_reviewed(item):
                return
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

    # ---- image classes (classify projects) ---------------------------------------------------
    def tags(self) -> dict:
        """{item: {"cls", "source", "score"}} for every image with a class (given or suggested)."""
        return {i: {"cls": c, "source": s, "score": sc} for i, c, s, sc in self.db.execute("SELECT * FROM tags")}

    def set_tags(self, items, cls, source: str = "manual", scores=None) -> None:
        """Give these images class `cls` (None clears it). Suggestions never replace a person's or imported class."""
        with self.db:
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
            return sum(self.db.execute("UPDATE tags SET source='manual' WHERE item=? AND source='suggested'",
                                       (i,)).rowcount for i in items)

    # ---- import / export ------------------------------------------------------------------
    def _match(self, stem: str):
        by_name = {it["name"]: k for k, it in enumerate(self.items)}
        for name in (stem, stem.replace("__", "/")):          # exports flatten sub-folders with "__"
            if name in by_name:
                return by_name[name]
        m = FRAME_RE.search(stem)
        if m and self.meta["kind"] == "video":
            by_frame = {it["frame"]: k for k, it in enumerate(self.items)}
            return by_frame.get(int(m.group(1)))
        return None

    def import_yolo(self, labels_dir, source: str = "imported") -> dict:
        """Load YOLO label files, matched to items by name, else by the _fNNNNNN frame number. Lines may be
        boxes (cls cx cy w h) or polygons (cls x1 y1 x2 y2 ...); polygons become outlines in outline projects."""
        matched = unmatched = boxes = 0
        obj = self.new_obj()
        for f in sorted(Path(labels_dir).glob("*.txt")):
            item = self._match(f.stem)
            if item is None:
                unmatched += 1
                continue
            W, H = Image.open(self.items[item]["path"]).size
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
        unless reviewed_only, any item with boxes. Unlabeled items are left out, as in YOLO."""
        st = self.statuses()
        return [k for k, s in enumerate(st) if s == 4 or (not reviewed_only and s >= 2)]

    def _export_rows(self, reviewed_only: bool):
        """(item, file stem, image path, W, H, [(cls, obj, x1, y1, x2, y2, mask)]) with boxes clipped to the
        image; suggestions nobody accepted are left out. Outline projects: `mask` is the part's mask, and parts
        without an outline are left out (counted in self.no_outline)."""
        self.no_outline = 0
        seg = self.task == "segment"
        for k in self._export_items(reviewed_only):
            it = self.items[k]
            with Image.open(it["path"]) as im:
                W, H = im.size
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
            yield k, it["name"].replace("/", "__"), Path(it["path"]), W, H, boxes

    @staticmethod
    def _copy_image(path: Path, folder: Path, stem: str) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        name = stem + path.suffix.lower()
        shutil.copy(path, folder / name)
        return name

    def export_yolo(self, out_dir, reviewed_only: bool = False) -> dict:
        out = Path(out_dir)
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        n_img = n_boxes = 0
        for _, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            self._copy_image(path, out / "images", stem)
            lines = []
            for c, _, x1, y1, x2, y2, mask in boxes:
                if mask is None:
                    lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                elif (poly := M.single_polygon(mask)) is not None:
                    lines.append(f"{c} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly))
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_boxes += len(lines)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        (out / "data.yaml").write_text("path: .\ntrain: images\nval: images\n\n"
                                       f"nc: {len(self.classes)}\nnames:\n" +
                                       "".join(f"  {i}: {n}\n" for i, n in enumerate(self.classes)), encoding="utf-8")
        return self._result(n_img, n_boxes, folder=str(out))

    def _result(self, images: int, boxes: int, **where) -> dict:
        res = {"images": images, "boxes": boxes, **where}
        if self.task == "segment":
            res["no_outline"] = self.no_outline
        return res

    def export_coco(self, out_json, reviewed_only: bool = False) -> dict:
        images, annotations = [], []
        for k, _, path, W, H, boxes in self._export_rows(reviewed_only):
            images.append({"id": k + 1, "file_name": str(path), "width": W, "height": H})
            for c, obj, x1, y1, x2, y2, mask in boxes:
                ann = {"id": len(annotations) + 1, "image_id": k + 1, "category_id": c + 1,
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "track_id": obj}
                if mask is not None:
                    ann.update(segmentation=M.coco_segmentation(mask), area=int(mask.sum()))
                annotations.append(ann)
        coco = {"images": images, "annotations": annotations,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(self.classes)]}
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(coco), encoding="utf-8")
        return self._result(len(images), len(annotations), file=str(out_json))

    def export_cvat(self, out_dir, reviewed_only: bool = False) -> dict:
        """CVAT for images 1.1: annotations.xml next to images/ (import both into a CVAT task)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        parts, n_img, n_box = [], 0, 0
        for _, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            name = self._copy_image(path, out / "images", stem)
            parts.append(f'  <image id="{n_img}" name={quoteattr(name)} width="{W}" height="{H}">')
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

    def export_voc(self, out_dir, reviewed_only: bool = False) -> dict:
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
        stems, n_box = [], 0
        for _, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            name = self._copy_image(path, out / "JPEGImages", stem)
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
        (out / "ImageSets" / "Main" / "default.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
        if seg:
            (out / "ImageSets" / "Segmentation" / "default.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
            pal = M.voc_palette()
            rgb = lambda i: ",".join(map(str, pal[3 * i:3 * i + 3]))
            (out / "labelmap.txt").write_text("# label:color_rgb:parts:actions\nbackground:0,0,0::\n" + "".join(
                f"{n}:{rgb(i + 1)}::\n" for i, n in enumerate(self.classes)), encoding="utf-8")
        else:
            (out / "labelmap.txt").write_text("".join(f"{n}:::\n" for n in self.classes), encoding="utf-8")
        return self._result(len(stems), n_box, folder=str(out))

    def export_labelstudio(self, out_dir, reviewed_only: bool = False) -> dict:
        """Label Studio: tasks.json (rectangle labels, percent coordinates), labeling_config.xml, images/."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tasks, n_box = [], 0
        for k, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            name = self._copy_image(path, out / "images", stem)
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
            tasks.append({"data": {"image": f"images/{name}"}, "annotations": [{"result": result}]})
            n_box += len(boxes)
        (out / "tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
        tag = "PolygonLabels" if self.task == "segment" else "RectangleLabels"
        config = (f'<View>\n  <Image name="image" value="$image"/>\n  <{tag} name="label" toName="image">\n'
                  + "".join(f"    <Label value={quoteattr(n)}/>\n" for n in self.classes)
                  + f"  </{tag}>\n</View>\n")
        (out / "labeling_config.xml").write_text(config, encoding="utf-8")
        return self._result(len(tasks), n_box, folder=str(out))

    # ---- image class exports -----------------------------------------------------------------
    def _tag_rows(self, reviewed_only: bool):
        """(item, file name, image path, class name) per image with a class; suggestions only if not reviewed_only
        (a suggestion nobody accepted is still a guess, so by default it is left out too)."""
        for item, t in sorted(self.tags().items()):
            if t["source"] == "suggested" or item >= len(self.items):
                continue
            it = self.items[item]
            path = Path(it["path"])
            yield item, it["name"].replace("/", "__") + path.suffix.lower(), path, self.classes[t["cls"]]

    @staticmethod
    def _safe_dir(name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .") or "_"

    def export_folders(self, out_dir, reviewed_only: bool = False) -> dict:
        """One folder per class holding its images (ImageFolder layout), plus classes.txt."""
        out = Path(out_dir)
        n = 0
        for _, name, path, cls in self._tag_rows(reviewed_only):
            d = out / self._safe_dir(cls)
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, d / name)
            n += 1
        out.mkdir(parents=True, exist_ok=True)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        return {"images": n, "folder": str(out)}

    def export_yolo_classify(self, out_dir, reviewed_only: bool = False, val_every: int = 5, block: int = 10) -> dict:
        """Ultralytics classification layout: train/<class>/ and val/<class>/. Items go to val in whole blocks of
        `block` neighbours (every `val_every`-th block), so near-identical neighbouring frames or burst shots do
        not end up on both sides."""
        out = Path(out_dir)
        n = {"train": 0, "val": 0}
        for item, name, path, cls in self._tag_rows(reviewed_only):
            split = "val" if (item // block) % val_every == val_every - 1 else "train"
            d = out / split / self._safe_dir(cls)
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, d / name)
            n[split] += 1
        out.mkdir(parents=True, exist_ok=True)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        return {"images": n["train"] + n["val"], **n, "folder": str(out)}

    def export_csv(self, out_dir, reviewed_only: bool = False) -> dict:
        """labels.csv (path of the original image, class), plus classes.txt; no images are copied."""
        import csv
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = [(str(path), cls) for _, _, path, cls in self._tag_rows(reviewed_only)]
        with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "class"])
            w.writerows(rows)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        return {"images": len(rows), "folder": str(out)}

    def export(self, fmt: str, out_dir, reviewed_only: bool = False) -> dict:
        """Write the dataset in one of self.formats into out_dir."""
        if fmt not in self.formats:
            raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(self.formats)}")
        if self.task == "classify":
            return getattr(self, "export_yolo_classify" if fmt == "yolo" else f"export_{fmt}")(out_dir, reviewed_only)
        if fmt == "coco":
            return self.export_coco(Path(out_dir) / "annotations.json", reviewed_only)
        return getattr(self, f"export_{fmt}")(out_dir, reviewed_only)

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
