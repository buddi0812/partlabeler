"""A PartLabeler project: one video or image folder, its class list, and every box.

Folder layout:
    project.json     kind ("video" | "images"), source path, sampling step, classes
    frames/          video only: the sampled frames as JPEG (f000000.jpg, f000005.jpg, ...)
    labels.sqlite    boxes and review status; every change is committed immediately

Boxes are stored in image pixels (x1, y1, x2, y2). `source` records where a box came from:
manual (drawn or clicked by a person), tracked, imported, suggested (not yet accepted).
"""
import json
import re
import shutil
import sqlite3
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import av
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
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
        """)
        self.items = self._list_items()

    # ---- creation -------------------------------------------------------------------------
    @classmethod
    def create(cls, folder, classes: list[str], video=None, images=None, every: int = 5, progress=None):
        folder = Path(folder)
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
        meta.update(name=folder.name, classes=list(classes), parent=None)
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

    def image(self, item: int) -> Image.Image:
        return Image.open(self.items[item]["path"]).convert("RGB")

    def boxes(self, item: int) -> list[dict]:
        rows = self.db.execute("SELECT obj, cls, x1, y1, x2, y2, source, score FROM boxes WHERE item=? ORDER BY obj",
                               (item,)).fetchall()
        return [{"obj": o, "cls": c, "box": [x1, y1, x2, y2], "source": s, "score": sc}
                for o, c, x1, y1, x2, y2, s, sc in rows]

    def is_reviewed(self, item: int) -> bool:
        return self.db.execute("SELECT 1 FROM reviewed WHERE item=?", (item,)).fetchone() is not None

    def statuses(self) -> list[int]:
        """Per item: 0 empty, 1 suggestions only, 2 tracked/imported, 3 has manual boxes, 4 reviewed."""
        out = [0] * len(self.items)
        rank = {"suggested": 1, "tracked": 2, "imported": 2, "manual": 3}
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

    def put(self, item: int, obj: int, cls: int, box, source: str = "manual", score=None) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO boxes VALUES (?,?,?,?,?,?,?,?,?)",
                            (item, obj, cls, *map(float, box), source, score))

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
        """Replace this item's tracked boxes for the given objects; manual/reviewed boxes are kept."""
        with self.db:
            if self.is_reviewed(item):
                return
            for obj, (cls, box, score) in results.items():
                row = self.db.execute("SELECT source FROM boxes WHERE item=? AND obj=?", (item, obj)).fetchone()
                if row and row[0] == "manual":
                    continue
                if box is None:
                    self.db.execute("DELETE FROM boxes WHERE item=? AND obj=? AND source='tracked'", (item, obj))
                else:
                    self.db.execute("INSERT OR REPLACE INTO boxes VALUES (?,?,?,?,?,?,?,?,?)",
                                    (item, obj, cls, *map(float, box), "tracked", score))

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
        """Load YOLO label files, matched to items by name, else by the _fNNNNNN frame number."""
        matched = unmatched = boxes = 0
        obj = self.new_obj()
        for f in sorted(Path(labels_dir).glob("*.txt")):
            item = self._match(f.stem)
            if item is None:
                unmatched += 1
                continue
            W, H = Image.open(self.items[item]["path"]).size
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()[:5]
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                self.put(item, obj, int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), source)
                obj += 1; boxes += 1
            matched += 1
        return {"files_matched": matched, "files_unmatched": unmatched, "boxes": boxes}

    def _export_items(self, reviewed_only: bool) -> list[int]:
        """Items that get a label file: reviewed ones (an empty file = checked, nothing there) and,
        unless reviewed_only, any item with boxes. Unlabeled items are left out, as in YOLO."""
        st = self.statuses()
        return [k for k, s in enumerate(st) if s == 4 or (not reviewed_only and s >= 2)]

    def _export_rows(self, reviewed_only: bool):
        """(item, file stem, image path, W, H, [(cls, obj, x1, y1, x2, y2)]) with boxes clipped to the
        image; suggestions nobody accepted are left out."""
        for k in self._export_items(reviewed_only):
            it = self.items[k]
            with Image.open(it["path"]) as im:
                W, H = im.size
            boxes = []
            for b in self.boxes(k):
                if b["source"] == "suggested":
                    continue
                x1, y1 = max(0.0, b["box"][0]), max(0.0, b["box"][1])
                x2, y2 = min(float(W), b["box"][2]), min(float(H), b["box"][3])
                if x2 > x1 and y2 > y1:
                    boxes.append((b["cls"], b["obj"], x1, y1, x2, y2))
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
            lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                     for c, _, x1, y1, x2, y2 in boxes]
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_boxes += len(lines)
        (out / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        (out / "data.yaml").write_text("path: .\ntrain: images\nval: images\n\n"
                                       f"nc: {len(self.classes)}\nnames:\n" +
                                       "".join(f"  {i}: {n}\n" for i, n in enumerate(self.classes)), encoding="utf-8")
        return {"images": n_img, "boxes": n_boxes, "folder": str(out)}

    def export_coco(self, out_json, reviewed_only: bool = False) -> dict:
        images, annotations = [], []
        for k, _, path, W, H, boxes in self._export_rows(reviewed_only):
            images.append({"id": k + 1, "file_name": str(path), "width": W, "height": H})
            for c, obj, x1, y1, x2, y2 in boxes:
                annotations.append({"id": len(annotations) + 1, "image_id": k + 1, "category_id": c + 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                    "iscrowd": 0, "track_id": obj})
        coco = {"images": images, "annotations": annotations,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(self.classes)]}
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(coco), encoding="utf-8")
        return {"images": len(images), "boxes": len(annotations), "file": str(out_json)}

    def export_cvat(self, out_dir, reviewed_only: bool = False) -> dict:
        """CVAT for images 1.1: annotations.xml next to images/ (import both into a CVAT task)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        parts, n_img, n_box = [], 0, 0
        for _, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            name = self._copy_image(path, out / "images", stem)
            parts.append(f'  <image id="{n_img}" name={quoteattr(name)} width="{W}" height="{H}">')
            for c, _, x1, y1, x2, y2 in boxes:
                parts.append(f'    <box label={quoteattr(self.classes[c])} occluded="0" source="manual" '
                             f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')
            parts.append("  </image>")
            n_img += 1
            n_box += len(boxes)
        labels = "".join(f"<label><name>{escape(n)}</name><type>rectangle</type><attributes></attributes></label>"
                         for n in self.classes)
        xml = ('<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n'
               f"  <meta><task><name>{escape(self.meta['name'])}</name><size>{n_img}</size>"
               f"<labels>{labels}</labels></task></meta>\n" + "\n".join(parts) + "\n</annotations>\n")
        (out / "annotations.xml").write_text(xml, encoding="utf-8")
        return {"images": n_img, "boxes": n_box, "folder": str(out)}

    def export_voc(self, out_dir, reviewed_only: bool = False) -> dict:
        """Pascal VOC: Annotations/*.xml, JPEGImages/, ImageSets/Main/default.txt, labelmap.txt."""
        out = Path(out_dir)
        (out / "Annotations").mkdir(parents=True, exist_ok=True)
        (out / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
        stems, n_box = [], 0
        for _, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            name = self._copy_image(path, out / "JPEGImages", stem)
            objs = "".join(f"<object><name>{escape(self.classes[c])}</name><pose>Unspecified</pose>"
                           f"<truncated>0</truncated><difficult>0</difficult><bndbox><xmin>{x1:.0f}</xmin>"
                           f"<ymin>{y1:.0f}</ymin><xmax>{x2:.0f}</xmax><ymax>{y2:.0f}</ymax></bndbox></object>"
                           for c, _, x1, y1, x2, y2 in boxes)
            (out / "Annotations" / f"{stem}.xml").write_text(
                f"<annotation><folder>JPEGImages</folder><filename>{escape(name)}</filename><size><width>{W}</width>"
                f"<height>{H}</height><depth>3</depth></size>{objs}</annotation>\n", encoding="utf-8")
            stems.append(stem)
            n_box += len(boxes)
        (out / "ImageSets" / "Main" / "default.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
        (out / "labelmap.txt").write_text("".join(f"{n}:::\n" for n in self.classes), encoding="utf-8")
        return {"images": len(stems), "boxes": n_box, "folder": str(out)}

    def export_labelstudio(self, out_dir, reviewed_only: bool = False) -> dict:
        """Label Studio: tasks.json (rectangle labels, percent coordinates), labeling_config.xml, images/."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tasks, n_box = [], 0
        for k, stem, path, W, H, boxes in self._export_rows(reviewed_only):
            name = self._copy_image(path, out / "images", stem)
            result = [{"id": f"{k}_{obj}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
                       "original_width": W, "original_height": H,
                       "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                                 "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [self.classes[c]]}}
                      for c, obj, x1, y1, x2, y2 in boxes]
            tasks.append({"data": {"image": f"images/{name}"}, "annotations": [{"result": result}]})
            n_box += len(result)
        (out / "tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
        config = ('<View>\n  <Image name="image" value="$image"/>\n  <RectangleLabels name="label" toName="image">\n'
                  + "".join(f"    <Label value={quoteattr(n)}/>\n" for n in self.classes)
                  + "  </RectangleLabels>\n</View>\n")
        (out / "labeling_config.xml").write_text(config, encoding="utf-8")
        return {"images": len(tasks), "boxes": n_box, "folder": str(out)}

    FORMATS = ("yolo", "coco", "cvat", "voc", "labelstudio")

    def export(self, fmt: str, out_dir, reviewed_only: bool = False) -> dict:
        """Write the dataset in one of FORMATS into out_dir."""
        if fmt not in self.FORMATS:
            raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(self.FORMATS)}")
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
                "reviewed": {k: self.is_reviewed(k) for k in items}}

    def restore(self, snap: dict) -> list[int]:
        """Put items back exactly as `snapshot` saw them; returns the items touched."""
        with self.db:
            for k, rows in snap["rows"].items():
                self.db.execute("DELETE FROM boxes WHERE item=?", (k,))
                self.db.executemany("INSERT INTO boxes VALUES (?,?,?,?,?,?,?,?,?)", rows)
                if snap["reviewed"][k]:
                    self.db.execute("INSERT OR IGNORE INTO reviewed VALUES (?)", (k,))
                else:
                    self.db.execute("DELETE FROM reviewed WHERE item=?", (k,))
        return list(snap["rows"])

    def mirrored_pairs(self) -> bool:
        return mirrored_pairs(self.classes)
