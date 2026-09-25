"""Teach & Transfer, part two: label other videos or image folders exactly as the source was labeled.

For each source: 1 frame in N (the source's step, counted from frame 0 like a project) -> optional
parent object (SAM 3, re-found every few frames) -> the taught RF-DETR inside it -> boxes in full-frame
YOLO. The output folder mirrors the source dataset (images/, labels/, classes.txt, data.yaml, the same
`<video stem>_fNNNNNN` naming) plus preview.mp4 (videos) and summary.json, whose frames_to_check lists
frames a person should look at. Nothing is filled in beyond what the model finds.
"""
import colorsys
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from engine.project import IMAGE_EXTS
from engine.teach import _save, write_data_yaml, yolo_line

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
FRAME_NAMING = "{stem}_f{frame:06d}"


def frames_to_check(rows, presence: dict, low: float = 0.5, usual: float = 0.9) -> list[dict]:
    """Frames worth a human look: far fewer boxes than this source's median (under `low` x median), or
    missing a class the source labels in at least `usual` of its frames. rows: [(name, [class of each box])]."""
    median = float(np.median([len(f) for _, f in rows])) if rows else 0.0
    usual_classes = [c for c, r in presence.items() if r >= usual]
    out = []
    for name, found in rows:
        missing = [c for c in usual_classes if c not in found]
        if len(found) < low * median or missing:
            out.append({"image": name, "boxes": len(found), "missing": missing})
    return out


def _video_frames(path: Path, every: int, naming: str, max_frames):
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if max_frames is not None and i >= max_frames:
                break
            if i % every == 0:
                yield naming.format(stem=path.stem, frame=i), fr.to_image(), None


def _folder_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _folder_frames(files, root: Path):
    for p in files:
        rel = p.relative_to(root).parent / p.stem
        with Image.open(p) as im:
            yield rel.as_posix().replace("/", "__"), im.convert("RGB"), p


class _Preview:
    """preview.mp4: the labeled frames with their boxes, 960 px wide, at the sampled frame rate."""

    def __init__(self, path: Path, fps: float, size, classes):
        W, H = size
        self.w, self.h, self.scale, self.classes = 960, max(2, round(960 * H / W / 2) * 2), 960 / W, classes
        self.colors = [tuple(int(255 * v) for v in colorsys.hsv_to_rgb(k / max(len(classes), 1), 0.9, 1.0))
                       for k in range(len(classes))]
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream("libx264", rate=max(1, round(fps)))
        self.s.width, self.s.height, self.s.pix_fmt = self.w, self.h, "yuv420p"

    def add(self, img: Image.Image, boxes, scores) -> None:
        view = img.resize((self.w, self.h))
        d = ImageDraw.Draw(view)
        for (c, x1, y1, x2, y2), s in zip(boxes, scores):
            x1, y1, x2, y2 = (v * self.scale for v in (x1, y1, x2, y2))
            d.rectangle([x1, y1, x2, y2], outline=self.colors[c], width=2)
            d.text((x1, max(0, y1 - 11)), f"{self.classes[c]} {s:.2f}", fill=self.colors[c])
        for pkt in self.s.encode(av.VideoFrame.from_image(view)):
            self.c.mux(pkt)

    def close(self) -> None:
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def _fresh(out: Path) -> None:
    """Start an empty output folder; a previous transfer output (it has summary.json) is replaced, any
    other non-empty folder is left alone."""
    if out.exists() and any(out.iterdir()):
        if not (out / "summary.json").exists():
            raise FileExistsError(f"{out} already exists and is not a transfer output; choose another --out")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()


def _label_source(src: Path, kind: str, out: Path, model, finder, s: dict, every: int, threshold: float,
                  parent_every: int, preview: bool, max_frames, progress, should_stop) -> dict:
    from engine import detector
    if finder:
        from engine.parent import crop_box
    classes = s["classes"]
    naming = s["naming"] if "{frame" in s.get("naming", "") else FRAME_NAMING
    _fresh(out)
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    write_data_yaml(out, classes)
    summary = {"source": str(src.resolve()), "kind": kind, "every": every if kind == "video" else 1,
               "threshold": threshold, "parent": s.get("parent"), "status": "running"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if kind == "video":
        with av.open(str(src)) as c:
            vs = c.streams.video[0]
            n, fps, size = vs.frames or 0, float(vs.average_rate or 25), (vs.codec_context.width, vs.codec_context.height)
        if max_frames:
            n = min(n, max_frames) if n else max_frames
        total, frames = -(-n // every), _video_frames(src, every, naming, max_frames)
        writer = _Preview(out / "preview.mp4", fps / every, size, classes) if preview else None
    else:
        files = _folder_images(src)[:max_frames]
        total, frames, writer = len(files), _folder_frames(files, src), None
    counts, rows, box, finds, lost, stopped, t = Counter(), [], None, 0, 0, False, time.perf_counter()
    try:
        for k, (name, img, path) in enumerate(frames):
            if should_stop and should_stop():
                stopped = True
                break
            W, H = img.size
            if finder and (kind == "images" or k % parent_every == 0):
                found = finder.find(img)
                box = crop_box(found, W, H, s.get("parent_margin", 0.12)) if found else None
                finds, lost = finds + 1, lost + (found is None)
            crop = box or (0, 0, W, H)
            det = detector.predict(model, img.crop(crop) if box else img, threshold)
            boxes = [(int(c), x1 + crop[0], y1 + crop[1], x2 + crop[0], y2 + crop[1])
                     for (x1, y1, x2, y2), c in zip(det.xyxy, det.class_id) if 0 <= c < len(classes)]
            scores = [float(v) for v, c in zip(det.confidence, det.class_id) if 0 <= c < len(classes)]
            lines = [l for l in (yolo_line(*b, W, H) for b in boxes) if l]
            if path:
                shutil.copyfile(path, out / "images" / (name + path.suffix.lower()))
            else:
                _save(img, out / "images" / (name + s.get("image_ext", ".jpg")))
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts.update(b[0] for b in boxes)
            rows.append((name, [classes[b[0]] for b in boxes]))
            if writer:
                writer.add(img, boxes, scores)
            if progress:
                progress(k + 1, total, f"Labeling {src.name}")
    finally:
        if writer:
            writer.close()
    per_frame = [len(f) for _, f in rows]
    presence = s.get("presence", {})
    summary.update({
        "status": "stopped" if stopped else "done", "frames": len(rows), "boxes": int(sum(counts.values())),
        "boxes_per_frame_median": float(np.median(per_frame)) if per_frame else 0.0,
        "empty_frames": int(sum(n == 0 for n in per_frame)),
        "boxes_per_class": {c: counts[k] for k, c in enumerate(classes)},
        "presence": {c: round(sum(c in f for _, f in rows) / max(len(rows), 1), 3) for c in classes},
        "source_presence": presence, "parent_searches": finds, "parent_not_found": lost,
        "seconds": round(time.perf_counter() - t, 1), "frames_to_check": frames_to_check(rows, presence)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def transfer(run_dir, sources, out_dir, every: int | None = None, threshold: float | None = None,
             parent: str | None = None, preview: bool = True, max_frames: int | None = None, parent_every: int = 4,
             progress=None, should_stop=None) -> dict:
    """Label each source (video file or image folder) into out_dir/<source stem>/ with the run's model.
    every, threshold and parent default to the run's settings; parent="" turns the parent crop off.
    max_frames limits each source to its first frames (or images), for a quick look."""
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    every = int(every or s.get("every") or 1)
    threshold = float(s["threshold"] if threshold is None else threshold)
    parent = s.get("parent") if parent is None else (parent or None)
    todo = []
    for src in map(Path, sources):
        if src.is_dir():
            todo.append((src, "images", Path(out_dir) / src.name))
        elif src.is_file() and src.suffix.lower() in VIDEO_EXTS:
            todo.append((src, "video", Path(out_dir) / src.stem))
        else:
            raise FileNotFoundError(f"{src} is neither a video file ({', '.join(sorted(VIDEO_EXTS))}) nor a folder")
    from engine import detector, hw
    model = detector.load(run_dir / s["checkpoint"], s["size"], s["resolution"])
    finder = None
    if parent:
        from engine.parent import ParentFinder
        finder = ParentFinder(parent)
    results = {}
    try:
        for src, kind, out in todo:
            if should_stop and should_stop():
                break
            results[str(src)] = _label_source(src, kind, out, model, finder, s | {"parent": parent}, every, threshold,
                                              parent_every, preview, max_frames, progress, should_stop)
            results[str(src)]["folder"] = str(out)
    finally:
        del model, finder
        hw.free_gpu_memory()
    return {"run_dir": str(run_dir), "out_dir": str(out_dir), "every": every, "threshold": threshold,
            "parent": parent, "sources": results}
