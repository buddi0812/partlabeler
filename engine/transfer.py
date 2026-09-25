"""Teach & Transfer, part two: label other videos or image folders exactly as the source was labeled.

For each source: 1 frame in N (the source's step, counted from frame 0 like a project) -> optional
parent object (SAM 3, re-found every few frames) -> the taught RF-DETR inside it -> boxes in full-frame
YOLO. The output folder mirrors the source dataset (images/, labels/, classes.txt, data.yaml, the same
`<video stem>_fNNNNNN` naming) plus preview.mp4 (videos) and summary.json, whose frames_to_check lists
frames a person should look at, including (videos, on by default) what the track check in engine/tracks.py
finds: a label that flips along a track, a part missing for a frame or two, a low-score box on one frame
only. Nothing is filled in or changed beyond what the model finds.
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
from engine.tracks import check

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
FRAME_NAMING = "{stem}_f{frame:06d}"


def frames_to_check(rows, presence: dict, low: float = 0.5, usual: float = 0.9, tracks: dict | None = None) -> list[dict]:
    """Frames worth a human look: far fewer boxes than this source's median (under `low` x median), missing
    a class the source labels in at least `usual` of its frames, or flagged by the track check (`tracks`:
    {name: {"label_flip": [...], "possible_miss": [...], "lone_box": [...]}}). rows: [(name, [class of each box])]."""
    median = float(np.median([len(f) for _, f in rows])) if rows else 0.0
    usual_classes = [c for c, r in presence.items() if r >= usual]
    out = []
    for name, found in rows:
        missing = [c for c in usual_classes if c not in found]
        track = {k: v for k, v in (tracks or {}).get(name, {}).items() if v}
        if len(found) < low * median or missing or track:
            out.append({"image": name, "boxes": len(found), "missing": missing} | ({"track": track} if track else {}))
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
                  parent_every: int, preview: bool, max_frames, progress, should_stop, tracks: bool = True,
                  predict=None) -> dict:
    """`predict(img)` -> [(cls, x1, y1, x2, y2, score)] replaces the taught detector (quick transfer)."""
    if predict is None:
        from engine import detector
    if finder:
        from engine.parent import crop_box
    classes = s["classes"]
    naming = s["naming"] if "{frame" in s.get("naming", "") else FRAME_NAMING
    _fresh(out)
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    write_data_yaml(out, classes)
    summary = {"source": str(src.resolve()), "kind": kind, "every": every if kind == "video" else 1, "mode": s.get("mode", "trained"),
               "threshold": threshold, "parent": s.get("parent"), "status": "running"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if kind == "video":
        with av.open(str(src)) as c:
            vs = c.streams.video[0]
            n, fps, size = vs.frames or 0, float(vs.average_rate or 25), (vs.codec_context.width, vs.codec_context.height)
        if max_frames:
            n = min(n, max_frames) if n else max_frames
        total, frames = -(-n // every), _video_frames(src, every, naming, max_frames)
    else:
        files = _folder_images(src)[:max_frames]
        total, frames = len(files), _folder_frames(files, src)
    # 1) detect every frame (boxes kept in memory: (cls, x1, y1, x2, y2, score)), 2) check along tracks,
    # 3) write the labels and the preview
    found, box, finds, lost, stopped, t = [], None, 0, 0, False, time.perf_counter()
    for k, (name, img, path) in enumerate(frames):
        if should_stop and should_stop():
            stopped = True
            break
        W, H = img.size
        if finder and (kind == "images" or k % parent_every == 0):
            hit = finder.find(img)
            box = crop_box(hit, W, H, s.get("parent_margin", 0.12)) if hit else None
            finds, lost = finds + 1, lost + (hit is None)
        if predict:
            dets = [d for d in predict(img) if 0 <= d[0] < len(classes)]
        else:
            crop = box or (0, 0, W, H)
            det = detector.predict(model, img.crop(crop) if box else img, threshold)
            dets = [(int(c), x1 + crop[0], y1 + crop[1], x2 + crop[0], y2 + crop[1], float(p))
                    for (x1, y1, x2, y2), c, p in zip(det.xyxy, det.class_id, det.confidence) if 0 <= c < len(classes)]
        image = out / "images" / (name + (path.suffix.lower() if path else s.get("image_ext", ".jpg")))
        if path:
            shutil.copyfile(path, image)
        else:
            _save(img, image)
        found.append((name, W, H, image, dets))
        if progress:
            progress(k + 1, total, f"Labeling {src.name}")
    flagged, track_check = {}, None
    if tracks and kind == "video" and found:
        flagged = {found[k][0]: v for k, v in check([f[4] for f in found], classes).items()}
        track_check = {w: sum(len(v[w]) for v in flagged.values()) for w in ("label_flip", "possible_miss", "lone_box")}
    counts, rows = Counter(), []
    for name, W, H, _, dets in found:
        lines = [l for l in (yolo_line(*d[:5], W, H) for d in dets) if l]
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts.update(d[0] for d in dets)
        rows.append((name, [classes[d[0]] for d in dets]))
    if kind == "video" and preview and found:
        writer = _Preview(out / "preview.mp4", fps / every, size, classes)
        try:
            for k, (_, _, _, image, dets) in enumerate(found):
                with Image.open(image) as im:
                    writer.add(im.convert("RGB"), [d[:5] for d in dets], [d[5] for d in dets])
                if progress and k % 10 == 0:
                    progress(k + 1, len(found), f"Writing the preview of {src.name}")
        finally:
            writer.close()
    per_frame = [len(f) for _, f in rows]
    presence = s.get("presence", {})
    summary.update({
        "status": "stopped" if stopped else "done", "frames": len(rows), "boxes": int(sum(counts.values())),
        "boxes_per_frame_median": float(np.median(per_frame)) if per_frame else 0.0,
        "empty_frames": int(sum(n == 0 for n in per_frame)),
        "boxes_per_class": {c: counts[k] for k, c in enumerate(classes)},
        "presence": {c: round(sum(c in f for _, f in rows) / max(len(rows), 1), 3) for c in classes},
        "source_presence": presence, "parent_searches": finds, "parent_not_found": lost, "track_check": track_check,
        "seconds": round(time.perf_counter() - t, 1), "frames_to_check": frames_to_check(rows, presence, tracks=flagged)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def transfer(run_dir, sources, out_dir, every: int | None = None, threshold: float | None = None,
             parent: str | None = None, preview: bool = True, max_frames: int | None = None, parent_every: int = 4,
             tracks: bool = True, progress=None, should_stop=None) -> dict:
    """Label each source (video file or image folder) into out_dir/<source stem>/ with the run's model.
    every, threshold and parent default to the run's settings; parent="" turns the parent crop off.
    max_frames limits each source to its first frames (or images), for a quick look. tracks=False skips
    the track check of videos."""
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
                                              parent_every, preview, max_frames, progress, should_stop, tracks)
            results[str(src)]["folder"] = str(out)
    finally:
        del model, finder
        hw.free_gpu_memory()
    return {"run_dir": str(run_dir), "out_dir": str(out_dir), "every": every, "threshold": threshold,
            "parent": parent, "tracks": tracks, "sources": results}
