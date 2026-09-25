"""Quick transfer (no training): label similar videos or image folders from a labeled dataset by matching.

A fast preview next to Teach & Transfer. Up to `refs` labeled images spread evenly over the source (YOLO:
images/, labels/, classes.txt or data.yaml) are the examples, and each target frame is matched with the
annotator's Suggest (DINOv3 patch matching, engine/suggest.py; inside the parent object when one is given).
Nothing is trained, and left/right twins are told apart by position, not appearance. Output: the Transfer
layout under <run>/labels/<source> (images/, labels/, classes.txt, data.yaml, preview.mp4, summary.json with
frames_to_check), and settings.json marks the run as quick so "Review in annotator" works. SAM 3 then fits
each matched box to the part (`tighten`). Match thresholds are calibrated per new video (`calibrate`):
examples all come from one video, so they resemble each other far more than a part in another colour or
light resembles them, and their own thresholds would find nothing there. Measured on 143 held-out frames (spikes/REPORT.md, S9): F1 0.69,
70% of parts found, 1.4 s a frame on an RTX 3060, against F1 0.96 for a taught detector. Check every
frame; Teach is the accurate path.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image

from engine.teach import analyse, read_labels, source_items


class SourceView:
    """What Suggester reads from a project, over a labeled dataset: its examples first, then the frames being
    labeled, each under a new index so cached features never mix."""

    def __init__(self, dataset_dir, classes, parent=None, refs: int = 30):
        labeled = [it for it in source_items(dataset_dir)
                   if it["label"] and read_labels(it["label"], len(classes))[0]]
        if not labeled:
            raise ValueError(f"{dataset_dir} has no labeled images to learn from")
        step = max(1, len(labeled) / refs)
        self.refs = [labeled[int(i * step)] for i in range(min(refs, len(labeled)))]
        self.classes, self.meta = classes, {"parent": parent}
        self.folder = Path(dataset_dir).resolve() / ".quick"          # cache key only
        self.targets = {}

    def statuses(self) -> list[int]:
        return [4] * len(self.refs) + [0] * len(self.targets)

    def image(self, k: int) -> Image.Image:
        if k < len(self.refs):
            with Image.open(self.refs[k]["image"]) as im:
                return im.convert("RGB")
        return self.targets[k]

    def boxes(self, k: int) -> list[dict]:
        if k >= len(self.refs):
            return []
        with Image.open(self.refs[k]["image"]) as im:
            W, H = im.size
        rows, _ = read_labels(self.refs[k]["label"], len(self.classes))
        return [{"cls": c, "source": "manual", "box": [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]}
                for c, cx, cy, w, h in rows]

    def add(self, img: Image.Image) -> int:
        """Register the next frame to label; only the newest one is kept in memory."""
        k = len(self.refs) + len(self.targets)
        self.targets = {**{i: None for i in self.targets}, k: img}
        return k


def tighten(seg, img, found, grow: float = 2.0):
    """SAM 3 fits each matched box (a typical-size box around the match) to the part under it. The fitted box
    is kept only when its size stays within `grow` times the matched one; otherwise the match stays as it was."""
    seg.set_image(img)
    out = []
    for c, box, score in found:
        mask, _ = seg.segment(box=[float(v) for v in box])
        ys, xs = mask.nonzero()
        if len(xs):
            fit = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            a0, a1 = (box[2] - box[0]) * (box[3] - box[1]), (fit[2] - fit[0]) * (fit[3] - fit[1])
            if a0 > 0 and 1 / grow <= a1 / a0 <= grow:
                box = fit
        out.append((c, box, score))
    return out


def sample(src: Path, kind: str, every: int, max_frames, n: int = 24) -> list[Image.Image]:
    """Up to n frames (or images) spread evenly over a source, for calibration."""
    from engine.transfer import _folder_images
    if kind == "images":
        files = _folder_images(src)[:max_frames]
        return [Image.open(f).convert("RGB") for f in files[:: max(1, len(files) // n)][:n]]
    with av.open(str(src)) as c:
        total = c.streams.video[0].frames or 0
    total = min(total, max_frames) if total and max_frames else total or max_frames or 10 ** 9
    idx = list(range(0, total, every))
    want = set(idx[:: max(1, len(idx) // n)][:n])
    out = []
    with av.open(str(src)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out.append(fr.to_image())
            if i >= max(want) or len(out) == len(want):
                break
    return out


def calibrate(suggester, view, frames, presence: dict, refs: int = 30) -> dict:
    """Thresholds for a new video, per group of classes: the match score reached in about as many sampled
    frames as the part appears in the source (Transfer's premise: similar videos show the same parts)."""
    from engine.suggest import group_of
    maxima, share = defaultdict(list), defaultdict(float)
    for img in frames:
        for g, m in suggester.maxima(view, view.add(img), refs).items():
            maxima[g].append(m)
    for name, p in presence.items():
        share[group_of(name)] = max(share[group_of(name)], p)
    return {g: float(np.quantile(ms, 1 - min(share[g], 0.98))) for g, ms in maxima.items() if share[g] > 0}


def quick_transfer(dataset_dir, sources, run_dir, parent: str | None = None, every: int | None = None,
                   refs: int = 30, preview: bool = True, max_frames: int | None = None, tracks: bool = True,
                   fit: bool = True, progress=None, should_stop=None, suggester=None, segmenter=None) -> dict:
    """Label each source into run_dir/labels/<source> by matching the dataset's examples; no training.
    Writes run_dir/settings.json ("mode": "quick") so the outputs can be reviewed like Transfer's.
    fit=False skips the SAM 3 box fitting (3x faster, looser boxes)."""
    from engine import hw
    from engine.transfer import VIDEO_EXTS, _label_source
    dataset_dir, run_dir = Path(dataset_dir).resolve(), Path(run_dir)
    a = analyse(dataset_dir)
    classes, sampling = a["classes"], a["sampling"]
    every = int(every or sampling["every"] or 1)
    s = {"mode": "quick", "classes": classes, "parent": parent, "every": every, "naming": sampling["naming"],
         "image_ext": sampling["image_ext"], "presence": {n: pc["presence"] for n, pc in a["per_class"].items()},
         "source": str(dataset_dir), "refs": refs, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    view = SourceView(dataset_dir, classes, parent, refs)
    if suggester is None or (fit and segmenter is None):
        from engine.api import model
        suggester = suggester or model("suggest")
        segmenter = segmenter or (model("seg") if fit else None)

    thresholds = {}

    def predict(img):
        found = suggester.suggest(view, view.add(img), max_refs=refs, thresholds=thresholds)
        if fit and found:
            found = tighten(segmenter, img, found)
        return [(int(c), *map(float, box), float(score)) for c, box, score in found]

    results = {}
    try:
        for src in map(Path, sources):
            if should_stop and should_stop():
                break
            kind = "images" if src.is_dir() else "video" if src.suffix.lower() in VIDEO_EXTS else None
            if kind is None:
                raise FileNotFoundError(f"{src} is neither a video file ({', '.join(sorted(VIDEO_EXTS))}) nor a folder")
            out = run_dir / "labels" / (src.name if kind == "images" else src.stem)
            if progress:
                progress(0, 0, f"Calibrating on {src.name}")
            thresholds.clear()
            thresholds.update(calibrate(suggester, view, sample(src, kind, every, max_frames), s["presence"], refs))
            results[str(src)] = _label_source(src, kind, out, None, None, s, every, 0.0, 1, preview, max_frames,
                                              progress, should_stop, tracks, predict=predict) | {
                "folder": str(out), "thresholds": {g: round(v, 3) for g, v in thresholds.items()}}
    finally:
        hw.free_gpu_memory()
    return {"run_dir": str(run_dir), "mode": "quick", "every": every, "parent": parent, "sources": results}
