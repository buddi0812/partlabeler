"""Teach & Transfer, part one: learn a labeled source dataset and prove the model reproduces it.

    analyse -> prepare (optional parent-object crop, time-block split) -> train (RF-DETR)
            -> prove on held-out frames -> run folder: settings.json, report.json, report.md, model/

The source is a YOLO folder (images/, labels/, classes.txt or data.yaml), usually frames of one video.
"Annotate as per source": settings.json records the source's classes, sampling step and file naming,
which `transfer` copies. Prepare and train are skipped on a rerun when their outputs are complete, and
an interrupted training run resumes from its last epoch.
"""
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image, ImageOps

from engine.project import FRAME_RE, IMAGE_EXTS, mirrored_pairs, read_classes

TARGETS = {"mAP50": 0.90, "recall": 0.90, "stress_drop": 10.0}   # stress drop in mAP50 points
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.1, 0.9, 0.05)]
TINY_PX = 8            # shorter box side, in pixels at the training resolution
FEW_BOXES = 10
FEW_FRAMES = 0.10      # a class present in under 10% of labeled frames
PARENT_MARGIN = 0.12


# ---- reading the source -------------------------------------------------------------------
def classes_of(dataset_dir) -> list[str]:
    for name in ("classes.txt", "data.yaml", "data.yml"):
        if (Path(dataset_dir) / name).exists():
            return read_classes(Path(dataset_dir) / name)
    raise FileNotFoundError(f"{dataset_dir} has no classes.txt or data.yaml")


def source_items(dataset_dir) -> list[dict]:
    """Every image under images/ with its label file (labels/<same path>.txt, None when missing), in time
    order: by video prefix and frame number when names end in _fNNNNNN, else by name."""
    images, labels = Path(dataset_dir) / "images", Path(dataset_dir) / "labels"
    if not images.is_dir():
        raise FileNotFoundError(f"{dataset_dir} has no images/ folder")
    items = []
    for p in images.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(images).parent / p.stem
        label = labels / rel.parent / (p.stem + ".txt")
        name = rel.as_posix()
        m = FRAME_RE.search(name)
        items.append({"name": name.replace("/", "__"), "image": p, "label": label if label.exists() else None,
                      "key": (name[:m.start()], int(m.group(1))) if m else (name, -1)})
    return sorted(items, key=lambda it: it["key"])


def parse_row(line: str):
    """(class, cx, cy, w, h) of one YOLO line; None if malformed."""
    parts = line.split()
    try:
        c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
    except (ValueError, IndexError, TypeError):
        return None
    return (c, cx, cy, w, h) if len(parts) >= 5 and w > 0 and h > 0 else None


def read_labels(path, nc: int) -> tuple[list[tuple], int]:
    """Valid rows of a YOLO file and the number of invalid lines (malformed, empty box, unknown class)."""
    rows, bad = [], 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = parse_row(line)
        if r is None or not 0 <= r[0] < nc:
            bad += 1
        else:
            rows.append(r)
    return rows, bad


def infer_sampling(names) -> dict:
    """Every-N step, frame offset and naming pattern from `<stem>_f<frame>` names. every is the most common
    gap between consecutive frames of one video (gaps from unlabeled frames do not change it); None when
    most names carry no frame number (e.g. a photo folder), whose files keep their own names."""
    frames, digits = defaultdict(list), Counter()
    for n in names:
        m = FRAME_RE.search(n)
        if m:
            frames[n[:m.start()]].append(int(m.group(1)))
            digits[len(m.group(1))] += 1
    if not names or sum(digits.values()) * 2 <= len(names):
        return {"every": None, "offset": None, "naming": "{name}"}
    steps = Counter(b - a for f in frames.values() for a, b in zip(sorted(f), sorted(f)[1:]) if b > a)
    every = min(s for s, n in steps.items() if n == max(steps.values())) if steps else None
    offset = Counter(x % every for f in frames.values() for x in f).most_common(1)[0][0] if every else None
    return {"every": every, "offset": offset, "naming": f"{{stem}}_f{{frame:0{digits.most_common(1)[0][0]}d}}"}


# ---- analyse --------------------------------------------------------------------------------
def analyse(dataset_dir, resolution: int = 640) -> dict:
    """Classes, box counts, presence and box sizes per class, labeled vs unlabeled images, sampling, and
    warnings about labels that will hurt training or its evaluation."""
    root = Path(dataset_dir)
    classes, items = classes_of(root), source_items(root)
    nc = len(classes)
    boxes, frames, tiny, dups = Counter(), Counter(), Counter(), Counter()
    sizes = defaultdict(list)
    invalid = background = 0
    for it in items:
        if it["label"] is None:
            continue
        rows, bad = read_labels(it["label"], nc)
        invalid += bad
        background += not rows
        with Image.open(it["image"]) as im:
            W, H = im.size
        scale, seen = resolution / max(W, H), set()
        for c, cx, cy, w, h in rows:
            key = (c, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
            if key in seen:
                dups[c] += 1
                continue
            seen.add(key)
            boxes[c] += 1
            sizes[c].append((w * W, h * H))
            tiny[c] += min(w * W, h * H) * scale < TINY_PX
        frames.update({c for c, *_ in rows})
    labeled = sum(it["label"] is not None for it in items)
    per_class = {}
    for c, name in enumerate(classes):
        wh = np.array(sizes[c]) if sizes[c] else np.zeros((0, 2))
        stat = lambda a: [round(float(a.min()), 1), round(float(np.median(a)), 1), round(float(a.max()), 1)] if len(a) else None
        per_class[name] = {"boxes": boxes[c], "frames": frames[c], "presence": round(frames[c] / max(labeled, 1), 3),
                           "width_px": stat(wh[:, 0]), "height_px": stat(wh[:, 1]),
                           "tiny": tiny[c], "duplicates": dups[c]}
    sampling = infer_sampling([it["name"] for it in items])
    sampling["image_ext"] = Counter(it["image"].suffix.lower() for it in items).most_common(1)[0][0] if items else ".jpg"

    warn = []
    if sampling["offset"]:
        warn.append(f"source frames start at offset {sampling['offset']} (not a multiple of {sampling['every']}); "
                    f"transfer and projects sample frames 0, {sampling['every']}, {2 * sampling['every']}, ...")
    if labeled < len(items):
        warn.append(f"{len(items) - labeled} image(s) have no label file: treated as unlabeled and left out "
                    "(an empty label file would mean 'checked, nothing there')")
    if invalid:
        warn.append(f"{invalid} label line(s) are invalid (malformed, zero size or class id outside 0-{nc - 1}) "
                    "and are ignored")
    pcs = per_class.items()
    groups = {
        "never labeled, so the model cannot learn them": {n: None for n, pc in pcs if not pc["boxes"]},
        f"with fewer than {FEW_BOXES} boxes": {n: pc["boxes"] for n, pc in pcs if 0 < pc["boxes"] < FEW_BOXES},
        f"in under {FEW_FRAMES:.0%} of the {labeled} labeled frames":
            {n: pc["frames"] for n, pc in pcs if pc["boxes"] and pc["presence"] < FEW_FRAMES},
        f"with tiny boxes, under {TINY_PX} px on the shorter side at {resolution} px input before any parent crop "
        "(a parent crop or a higher resolution helps)": {n: pc["tiny"] for n, pc in pcs if pc["tiny"]},
        "with exact duplicate boxes, dropped for training": {n: pc["duplicates"] for n, pc in pcs if pc["duplicates"]}}
    for text, found in groups.items():
        if found:
            warn.append(f"classes {text}: " + ", ".join(n if v is None else f"{n} ({v})" for n, v in found.items()))
    return {"source": str(root), "classes": classes, "images": len(items), "labeled": labeled,
            "with_boxes": labeled - background, "background": background, "unlabeled": len(items) - labeled,
            "boxes": sum(boxes.values()), "invalid_lines": invalid, "per_class": per_class,
            "mirrored_pairs": mirrored_pairs(classes), "sampling": sampling, "warnings": warn}


# ---- prepare --------------------------------------------------------------------------------
def time_blocks(n: int, held_out: float = 0.2, blocks: int = 10) -> list[bool]:
    """Held-out flags for n frames in time order: whole blocks of neighbouring frames, spread over the
    source, so the test is not near-duplicates of training frames."""
    if not 0 < held_out < 1:
        raise ValueError("held_out must be between 0 and 1")
    blocks = max(2, min(blocks, n))
    k = min(blocks - 1, max(1, round(held_out * blocks)))
    chosen = {int((j + 0.5) * blocks / k) for j in range(k)}
    return [int(i * blocks / n) in chosen for i in range(n)]


def yolo_line(c: int, x1, y1, x2, y2, W, H) -> str | None:
    """YOLO row for a pixel box, clipped to the image; None when nothing is left."""
    x1, y1, x2, y2 = max(0.0, float(x1)), max(0.0, float(y1)), min(float(W), float(x2)), min(float(H), float(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def to_crop(row, crop, W, H) -> str | None:
    """A full-image YOLO row in the coordinates of crop (x1, y1, x2, y2 pixels): clipped, None if outside."""
    c, cx, cy, w, h = row
    X1, Y1, X2, Y2 = crop
    x1, y1 = (cx - w / 2) * W - X1, (cy - h / 2) * H - Y1
    return yolo_line(c, x1, y1, x1 + w * W, y1 + h * H, X2 - X1, Y2 - Y1)


def from_crop(row, crop, W, H) -> str | None:
    """The inverse of to_crop: a row relative to the crop, back in full-image coordinates."""
    c, cx, cy, w, h = row
    X1, Y1, X2, Y2 = crop
    cw, ch = X2 - X1, Y2 - Y1
    x1, y1 = (cx - w / 2) * cw + X1, (cy - h / 2) * ch + Y1
    return yolo_line(c, x1, y1, x1 + w * cw, y1 + h * ch, W, H)


def _save(img: Image.Image, path: Path) -> None:
    jpg = path.suffix.lower() in (".jpg", ".jpeg")
    (img.convert("RGB") if jpg else img).save(path, **({"quality": 95} if jpg else {}))


def write_data_yaml(folder, classes, train: str = "images", val: str = "images") -> None:
    Path(folder, "data.yaml").write_text(f"train: {train}\nval: {val}\n\nnc: {len(classes)}\nnames:\n" +
                                         "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)), encoding="utf-8")


def prepare(dataset_dir, out, parent: str | None = None, held_out: float = 0.2, progress=None,
            should_stop=None) -> dict | None:
    """Training set in `out`: train/ and valid/ (images + YOLO labels) and data.yaml. With a parent (text such
    as "engine block"), each image is cut to that object (SAM 3) and its labels move into the crop. Returns
    the record also saved as prepared.json (reused when source and settings are unchanged); None if stopped."""
    root, out = Path(dataset_dir), Path(out)
    classes = classes_of(root)
    items = [it for it in source_items(root) if it["label"] is not None]
    if len(items) < 2:
        raise ValueError(f"{root}: need at least 2 labeled images")
    held = time_blocks(len(items), held_out)
    key = hashlib.sha1(json.dumps([str(root.resolve()), parent, held_out, classes,
                                   [it["name"] for it in items]]).encode()).hexdigest()[:12]
    if (out / "prepared.json").exists():
        rec = json.loads((out / "prepared.json").read_text(encoding="utf-8"))
        if rec.get("key") == key:
            return rec
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
    finder = None
    if parent:
        from engine.parent import ParentFinder, crop_box
        finder = ParentFinder(parent)
    crops, missing, n = {}, [], Counter()
    try:
        for k, (it, test) in enumerate(zip(items, held)):
            if should_stop and should_stop():
                return None
            img = Image.open(it["image"])
            W, H = img.size
            crop = (0, 0, W, H)
            if finder:
                box = finder.find(img.convert("RGB"))
                if box:
                    crop = crop_box(box, W, H, PARENT_MARGIN)
                else:
                    missing.append(it["name"])
            crops[it["name"]] = list(crop)
            rows, _ = read_labels(it["label"], len(classes))
            n["duplicates_removed"] += len(rows) - len(set(rows))
            lines = []
            for r in dict.fromkeys(rows):                 # exact duplicates dropped, order kept
                c, cx, cy, w, h = r
                x1, y1 = max(0.0, (cx - w / 2) * W), max(0.0, (cy - h / 2) * H)
                x2, y2 = min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)
                inside = x1 >= crop[0] and y1 >= crop[1] and x2 <= crop[2] and y2 <= crop[3]
                line = to_crop(r, crop, W, H)
                n["kept" if line and inside else "clipped" if line else "outside_parent"] += 1
                lines += [line] if line else []
            split = out / ("valid" if test else "train")
            dst = split / "images" / (it["name"] + it["image"].suffix.lower())
            if crop == (0, 0, W, H):
                shutil.copyfile(it["image"], dst)
            else:
                _save(img.crop(crop), dst)
            (split / "labels" / (it["name"] + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            if progress and k % 10 == 0:
                progress(k, len(items), "Preparing the training set" + (f" (finding '{parent}')" if parent else ""))
    finally:
        if finder:
            from engine import hw
            del finder
            hw.free_gpu_memory()
    write_data_yaml(out, classes, "train/images", "valid/images")
    total = sum(n[k] for k in ("kept", "clipped", "outside_parent"))
    rec = {"key": key, "source": str(root.resolve()), "parent": parent, "held_out": held_out,
           "train": [it["name"] for it, h in zip(items, held) if not h],
           "valid": [it["name"] for it, h in zip(items, held) if h],
           "boxes": dict(n), "parent_coverage": round(n["kept"] / total, 3) if parent and total else None,
           "parent_missing": missing, "crops": crops}
    (out / "prepared.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


# ---- prove ----------------------------------------------------------------------------------
def load_targets(label: Path, W: int, H: int) -> sv.Detections:
    rows = [r for r in map(parse_row, label.read_text().splitlines()) if r] if label.exists() else []
    if not rows:
        return sv.Detections(xyxy=np.zeros((0, 4)), class_id=np.zeros(0, int))
    a = np.array(rows, float)
    xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                     (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H], axis=1)
    return sv.Detections(xyxy=xyxy, class_id=a[:, 0].astype(int))


def count_matches(preds, targets, iou: float = 0.5) -> tuple[Counter, Counter, Counter]:
    """Per class true positives, false positives and misses: each label matched at most once, most
    confident prediction first."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for p, t in zip(preds, targets):
        for c in set(p.class_id.tolist()) | set(t.class_id.tolist()):
            pc, tc = p[p.class_id == c], t[t.class_id == c]
            hits = 0
            if len(pc) and len(tc):
                m, used = sv.box_iou_batch(tc.xyxy, pc.xyxy), np.zeros(len(tc), bool)
                order = np.argsort(-pc.confidence) if pc.confidence is not None else range(len(pc))
                for i in order:
                    ious = np.where(used, -1.0, m[:, i])
                    j = int(ious.argmax())
                    if ious[j] >= iou:
                        used[j], hits = True, hits + 1
            tp[c] += hits
            fp[c] += len(pc) - hits
            fn[c] += len(tc) - hits
    return tp, fp, fn


def f1_sweep(preds, targets, thresholds=THRESHOLDS) -> dict[float, float]:
    """Micro F1 over all classes at each confidence threshold."""
    out = {}
    for th in thresholds:
        tp, fp, fn = (sum(c.values()) for c in count_matches([p[p.confidence >= th] for p in preds], targets))
        out[th] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
    return out


def pick_threshold(sweep: dict) -> float:
    """Threshold with the best micro F1; when several share it, the middle one (safest on a plateau)."""
    best = max(sweep.values())
    tied = sorted(t for t, f in sweep.items() if f >= best - 1e-9)
    return tied[len(tied) // 2]


def repaint_outside(img: Image.Image, det: sv.Detections, tint=(0.25, 0.25, 0.27)) -> Image.Image:
    """Everything outside the labeled boxes repainted in a dark tint, keeping its shading."""
    a = np.asarray(img).astype(np.float32)
    keep = np.zeros(a.shape[:2], bool)
    for x1, y1, x2, y2 in det.xyxy.astype(int):
        keep[max(0, y1):y2, max(0, x1):x2] = True
    painted = a.mean(axis=2, keepdims=True) * np.array(tint, np.float32)
    a[~keep] = painted[~keep]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def hue_rotate(img: Image.Image, degrees: float = 180) -> Image.Image:
    h, s, v = img.convert("HSV").split()
    shift = int(round(degrees / 360 * 256))
    return Image.merge("HSV", (h.point(lambda x: (x + shift) % 256), s, v)).convert("RGB")


STRESS = {"grayscale": lambda im, t: ImageOps.grayscale(im).convert("RGB"),
          "hue_180": lambda im, t: hue_rotate(im, 180),
          "dark_background": repaint_outside}


def _map(preds, targets):
    from supervision.metrics import MeanAveragePrecision
    return MeanAveragePrecision().update(preds, targets).compute()


def prove(model, valid_dir, classes, progress=None) -> dict:
    """Agreement with the source labels on held-out frames: mAP50, mAP50-95, the confidence threshold with
    the best micro F1, recall/precision per class at it, and the colour stress test."""
    from engine import detector
    valid_dir = Path(valid_dir)
    files = sorted(p for p in (valid_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    images = [Image.open(f).convert("RGB") for f in files]
    targets = [load_targets(valid_dir / "labels" / (f.stem + ".txt"), *im.size) for im, f in zip(images, files)]
    preds = detector.predict(model, images, threshold=0.01)
    m = _map(preds, targets)
    sweep = f1_sweep(preds, targets)
    th = pick_threshold(sweep)
    tp, fp, fn = count_matches([p[p.confidence >= th] for p in preds], targets)
    ap50 = {int(c): float(a[0]) for c, a in zip(m.matched_classes, m.ap_per_class)}
    per_class = {name: {"labels": tp[c] + fn[c], "found": tp[c], "extra": fp[c],
                        "recall": round(tp[c] / (tp[c] + fn[c]), 3) if tp[c] + fn[c] else None,
                        "precision": round(tp[c] / (tp[c] + fp[c]), 3) if tp[c] + fp[c] else None,
                        "AP50": round(ap50[c], 3) if c in ap50 else None}
                 for c, name in enumerate(classes)}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    stress = {}
    for k, (name, fn_) in enumerate(STRESS.items()):
        if progress:
            progress(k, len(STRESS), f"Colour stress test: {name}")
        vm = _map(detector.predict(model, [fn_(im, t) for im, t in zip(images, targets)], threshold=0.01), targets)
        stress[name] = {"mAP50": round(float(vm.map50), 3), "drop": round(100 * float(m.map50 - vm.map50), 1)}
    return {"frames": len(files), "labels": TP + FN, "mAP50": round(float(m.map50), 3),
            "mAP50_95": round(float(m.map50_95), 3), "threshold": th, "f1_sweep": sweep,
            "f1": sweep[th], "recall": round(TP / max(TP + FN, 1), 3), "precision": round(TP / max(TP + FP, 1), 3),
            "per_class": per_class, "stress": stress}


# ---- run folder -----------------------------------------------------------------------------
def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def _train_step(run_dir: Path, prep: dict, size, epochs, resolution, aug, progress, should_stop) -> Path | None:
    """Train into run_dir/model, or reuse a finished model trained with the same data and settings, or
    resume an interrupted one from its last epoch."""
    from engine import detector
    out = run_dir / "model"
    params = {"dataset": prep["key"], "size": size, "epochs": epochs, "resolution": resolution, "aug": aug}
    done, started = out / "trained.json", out / "started.json"
    if done.exists() and json.loads(done.read_text())["params"] == params and detector.best_checkpoint(out):
        return detector.best_checkpoint(out)
    resume = None
    if started.exists() and json.loads(started.read_text()) == params:
        resume = next((p for p in (out / "last.ckpt", *sorted(out.glob("checkpoint_*.ckpt"))) if p.exists()), None)
    if resume is None:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(started, params)
    t = time.perf_counter()
    ckpt = detector.train(run_dir / "dataset", out, size, epochs, resolution, aug, resume, progress, should_stop)
    if ckpt is None:
        return None
    _write_json(done, {"params": params, "checkpoint": ckpt.name, "minutes": round((time.perf_counter() - t) / 60, 1),
                       "resumed_from": resume.name if resume else None})
    for f in out.glob("*.ckpt"):             # resume state only (~0.5 GB each); the .pth weights stay
        f.unlink()
    return ckpt


def teach(dataset_dir, run_dir, parent: str | None = None, size: str = "small", epochs: int = 30,
          resolution: int = 640, held_out: float = 0.2, aug: str = "strong", progress=None,
          should_stop=None) -> dict:
    """Analyse, prepare, train and prove; writes settings.json, report.json and report.md into run_dir and
    returns the report ({"status": "stopped", ...} when should_stop ended it early)."""
    from engine import detector, hw
    detector.check_resolution(resolution)
    detector.aug_config([], aug)
    if size not in detector.SIZES:
        raise ValueError(f"unknown model size {size!r}; choose one of {', '.join(detector.SIZES)}")
    dataset_dir, run_dir = Path(dataset_dir).resolve(), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda *a: None)
    say(0, 4, "Analysing the source labels")
    analysis = analyse(dataset_dir, resolution)
    _write_json(run_dir / "analysis.json", analysis)
    say(1, 4, "Preparing the training set")
    prep = prepare(dataset_dir, run_dir / "dataset", parent, held_out, progress, should_stop)
    if prep is None:
        return {"status": "stopped", "step": "prepare", "run_dir": str(run_dir)}
    say(2, 4, "Training")
    ckpt = _train_step(run_dir, prep, size, epochs, resolution, aug, progress, should_stop)
    if ckpt is None:
        return {"status": "stopped", "step": "train", "run_dir": str(run_dir)}
    say(3, 4, "Proving on held-out frames")
    model = detector.load(ckpt, size, resolution)
    try:
        proof = prove(model, run_dir / "dataset" / "valid", analysis["classes"], progress)
    finally:
        del model
        hw.free_gpu_memory()

    classes, sampling = analysis["classes"], analysis["sampling"]
    worst = max((s["drop"] for s in proof["stress"].values()), default=0.0)
    targets = {"held-out mAP50": {"value": proof["mAP50"], "target": f">= {TARGETS['mAP50']:.2f}",
                                  "pass": proof["mAP50"] >= TARGETS["mAP50"]},
               "recall": {"value": proof["recall"], "target": f">= {TARGETS['recall']:.2f}",
                          "pass": proof["recall"] >= TARGETS["recall"]},
               "colour stress drop": {"value": worst, "target": f"<= {TARGETS['stress_drop']:.0f} points",
                                      "pass": worst <= TARGETS["stress_drop"]}}
    warnings = list(analysis["warnings"])
    if parent and prep["parent_coverage"] is not None and prep["parent_coverage"] < 0.9:
        warnings.append(f"only {prep['parent_coverage']:.0%} of labeled boxes lie wholly inside the '{parent}' crop: "
                        "check the parent text")
    if prep["parent_missing"]:
        warnings.append(f"'{parent}' was not found in {len(prep['parent_missing'])} image(s); they were used uncropped")
    unmeasured = [n for n, pc in proof["per_class"].items() if analysis["per_class"][n]["boxes"] and not pc["labels"]]
    if unmeasured:
        warnings.append(f"classes with no held-out labels, so their recall is not measured: {', '.join(unmeasured)}")
    low = [f"{n} ({pc['recall']:.2f})" for n, pc in proof["per_class"].items()
           if pc["recall"] is not None and pc["recall"] < TARGETS["recall"]]
    if low:
        warnings.append(f"classes with held-out recall under {TARGETS['recall']:.2f} (check how consistently the "
                        f"source labels them): {', '.join(low)}")
    settings = {"classes": classes, "parent": parent, "parent_margin": PARENT_MARGIN, "size": size,
                "resolution": resolution, "threshold": proof["threshold"], "every": sampling["every"],
                "naming": sampling["naming"], "image_ext": sampling["image_ext"],
                "presence": {n: pc["presence"] for n, pc in analysis["per_class"].items()},
                "checkpoint": ckpt.relative_to(run_dir).as_posix(), "source": str(dataset_dir),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    report = {"status": "done", "run_dir": str(run_dir), "source": str(dataset_dir),
              "knowledge_score": round(100 * proof["mAP50"], 1), "passed": all(t["pass"] for t in targets.values()),
              "targets": targets, "held_out": proof, "warnings": warnings, "checkpoint": str(ckpt),
              "training": json.loads((run_dir / "model" / "trained.json").read_text()),
              "prepare": {k: prep[k] for k in ("parent", "held_out", "boxes", "parent_coverage")} |
                         {"train_frames": len(prep["train"]), "held_out_frames": len(prep["valid"]),
                          "parent_missing": len(prep["parent_missing"])},
              "analysis": {k: v for k, v in analysis.items() if k != "warnings"}, "settings": settings}
    _write_json(run_dir / "settings.json", settings)
    _write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(report_md(report), encoding="utf-8")
    say(4, 4, "Done")
    return report


def report_md(r: dict) -> str:
    """The report as a page a person reads: verdict first, then per class, stress test and warnings."""
    p, a, s, ho = r["prepare"], r["analysis"], r["settings"], r["held_out"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    out = [f"# Teach report: {Path(r['source']).name}", "",
           f"**Knowledge score: {r['knowledge_score']:.1f} / 100** (held-out mAP50: agreement with the source "
           f"labels on {ho['frames']} frames the model never saw). **{'PASS' if r['passed'] else 'FAIL'}** "
           "against the targets.", "",
           "| Target | Value | Required | Result |", "|---|---|---|---|"]
    out += [f"| {k} | {fmt(t['value'])} | {t['target']} | {'pass' if t['pass'] else 'FAIL'} |" for k, t in r["targets"].items()]
    out += ["", f"mAP50-95 {ho['mAP50_95']:.3f} · confidence threshold {ho['threshold']:.2f} (best micro F1 "
            f"{ho['f1']:.3f}) · recall {ho['recall']:.3f} · precision {ho['precision']:.3f}", "",
            "## Per class (held-out frames, at the threshold)", "",
            "| Class | Source boxes | In % of frames | Held-out labels | Recall | Precision | AP50 |",
            "|---|---|---|---|---|---|---|"]
    for name, pc in ho["per_class"].items():
        src = a["per_class"][name]
        low = pc["recall"] is not None and pc["recall"] < TARGETS["recall"]
        recall = f"**{fmt(pc['recall'])}**" if low else fmt(pc["recall"])
        out.append(f"| {name} | {src['boxes']} | {src['presence']:.0%} | {pc['labels']} | {recall} | "
                   f"{fmt(pc['precision'])} | {fmt(pc['AP50'])} |")
    out += ["", "## Colour stress test (same held-out frames)", "", "| Variant | mAP50 | Drop (points) |", "|---|---|---|"]
    out += [f"| {k} | {v['mAP50']:.3f} | {v['drop']:.1f} |" for k, v in ho["stress"].items()]
    out += ["", "## Warnings", ""] + ([f"- {w}" for w in r["warnings"]] or ["- none"])
    every = f"1 frame in {s['every']}" if s["every"] else "not a frame sequence (files keep their names)"
    out += ["", "## Source and settings", "",
            f"- Source: `{r['source']}`: {a['images']} images, {a['labeled']} labeled ({a['background']} empty = "
            f"background), {a['unlabeled']} without a label file, {a['boxes']} boxes, {len(a['classes'])} classes",
            f"- Sampling: {every}; file names `{s['naming']}{s['image_ext']}`",
            f"- Parent object: {s['parent'] or 'none (whole image)'}"
            + (f"; {p['parent_coverage']:.0%} of labeled boxes wholly inside" if p["parent_coverage"] is not None else ""),
            f"- Split: {p['train_frames']} train / {p['held_out_frames']} held-out frames, time blocks",
            f"- Model: RF-DETR {s['size']}, {s['resolution']} px, horizontal flip "
            f"{'off (left_/right_ classes)' if a['mirrored_pairs'] else 'on'}; {r['training']['minutes']} min training",
            f"- Checkpoint: `{r['checkpoint']}`", ""]
    return "\n".join(out)
