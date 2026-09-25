"""S7 step 3: score the trained detector.

1. Agreement with the booth YOLO on held-out White frames (mAP50, mAP50-95, per-class
   recall/precision). The labels come from that YOLO and have gaps, so this is agreement, not accuracy.
2. Colour stress test on the same frames: grayscale, and the area outside the labeled parts
   repainted dark ("black body") or dark green ("green body").
3. Transfer to the station-2 videos (no labels): per-class detection rates versus the white car,
   plus contact sheets for visual checking.

    python spikes/s7_eval.py --ckpt data/example/s7_runs/baseline/checkpoint_best_total.pth
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageOps
from rfdetr import RFDETRSmall
from supervision.metrics import MeanAveragePrecision

from engine.parent import ParentFinder, crop_box

ROOT = Path("data/example")
OUT = Path("spikes/out/s7")
TARGETS = ["black_car_front.mp4", "green_car_front.mp4"]


def load_targets(img: Image.Image, label: Path) -> sv.Detections:
    rows = [list(map(float, l.split())) for l in label.read_text().splitlines() if l.strip()]
    if not rows:
        return sv.Detections.empty()
    a = np.array(rows)
    W, H = img.size
    xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                     (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H], axis=1)
    return sv.Detections(xyxy=xyxy, class_id=a[:, 0].astype(int))


def repaint_outside(img: Image.Image, det: sv.Detections, tint) -> Image.Image:
    """Repaint everything outside the labeled parts: keep shading, apply a dark body colour."""
    a = np.asarray(img).astype(np.float32)
    keep = np.zeros(a.shape[:2], bool)
    for x1, y1, x2, y2 in det.xyxy.astype(int):
        keep[max(0, y1):y2, max(0, x1):x2] = True
    lum = a.mean(axis=2, keepdims=True) / 255.0
    painted = lum * np.array(tint, np.float32) * 255.0
    a[~keep] = painted[~keep]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def recall_precision(preds, targets, names, iou=0.5):
    tp, fp, fn = Counter(), Counter(), Counter()
    for p, t in zip(preds, targets):
        for c in set(p.class_id.tolist()) | set(t.class_id.tolist()):
            pc, tc = p[p.class_id == c], t[t.class_id == c]
            if len(pc) == 0 or len(tc) == 0:
                fp[c] += len(pc); fn[c] += len(tc); continue
            m = sv.box_iou_batch(pc.xyxy, tc.xyxy)
            matched_t, hits = set(), 0
            for i in np.argsort(-pc.confidence):
                j = int(np.argmax(np.where([k in matched_t for k in range(len(tc))], -1, m[i])))
                if m[i, j] >= iou and j not in matched_t:
                    matched_t.add(j); hits += 1
            tp[c] += hits; fp[c] += len(pc) - hits; fn[c] += len(tc) - hits
    return {names[c]: {"recall": round(tp[c] / max(tp[c] + fn[c], 1), 3),
                       "precision": round(tp[c] / max(tp[c] + fp[c], 1), 3),
                       "labels": tp[c] + fn[c]} for c in sorted(set(tp) | set(fn) | set(fp))}


def predict_all(model, images, threshold, batch=8):
    out = []
    for i in range(0, len(images), batch):
        r = model.predict(images[i:i + batch], threshold=threshold)
        out.extend(r if isinstance(r, list) else [r])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--frames", type=int, default=40, help="frames sampled per target video")
    args = ap.parse_args()
    names = (ROOT / "dataset/classes.txt").read_text().split()
    OUT.mkdir(parents=True, exist_ok=True)
    model = RFDETRSmall(pretrain_weights=args.ckpt, resolution=args.resolution)

    # 1. held-out agreement
    files = sorted((ROOT / "s7_ds/valid/images").glob("*.jpg"))
    images = [Image.open(f).convert("RGB") for f in files]
    targets = [load_targets(im, ROOT / "s7_ds/valid/labels" / (f.stem + ".txt")) for im, f in zip(images, files)]
    preds = predict_all(model, images, threshold=0.01)          # low threshold for mAP
    res = {"checkpoint": args.ckpt, "held_out_frames": len(files)}
    m = MeanAveragePrecision().update(preds, targets).compute()
    res["held_out"] = {"mAP50": round(float(m.map50), 3), "mAP50_95": round(float(m.map50_95), 3)}
    # confidence cut-off that best reproduces the source labels (micro F1 over all classes)
    sweep = {}
    for th in np.arange(0.2, 0.75, 0.05):
        pc = recall_precision([p[p.confidence >= th] for p in preds], targets, names)
        tp = sum(v["recall"] * v["labels"] for v in pc.values())
        n_pred = sum(int((p.confidence >= th).sum()) for p in preds)
        n_lab = sum(v["labels"] for v in pc.values())
        sweep[round(float(th), 2)] = round(2 * tp / max(n_pred + n_lab, 1), 3)
    args.threshold = max(sweep, key=sweep.get)
    res["threshold_f1"] = sweep
    res["threshold"] = args.threshold
    confident = [p[p.confidence >= args.threshold] for p in preds]
    res["per_class"] = recall_precision(confident, targets, names)

    # 2. colour stress on the same frames
    variants = {"grayscale": lambda im, t: ImageOps.grayscale(im).convert("RGB"),
                "black_body": lambda im, t: repaint_outside(im, t, (0.25, 0.25, 0.27)),
                "green_body": lambda im, t: repaint_outside(im, t, (0.12, 0.30, 0.24))}
    res["colour_stress"] = {}
    for name, fn in variants.items():
        vimgs = [fn(im, t) for im, t in zip(images, targets)]
        vm = MeanAveragePrecision().update(predict_all(model, vimgs, 0.01), targets).compute()
        res["colour_stress"][name] = {"mAP50": round(float(vm.map50), 3),
                                      "drop": round(float(m.map50 - vm.map50), 3)}
        if name != "grayscale":
            vimgs[len(vimgs) // 2].save(OUT / f"stress_{name}.jpg", quality=85)

    # 3. transfer to station-2 videos
    white_rate = {names[c]: round(sum(int((t.class_id == c).any()) for t in targets) / len(targets), 2)
                  for c in sorted({int(c) for t in targets for c in t.class_id})}
    res["white_label_presence"] = white_rate
    finder = ParentFinder("car")
    res["transfer"] = {}
    for vid in TARGETS:
        with av.open(str(ROOT / "videos" / vid)) as c:
            n = c.streams.video[0].frames
            want = set(np.linspace(0, n - 1, args.frames).astype(int).tolist())
            frames = [(i, fr.to_image()) for i, fr in enumerate(c.decode(video=0)) if i in want]
        crops = []
        for i, im in frames:
            box = finder.find(im)
            crops.append(im.crop(crop_box(box, *im.size) if box else (0, 0, *im.size)))
        dets = [p[p.confidence >= args.threshold] for p in predict_all(model, crops, 0.01)]
        rate = {names[c]: round(sum(int((d.class_id == c).any()) for d in dets) / len(dets), 2)
                for c in sorted({int(c) for d in dets for c in d.class_id})}
        res["transfer"][vid] = {"frames": len(dets), "boxes_per_frame": round(float(np.mean([len(d) for d in dets])), 1),
                                "presence": rate}
        sheet = Image.new("RGB", (640 * 3, 400 * 2))
        for k, idx in enumerate(np.linspace(0, len(crops) - 1, 6).astype(int)):
            im = crops[idx].copy(); d = ImageDraw.Draw(im)
            for (x1, y1, x2, y2), c, s in zip(dets[idx].xyxy, dets[idx].class_id, dets[idx].confidence):
                d.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)
                d.text((x1, max(0, y1 - 12)), f"{names[c]} {s:.2f}", fill=(255, 255, 0))
            sheet.paste(im.resize((640, 400)), ((k % 3) * 640, (k // 3) * 400))
        sheet.save(OUT / f"transfer_{vid.split('_')[2]}.jpg", quality=85)

    res["white_boxes_per_frame_labels"] = round(float(np.mean([len(t) for t in targets])), 1)
    print(json.dumps(res, indent=1))
    (OUT / f"eval_{Path(args.ckpt).parent.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
