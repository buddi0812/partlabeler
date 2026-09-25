"""S3 + S6: find parts in new images from a few labeled examples, with no training.

30 random labeled White frames act as an unordered image folder: 5 are references, the
other 25 are predicted. Per part group, DINOv3 features of the reference boxes are matched
against each new image; similarity peaks above a threshold calibrated on the references become
detections, sized either by the reference boxes or by a SAM 3 mask from a click at the peak.
Left/right twins share one appearance model and get their side from position (the car's right
is on the image's left). Scored against the booth-YOLO labels.

    python spikes/s3_discovery.py [--refs 5] [--images 30]
"""
import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from engine.embed import Embedder
from engine.parent import ParentFinder, crop_box
from engine.segmenter import Segmenter
from s7_prepare import to_crop

ROOT = Path("data/example")
OUT = Path("spikes/out/s3")
PAIRS = {"drl": ("left_drl", "right_drl"), "head_lamp": ("left_head_lamp", "right_head_lamp"),
         "mirror_lamp": ("left_mirror_lamp", "right_mirror_lamp")}
MULTI = {"roof_rack": 4, "roof_granish": 4}


def group_of(name: str) -> str:
    return next((g for g, pair in PAIRS.items() if name in pair), name)


def max_count(group: str) -> int:
    return 2 if group in PAIRS else MULTI.get(group, 1)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def box_vector(feats, scale, box):
    """Mean feature of the patches whose centres fall inside `box` (crop pixels)."""
    p = Embedder.patch
    x1, y1, x2, y2 = box[0] * scale[0] / p, box[1] * scale[1] / p, box[2] * scale[0] / p, box[3] * scale[1] / p
    r0, r1 = int(np.floor(y1 - 0.5)) + 1, int(np.ceil(y2 - 0.5))
    c0, c1 = int(np.floor(x1 - 0.5)) + 1, int(np.ceil(x2 - 0.5))
    rows, cols = feats.shape[:2]
    if r1 <= r0 or c1 <= c0:                          # tiny box: the patch holding its centre
        r0 = min(rows - 1, int((y1 + y2) / 2)); c0 = min(cols - 1, int((x1 + x2) / 2)); r1, c1 = r0 + 1, c0 + 1
    v = feats[max(0, r0):min(rows, r1), max(0, c0):min(cols, c1)].reshape(-1, feats.shape[-1]).mean(0)
    return F.normalize(v, dim=0)


def score_map(feats, vectors):
    return (feats @ torch.stack(vectors).T).amax(-1)            # (rows, cols)


def peaks(smap, threshold, k, radius):
    m = smap[None, None]
    is_max = (F.max_pool2d(m, 3, 1, 1) == m)[0, 0] & (smap >= threshold)
    rc = torch.nonzero(is_max)
    order = torch.argsort(smap[is_max], descending=True)
    kept = []
    for r, c in rc[order].tolist():
        if all(max(abs(r - a), abs(c - b)) > radius for a, b, _ in kept):
            kept.append((r, c, float(smap[r, c])))
        if len(kept) == k:
            break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=30)
    ap.add_argument("--refs", type=int, default=5)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = (ROOT / "dataset/classes.txt").read_text().split()
    rng = random.Random(args.seed)
    files = sorted(rng.sample(sorted((ROOT / "dataset/labels").glob("*.txt")), args.images))
    order = files[:]; rng.shuffle(order)
    refs, queries = order[:args.refs], order[args.refs:]

    finder, emb, seg = ParentFinder("car"), Embedder("small"), Segmenter()
    data = {}
    for f in files:
        img = Image.open(ROOT / "dataset/images" / (f.stem + ".jpg")).convert("RGB")
        box = finder.find(img)
        crop = crop_box(box, *img.size) if box else (0, 0, *img.size)
        cimg = img.crop(crop)
        gt = []
        for line in f.read_text().splitlines():
            r = to_crop(line, crop, *img.size)
            if r:
                c, cx, cy, w, h = r.split(); c = int(c)
                W, H = cimg.size; cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                gt.append((names[c], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        feats, scale = emb.patches(cimg, args.width)
        data[f.stem] = {"img": cimg, "gt": gt, "feats": feats, "scale": scale}

    # appearance model per group from the reference frames
    vecs, sizes, per_ref = defaultdict(list), defaultdict(list), defaultdict(dict)
    for f in refs:
        d = data[f.stem]
        for name, box in d["gt"]:
            g = group_of(name)
            v = box_vector(d["feats"], d["scale"], box)
            vecs[g].append(v); per_ref[g].setdefault(f.stem, []).append(v)
            sizes[g].append((box[2] - box[0], box[3] - box[1]))
    # threshold per group: leave-one-reference-out scores at labeled boxes vs. best background score
    thresholds = {}
    for g in vecs:
        pos, neg = [], []
        for f in refs:
            others = [v for s, vs in per_ref[g].items() if s != f.stem for v in vs]
            if not others:
                continue
            d = data[f.stem]; smap = score_map(d["feats"], others)
            inside = torch.zeros_like(smap, dtype=torch.bool)
            for name, box in d["gt"]:
                if group_of(name) == g:
                    pos.append(float((box_vector(d["feats"], d["scale"], box) @ torch.stack(others).T).max()))
                    sx, sy, p = d["scale"][0], d["scale"][1], Embedder.patch
                    inside[max(0, int(box[1] * sy / p) - 1):int(box[3] * sy / p) + 2,
                           max(0, int(box[0] * sx / p) - 1):int(box[2] * sx / p) + 2] = True
            neg.append(float(smap[~inside].max()))
        if pos and neg:
            thresholds[g] = max(statistics.median(neg), 0.5 * (statistics.median(pos) + statistics.median(neg)))
        elif pos:
            thresholds[g] = 0.9 * min(pos)
        else:                                          # one example only: fall back to a fixed bar
            thresholds[g] = 0.6

    # predict the query frames
    results = {"median_box": [], "sam_box": []}
    t0 = time.perf_counter()
    for f in queries:
        d = data[f.stem]; W = d["img"].size[0]
        seg.set_image(d["img"])
        preds = {"median_box": [], "sam_box": []}
        for g, vs in vecs.items():
            smap = score_map(d["feats"], vs)
            mw = statistics.median(s[0] for s in sizes[g]); mh = statistics.median(s[1] for s in sizes[g])
            radius = max(1, int(0.5 * max(mw * d["scale"][0], mh * d["scale"][1]) / Embedder.patch))
            for r, c, s in peaks(smap, thresholds[g], max_count(g), radius):
                cx = (c + 0.5) * Embedder.patch / d["scale"][0]; cy = (r + 0.5) * Embedder.patch / d["scale"][1]
                name = (PAIRS[g][1] if cx < W / 2 else PAIRS[g][0]) if g in PAIRS else g
                mbox = (cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2)
                preds["median_box"].append((name, mbox, s))
                best, best_err = mbox, np.log(4)
                for mask, _ in seg.candidates([[cx, cy]], [1]):
                    ys, xs = np.nonzero(mask)
                    if len(xs):
                        b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                        err = abs(np.log(((b[2] - b[0]) * (b[3] - b[1]) + 1) / (mw * mh + 1)))
                        if err < best_err:
                            best, best_err = b, err
                preds["sam_box"].append((name, tuple(float(v) for v in best), s))
        for k in results:
            results[k].append((d["gt"], preds[k]))
    per_img = (time.perf_counter() - t0) / len(queries)

    def score(pairs, thr):
        tp, fn, fp, per = 0, 0, 0, defaultdict(lambda: [0, 0, 0])
        for gt, pr in pairs:
            used = set()
            for name, gbox in gt:
                cands = [(iou(gbox, b), j) for j, (n, b, _) in enumerate(pr) if n == name and j not in used]
                hit = max(cands, default=(0, -1))
                if hit[0] >= thr:
                    used.add(hit[1]); tp += 1; per[name][0] += 1
                else:
                    fn += 1; per[name][1] += 1
            for j, (n, _, _) in enumerate(pr):
                if j not in used:
                    fp += 1; per[n][2] += 1
        return tp / max(tp + fn, 1), tp / max(tp + fp, 1), per

    out = {"images": len(files), "references": len(refs), "queries": len(queries),
           "seconds_per_query": round(per_img, 2),
           "groups_with_examples": sorted(vecs), "groups_without_examples":
               sorted({group_of(n) for f in queries for n, _ in data[f.stem]["gt"]} - set(vecs))}
    for k in results:
        for thr in (0.5, 0.3):
            rec, prec, per = score(results[k], thr)
            out[f"{k}@iou{thr}"] = {"recall": round(rec, 3), "precision": round(prec, 3)}
        out[f"{k}_per_class@0.3"] = {n: {"found": v[0], "missed": v[1], "extra": v[2]} for n, v in sorted(per.items())}
    rec_seen = score([(gt_, pr) for gt_, pr in [((([(n, b) for n, b in gt if group_of(n) in vecs])), pr)
                      for gt, pr in results["sam_box"]]], 0.3)[0]
    out["sam_box_recall@iou0.3_on_parts_with_examples"] = round(rec_seen, 3)
    print(json.dumps(out, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "discovery.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
