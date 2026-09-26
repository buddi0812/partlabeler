"""S11: does the smart brush / smart eraser (Segmenter.smart_edit) beat the plain ones at part edges?

For labeled parts of a YOLO dataset, SAM 3 outlines each part from its box (the reference outline G). Then:
  brush:  a notch is cut out of G at its edge, and a rough stroke along the edge paints it back with a brush as
          wide as the notch, so it also spills over the edge onto the background;
  eraser: a blob leaks out of G at its edge, and a rough stroke along the edge erases it with a brush as wide
          as the blob, so it also cuts into the part.
Both are scored against G (IoU) with the plain tools (paint exactly the footprint) and the smart ones, plus how
much of the result lies outside the person's labeled box (spill).

    python spikes/s11_smart_brush.py DATASET_DIR [PARTS]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import masks as M  # noqa: E402
from engine.segmenter import Segmenter  # noqa: E402

root, want = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 60
rng = np.random.default_rng(0)
rows = []
for lab in sorted((root / "labels").glob("*.txt")):
    rows += [(root / "images" / f"{lab.stem}.jpg", l.split()) for l in lab.read_text().splitlines() if l.strip()]
rows = [rows[i] for i in rng.permutation(len(rows))]
seg, res, last = Segmenter(), {"brush": [], "eraser": []}, None


def edge_point(g):
    """A random point on g's edge, the outward normal there and g's size."""
    cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0]
    k = int(rng.integers(len(c)))
    p, a, b = c[k].astype(float), c[(k - 3) % len(c)].astype(float), c[(k + 3) % len(c)].astype(float)
    t = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    n = np.array([t[1], -t[0]])
    if g[int(np.clip(p[1] + 3 * n[1], 0, g.shape[0] - 1)), int(np.clip(p[0] + 3 * n[0], 0, g.shape[1] - 1))]:
        n = -n                                             # make it point out of the part
    return p, t, n


def outside(m, box):
    x1, y1, x2, y2 = map(int, box)
    inside = np.zeros_like(m); inside[max(0, y1):y2, max(0, x1):x2] = True
    return float((m & ~inside).sum())


for f, (c, cx, cy, w, h) in rows:
    if len(res["brush"]) >= 2 * want:
        break
    img = Image.open(f).convert("RGB")
    W, H = img.size
    box = [(float(cx) - float(w) / 2) * W, (float(cy) - float(h) / 2) * H, (float(cx) + float(w) / 2) * W, (float(cy) + float(h) / 2) * H]
    if f != last:
        seg.set_image(img); last = f
    g = M.clean(seg.segment(box=box)[0])
    gb = M.box_of(g)
    if gb is None or g.sum() < 400:
        continue
    iw = max(0, min(gb[2], box[2]) - max(gb[0], box[0])) * max(0, min(gb[3], box[3]) - max(gb[1], box[1]))
    if iw / ((gb[2] - gb[0]) * (gb[3] - gb[1]) + (box[2] - box[0]) * (box[3] - box[1]) - iw) < 0.7:
        continue                                           # SAM's outline disagrees with the label: skip
    r = max(3.0, 0.2 * min(gb[2] - gb[0], gb[3] - gb[1]))
    p, t, n = edge_point(g)
    area = float(g.sum())
    notch, leak = p - 0.5 * r * n, p + 0.6 * r * n
    jit = lambda: rng.normal(0, 0.15 * r, 2)
    strokes = {  # aimed: the centre line runs over the spot to fix; sloppy: along the edge, partly on the wrong side
        "aimed": {"brush": [notch - 1.2 * r * t + jit(), notch + jit(), notch + 1.2 * r * t + jit()],
                  "eraser": [leak - 1.2 * r * t + jit(), leak + jit(), leak + 1.2 * r * t + jit()]},
        "sloppy": {"brush": [p - 1.5 * r * t + 0.3 * r * n, p, p + 1.5 * r * t - 0.2 * r * n],
                   "eraser": [p - 1.5 * r * t + 0.3 * r * n, p, p + 1.5 * r * t - 0.2 * r * n]}}
    for kind, st in strokes.items():
        # brush: a notch cut into the part, painted back
        old = g & ~M.footprint([notch], r, g.shape)
        foot = M.footprint(st["brush"], r, g.shape)
        part_only = seg.segment(M.inner_points(old & ~foot), [1] * len(M.inner_points(old & ~foot)))[0]
        res["brush"].append({"stroke": kind, "before": M.iou(old, g), "plain": M.iou(old | foot, g),
                             "smart": M.iou(seg.smart_edit(old, st["brush"], r), g),
                             "part_only": M.iou(M.clean(old | (foot & part_only)), g),
                             "spill_plain": outside(old | foot, box) / area,
                             "spill_smart": outside(seg.smart_edit(old, st["brush"], r), box) / area})
        # eraser: a blob leaking out of the part, erased
        old = g | M.footprint([leak], r, g.shape)
        foot = M.footprint(st["eraser"], r, g.shape)
        ins = M.inner_points(old & ~foot)
        part_only = seg.segment(ins, [1] * len(ins))[0]
        res["eraser"].append({"stroke": kind, "before": M.iou(old, g), "plain": M.iou(old & ~foot, g),
                              "smart": M.iou(seg.smart_edit(old, st["eraser"], r, erase=True), g),
                              "part_only": M.iou(M.clean(old & ~(foot & ~part_only)), g)})

out = {}
for tool, rs in res.items():
    for kind in ("aimed", "sloppy"):
        sub = [x for x in rs if x["stroke"] == kind]
        s_ = {k: round(float(np.mean([x[k] for x in sub])), 3) for k in sub[0] if k != "stroke"}
        s_["parts"] = len(sub)
        s_["smart_better"] = round(float(np.mean([x["smart"] > x["plain"] + 0.005 for x in sub])), 2)
        s_["smart_worse"] = round(float(np.mean([x["smart"] < x["plain"] - 0.005 for x in sub])), 2)
        out[f"{tool}_{kind}"] = s_
print(json.dumps(out, indent=1))
Path(__file__).with_name("out").mkdir(exist_ok=True)
(Path(__file__).with_name("out") / "s11_smart_brush.json").write_text(json.dumps({"summary": out, "parts": res}, indent=1))
