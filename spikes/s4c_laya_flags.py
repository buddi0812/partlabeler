"""S4c: can Laya (text/JSON decision model) spot tracked boxes that went wrong, from numbers alone?

Seeds a throwaway project with the source labels on several frames, tracks each 30 frames ahead,
and records per tracked box: movement and size change since the previous frame, size change since
the person's box, tracker confidence, frames since the person's box. Truth: the tracked box matches
a source label of its class (IoU >= 0.5). Frames where the source has no label of that class are left
out (the source YOLO has gaps). Compared: the app's current rule, each number alone, and Laya
answering one yes/no question over the numbers as JSON (zero-shot).

    python spikes/s4c_laya_flags.py
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from engine.project import Project, read_classes
from engine.tracker import Tracker

ROOT = Path("data/example")
PROJ = Path("data/projects/_flagtest")
STARTS, AHEAD = [100, 200, 300, 400, 500, 600], 30


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy + 1e-9)


def truth(frame, W, H):
    f = ROOT / "dataset/labels" / f"video_A_cam_f{frame:06d}.txt"
    out = []
    for line in f.read_text().splitlines() if f.exists() else []:
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append((int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def collect():
    names = read_classes(ROOT / "dataset/classes.txt")
    p = Project(PROJ) if (PROJ / "project.json").exists() else \
        Project.create(PROJ, names, video=ROOT / "videos/white_car_front.mp4", every=5)
    W, H = Image.open(p.items[0]["path"]).size
    tracker, rows = Tracker(), []
    for start in STARTS:
        for b in p.boxes(start):
            p.delete(start, b["obj"])
        obj0 = p.new_obj()
        anchor = {}
        for k, (c, box) in enumerate(truth(p.items[start]["frame"], W, H)):
            p.put(start, obj0 + k, c, box, "manual")
            anchor[obj0 + k] = box
        prev = dict(anchor)

        def on_item(item, res, start=start):
            gt = truth(p.items[item]["frame"], W, H)
            for obj, (cls, box, score) in res.items():
                if box is None:
                    continue
                same = [g for c, g in gt if c == cls]
                pb, ab = prev.get(obj, box), anchor[obj]
                diag = math.hypot(pb[2] - pb[0], pb[3] - pb[1]) or 1.0
                moved = math.hypot((box[0] + box[2] - pb[0] - pb[2]) / 2, (box[1] + box[3] - pb[1] - pb[3]) / 2) / diag
                area = (box[2] - box[0]) * (box[3] - box[1])
                rows.append({"part": names[cls], "frames_since_person_box": item - start,
                             "tracker_confidence": round(score or 0.0, 3),
                             "moved_since_previous_frame_in_box_diagonals": round(moved, 3),
                             "size_ratio_vs_previous_frame": round(area / max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1])), 3),
                             "size_ratio_vs_person_box": round(area / max(1.0, (ab[2] - ab[0]) * (ab[3] - ab[1])), 3),
                             "has_source_label": bool(same),
                             "wrong": (max(iou(box, g) for g in same) < 0.5) if same else None})
                prev[obj] = box

        tracker.track(p, start, AHEAD, on_item)
    return rows


def main():
    cache = Path("spikes/out/s4c_rows.json")
    rows = json.loads(cache.read_text()) if cache.exists() else collect()
    cache.parent.mkdir(exist_ok=True); cache.write_text(json.dumps(rows))
    ev = [r for r in rows if r["wrong"] is not None]
    y = np.array([r["wrong"] for r in ev], int)
    out = {"tracked_boxes": len(rows), "with_source_label": len(ev), "wrong": int(y.sum())}

    rule = np.array([r["moved_since_previous_frame_in_box_diagonals"] > 0.5 or
                     not 0.5 <= r["size_ratio_vs_previous_frame"] <= 2.0 for r in ev], int)
    tp = int((rule & y).sum())
    out["app_rule"] = {"flagged": int(rule.sum()), "precision": round(tp / max(rule.sum(), 1), 3),
                       "recall": round(tp / max(y.sum(), 1), 3)}
    single = {"1 - tracker_confidence": [1 - r["tracker_confidence"] for r in ev],
              "moved": [r["moved_since_previous_frame_in_box_diagonals"] for r in ev],
              "|log size vs previous|": [abs(math.log(max(r["size_ratio_vs_previous_frame"], 1e-3))) for r in ev],
              "|log size vs person box|": [abs(math.log(max(r["size_ratio_vs_person_box"], 1e-3))) for r in ev],
              "frames since person box": [r["frames_since_person_box"] for r in ev]}
    out["auroc_single_number"] = {k: round(roc_auc_score(y, v), 3) for k, v in single.items()}

    from laya import Router
    router = Router(preload=True)
    q = {"wrong": {"type": "noul", "instructions":
                   "This bounding box was placed automatically by a video tracker following a machine part "
                   "that a person boxed earlier. Has the box probably drifted off the part or become the wrong "
                   "size, so a person should check it?"}}
    keys = ["part", "frames_since_person_box", "tracker_confidence", "moved_since_previous_frame_in_box_diagonals",
            "size_ratio_vs_previous_frame", "size_ratio_vs_person_box"]
    router.predict({k: ev[0][k] for k in keys}, q)                      # warm-up
    t = time.perf_counter()
    probs = [float(router.predict({k: r[k] for k in keys}, q)["answers"]["wrong"]["noul"]) for r in ev]
    out["laya"] = {"auroc": round(roc_auc_score(y, probs), 3), "ms_per_box": round((time.perf_counter() - t) / len(ev) * 1000, 1),
                   "flagged_at_0.5": int(sum(p >= 0.5 for p in probs)),
                   "precision_at_0.5": round(sum(p >= 0.5 and w for p, w in zip(probs, y)) / max(1, sum(p >= 0.5 for p in probs)), 3),
                   "recall_at_0.5": round(sum(p >= 0.5 and w for p, w in zip(probs, y)) / max(1, int(y.sum())), 3)}
    print(json.dumps(out, indent=1))
    Path("spikes/out/s4c_laya_flags.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
