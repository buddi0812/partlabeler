"""M1 check: how closely does "track ahead" from one labeled frame reproduce the source labels?

Creates a project for the White video (every 5th frame), puts only the start frame's source
labels in as manual boxes, tracks ahead, and compares each tracked box with the source label of the
same class on that frame (best IoU).

    python spikes/m1_track_check.py --start 0 --count 60
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from engine.project import Project, read_classes
from engine.tracker import Tracker

ROOT = Path("data/example")
PROJ = Path("data/projects/example_white")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def truth(frame: int, W: int, H: int):
    f = ROOT / "dataset/labels" / f"video_A_cam_f{frame:06d}.txt"
    out = []
    for line in f.read_text().splitlines() if f.exists() else []:
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append((int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=60)
    args = ap.parse_args()
    classes = read_classes(ROOT / "dataset/classes.txt")
    t = time.perf_counter()
    p = Project(PROJ) if (PROJ / "project.json").exists() else \
        Project.create(PROJ, classes, video=ROOT / "videos/white_car_front.mp4", every=5)
    print(f"project ready: {len(p.items)} items ({time.perf_counter() - t:.0f} s)", flush=True)
    for b in p.boxes(args.start):
        p.delete(args.start, b["obj"])
    W, H = Image.open(p.items[args.start]["path"]).size
    obj = p.new_obj()
    for k, (c, box) in enumerate(truth(p.items[args.start]["frame"], W, H)):
        p.put(args.start, obj + k, c, box, "manual")

    got = {}
    t = time.perf_counter()
    n = Tracker().track(p, args.start, args.count, on_item=lambda item, res: got.__setitem__(item, res))
    secs = time.perf_counter() - t

    ious, matched, tracked, labels = [], 0, 0, 0
    for item, res in got.items():
        gt = truth(p.items[item]["frame"], W, H)
        labels += len(gt)
        for cls, box, _ in res.values():
            if box is None:
                continue
            tracked += 1
            best = max((iou(box, g) for c, g in gt if c == cls), default=0.0)
            ious.append(best)
            matched += best >= 0.5
    out = {"start_item": args.start, "items_tracked": n, "seconds": round(secs, 1),
           "items_per_second": round(n / secs, 2), "anchor_boxes": len(truth(p.items[args.start]["frame"], W, H)),
           "tracked_boxes": tracked, "source_labels_on_those_items": labels,
           "tracked_boxes_matching_a_label_iou50": round(matched / max(tracked, 1), 3),
           "labels_covered": round(matched / max(labels, 1), 3),
           "median_iou": round(sorted(ious)[len(ious) // 2], 3) if ious else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
