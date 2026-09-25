"""S9: how good is quick transfer (no training)? Examples from a Teach run's training split, matched in its
held-out frames by the annotator's Suggest (DINOv3), scored against the held-out labels at IoU 0.5.

    python spikes/s9_quick_transfer.py DATASET_DIR [REFS] [--refine]   # DATASET_DIR holds train/ and valid/ (YOLO)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.api import model  # noqa: E402
from engine.quick import SourceView  # noqa: E402
from engine.teach import classes_of, count_matches, load_targets  # noqa: E402

root, refs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
classes = classes_of(root) if (root / "data.yaml").exists() or (root / "classes.txt").exists() else classes_of(root / "train")
view = SourceView(root / "train", classes, refs=refs)
files = sorted((root / "valid" / "images").glob("*.jpg"))
refine = "--refine" in sys.argv
sug, preds, targets, t0 = model("suggest"), [], [], time.perf_counter()
for f in files:
    img = Image.open(f).convert("RGB")
    found = sug.suggest(view, view.add(img), max_refs=refs)
    if refine and found:
        from engine.quick import tighten
        found = tighten(model("seg"), img, found)
    a = np.array([[*box, c, s] for c, box, s in found], float).reshape(-1, 6)
    preds.append(sv.Detections(xyxy=a[:, :4], class_id=a[:, 4].astype(int), confidence=a[:, 5]))
    targets.append(load_targets(root / "valid" / "labels" / (f.stem + ".txt"), *img.size))
loose = [sum(c.values()) for c in count_matches(preds, targets, iou=0.3)]
tp, fp, fn = count_matches(preds, targets)
TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
p, r = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
lr = [c for c in range(len(classes)) if classes[c].startswith(("left_", "right_"))]
print(json.dumps({"frames": len(files), "examples": len(view.refs), "seconds_per_frame": round((time.perf_counter() - t0) / len(files), 2),
                  "TP": TP, "FP": FP, "FN": FN, "precision": round(p, 3), "recall": round(r, 3),
                  "F1": round(2 * p * r / max(p + r, 1e-9), 3),
                  "F1 at IoU 0.3": round(2 * loose[0] / max(2 * loose[0] + loose[1] + loose[2], 1), 3),
                  "left/right classes recall": round(sum(tp[c] for c in lr) / max(sum(tp[c] + fn[c] for c in lr), 1), 3)}, indent=1))
