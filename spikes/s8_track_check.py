"""S8: is a track check worth it in Transfer? On a Teach run's held-out frames: raw detections vs detections
changed along tracks, and which frames the check lists vs where the detector is wrong (against the labels).

    python spikes/s8_track_check.py RUN_DIR VALID_DIR       # VALID_DIR holds images/ and labels/ in time order
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import detector  # noqa: E402
from engine.teach import count_matches, load_targets  # noqa: E402
from engine.tracks import check, iou, link  # noqa: E402

S = json.loads((Path(sys.argv[1]) / "settings.json").read_text())
VALID = Path(sys.argv[2])
files = sorted((VALID / "images").glob("*.jpg"))
images = [Image.open(f).convert("RGB") for f in files]
targets = [load_targets(VALID / "labels" / (f.stem + ".txt"), *im.size) for im, f in zip(images, files)]
model = detector.load(S["checkpoint"], S["size"], S["resolution"])


def score(preds):
    tp, fp, fn = (sum(c.values()) for c in count_matches(preds, targets))
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"TP": tp, "FP": fp, "FN": fn, "F1": round(2 * p * r / max(p + r, 1e-9), 4)}


def runs():
    """Contiguous stretches of sampled frames (the held-out set is time blocks)."""
    nums = [int(re.search(r"_f(\d+)$", f.stem).group(1)) for f in files]
    out, cur = [], [0]
    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] == S["every"]:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    return out + [cur]


def tuples(d):
    return [(int(c), *map(float, b), float(s)) for b, c, s in zip(d.xyxy, d.class_id, d.confidence)]


def detections(rows):
    if not rows:
        return sv.Detections.empty()
    a = np.array(rows, float)
    return sv.Detections(xyxy=a[:, 1:5], class_id=a[:, 0].astype(int), confidence=a[:, 5])


def fill_gaps(frames, max_gap=2):
    """The rejected fix: interpolate a track's box into the frames where it was missing."""
    out = [list(f) for f in frames]
    for t in link(frames, max_gap):
        ks = sorted(t["obs"])
        cls = t["obs"][ks[0]][0]
        for a, b in zip(ks, ks[1:]):
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                box = np.array(t["obs"][a][1:5]) * (1 - w) + np.array(t["obs"][b][1:5]) * w
                if not any(d[0] == cls and iou(box, d[1:5]) > 0.5 for d in frames[k]):
                    out[k].append((cls, *box, min(t["obs"][a][5], t["obs"][b][5])))
    return out


raw = [d[d.confidence >= S["threshold"]] for d in detector.predict(model, images, threshold=0.01)]
filled = [None] * len(raw)
flag = set()
for idx in runs():
    frames = [tuples(raw[i]) for i in idx]
    for j, f in enumerate(fill_gaps(frames)):
        filled[idx[j]] = detections(f)
    flag |= {idx[k] for k in check(frames, S["classes"])}
err = []
for p, t in zip(raw, targets):
    tp, fp, fn = (sum(c.values()) for c in count_matches([p], [t]))
    err.append(fp + fn)
bad = {i for i, e in enumerate(err) if e}
print(json.dumps({"frames": len(files), "raw": score(raw), "gaps filled": score(filled),
                  "frames with an error": len(bad), "errors": sum(err), "listed by the check": len(flag),
                  "listed with an error": len(flag & bad), "errors in listed frames": sum(err[i] for i in flag)}, indent=1))
