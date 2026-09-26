"""S10: smart sorting on real part crops. Every labeled box of a YOLO dataset is cut out (a square around the part
with some context), embedded with DINOv3 and then:
  - grouped with engine.sort.Groups: does the automatic number of groups and its purity match the true classes?
    (NMI / ARI; left_/right_ twins are mirror images, so scores are also given with each pair merged)
  - classified from a few examples per class (engine.sort.predict), 5 random draws each
  - checked for wrong labels: 3% of labels flipped at random, how many does engine.sort.odd_ones find?

    python spikes/s10_sorting.py DATASET_DIR [MAX_CROPS]      # DATASET_DIR: images/, labels/, classes.txt or data.yaml
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.embed import Embedder  # noqa: E402
from engine.sort import Groups, odd_ones, predict  # noqa: E402
from engine.teach import classes_of  # noqa: E402

root = Path(sys.argv[1])
max_crops = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
classes = classes_of(root)
twin = [re.sub(r"^(left|right)_", "", c) for c in classes]
merged_id = {n: i for i, n in enumerate(dict.fromkeys(twin))}

crops, truth = [], []
files = sorted((root / "images").glob("*.jpg"))
rng = np.random.default_rng(0)
rows = []
for f in files:
    lab = root / "labels" / f"{f.stem}.txt"
    if lab.exists():
        rows += [(f, line.split()) for line in lab.read_text().splitlines() if line.strip()]
rows = [rows[i] for i in sorted(rng.choice(len(rows), min(max_crops, len(rows)), replace=False))]
cache = {}
for f, (c, cx, cy, w, h) in rows:
    img = cache.get(f) or cache.setdefault(f, Image.open(f).convert("RGB"))
    if len(cache) > 8:
        cache.pop(next(iter(cache)))
    W, H = img.size
    cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
    side = max(w, h) * 1.2
    crops.append(img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))))
    truth.append(int(c))
truth = np.array(truth)
merged = np.array([merged_id[twin[c]] for c in truth])
print(f"{len(crops)} crops, {len(set(truth))} classes ({len(set(merged))} with twins merged)", flush=True)


def scores(labels):
    return {"nmi": round(normalized_mutual_info_score(truth, labels), 3), "ari": round(adjusted_rand_score(truth, labels), 3),
            "nmi_twins_merged": round(normalized_mutual_info_score(merged, labels), 3),
            "ari_twins_merged": round(adjusted_rand_score(merged, labels), 3)}


out = {}
for size in ("small", "base"):
    emb = Embedder(size)
    for pool in ("cls", "avg", "both"):
        t0 = time.perf_counter()
        v = emb.vectors(crops, pool=pool)
        secs = time.perf_counter() - t0
        g = Groups(v)
        auto = g.cut()
        lab = np.empty(len(v), int)
        for gi, items in enumerate(auto["groups"]):
            lab[items] = gi
        at_true = g.labels(len(set(merged)))
        res = {"crops_per_s": round(len(v) / secs), "auto_k": auto["k"], "auto": scores(lab),
               "unsure_share": round(len(auto["unsure"]) / len(v), 3), "at_true_k_twins_merged": scores(at_true)}
        for shots in (1, 3, 5, 10):
            accs, accs_m = [], []
            for seed in range(5):
                r = np.random.default_rng(seed)
                labeled = {}
                for c in set(truth):
                    idx = np.flatnonzero(truth == c)
                    for i in r.choice(idx, min(shots, len(idx)), replace=False):
                        labeled[int(i)] = int(c)
                cls = predict(v, labeled)[0]
                rest = np.array([i for i in range(len(v)) if i not in labeled])
                accs.append(float((cls[rest] == truth[rest]).mean()))
                accs_m.append(float((np.array([merged_id[twin[c]] for c in cls[rest]]) == merged[rest]).mean()))
            res[f"acc_{shots}_shot"] = round(float(np.mean(accs)), 3)
            res[f"acc_{shots}_shot_twins_merged"] = round(float(np.mean(accs_m)), 3)
        r = np.random.default_rng(1)
        noisy = {i: int(merged[i]) for i in range(len(v))}
        flip = r.choice(len(v), max(1, len(v) * 3 // 100), replace=False)
        for i in flip:
            noisy[int(i)] = int((merged[i] + 1 + r.integers(len(set(merged)) - 1)) % len(set(merged)))
        found = {o[0] for o in odd_ones(v, noisy)}
        hit = len(found & set(flip.tolist()))
        res["odd_ones"] = {"flipped": len(flip), "flagged": len(found), "caught": hit,
                           "recall": round(hit / len(flip), 3), "precision": round(hit / max(len(found), 1), 3)}
        out[f"{size}_{pool}"] = res
        print(f"{size}_{pool}", json.dumps(res), flush=True)
    del emb
Path(__file__).with_name("out").mkdir(exist_ok=True)
(Path(__file__).with_name("out") / "s10_sorting.json").write_text(json.dumps(out, indent=1))
