"""S25: keeping tracked outlines of small and thin parts faithful - candidates measured on held-out synthetic clips.

    python spikes/s25_edge_fidelity.py make  <folder> <clip>   # clip: A (S24's, development), B, C (held out)
    python spikes/s25_edge_fidelity.py track <folder> <clip>   # SAM 3 tracks from frame 0 only; raw masks saved
    python spikes/s25_edge_fidelity.py score <folder> <clip>   # offline: every candidate against the true outlines

Candidates on top of SAM 3's tracked mask (all from the same tracking run):
  raw       SAM 3 as today
  transfer  the frame-0 outline fitted onto SAM's mask (moments, then affine ECC); SAM's mask kept when they agree
            under 0.8 IoU or SAM's area is under 85 % of the fitted one (something covers the part) - set on clip A,
            judged on B and C
  zoom      the part outlined again by SAM 3's image model in a crop around the tracked box, enlarged to ~1008 px
  snap      SAM's mask pulled to image edges by a guided filter (radius 4 px), thresholded at 0.5
  zoom+transfer / snap+transfer  the transfer fitted onto the refined mask instead
Bar (unchanged from S24, set before looking): 3 px boundary IoU +0.05 or more, overall IoU not lower (>= raw - 0.005),
under 2 % of frames where a candidate replaces SAM's mask worse than SAM by more than 0.05 IoU. Seconds per part-frame
are measured for each candidate.
"""
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s24_shape_transfer as S24  # noqa: E402

W, H, N = 1920, 1080, 100
CLASSES = ["park_lamp", "chrome_strip", "grill", "thin_part"]
THIN = np.array([[0, 0], [110, 2], [124, 5], [110, 8], [0, 9]], np.float32)       # a long thin part, tapered end
CLIPS = {                                                       # seed, start, travel, scale change, turn, hand frames
    "A": dict(seed=7, start=(700, 460), travel=(400, -30), grow=0.15, turn=3, hand=(40, 55), recolour=70),
    "B": dict(seed=11, start=(1150, 420), travel=(-380, 25), grow=-0.10, turn=-4, hand=(62, 76), recolour=30),
    "C": dict(seed=23, start=(820, 300), travel=(60, 330), grow=0.20, turn=2, hand=(18, 31), recolour=85),
}


def pose(c, f):
    t = f / (N - 1)
    return (np.array(c["start"]) + np.array(c["travel"]) * t), 1.0 + c["grow"] * t, np.deg2rad(c["turn"] * t)


def place(c, poly, f, extra=(0, 0)):
    (ox, oy), s, a = pose(c, f)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (poly * s) @ R.T + np.array([ox + extra[0] * s, oy + extra[1] * s])


def frame(c, f, rng):
    img = np.dstack([np.linspace(55, 95, W, dtype=np.float32)[None].repeat(H, 0)] * 3) + rng.normal(0, 3, (H, W, 3))
    grill = S24.fill(S24.GRILL + np.array([1150, 640]))
    img[grill] = (70, 74, 72)
    strip = S24.fill(place(c, S24.STRIP, f, (100, 12)))
    img[strip] = (165, 165, 170)
    lamp = S24.fill(place(c, S24.LAMP, f))
    img[lamp] = (235, 235, 225) if f < c["recolour"] else (240, 160, 40)
    thin = S24.fill(place(c, THIN, f, (-10, 60)))
    img[thin] = (215, 215, 205)
    hand = np.zeros((H, W), np.uint8)
    a, b = c["hand"]
    if a <= f < b:
        (ox, oy), _, _ = pose(c, f)
        cx = int(ox - 60 + (f - a) / (b - a) * 220)
        cv2.ellipse(hand, (cx, int(oy + 25)), (55, 30), 15, 0, 360, 1, -1)
        img[hand.astype(bool)] = (200, 155, 125)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 1.2)
    seen = ~hand.astype(bool)
    return img, {1: (0, lamp & seen), 2: (1, strip & seen), 3: (2, grill & seen), 4: (3, thin & seen)}


def make(folder, clip):
    from engine.project import Project
    c, d = CLIPS[clip], folder / clip
    d.mkdir(parents=True, exist_ok=True)
    rng, truth = np.random.default_rng(c["seed"]), {}
    with av.open(str(d / "clip.mp4"), "w") as out:
        s = out.add_stream("libx264", rate=25, options={"crf": "23"})
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for f in range(N):
            img, parts = frame(c, f, rng)
            truth[f] = parts
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    p = Project.create(d / "proj", CLASSES, video=d / "clip.mp4", every=1, task="segment")
    for o, (cl, m) in truth[0].items():
        ys, xs = np.nonzero(m)
        p.put(0, o, cl, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), mask=m)
    np.savez_compressed(d / "truth.npz", **{f"{f}_{o}": m for f, parts in truth.items() for o, (_, m) in parts.items()})
    print("made", d)


def track(folder, clip):
    from engine.project import Project
    from engine.tracker import make_tracker
    d = folder / clip
    p, raw = Project(d / "proj"), {}
    t0 = time.perf_counter()
    n = make_tracker().track(p, 0, N - 1, lambda item, res: raw.update(
        {f"{item}_{o}": r[3] for o, r in res.items() if len(r) > 3 and r[3] is not None}), masks=True)
    (d / "track_s.txt").write_text(str((time.perf_counter() - t0) / max(1, n) / 4))   # seconds per part-frame
    np.savez_compressed(d / "raw.npz", **raw)
    print("tracked", clip, len(raw), "part-frames")


# ---- refinements ------------------------------------------------------------------------------
def guided(I, p, r=4, eps=1e-3):
    """He et al.'s guided filter (box filters): `p` smoothed along the edges of the grey image `I` (both 0..1)."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def snap(img, m):
    ys, xs = np.nonzero(m)
    x0, y0, x1, y1 = max(0, xs.min() - 20), max(0, ys.min() - 20), min(W, xs.max() + 21), min(H, ys.max() + 21)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = guided(g, m[y0:y1, x0:x1].astype(np.float32)) > 0.5
    return out


class Zoom:
    """The part outlined again by SAM 3's image model in an enlarged crop around its tracked box."""

    def __init__(self):
        from engine.segmenter import Segmenter
        self.seg = Segmenter()

    def __call__(self, img, m):
        from PIL import Image
        ys, xs = np.nonzero(m)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        pad = int(max(bw, bh) * 0.5) + 8
        x0, y0, x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad)
        k = 1008 / max(x1 - x0, y1 - y0)
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        self.seg.set_image(Image.fromarray(crop))
        box = [(xs.min() - x0) * k, (ys.min() - y0) * k, (xs.max() + 1 - x0) * k, (ys.max() + 1 - y0) * k]
        mk, _ = self.seg.segment(box=box)
        out = np.zeros_like(m)
        out[y0:y1, x0:x1] = cv2.resize(mk.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        return out


def transfer(key, pred, img=None):
    """S24's affine transfer with the cover test: None (keep `pred`) when the fit disagrees or something covers the part."""
    t = S24.transfer(key, pred, "affine")
    if t is None or S24.iou(t, pred) < 0.8 or pred.sum() < 0.85 * t.sum():
        return None
    return t


def score(folder, clip):
    import torch
    d = folder / clip
    truth, raw = np.load(d / "truth.npz"), np.load(d / "raw.npz")
    from engine.project import Project
    p = Project(d / "proj")
    zoom = Zoom() if torch.cuda.is_available() else None
    key = {o: truth[f"0_{o}"] for o in (1, 2, 3, 4)}
    hand = CLIPS[clip]["hand"]
    rows, cost = [], {"snap": [], "zoom": [], "transfer": []}
    for name in raw.files:
        f, o = map(int, name.split("_"))
        gt, pred = truth[name], raw[name]
        if not gt.any() or not pred.any():
            continue
        img = np.asarray(p.image(f).convert("RGB"))
        got = {"raw": pred}
        t0 = time.perf_counter(); got["snap"] = snap(img, pred); cost["snap"].append(time.perf_counter() - t0)
        if zoom:
            t0 = time.perf_counter(); got["zoom"] = zoom(img, pred); cost["zoom"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); tr = transfer(key[o], pred); cost["transfer"].append(time.perf_counter() - t0)
        got["transfer"] = tr if tr is not None else pred
        for base in ("snap", "zoom"):
            if base in got:
                tb = transfer(key[o], got[base])
                got[f"{base}+transfer"] = tb if tb is not None else got[base]
        rows.append({"f": f, "o": o, "hand": hand[0] <= f < hand[1], "replaced": {k: not np.array_equal(v, pred) for k, v in got.items()},
                     "s": {k: (S24.iou(v, gt), S24.iou(S24.band(v), S24.band(gt))) for k, v in got.items()}})
    out = {"track s/part-frame": round(float((d / "track_s.txt").read_text()), 3),
           "extra s/part-frame": {k: round(float(np.mean(v)), 3) for k, v in cost.items() if v}}
    for o in (1, 2, 3, 4):
        mine = [r for r in rows if r["o"] == o]
        res = {}
        for k in mine[0]["s"]:
            base = [r["s"]["raw"] for r in mine]
            cand = [r["s"][k] for r in mine]
            rep = [r["replaced"][k] for r in mine]
            worse = sum(c[0] < b[0] - 0.05 for c, b, x in zip(cand, base, rep) if x)
            hand_d = [c[0] - b[0] for c, b, r in zip(cand, base, mine) if r["hand"]]
            res[k] = {"iou": round(float(np.mean([c[0] for c in cand])), 3), "edge": round(float(np.mean([c[1] for c in cand])), 3),
                      "replaced": sum(rep), "worse": worse, "worst": round(float(min(c[0] - b[0] for c, b in zip(cand, base))), 3),
                      "hand": round(float(np.mean(hand_d)), 3) if hand_d else None}
        out[CLASSES[o - 1]] = res
    (d / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"make": make, "track": track, "score": score}[sys.argv[1]](Path(sys.argv[2]), sys.argv[3])
