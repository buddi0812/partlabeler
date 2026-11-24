"""S24: does carrying the drawn outline (shape transfer) keep tracked outlines truer than SAM 3's own masks?

    python spikes/s24_shape_transfer.py make  <folder>     # a 1080p synthetic clip + outline project (true outlines known)
    python spikes/s24_shape_transfer.py track <folder>     # SAM 3 tracks from frame 0 only; raw masks saved (GPU, once)
    python spikes/s24_shape_transfer.py score <folder>     # offline: raw SAM vs the drawn outline fitted onto it

SAM 3 predicts masks on a 288 x 288 grid (1008 px input): on 1920 x 1080 a cell is ~6.7 x 3.8 px, so small parts lose
notches and corners. The clip: a small lamp-like part (~95 x 38 px, sharp corners, a 12 px notch) beside a look-alike
chrome strip, moving right and growing 15 %, soft compressed edges; a hand sweeps over it, then it changes colour; a big
simple part for comparison. Only frame 0's outline is given (drawn = true). Shape transfer fits the frame-0 outline
onto SAM's mask (translation / similarity / affine) and keeps SAM's mask when the two disagree (fallback).
Bar (set before looking): boundary IoU (3 px) +0.05 or more, overall IoU not lower (>= raw - 0.005), and under 2 % of
frames where the transfer is used worse than raw SAM by more than 0.05 IoU.
"""
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

W, H, N = 1920, 1080, 100
CLASSES = ["park_lamp", "chrome_strip", "grill"]
LAMP = np.array([[0, 6], [60, 0], [95, 10], [95, 38], [48, 38], [48, 26], [36, 26], [36, 38], [4, 38]], np.float32)  # notch
STRIP = np.array([[0, 0], [70, 0], [70, 14], [0, 14]], np.float32)
GRILL = np.array([[0, 0], [420, 0], [400, 160], [20, 160]], np.float32)


def pose(f):
    """Frame f: the lamp's offset, scale and rotation (moving right, growing 15 %, turning 3 degrees)."""
    t = f / (N - 1)
    return np.array([700 + 400 * t, 460 - 30 * t]), 1.0 + 0.15 * t, np.deg2rad(3 * t)


def place(poly, f, extra=(0, 0)):
    (ox, oy), s, a = pose(f)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (poly * s) @ R.T + np.array([ox + extra[0] * s, oy + extra[1] * s])


def fill(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def frame(f, rng):
    img = np.dstack([np.linspace(55, 95, W, dtype=np.float32)[None].repeat(H, 0)] * 3) + rng.normal(0, 3, (H, W, 3))
    grill = fill(GRILL + np.array([1150, 600]))
    img[grill] = (70, 74, 72)
    strip = fill(place(STRIP, f, (100, 12)))                      # chrome, right beside the lamp
    img[strip] = (165, 165, 170)
    lamp = fill(place(LAMP, f))
    img[lamp] = (235, 235, 225) if f < 70 else (240, 160, 40)     # changes colour at frame 70
    hand = np.zeros((H, W), np.uint8)
    if 40 <= f < 55:
        (ox, oy), _, _ = pose(f)
        cx = int(ox - 60 + (f - 40) / 15 * 220)
        cv2.ellipse(hand, (cx, int(oy + 20)), (55, 30), 15, 0, 360, 1, -1)
        img[hand.astype(bool)] = (200, 155, 125)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 1.2)   # soft edges, as in real footage
    seen = ~hand.astype(bool)
    return img, {1: (0, lamp & seen), 2: (1, strip & seen), 3: (2, grill & seen)}


def make(folder):
    from engine.project import Project
    folder.mkdir(parents=True, exist_ok=True)
    rng, truth = np.random.default_rng(7), {}
    with av.open(str(folder / "clip.mp4"), "w") as c:
        s = c.add_stream("libx264", rate=25, options={"crf": "23"})
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for f in range(N):
            img, parts = frame(f, rng)
            truth[f] = parts
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    p = Project.create(folder / "proj", CLASSES, video=folder / "clip.mp4", every=1, task="segment")
    for o, (c, m) in truth[0].items():
        ys, xs = np.nonzero(m)
        p.put(0, o, c, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), mask=m)
    np.savez_compressed(folder / "truth.npz", **{f"{f}_{o}": m for f, parts in truth.items() for o, (_, m) in parts.items()})
    print("made", folder, len(p.items), "frames")


def track(folder):
    from engine.project import Project
    from engine.tracker import make_tracker
    p, raw = Project(folder / "proj"), {}
    make_tracker().track(p, 0, N - 1, lambda item, res: raw.update(
        {f"{item}_{o}": r[3] for o, r in res.items() if len(r) > 3 and r[3] is not None}), masks=True)
    np.savez_compressed(folder / "raw.npz", **raw)
    print("tracked", len(raw), "part-frames")


# ---- shape transfer -------------------------------------------------------------------------
def _moments(m):
    mo = cv2.moments(m.astype(np.uint8), True)
    if mo["m00"] < 16:
        return None
    c = np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]])
    ang = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
    ecc = np.hypot(mo["mu20"] - mo["mu02"], 2 * mo["mu11"]) / (mo["mu20"] + mo["mu02"] + 1e-9)
    return mo["m00"], c, ang, ecc


def transfer(key, pred, kind):
    """`key` (the drawn outline) fitted onto `pred` (SAM's mask): translation, similarity (moments) or affine (ECC)."""
    a, b = _moments(key), _moments(pred)
    if a is None or b is None:
        return None
    s = np.sqrt(b[0] / a[0]) if kind != "shift" else 1.0
    rot = 0.0
    if kind != "shift" and min(a[3], b[3]) > 0.2:                   # orientation only for elongated shapes
        rot = (b[2] - a[2] + np.pi / 2) % np.pi - np.pi / 2
    M = cv2.getRotationMatrix2D((float(a[1][0]), float(a[1][1])), -np.rad2deg(rot), s)
    M[:, 2] += b[1] - a[1]
    out = cv2.warpAffine(key.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
    if kind == "affine":                                            # refine on the blurred masks (same-size images;
        tpl = cv2.GaussianBlur(pred.astype(np.float32), (0, 0), 2)  # ECC's warp maps template -> input, so start
        inp = cv2.GaussianBlur(key.astype(np.float32), (0, 0), 2)   # from the inverse of the moments fit)
        Wm = cv2.invertAffineTransform(M).astype(np.float32)
        try:
            _, Wm = cv2.findTransformECC(tpl, inp, Wm, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6), None, 5)
            out = cv2.warpAffine(key.astype(np.uint8), Wm, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        except cv2.error as e:
            transfer.failed = getattr(transfer, "failed", 0) + 1
            print("ECC failed:", str(e)[:80])
    return out.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def band(m, d=3):
    """The inner boundary band of width d (Boundary IoU, Cheng et al. 2021)."""
    return m & ~cv2.erode(m.astype(np.uint8), np.ones((2 * d + 1, 2 * d + 1), np.uint8)).astype(bool)


def score(folder):
    truth, raw = np.load(folder / "truth.npz"), np.load(folder / "raw.npz")
    key = {o: truth[f"0_{o}"] for o in (1, 2, 3)}
    rows = []
    for name in raw.files:
        f, o = map(int, name.split("_"))
        gt, pred = truth[name], raw[name]
        if not gt.any():
            continue
        r = {"f": f, "o": o, "hand": 40 <= f < 55, "raw": (iou(pred, gt), iou(band(pred), band(gt)))}
        for kind in ("shift", "similarity", "affine"):
            t = transfer(key[o], pred, kind)
            r[kind] = (iou(t, gt), iou(band(t), band(gt)), iou(t, pred)) if t is not None else None
        rows.append(r)
    out = {}
    for kind in ("shift", "similarity", "affine"):
        for T in (0.0, 0.7, 0.8):                                    # fallback: SAM's mask when agreement < T
            for o in (1, 2, 3):
                sel = [r for r in rows if r["o"] == o and r[kind]]
                use = [r[kind][2] >= T for r in sel]
                got = [(r[kind][0], r[kind][1]) if u else r["raw"] for r, u in zip(sel, use)]
                base = [r["raw"] for r in sel]
                worse = [g[0] < b[0] - 0.05 for g, b, u in zip(got, base, use) if u]
                hand = [g[0] - b[0] for g, b, r in zip(got, base, sel) if r["hand"]]
                buck = {}
                for (g, b), r in zip(zip(got, base), sel):
                    k = "1-10" if r["f"] <= 10 else "11-30" if r["f"] <= 30 else "31-75" if r["f"] <= 75 else "76+"
                    buck.setdefault(k, []).append(g[1] - b[1])
                out[f"{kind} T={T} {CLASSES[o - 1]}"] = {
                    "n": len(sel), "iou raw": round(np.mean([b[0] for b in base]), 3), "iou": round(np.mean([g[0] for g in got]), 3),
                    "edge raw": round(np.mean([b[1] for b in base]), 3), "edge": round(np.mean([g[1] for g in got]), 3),
                    "fallback": f"{use.count(False)}/{len(use)}", "used & worse": f"{sum(worse)}/{use.count(True)}",
                    "worst": round(min([g[0] - b[0] for g, b in zip(got, base)] or [0]), 3),
                    "hand frames iou change": round(float(np.mean(hand)), 3) if hand else None,
                    "edge gain by frames from key": {k: round(float(np.mean(v)), 3) for k, v in sorted(buck.items())}}
    (folder / "score.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    {"make": make, "track": track, "score": score}[sys.argv[1]](Path(sys.argv[2]))
