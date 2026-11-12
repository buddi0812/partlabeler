"""S23 stress set: a synthetic outline project (no customer footage) made to provoke the three tracking failures the
lamp videos did not show: bleeding (a grey housing slides up against a neighbour of nearly the same grey), deforming
(a skin-coloured hand sweeps across it) and the wrong state (a lamp in it turns red, amber, then white).

    python spikes/s23_synthetic.py <folder>        # then: s23_tracking_bench.py --projects <folder>/stress --tasks 2

Two videos of the same station: "labeled" (the housing moves left; every frame drawn and confirmed, the lamp's three
states included: what a person has already annotated) and "new" (moves right; every frame confirmed too, so it can
be scored, but the benchmark hides everything after its first frame). Every part's true outline is known exactly.
"""
import sys
from pathlib import Path

import av
import cv2
import numpy as np

from engine.project import Project

W, H, N = 640, 360, 120
CLASSES = ["housing", "bracket", "lamp_red", "lamp_amber", "lamp_white"]
COLOUR = {2: (220, 30, 30), 3: (240, 150, 20), 4: (250, 250, 240)}           # RGB per lamp state


def scene(f, direction, rng):
    """Frame f: the image and the true outline of each part {obj: (cls, mask)}."""
    img = np.dstack([np.linspace(60, 110, W, dtype=np.float32)[None].repeat(H, 0)] * 3)
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    x = 60 + min(2 * f, 200) if direction > 0 else 460 - min(2 * f, 200)       # stops beside the bracket
    hx, hy = x, 140
    housing = np.zeros((H, W), np.uint8)
    cv2.rectangle(housing, (hx, hy), (hx + 120, hy + 80), 1, -1)
    bracket = np.zeros((H, W), np.uint8)
    bx = 382 if direction > 0 else 168                                        # 2 px from where the housing stops
    cv2.rectangle(bracket, (bx, 120), (bx + 90, 240), 1, -1)
    img[bracket.astype(bool)] = (132, 132, 136)                               # nearly the housing's grey
    img[housing.astype(bool)] = (128, 128, 128)
    lamp = np.zeros((H, W), np.uint8)
    cv2.circle(lamp, (hx + 60, hy + 40), 22, 1, -1)
    state = 2 if f < 40 else 3 if f < 80 else 4
    glow = cv2.GaussianBlur(lamp.astype(np.float32), (0, 0), 8)[..., None]
    img = img * (1 - 0.5 * glow) + np.array(COLOUR[state], np.float32) * 0.5 * glow
    img[lamp.astype(bool)] = COLOUR[state]
    hand = np.zeros((H, W), np.uint8)
    if 30 <= f < 55:                                                           # sweeps across the housing
        cx = hx - 40 + int((f - 30) / 25 * 200)
        cv2.ellipse(hand, (cx, hy + 50), (45, 28), 20, 0, 360, 1, -1)
        img[hand.astype(bool)] = (205, 160, 130)
    seen = ~hand.astype(bool)
    parts = {1: (0, housing.astype(bool) & seen), 2: (1, bracket.astype(bool) & seen),
             3: (state, lamp.astype(bool) & seen)}
    return np.clip(img, 0, 255).astype(np.uint8), parts


def write(folder, name, direction, seed):
    rng = np.random.default_rng(seed)
    path = folder / f"{name}.mp4"
    truth = []
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=25, options={"crf": "12"})
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for f in range(N):
            img, parts = scene(f, direction, rng)
            truth.append(parts)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path, truth


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    first, truth_a = write(folder, "labeled", -1, 1)
    second, truth_b = write(folder, "new", 1, 2)
    p = Project.create(folder / "stress", CLASSES, video=first, every=1, task="segment")
    p.add_source(video=second, every=1)
    p.set_meta(states=[["lamp_red", "lamp_amber", "lamp_white"]])          # as Settings, or a relabel, would set
    for task, truth, base_obj in ((0, truth_a, 0), (1, truth_b, 10)):
        lo, _ = p.ranges[p.meta["sources"][task]["id"]]
        for f, parts in enumerate(truth):
            for o, (c, m) in parts.items():
                if m.sum() >= 16:
                    ys, xs = np.nonzero(m)
                    p.put(lo + f, base_obj + o, c, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), mask=m)
            p.set_reviewed(lo + f)
    print(f"{folder / 'stress'}: {len(p.items)} frames, 2 tasks")


if __name__ == "__main__":
    main(sys.argv[1])
