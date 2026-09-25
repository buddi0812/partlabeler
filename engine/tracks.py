"""Transfer track check: link a video's per-frame detections into tracks and list what looks inconsistent.

Along each track three things are worth a human look: a frame where the label differs from the track's
score-weighted majority (e.g. a left/right flip), a short gap where the part was found before and after
but not in between (a possible miss), and a low-score box seen on a single frame only (a possible false
box). The labels themselves are never changed: on the example dataset's 143 held-out frames, filling gaps
turned 51 boxes into false positives (F1 0.962 -> 0.948) and voting, smoothing and dropping changed F1 by
under 0.001 (spikes/REPORT.md, S8). A person decides, with these frames on the check list.

Linking is greedy IoU matching against each track's constant-velocity prediction. A detection joins a
track of another class only when the boxes nearly coincide (that is what a flip looks like), so nested
parts of different classes stay apart. Suits parts that appear once or a few times per frame.
"""
from collections import Counter

import numpy as np


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _predict(track: dict, k: int) -> np.ndarray:
    ks = sorted(track["obs"])[-2:]
    last = np.array(track["obs"][ks[-1]][1:5], float)
    if len(ks) == 1:
        return last
    prev = np.array(track["obs"][ks[0]][1:5], float)
    return last + (last - prev) / (ks[1] - ks[0]) * (k - ks[1])


def _majority(track: dict) -> int:
    votes = Counter()
    for c, *_, score in track["obs"].values():
        votes[c] += score
    return votes.most_common(1)[0][0]


def link(frames, max_gap: int = 2, same_iou: float = 0.3, cross_iou: float = 0.6) -> list[dict]:
    """frames: per sampled frame (in time order), a list of (cls, x1, y1, x2, y2, score).
    Tracks as {"obs": {frame index: [cls, x1, y1, x2, y2, score]}}."""
    tracks = []
    for k, dets in enumerate(frames):
        live = [t for t in tracks if k - t["last"] <= max_gap + 1]
        pairs = []
        for ti, t in enumerate(live):
            p, cls = _predict(t, k), _majority(t)
            for di, d in enumerate(dets):
                o = iou(p, d[1:5])
                if o >= (same_iou if d[0] == cls else cross_iou):
                    pairs.append((o + (d[0] == cls), ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs, reverse=True):
            if ti not in used_t and di not in used_d:
                used_t.add(ti), used_d.add(di)
                live[ti]["obs"][k], live[ti]["last"] = list(dets[di]), k
        tracks += [{"obs": {k: list(d)}, "last": k} for di, d in enumerate(dets) if di not in used_d]
    return tracks


def check(frames, classes=None, max_gap: int = 2, lone_score: float = 0.7) -> dict:
    """{frame index: {"label_flip": [[seen, usual]], "possible_miss": [...], "lone_box": [...]}} for the
    frames worth a look (class names when `classes` is given). Nothing is changed."""
    name = (lambda c: classes[c]) if classes else (lambda c: c)
    found = {}

    def note(k, kind, value):
        found.setdefault(k, {"label_flip": [], "possible_miss": [], "lone_box": []})[kind].append(value)

    for t in link(frames, max_gap):
        obs, ks = t["obs"], sorted(t["obs"])
        if len(ks) == 1:
            if obs[ks[0]][5] < lone_score:
                note(ks[0], "lone_box", name(obs[ks[0]][0]))
            continue
        cls = _majority(t)
        for k in ks:
            if obs[k][0] != cls:
                note(k, "label_flip", [name(obs[k][0]), name(cls)])
        for a, b in zip(ks, ks[1:]):
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                box = np.array(obs[a][1:5], float) * (1 - w) + np.array(obs[b][1:5], float) * w
                if not any(d[0] == cls and iou(box, d[1:5]) > 0.5 for d in frames[k]):   # not on another track
                    note(k, "possible_miss", name(cls))
    return found
