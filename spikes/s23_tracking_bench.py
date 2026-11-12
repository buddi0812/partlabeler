"""S23: tracking quality, speed and memory, measured against outlines a person confirmed. Read-only: tracks in memory
and never writes to the projects it reads.

    python spikes/s23_tracking_bench.py --label before --projects <outline project folder> [...] [--tasks 3] [--frames 90]

Per project, the tasks with the most confirmed frames; each run starts on the task's first confirmed frame with outlines
and tracks `--frames` ahead with the app's tracker (outlines on). On every confirmed frame reached, each tracked part is
compared with the confirmed outline of the same part: IoU, area ratio, leaked share (tracked pixels outside the
confirmed outline grown by 3 px, over its area). States (projects with parts that change look): the class the app's
state guide gives, as the app builds it today ("app": examples anywhere in the task) and "fair" (only examples up to
the start frame: what a new video gets). Also frames/s, model load time and peak GPU memory. Writes
spikes/out/s23_<label>.json (gitignored) and prints a summary; project names are passed in, never stored here.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from engine import masks as M
from engine import states
from engine.project import Project
from engine.tracker import make_tracker

OUT = Path(__file__).parent / "out"


def connected(mask, prev, grow=15):
    """Only the pieces of `mask` that touch `prev` grown by `grow` px (all of it when none or no previous)."""
    m = M.clean(mask)
    if prev is None or not prev.any() or not m.any():
        return m
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    near = cv2.dilate(prev.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
    keep = [k for k in np.unique(lab[near & m]) if k]
    return np.isin(lab, keep) if keep else m


VARIANTS = {"clean": lambda m, prev: M.clean(m), "connected": connected}   # --variants: the same run, cleaned after


def leak(pred, gt):
    grown = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return float((pred & ~grown).sum() / max(1, gt.sum()))


def pick_runs(p, n_tasks, frames):
    """Per task, the start (a confirmed frame with outlines) whose next `frames` frames hold the most confirmed
    frames with outlines; the tasks where that is highest."""
    reviewed = {r[0] for r in p.db.execute("SELECT item FROM reviewed")}
    with_masks = {r[0] for r in p.db.execute("SELECT DISTINCT item FROM boxes WHERE rle IS NOT NULL")}
    good = np.array(sorted(reviewed & with_masks))
    runs = []
    for t in p.meta["sources"]:
        lo, hi = p.ranges[t["id"]]
        mine = good[(good >= lo) & (good < hi)]
        if len(mine) < 20:
            continue
        ahead = np.searchsorted(mine, np.minimum(mine + frames, hi - 1), side="right") - np.arange(len(mine)) - 1
        best = int(ahead.argmax())
        if ahead[best] >= 20:
            runs.append((int(ahead[best]), t["id"], int(mine[best])))
    return [(tid, start) for _, tid, start in sorted(runs, reverse=True)[:n_tasks]]


def fair_guide(p, start):
    """The state guide without examples from this task after the start frame (a new video's situation)."""
    lo, hi = p.ranges[p.task_of(start)["id"]]

    class Fair(states.Guide):
        def _rows(self, want, s):
            return [r for r in super()._rows(want, s) if not start < r[0] < hi]

    if "_rows" in states.Guide.__dict__:                       # the guide reads examples from the whole project
        return Fair(p, start)
    saved = p.ranges[p.task_of(start)["id"]]                   # older guide: this task only
    p.ranges[p.task_of(start)["id"]] = (lo, start + 1)
    try:
        return states.Guide(p, start)
    finally:
        p.ranges[p.task_of(start)["id"]] = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--projects", nargs="+", required=True)
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--variants", action="store_true", help="also score each outline cleaned (VARIANTS)")
    ap.add_argument("--anchor-every", type=int, default=0, help="let the tracker see every Nth confirmed frame ahead "
                    "(a person confirming now and then); those frames are not scored")
    a = ap.parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tracker = make_tracker()
    load_s = time.perf_counter() - t0
    load_gb = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    report = {"label": a.label, "tracker": tracker.name, "load_s": round(load_s, 1), "load_gb": round(load_gb, 2),
              "runs": []}
    for set_no, folder in enumerate(a.projects):
        p = Project(folder)
        reviewed = {r[0] for r in p.db.execute("SELECT item FROM reviewed")}
        for tid, start in pick_runs(p, a.tasks, a.frames):
            guides = {"app": states.Guide(p, start), "fair": fair_guide(p, start)} if p.meta.get("states") else {}
            grouped = set(guides["app"].group) if guides else set()
            spent = [0.0]                                      # time spent scoring, taken out of frames/s
            rows, hits, var, prev_out = {}, {}, {k: {} for k in VARIANTS}, {}                                # keyed by frame and part: a frame tracked again
                                                               # after running out of memory counts once

            def on_item(item, res):
                t2 = time.perf_counter()
                try:
                    score(item, res)
                finally:
                    spent[0] += time.perf_counter() - t2

            def score(item, res):
                try:
                    score_one(item, res)
                finally:
                    prev_out.update({o: r[3] for o, r in res.items() if len(r) > 3 and r[3] is not None})

            def score_one(item, res):
                if item not in reviewed or (a.anchor_every and (item - start) % a.anchor_every == 0):
                    return
                truth = {b["obj"]: b for b in p.boxes(item)}
                for name, g in guides.items():
                    got = g.apply(item, res) if g.examples else res
                    for o, r in got.items():
                        if o in truth and r[1] is not None and truth[o]["cls"] in grouped:
                            hits[(name, item, o)] = r[0] == truth[o]["cls"]
                for o, r in res.items():
                    gt = p.mask(item, o)
                    if gt is None or not gt.any():
                        continue
                    pred = r[3] if len(r) > 3 and r[3] is not None else np.zeros_like(gt)
                    inter, union = (pred & gt).sum(), (pred | gt).sum()
                    rows[(item, o)] = (item - start, float(inter / union), float(pred.sum() / gt.sum()), leak(pred, gt))
                    if a.variants:
                        for name, fn in VARIANTS.items():
                            v = fn(pred, prev_out.get(o))
                            var[name][(item, o)] = float((v & gt).sum() / max(1, (v | gt).sum()))

            every = a.anchor_every
            seen = lambda k: k <= start or (every and (k - start) % every == 0)
            hidden = Project.anchors                           # confirmed frames ahead are the answers: hidden
            p.anchors = lambda o, _h=hidden, _p=p: [k for k in _h(_p, o) if seen(k)]
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            n = tracker.track(p, start, a.frames, on_item, masks=True)
            del p.anchors
            dt = time.perf_counter() - t1 - spent[0]
            r = np.array(list(rows.values())) if rows else np.zeros((0, 4))
            run = {"set": "AB"[set_no] if set_no < 2 else str(set_no), "task": tid, "start": start, "frames": n,
                   "parts": len(p.boxes(start)), "fps": round(n / dt, 2), "compared": len(r)}
            if len(r):
                run.update(iou_mean=round(float(r[:, 1].mean()), 3), iou_p10=round(float(np.percentile(r[:, 1], 10)), 3),
                           area_ratio=round(float(np.median(r[:, 2])), 3),
                           ballooned=int(((r[:, 2] > 1.5) | (r[:, 3] > 0.25)).sum()),
                           leak_mean=round(float(r[:, 3].mean()), 4), lost=int((r[:, 1] == 0).sum()))
            for name, got in var.items():
                if got:
                    run[f"iou_{name}"] = round(float(np.mean(list(got.values()))), 3)
            for name in guides:
                mine = [v for (g, _, _), v in hits.items() if g == name]
                if mine:
                    run[f"state_{name}"] = f"{sum(mine)}/{len(mine)}"
            report["runs"].append(run)
            print(json.dumps(run), flush=True)
    report["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0.0
    OUT.mkdir(exist_ok=True)
    (OUT / f"s23_{a.label}.json").write_text(json.dumps(report, indent=1))
    runs = [r for r in report["runs"] if r.get("compared")]
    tot = lambda k: sum(r[k] for r in runs)
    print(f"{a.label}: {report['tracker']}, load {report['load_s']} s / {report['load_gb']} GB, peak {report['peak_gb']} GB, "
          f"fps {np.mean([r['fps'] for r in report['runs']]):.2f}, IoU {np.mean([r['iou_mean'] for r in runs]):.3f}, "
          f"ballooned {tot('ballooned')}/{tot('compared')}, lost {tot('lost')}")


if __name__ == "__main__":
    main()
