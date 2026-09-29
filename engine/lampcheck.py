"""Reference profiles of lit parts (lamps, LEDs) and a check of a detected part against them: a second layer that
says whether a part a detector found is lit as it should be (a dark section, too dim, wrong shape or colour, left and
right not matching). Standalone on purpose (numpy, OpenCV, the standard library), so production can copy this file:

    ref = build(samples)                        # healthy examples, e.g. confirmed frames of an annotated dataset
    write_xml(ref, "reference.xml"); ref = read_xml("reference.xml")
    check(ref, group, part, image_rgb, box)     # -> {"ok": bool, "reasons": [...], "values": {...}}
    check_pair(ref, group, part_a, image_rgb, box_a, part_b, box_b)
    check_sweep(ref, group, part, [(image_rgb, box), ...])   # one blink of a part that lights up in a sweep
    check_blinks(ref, group, part, [blink1_frames, blink2_frames, ...])   # several blinks: 2 of 3 must fail

Parts that light up in stages (a progressive indicator sweeping from its tip to full length) are built with
build(..., sweep=[names]) from their fully swept frames only, and checked over a few blinks: the frame where the part
reaches furthest must show the full shape (NOT_FULL when it never gets there, DARK_SEGMENT for a gap in it).

Measurement (the same for building and checking): inside the part's box, a pixel is lit when its brightest colour
channel reaches the part's threshold, calibrated per part on the labeled outlines. Values: box centre (share of the
frame) and size, lit pixels, lit share of the box, lit share in N segments along the part's long side, the lit area
on a fixed grid (compared with the reference template), and the colour of the lit pixels. Ranges are the healthy
examples' 0.5-99.5 percentiles, widened by a margin.
"""
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np

SEGMENTS = 20
GRID = (12, 48)                 # rows x cols of the template, cols along the long side
PCT = (0.2, 99.8)
MARGIN = 0.1                    # ranges widened by this share of their width
DARK_RUN = 2                    # a dark section: at least this many neighbouring segments below their minimum
DARK_SHARE = 0.3                # sweeping (thin) parts: a segment is also dark below this share of its usual lit share
SEGMENT_LIT = 0.05              # sweeping parts: segments checked when usually at least this share lit
STEADY_LIT = 0.3                # steady parts: segments checked when usually at least this share lit
VOTE = (2, 3)                   # sweeping parts over several blinks: not OK when at least 2 of 3 blinks fail
RANGES = ("cx", "cy", "w", "h", "long", "lit_px", "fill", "chroma")
# the least half-width of a range, so a tolerance is never zero: absolute, or relative (r) to the value
MIN_HALF = {"cx": 0.01, "cy": 0.01, "w": ("r", 0.03), "h": ("r", 0.03), "long": ("r", 0.03), "lit_px": ("r", 0.05),
            "fill": 0.03, "chroma": 3.0, "hue": 5.0}
NON_NEGATIVE = {"w", "h", "long", "lit_px", "fill", "chroma"}
FULL = 0.9                      # sweeping parts: frames whose long side reaches 90% of the usual full length
REACH = 0.85                    # ... and a blink counts as full when it reaches 85% of it (sampling can miss the end)
COLOURED = 15.0                 # parts whose lit pixels are this colourful (Lab chroma) are lit only in their own hue
HUE_TOLERANCE = 30.0


def _patch(image, box):
    H, W = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def _long(a: np.ndarray) -> np.ndarray:
    """The patch with its long side as columns."""
    return a if a.shape[1] >= a.shape[0] else np.swapaxes(a, 0, 1)


def _lit(patch: np.ndarray, threshold: int, hue=None, lab=None) -> np.ndarray:
    """Lit pixels: bright (brightest channel at the threshold); for a coloured part (hue: its hue in degrees) also of
    that colour, so bright white surroundings (a white body, a neighbouring white lamp) do not count."""
    lit = patch.max(axis=2) >= threshold
    if hue is not None:
        lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32) if lab is None else lab
        a, b = lab[..., 1] - 128, lab[..., 2] - 128
        off = np.abs((np.degrees(np.arctan2(b, a)) % 360 - hue + 180) % 360 - 180)
        lit &= (np.hypot(a, b) >= COLOURED * 0.8) & (off <= HUE_TOLERANCE)
    return lit


def measure(image: np.ndarray, box, threshold: int, hue=None) -> dict | None:
    """image: H x W x 3 uint8 RGB; box: pixel [x1, y1, x2, y2]; hue: a coloured part's hue (lit pixels must match)."""
    patch, (x1, y1, x2, y2) = _patch(image, box)
    if patch.size == 0 or min(patch.shape[:2]) < 2:
        return None
    H, W = image.shape[:2]
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    lit = _lit(patch, threshold, hue, lab)
    long = _long(lit)
    cols = np.array_split(np.arange(long.shape[1]), SEGMENTS)
    a, b = (lab[..., 1][lit].mean() - 128, lab[..., 2][lit].mean() - 128) if lit.any() else (0.0, 0.0)
    grid = cv2.resize(long.astype(np.float32), GRID[::-1], interpolation=cv2.INTER_AREA)
    return {"cx": (x1 + x2) / 2 / W, "cy": (y1 + y2) / 2 / H, "w": float(x2 - x1), "h": float(y2 - y1),
            "long": float(max(x2 - x1, y2 - y1)),
            "lit_px": float(lit.sum()), "fill": float(lit.mean()), "chroma": float(math.hypot(a, b)),
            "hue": float(math.degrees(math.atan2(b, a)) % 360), "segments": [float(long[:, c].mean()) for c in cols],
            "grid": grid}


def _iou(grid: np.ndarray, template: np.ndarray) -> float:
    a, t = grid >= 0.5, template >= 0.5
    u = (a | t).sum()
    return float((a & t).sum() / u) if u else 1.0


def part_hue(crops):
    """A coloured part's hue (degrees) from its labeled pixels, or None for a white one."""
    ab = [cv2.cvtColor(p, cv2.COLOR_RGB2LAB).astype(np.float32)[o][:, 1:] - 128 for p, o in crops if o.any()]
    if not ab:
        return None
    ab = np.concatenate(ab)
    if np.median(np.hypot(ab[:, 0], ab[:, 1])) < COLOURED:
        return None
    return float(np.degrees(np.arctan2(np.median(ab[:, 1]), np.median(ab[:, 0]))) % 360)


def calibrate(crops, hue=None) -> int:
    """The lit threshold that best reproduces the labeled outlines: crops [(patch RGB, outline mask of the patch)].
    Of the thresholds within 2% of the best, the middle one: robust, and a dimmed part falls below it."""
    score = {}
    for t in range(150, 255, 5):
        scores = []
        for patch, ref in crops:
            lit = _lit(patch, t, hue)
            u = (lit | ref).sum()
            scores.append((lit & ref).sum() / u if u else 1.0)
        if scores:
            score[t] = float(np.mean(scores))
    if not score:
        return 240
    good = [t for t, v in score.items() if v >= max(score.values()) - 0.02]
    return int(good[len(good) // 2])


def _range(values, key=None):
    lo, hi = np.percentile(values, PCT)
    m = (hi - lo) * MARGIN
    lo, hi = lo - m, hi + m
    least = MIN_HALF.get(key)
    if least is not None:
        mid = (lo + hi) / 2
        half = max((hi - lo) / 2, least[1] * abs(mid) if isinstance(least, tuple) else least)
        lo, hi = mid - half, mid + half
    if key in NON_NEGATIVE:
        lo = max(0.0, lo)
    return [float(lo), float(hi)]


def build(samples, pairs=(), calibration: int = 80, sweep=()) -> dict:
    """samples: a function giving a fresh iterator of (frame, group, part, image, box, outline or None) over healthy
    parts; it is called twice (calibrating, then measuring), so frames can be read one at a time. frame: any key the
    parts of one picture share; group: e.g. a camera or station; outline: the part's labeled pixels (full-frame bool
    mask). pairs: [(part_a, part_b)] seen together whose lit areas should match (e.g. left and right). sweep: parts
    that light up in stages; their profile is built from their fully swept frames. Returns
    {group: {"parts": {part: profile}, "pairs": {(a, b): {"area_ratio": [lo, hi]}}}}."""
    crops, seen = defaultdict(list), defaultdict(int)
    for _, group, part, image, box, outline in samples():            # calibration: small crops, spread out
        if outline is None:
            continue
        seen[(group, part)] += 1
        if len(crops[(group, part)]) < calibration and seen[(group, part)] % 3 == 1:
            patch, (x1, y1, x2, y2) = _patch(image, box)
            if patch.size:
                crops[(group, part)].append((patch.copy(), outline[y1:y2, x1:x2].astype(bool)))
    hues = {k: part_hue(v) for k, v in crops.items()}
    thresholds = {k: calibrate(v, hues[k]) for k, v in crops.items()}
    vals, lit = defaultdict(list), defaultdict(dict)
    for frame, group, part, image, box, _ in samples():
        v = measure(image, box, thresholds.get((group, part), 240), hues.get((group, part)))
        if v:
            vals[(group, part)].append(v)
            lit[(group, frame)][part] = v["lit_px"]
    ref = defaultdict(lambda: {"parts": {}, "pairs": {}})
    for (group, part), vs in vals.items():
        full_long = None
        if part in sweep and vs:                              # the fully swept frames only
            cut = FULL * np.percentile([v["long"] for v in vs], 95)
            vs = [v for v in vs if v["long"] >= cut]
            full_long = float(np.median([v["long"] for v in vs])) if vs else None
        if len(vs) < 5:
            continue
        seg = np.array([v["segments"] for v in vs])
        med = np.median(seg, axis=0)
        template = np.mean([v["grid"] for v in vs], axis=0)
        ious = [_iou(v["grid"], template) for v in vs]
        ref[group]["parts"][part] = {
            "samples": len(vs), "threshold": thresholds.get((group, part), 240), "sweep": part in sweep,
            "full_long": full_long, "lit_hue": hues.get((group, part)),
            **{k: _range([v[k] for v in vs], k) for k in RANGES},
            "hue": _range([v["hue"] for v in vs], "hue") if np.median([v["chroma"] for v in vs]) > 15 else None,
            "segment_median": [round(float(x), 3) for x in med],
            "segment_min": [round(float(max(0.0, np.percentile(seg[:, i], PCT[0]) - MARGIN * med[i])), 3)
                            if med[i] >= SEGMENT_LIT else None for i in range(SEGMENTS)],   # normally lit ones only
            "template": template, "overlap_min": float(max(0.0, np.percentile(ious, PCT[0]) - MARGIN))}
    for a, b in pairs:
        for group in ref:
            ratios = [f[a] / f[b] for (g, _), f in lit.items() if g == group and a in f and b in f and f[b] > 0]
            if len(ratios) >= 5 and a in ref[group]["parts"] and b in ref[group]["parts"]:
                ref[group]["pairs"][(a, b)] = {"area_ratio": _range(ratios)}
    return dict(ref)


def check(ref: dict, group: str, part: str, image: np.ndarray, box) -> dict:
    """OK or not. Reasons (not OK): AREA_LOW, FILL_LOW (too little lit), DARK_SEGMENT (a dark section: the segment
    numbers), SHAPE, COLOUR, UNKNOWN_PART, NO_PIXELS. Notes (still OK: they say nothing about the part working):
    AREA_HIGH, FILL_HIGH (glare, bloom), POSITION, SIZE (where the object stopped)."""
    r = ref.get(group, {}).get("parts", {}).get(part)
    if r is None:
        return {"ok": False, "reasons": ["UNKNOWN_PART"], "values": {}}
    v = measure(image, box, r["threshold"], r.get("lit_hue"))
    if v is None:
        return {"ok": False, "reasons": ["NO_PIXELS"], "values": {}}
    reasons, notes = [], []
    for key, code in (("lit_px", "AREA"), ("fill", "FILL")):
        lo, hi = r[key]
        if v[key] < lo:
            reasons.append(f"{code}_LOW")
        elif v[key] > hi:
            notes.append(f"{code}_HIGH")
    if not (r["cx"][0] <= v["cx"] <= r["cx"][1] and r["cy"][0] <= v["cy"] <= r["cy"][1]):
        notes.append("POSITION")
    if not (r["w"][0] <= v["w"] <= r["w"][1] and r["h"][0] <= v["h"] <= r["h"][1]):
        notes.append("SIZE")
    if r.get("sweep") and r.get("full_long") and v["long"] < REACH * r["full_long"]:   # never reached full length
        reasons.append("NOT_FULL")
    if r.get("sweep"):                                    # thin strips: also relative to the usual share
        dark = [i for i, (lo, m, s) in enumerate(zip(r["segment_min"], r["segment_median"], v["segments"]))
                if lo is not None and s < max(lo, DARK_SHARE * m)]
    else:
        dark = [i for i, (lo, m, s) in enumerate(zip(r["segment_min"], r["segment_median"], v["segments"]))
                if lo is not None and m >= STEADY_LIT and s < lo]
    runs, cur = [], []
    for i in dark:                                        # neighbouring dark segments: a dead section
        cur = cur + [i] if cur and i == cur[-1] + 1 else [i]
        if len(cur) == DARK_RUN:
            runs.append(cur)
        elif len(cur) > DARK_RUN:
            runs[-1] = cur
    if runs:
        reasons.append("DARK_SEGMENT " + ",".join(str(i + 1) for run in runs for i in run))
    overlap = _iou(v["grid"], r["template"])
    if overlap < r["overlap_min"]:
        reasons.append("SHAPE")
    if v["chroma"] > r["chroma"][1] or (r["hue"] and not _hue_in(v["hue"], r["hue"])):
        reasons.append("COLOUR")
    return {"ok": not reasons, "reasons": reasons, "notes": notes,
            "values": {k: round(v[k], 4) for k in (*RANGES, "hue")} | {"overlap": round(overlap, 3)}}


def check_sweep(ref: dict, group: str, part: str, frames) -> dict:
    """A part that lights up in stages, over a few of its blinks: frames [(image, box)] where it was found. The frame
    where it reaches furthest is checked as a full part (NOT_FULL, DARK_SEGMENT, AREA_LOW, SHAPE, COLOUR...);
    NOT_SEEN when it was never found."""
    frames = [(img, box) for img, box in frames if box is not None]
    if not frames:
        return {"ok": False, "reasons": ["NOT_SEEN"], "notes": [], "values": {}, "peak": None}
    peak = max(range(len(frames)), key=lambda k: max(frames[k][1][2] - frames[k][1][0], frames[k][1][3] - frames[k][1][1]))
    return check(ref, group, part, *frames[peak]) | {"peak": peak}


def check_blinks(ref: dict, group: str, part: str, blinks, vote=VOTE) -> dict:
    """A sweeping part over consecutive blinks (each a list of (image, box)): not OK when at least vote[0] of any
    vote[1] neighbouring blinks fail check_sweep (a dead LED fails every blink; a missed moment only one)."""
    results = [check_sweep(ref, group, part, frames) for frames in blinks]
    need, of = vote
    bad = [not r["ok"] for r in results]
    failing = any(sum(bad[k:k + of]) >= need for k in range(max(1, len(bad) - of + 1))) if len(bad) >= need else all(bad)
    reasons = sorted({x.split(" ")[0] for r, b in zip(results, bad) if b for x in r["reasons"]})
    return {"ok": not failing, "reasons": reasons if failing else [], "blinks": len(results), "failed": sum(bad),
            "per_blink": results}


def _hue_in(h, rng):
    lo, hi = rng
    return lo <= h <= hi


def check_pair(ref, group, part_a, image, box_a, part_b, box_b) -> dict:
    """The two parts' lit areas match as usual (e.g. left and right): LEFT_RIGHT_MISMATCH otherwise."""
    p = ref.get(group, {}).get("pairs", {}).get((part_a, part_b))
    ra, rb = (ref.get(group, {}).get("parts", {}).get(x) for x in (part_a, part_b))
    if not (p and ra and rb):
        return {"ok": True, "reasons": [], "ratio": None}
    va, vb = measure(image, box_a, ra["threshold"], ra.get("lit_hue")), measure(image, box_b, rb["threshold"], rb.get("lit_hue"))
    ratio = va["lit_px"] / vb["lit_px"] if va and vb and vb["lit_px"] else 0.0
    ok = p["area_ratio"][0] <= ratio <= p["area_ratio"][1]
    return {"ok": ok, "reasons": [] if ok else ["LEFT_RIGHT_MISMATCH"], "ratio": round(ratio, 3)}


# ---- XML ---------------------------------------------------------------------------------------------------------
def write_xml(ref: dict, path, meta: dict | None = None, validation: dict | None = None) -> None:
    root = ET.Element("PartReference", {"version": "1", "segments": str(SEGMENTS), "grid": f"{GRID[0]}x{GRID[1]}",
                                        "measurement": "lit = max(R,G,B) >= threshold inside the part's box"}
                      | {k: str(v) for k, v in (meta or {}).items()})
    for group, g in ref.items():
        ge = ET.SubElement(root, "Group", {"name": str(group)})
        for part, r in g["parts"].items():
            pe = ET.SubElement(ge, "Part", {"class": part, "samples": str(r["samples"]), "threshold": str(r["threshold"]),
                                            "sweep": "1" if r.get("sweep") else "0"}
                               | ({"full_long": f"{r['full_long']:.1f}"} if r.get("full_long") else {})
                               | ({"lit_hue": f"{r['lit_hue']:.1f}"} if r.get("lit_hue") is not None else {}))
            for k in RANGES:
                ET.SubElement(pe, "Range", {"name": k, "min": f"{r[k][0]:.7g}", "max": f"{r[k][1]:.7g}"})
            if r["hue"]:
                ET.SubElement(pe, "Range", {"name": "hue", "min": f"{r['hue'][0]:.4g}", "max": f"{r['hue'][1]:.4g}"})
            ET.SubElement(pe, "Segments", {"count": str(SEGMENTS), "dark_run": str(DARK_RUN),
                                           "min": " ".join("-" if x is None else f"{x:.3f}" for x in r["segment_min"])}
                          ).text = " ".join(f"{x:.3f}" for x in r["segment_median"])
            ET.SubElement(pe, "Template", {"rows": str(GRID[0]), "cols": str(GRID[1]),
                                           "overlap_min": f"{r['overlap_min']:.3f}"}).text = \
                " ".join(str(int(round(x * 100))) for x in r["template"].ravel())
        for (a, b), p in g["pairs"].items():
            ET.SubElement(ge, "Pair", {"a": a, "b": b, "area_ratio_min": f"{p['area_ratio'][0]:.4f}",
                                       "area_ratio_max": f"{p['area_ratio'][1]:.4f}"})
    if validation:
        ve = ET.SubElement(root, "Validation")
        for key, value in validation.items():
            ET.SubElement(ve, "Result", {"name": key}).text = str(value)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_xml(path) -> dict:
    ref = {}
    root = ET.parse(path).getroot()
    rows, cols = (int(x) for x in root.get("grid", f"{GRID[0]}x{GRID[1]}").split("x"))
    for ge in root.iter("Group"):
        g = ref.setdefault(ge.get("name"), {"parts": {}, "pairs": {}})
        for pe in ge.iter("Part"):
            rng = {e.get("name"): [float(e.get("min")), float(e.get("max"))] for e in pe.iter("Range")}
            seg, tpl = pe.find("Segments"), pe.find("Template")
            g["parts"][pe.get("class")] = {
                "samples": int(pe.get("samples")), "threshold": int(pe.get("threshold")),
                "sweep": pe.get("sweep") == "1",
                "full_long": float(pe.get("full_long")) if pe.get("full_long") else None,
                "lit_hue": float(pe.get("lit_hue")) if pe.get("lit_hue") else None,
                **{k: rng[k] for k in RANGES}, "hue": rng.get("hue"),
                "segment_median": [float(x) for x in seg.text.split()],
                "segment_min": [None if x == "-" else float(x) for x in seg.get("min").split()],
                "template": np.array([int(x) / 100 for x in tpl.text.split()], np.float32).reshape(rows, cols),
                "overlap_min": float(tpl.get("overlap_min"))}
        for pr in ge.iter("Pair"):
            g["pairs"][(pr.get("a"), pr.get("b"))] = {"area_ratio": [float(pr.get("area_ratio_min")),
                                                                      float(pr.get("area_ratio_max"))]}
    return ref
