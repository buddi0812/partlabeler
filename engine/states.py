"""Parts that change look: classes that are states of one part, such as a lamp that is bright (one class) or dim
(another), low or high beam, a light that is on or off. The project lists them (project.json "states": groups of
class names, any number of groups and of states per group). Tracking follows the part itself; on every frame it
gets the state whose examples look nearest: parts a person drew or confirmed in the same task, compared by the
brightness and colour inside the outline (or box) and the glow around it. Not by size: right after a switch the
tracked outline still has the old state's shape. S16 (person-checked outlines of one lamp's bright and dim states,
each classified from earlier examples): 780 of 782 bright, 140 of 140 dim, 4 of 4 switch frames. S19 (tracking
through a switch, the tracked frames hidden): every frame's state right with SAM 3, bright to dim and back.

Groups are the project's (Settings); a person changing a part's class on one frame where it clearly looks different
adds one (api.on_set_class). Classes a person gave the same part on different frames are only suggested
(found_groups, shown under the setting; on the lamp projects this found exactly the groups set by hand), never used
on their own: a fixed left/right swap looks the same in the labels. Examples come from the whole project,
the start item's task first, so a new video follows the states labeled in others. Colour is read inside the outline
shrunk by a few pixels, so an outline that bleeds onto the background does not drag the colour with it. No
smoothing over frames: an indicator's blink can last a single kept frame.
"""
import re

import cv2
import numpy as np

from engine import masks as M

PAIR = re.compile(r"^(left|right)_(.+)$")
PER_CLASS = 20      # examples per state, the ones nearest to where tracking starts
MIN_FRAMES = 2      # vouched frames each class needs on one part before two classes count as its states
INNER = 5           # px kernel the outline is shrunk by before reading its colour (away from a bleeding edge)


def parse(text: str, classes: list[str]) -> list[list[str]]:
    """One group per line, names separated by commas: "left_drl, left_park". With left_/right_ classes a line may
    use the shared name ("drl, park"), which makes one group per side. Unknown names are an error."""
    groups = []
    for line in text.splitlines():
        names = [n.strip() for n in line.replace(";", ",").split(",") if n.strip()]
        if not names:
            continue
        if len(names) < 2:
            raise ValueError(f"'{line.strip()}': a group needs two or more classes")
        if all(n in classes for n in names):
            groups.append(names)
            continue
        sides = [[f"{s}_{n}" for n in names] for s in ("left", "right")]
        if not all(c in classes for g in sides for c in g):
            missing = [n for n in names if n not in classes and not {f"left_{n}", f"right_{n}"} <= set(classes)]
            raise ValueError(f"Not a class of this project: {', '.join(missing) or line.strip()}")
        groups += sides
    return groups


def text(groups: list[list[str]], classes: list[str]) -> str:
    """The groups as the settings box shows them (a left/right pair of groups as one shared line)."""
    lines, done = [], set()
    for g in groups:
        if tuple(g) in done:
            continue
        m = [PAIR.match(n) for n in g]
        twin = [f"right_{x.group(2)}" for x in m] if all(x and x.group(1) == "left" for x in m) else None
        if twin and twin in groups:
            lines.append(", ".join(x.group(2) for x in m))
            done.add(tuple(twin))
        else:
            lines.append(", ".join(g))
    return "\n".join(lines)


def features(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    """img: H x W x 3 floats 0..1; region: H x W bool (the outline, or the box filled in). Colour and brightness
    are read inside the region shrunk by INNER px (when enough is left), the glow around the region itself."""
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    inner = cv2.erode(region.astype(np.uint8), np.ones((INNER, INNER), np.uint8)).astype(bool)
    core = inner if inner.sum() >= 16 else region
    px, lv = img[core], lum[core]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    ring = lum[cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~region]
    if not len(ring):
        ring = np.zeros(1, np.float32)
    return np.r_[px.mean(0), np.percentile(lv, [10, 50, 90]), [(lv > t).mean() for t in (0.8, 0.9, 0.95, 0.98)],
                 (px.max(1) - px.min(1)).mean(), np.median(lab, 0), ring.mean(),
                 [(ring > t).mean() for t in (0.5, 0.8, 0.9)]]


LOOK_CHANGE = 20    # Lab units: a part this much brighter/darker or this far off in colour has changed state


def _look(project, item, box, rle):
    """Median Lab colour inside a part's outline (or box), shrunk away from its edge."""
    img = _array(project.image(item))
    if not rle:                                       # a box holds background too: its middle half only
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) / 4, (y2 - y1) / 4
        box = (x1 + dx, y1 + dy, x2 - dx, y2 - dy)
    region = _region(img.shape, box, M.decode(rle) if rle else None)
    if region is None:
        return None
    inner = cv2.erode(region.astype(np.uint8), np.ones((INNER, INNER), np.uint8)).astype(bool)
    px = img[inner if inner.sum() >= 16 else region]
    return np.median(cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3), 0)


def looks_changed(project, obj: int, cls: int, item: int, n: int = 5) -> bool:
    """This part on `item` looks clearly unlike the frames where a person drew or confirmed it as `cls` (brighter,
    darker or another colour inside the outline): a change of state, not a wrong class."""
    q = ("SELECT item, x1, y1, x2, y2, rle FROM boxes WHERE obj=? AND {} AND (source='manual' OR item IN "
         "(SELECT item FROM reviewed))")
    rows = project.db.execute(q.format("cls=? AND item!=?") + " ORDER BY ABS(item - ?) LIMIT ?",
                              (obj, cls, item, item, n)).fetchall()
    here = project.db.execute("SELECT item, x1, y1, x2, y2, rle FROM boxes WHERE obj=? AND item=?", (obj, item)).fetchone()
    if not rows or here is None:
        return False
    refs = [v for v in (_look(project, r[0], r[1:5], r[5]) for r in rows) if v is not None]
    now = _look(project, here[0], here[1:5], here[5])
    if not refs or now is None:
        return False
    d = now - np.median(refs, 0)
    return abs(d[0]) > LOOK_CHANGE or float(np.hypot(d[1], d[2])) > LOOK_CHANGE


def found_groups(project) -> list[list[int]]:
    """Classes a person gave the same part on different frames, each on MIN_FRAMES or more frames they drew or
    confirmed: the states of one part. Groups that share a class are merged."""
    rows = project.db.execute(
        "SELECT obj, cls FROM boxes WHERE source='manual' OR item IN (SELECT item FROM reviewed) "
        "GROUP BY obj, cls HAVING COUNT(*) >= ?", (MIN_FRAMES,)).fetchall()
    per = {}
    for o, c in rows:
        per.setdefault(o, set()).add(c)
    one_side = lambda cs: len({m.group(1) for m in (PAIR.match(project.classes[c]) for c in cs) if m}) <= 1
    return merge([sorted(cs) for cs in per.values() if len(cs) >= 2 and one_side(cs)])   # left/right: a swap fixed


def merge(groups) -> list[list[int]]:
    """Groups of classes, the ones sharing a class joined into one (where the first of them was)."""
    out = []
    for g in map(set, groups):
        hits = [i for i, o in enumerate(out) if o & g]
        for i in reversed(hits):
            g |= out.pop(i)
        out.insert(hits[0] if hits else len(out), g)
    return [sorted(g) for g in out]



def _region(shape, box, mask=None) -> np.ndarray | None:
    if mask is not None and mask.any():
        return mask.astype(bool)
    r = np.zeros(shape[:2], bool)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    r[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return r if r.sum() >= 16 else None


def _array(img) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255


class Guide:
    """Picks each tracked part's state, from examples a person drew or confirmed: in the start item's task first,
    then anywhere in the project. Groups: the project's. Parts whose class is in no group, or whose group has
    examples of fewer than two states, keep the tracker's class."""

    def __init__(self, project, start: int):
        classes = project.classes
        self.project, self.group = project, {}
        for g in project.meta.get("states") or []:
            ids = [classes.index(n) for n in g if n in classes]
            for c in ids:
                self.group[c] = ids
        self.examples = {}                               # class -> feature vectors
        if not self.group:
            return
        lo, hi = project.ranges[project.task_of(start)["id"]]
        want = sorted(self.group)
        rows = self._rows(want, start)
        picked = []
        for c in want:                                   # this task's nearest first, then the rest of the project
            mine = sorted((r for r in rows if r[1] == c), key=lambda r: (not lo <= r[0] < hi, abs(r[0] - start)))
            picked += mine[:PER_CLASS]
        for item in sorted({r[0] for r in picked}):          # in order: a video is read forward
            img = _array(project.image(item))
            for _, c, x1, y1, x2, y2, rle in (r for r in picked if r[0] == item):
                region = _region(img.shape, (x1, y1, x2, y2), M.decode(rle) if rle else None)
                if region is not None:
                    self.examples.setdefault(c, []).append(features(img, region))
        self.examples = {c: np.array(v) for c, v in self.examples.items()}

    def _rows(self, want, start):
        """Every drawn or confirmed part of these classes: (item, cls, x1, y1, x2, y2, rle)."""
        return self.project.db.execute(
            f"SELECT item, cls, x1, y1, x2, y2, rle FROM boxes WHERE cls IN ({','.join('?' * len(want))}) "
            "AND (source='manual' OR item IN (SELECT item FROM reviewed))", want).fetchall()

    def apply(self, item: int, results: dict, allowed=None) -> dict:
        """results as trackers give them, {obj: (cls, box or None, score[, mask])}; returns them with states set.
        allowed: the only classes a state may become (classes tracked alone)."""
        ok = lambda c: c in self.examples and (not allowed or c in allowed)
        todo = [o for o, r in results.items() if r[1] is not None and r[0] in self.group
                and sum(ok(c) for c in self.group[r[0]]) >= 2]
        if not todo:
            return results
        img, out = _array(self.project.image(item)), dict(results)
        for o in todo:
            cls, box, *rest = results[o]
            region = _region(img.shape, box, rest[1] if len(rest) > 1 else None)
            if region is None:
                continue
            f = features(img, region)
            cands = [c for c in self.group[cls] if ok(c)]
            sd = np.concatenate([self.examples[c] for c in cands]).std(0) + 1e-3
            best = min(cands, key=lambda c: np.linalg.norm((self.examples[c] - f) / sd, axis=1).min())
            out[o] = (best, box, *rest)
        return out
