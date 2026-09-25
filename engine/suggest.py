"""Training-free suggestions: find parts in an image by matching DINOv3 features of the parts
already labeled in the project (S3/S6: ~72% of parts found from 5 labeled images).

Left/right twins (classes named left_X / right_X) share one appearance model; each suggestion
takes the side whose labeled examples sit closer in horizontal position. When the project names a
parent object (its "parent" setting, e.g. "engine block"), matching runs inside that object's crop
(S3: 72% found with the crop vs 67% on the full frame).
"""
import re
import statistics
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F

from engine.embed import Embedder
from engine.parent import crop_box

PAIR = re.compile(r"^(left|right)_(.+)$")


def group_of(name: str) -> str:
    m = PAIR.match(name)
    return m.group(2) if m else name


def box_vector(feats, scale, box):
    p = Embedder.patch
    x1, y1, x2, y2 = box[0] * scale[0] / p, box[1] * scale[1] / p, box[2] * scale[0] / p, box[3] * scale[1] / p
    rows, cols = feats.shape[:2]
    r0, r1 = int(y1 + 0.5), max(int(y1 + 0.5) + 1, int(y2 + 0.5))
    c0, c1 = int(x1 + 0.5), max(int(x1 + 0.5) + 1, int(x2 + 0.5))
    v = feats[min(r0, rows - 1):min(r1, rows), min(c0, cols - 1):min(c1, cols)].reshape(-1, feats.shape[-1]).mean(0)
    return F.normalize(v, dim=0)


def shift(box, off, sign: int = -1):
    return [box[0] + sign * off[0], box[1] + sign * off[1], box[2] + sign * off[0], box[3] + sign * off[1]]


class Suggester:
    def __init__(self, width: int = 1280, device: str | None = None, finder=None, cache_size: int = 64):
        """`finder`: callable returning a ConceptFinder, used only for projects with a parent object."""
        self.emb, self.width, self.finder = Embedder("small", device), width, finder
        self.cache, self.cache_size = OrderedDict(), cache_size

    def _feats(self, project, item):
        """(features, scale, offset): features of the item's image, or of its parent-object crop, whose
        top-left corner is `offset` (boxes are shifted by it on the way in and out)."""
        parent = project.meta.get("parent")
        key = (str(project.folder), item, parent)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        img, off = project.image(item), (0, 0)
        if parent and self.finder:
            box = self.finder().largest(img, parent)
            if box:
                c = crop_box(box, *img.size)
                img, off = img.crop(c), c[:2]
        self.cache[key] = (*self.emb.patches(img, self.width), off)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[key]

    def suggest(self, project, item: int, max_refs: int = 30) -> list[tuple[int, list, float]]:
        names = project.classes
        statuses = project.statuses()
        refs = [k for k, s in enumerate(statuses) if s >= 3 and k != item][-max_refs:]
        if not refs:
            return []
        vecs, per_ref, sizes, xs, counts = (defaultdict(list), defaultdict(dict), defaultdict(list),
                                            defaultdict(list), defaultdict(int))
        for k in refs:
            feats, scale, off = self._feats(project, k)
            W = feats.shape[1] * Embedder.patch / scale[0]
            per_img = defaultdict(int)
            for b in project.boxes(k):
                if b["source"] == "suggested":
                    continue
                g = group_of(names[b["cls"]])
                box = shift(b["box"], off)
                v = box_vector(feats, scale, box)
                vecs[g].append(v); per_ref[g].setdefault(k, []).append((v, box))
                sizes[g].append((box[2] - box[0], box[3] - box[1]))
                xs[b["cls"]].append((box[0] + box[2]) / 2 / W)
                per_img[g] += 1
            for g, n in per_img.items():
                counts[g] = max(counts[g], n)
        feats, scale, off = self._feats(project, item)
        W = feats.shape[1] * Embedder.patch / scale[0]
        out = []
        for g, vs in vecs.items():
            thr = self._threshold(project, g, per_ref)
            smap = (feats @ torch.stack(vs).T).amax(-1)
            mw = statistics.median(s[0] for s in sizes[g]); mh = statistics.median(s[1] for s in sizes[g])
            radius = max(1, int(0.5 * max(mw * scale[0], mh * scale[1]) / Embedder.patch))
            members = [c for c, n in enumerate(names) if group_of(n) == g and xs.get(c)]
            for r, c, s in self._peaks(smap, thr, counts[g], radius):
                cx = (c + 0.5) * Embedder.patch / scale[0]; cy = (r + 0.5) * Embedder.patch / scale[1]
                cls = min(members, key=lambda m: abs(statistics.mean(xs[m]) - cx / W))
                out.append((cls, shift([cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2], off, +1), s))
        return out

    def _threshold(self, project, g, per_ref) -> float:
        """Midway between labeled parts' scores (leave one image out) and the best background score."""
        pos, neg = [], []
        p = Embedder.patch
        for k, own in per_ref[g].items():
            others = [v for kk, vs in per_ref[g].items() if kk != k for v, _ in vs]
            if not others:
                continue
            feats, scale, _ = self._feats(project, k)
            O = torch.stack(others).T
            smap = (feats @ O).amax(-1)
            inside = torch.zeros_like(smap, dtype=torch.bool)
            for v, box in own:
                pos.append(float((v @ O).max()))
                inside[max(0, int(box[1] * scale[1] / p) - 1):int(box[3] * scale[1] / p) + 2,
                       max(0, int(box[0] * scale[0] / p) - 1):int(box[2] * scale[0] / p) + 2] = True
            if (~inside).any():
                neg.append(float(smap[~inside].max()))
        if pos and neg:
            return max(statistics.median(neg), 0.5 * (statistics.median(pos) + statistics.median(neg)))
        return 0.9 * min(pos) if pos else 0.6

    @staticmethod
    def _peaks(smap, threshold, k, radius):
        m = smap[None, None]
        is_max = (F.max_pool2d(m, 3, 1, 1) == m)[0, 0] & (smap >= threshold)
        rc = torch.nonzero(is_max)
        kept = []
        for r, c in rc[torch.argsort(smap[is_max], descending=True)].tolist():
            if all(max(abs(r - a), abs(c - b)) > radius for a, b, _ in kept):
                kept.append((r, c, float(smap[r, c])))
            if len(kept) >= max(k, 1):
                break
        return kept
