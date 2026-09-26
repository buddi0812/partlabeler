"""Masks for outline (segmentation) projects: storage, transport to the page, and polygons for export.

Masks are the source of truth, stored per object as COCO compressed RLE (about 1 KB); the box is always the
mask's own box. Polygons are made only at export, simplified as far as the redrawn polygon still matches the
mask (IoU >= 0.98, trying 1 px, 0.5 px, then the raw contour). Formats that cannot hold holes or several
pieces get a zero-width bridge (YOLO, as Ultralytics does itself) or a mask instead (COCO RLE, CVAT <mask>).
"""
import base64
import io
import json

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu

TOLERANCES = (1.0, 0.5, 0.0)          # px, tried in turn until the polygon matches the mask
MIN_IOU = 0.98


# ---- storage --------------------------------------------------------------------------------------
def encode(mask: np.ndarray) -> str:
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return json.dumps({"size": r["size"], "counts": r["counts"].decode("ascii")})


def _rle(text: str) -> dict:
    r = json.loads(text)
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def decode(text: str) -> np.ndarray:
    return mu.decode(_rle(text)).astype(bool)


def box_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def clean(mask: np.ndarray, area: int = 16) -> np.ndarray:
    """Fill holes and drop specks smaller than `area` px (SAM 3 video uses 16 for both)."""
    m = mask.astype(np.uint8)
    for value in (0, 1):                                  # 0: holes (specks of background), 1: specks of mask
        src = (m == value).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < area]
        if value == 0:                                    # background touching the border is not a hole
            small = [i for i in small if not (stats[i, 0] == 0 or stats[i, 1] == 0 or
                     stats[i, 0] + stats[i, 2] == m.shape[1] or stats[i, 1] + stats[i, 3] == m.shape[0])]
        if small:
            m[np.isin(lab, small)] = 1 - value
    return m.astype(bool)


# ---- smart brush / smart eraser -----------------------------------------------------------------------
def footprint(points, radius: float, shape) -> np.ndarray:
    """What a round brush of `radius` px covers along a stroke (image-pixel points)."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(points, float)).astype(np.int32).reshape(-1, 2)
    r = max(1, int(round(radius)))
    for p in pts:
        cv2.circle(m, (int(p[0]), int(p[1])), r, 1, -1)
    if len(pts) > 1:
        cv2.polylines(m, [pts], False, 1, thickness=2 * r)
    return m.astype(bool)


def along(points, n: int = 8) -> list[list[float]]:
    """Up to n points spread evenly along a stroke (by length), ends included: prompts for the outline model."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) < 2:
        return pts.tolist()
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if seg[-1] == 0:
        return pts[:1].tolist()
    at = np.linspace(0, seg[-1], min(n, max(2, int(seg[-1] // 4) + 1)))
    return np.c_[np.interp(at, seg, pts[:, 0]), np.interp(at, seg, pts[:, 1])].tolist()


def inner_points(mask: np.ndarray, n: int = 3) -> list[list[float]]:
    """Up to n points deep inside the mask, far apart (its most interior spots): 'this is the part' prompts."""
    if not mask.any():
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    out = []
    for _ in range(n):
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        if dist[y, x] < 1:
            break
        out.append([float(x), float(y)])
        cv2.circle(dist, (int(x), int(y)), max(4, int(dist[y, x] * 3)), 0, -1)
    return out


# ---- the page ---------------------------------------------------------------------------------------
def crop_png(mask: np.ndarray):
    """(x, y, data URL of a 1-bit PNG of the mask's box): what the page tints and draws."""
    box = box_of(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = map(int, box)
    buf = io.BytesIO()
    Image.fromarray(mask[y1:y2, x1:x2]).convert("1").save(buf, "PNG", optimize=True)
    return [x1, y1, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()]


def from_png(data_url: str, x: int, y: int, height: int, width: int) -> np.ndarray:
    """A painted crop from the page (opaque = mask) back into a full-size mask."""
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    img = Image.open(io.BytesIO(raw))
    a = np.array(img.convert("RGBA"))[..., 3] > 127 if img.mode in ("RGBA", "LA", "P") else np.array(img.convert("L")) > 127
    out = np.zeros((height, width), bool)
    x, y = int(x), int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + a.shape[1]), min(height, y + a.shape[0])
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = a[y0 - y:y1 - y, x0 - x:x1 - x]
    return out


# ---- polygons -----------------------------------------------------------------------------------------
def draw(parts, shape) -> np.ndarray:
    """Fill [(outer, [holes])] (point arrays) back into a mask, as OpenCV and Ultralytics draw polygons."""
    m = np.zeros(shape, np.uint8)
    for outer, holes in parts:
        cv2.fillPoly(m, [np.round(outer).astype(np.int32)], 1)
        for h in holes:                                   # a hole's contour runs over mask pixels: keep them
            h = np.round(h).astype(np.int32)
            cv2.fillPoly(m, [h], 0)
            cv2.polylines(m, [h], True, 1)
    return m.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def polygons(mask: np.ndarray):
    """[(outer, [holes])] as float (N, 2) arrays in pixels, each ring with at least 3 points."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    hier = hier[0]
    best = None
    for tol in TOLERANCES:
        rings = [cv2.approxPolyDP(c, tol, True)[:, 0].astype(float) if tol else c[:, 0].astype(float) for c in cs]
        parts = [(rings[i], [rings[j] for j in range(len(cs)) if hier[j][3] == i and len(rings[j]) >= 3])
                 for i in range(len(cs)) if hier[i][3] == -1 and len(rings[i]) >= 3]
        best = parts
        if iou(draw(parts, mask.shape), mask) >= MIN_IOU:
            break
    return best


def _bridge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Join ring b into ring a with a zero-width cut at their closest points (in to b, round it, back out)."""
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(d.argmin()), d.shape)
    return np.concatenate([a[:i + 1], b[j:], b[:j + 1], a[i:]])


def single_polygon(mask: np.ndarray):
    """One polygon for the whole mask (YOLO segmentation labels): pieces and holes bridged in; None if empty."""
    rings = [r for outer, holes in polygons(mask) for r in (outer, *holes)]
    if not rings:
        return None
    poly = rings[0]
    for r in rings[1:]:
        # ponytail: O(n*m) closest-point search per ring; fine for part outlines (hundreds of points)
        poly = _bridge(poly, r)
    return poly


def coco_segmentation(mask: np.ndarray):
    """COCO polygons ([[x, y, ...], ...]) when they redraw the mask well (pycocotools draws them about half a
    pixel small) and there are no holes; otherwise RLE with iscrowd 0 (RF-DETR ignores iscrowd 1)."""
    h, w = mask.shape
    parts = polygons(mask)
    polys = [outer.flatten().tolist() for outer, _ in parts]
    if polys and not any(holes for _, holes in parts):
        redrawn = mu.decode(mu.merge(mu.frPyObjects(polys, h, w))).astype(bool)
        if iou(redrawn, mask) >= 0.95:
            return [[round(v, 2) for v in p] for p in polys]
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [h, w], "counts": r["counts"].decode("ascii")}


def cvat_rle(mask: np.ndarray):
    """CVAT's <mask>: (runs, left, top, width, height), runs row-major over the mask's box, zeros first."""
    x1, y1, x2, y2 = map(int, box_of(mask))
    flat = mask[y1:y2, x1:x2].ravel().astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
    if flat[0]:
        runs = [0] + runs
    return runs, x1, y1, x2 - x1, y2 - y1


def voc_palette() -> list[int]:
    """The Pascal VOC colour map: index 0 black (background), 255 cream (void)."""
    pal = []
    for i in range(256):
        r = g = b = 0
        c = i
        for k in range(8):
            r |= ((c >> 0) & 1) << (7 - k)
            g |= ((c >> 1) & 1) << (7 - k)
            b |= ((c >> 2) & 1) << (7 - k)
            c >>= 3
        pal += [r, g, b]
    return pal
