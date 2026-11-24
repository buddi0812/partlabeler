"""Sharper outline edges when tracking (masks.snap, Project.snap_tracked, S25): a coarse mask like SAM 3's (drawn on a
288 x 288 grid) gets its notch and corners back from the picture; off by default, and every change is logged."""
import json

import cv2
import numpy as np
from PIL import Image

from engine import api
from engine import masks as M
from engine.api import Session
from engine.project import Project

W, H = 320, 180
NOTCHED = np.array([[60, 70], [150, 60], [190, 75], [190, 110], [140, 110], [140, 95], [118, 95], [118, 110], [64, 110]])


def picture():
    img = np.full((H, W, 3), 70, np.uint8)
    truth = np.zeros((H, W), np.uint8)
    cv2.fillPoly(truth, [NOTCHED], 1)
    img[truth.astype(bool)] = (235, 235, 225)
    return cv2.GaussianBlur(img, (0, 0), 1.0), truth.astype(bool)


def coarse(mask, cell=7):
    """What SAM 3 gives on a 1080p frame: the mask seen through a grid of `cell` px, upsampled and thresholded."""
    small = cv2.resize(mask.astype(np.float32), (W // cell, H // cell), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def band(m, d=3):
    return m & ~cv2.erode(m.astype(np.uint8), np.ones((2 * d + 1, 2 * d + 1), np.uint8)).astype(bool)


def test_snap_puts_a_notch_back_where_the_picture_shows_it():
    img, truth = picture()
    rough = coarse(truth)
    sharp = M.snap(img, rough)
    assert iou(band(sharp), band(truth)) >= iou(band(rough), band(truth)) + 0.05
    assert iou(sharp, truth) >= iou(rough, truth)
    notch = (slice(95, 110), slice(118, 140))                                    # the coarse mask fills it in
    assert (sharp & ~truth)[notch].sum() < (rough & ~truth)[notch].sum() / 2
    assert np.array_equal(M.snap(img, np.zeros_like(truth)), np.zeros_like(truth))


class CoarseTracker:
    """Gives every frame SAM 3's kind of mask: the true shape seen through a coarse grid."""
    def track(self, project, start, count, on_item, should_stop, direction=1, masks=False, only=None):
        _, truth = picture()
        m = coarse(truth)
        span = range(start + 1, start + count + 1)
        for k in span:
            on_item(k, {1: (0, M.box_of(m), 0.9, m)})
        return len(span)


def project(tmp_path, monkeypatch):
    monkeypatch.setitem(api._MODELS, "track", CoarseTracker())
    (tmp_path / "imgs").mkdir()
    img, truth = picture()
    for k in range(4):
        Image.fromarray(img).save(tmp_path / "imgs" / f"{k:02d}.png")
    p = Project.create(tmp_path / "proj", ["lamp"], images=tmp_path / "imgs", task="segment")
    p.put(0, 1, 0, (0, 0, 1, 1), mask=truth)
    return p, Session(p, lambda m: None), truth


def test_off_by_default_the_tracked_outline_is_stored_as_given(tmp_path, monkeypatch):
    p, s, truth = project(tmp_path, monkeypatch)
    assert s.project_msg()["sharp_edges"] is False
    s.call({"type": "track", "item": 0, "count": 3})
    assert np.array_equal(p.mask(2, 1), coarse(truth)) and not (p.folder / "edge_snap.jsonl").exists()


def test_switched_on_tracked_outlines_are_sharper_and_logged(tmp_path, monkeypatch):
    p, s, truth = project(tmp_path, monkeypatch)
    s.call({"type": "settings", "sharp_edges": True})
    assert s.project_msg()["sharp_edges"] is True
    s.call({"type": "track", "item": 0, "count": 3})
    got = p.mask(2, 1)
    assert iou(band(got), band(truth)) >= iou(band(coarse(truth)), band(truth)) + 0.05
    assert p.boxes(2)[0]["box"] == M.box_of(got)                                # the box follows the outline
    log = [json.loads(line) for line in (p.folder / "edge_snap.jsonl").read_text().splitlines()]
    assert [(r["item"], r["obj"]) for r in log] == [(1, 1), (2, 1), (3, 1)] and all(r["changed"] > 0 for r in log)
