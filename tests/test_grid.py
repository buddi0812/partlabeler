"""The grid (image classes, sorting parts) and outline editing through the session, with a stand-in feature model:
images are red, green or blue, and the stand-in's features are their mean colours."""
import base64
import io

import numpy as np
from PIL import Image

from engine import api, masks as M
from engine.api import Session
from engine.project import Project
from tests.test_masks import ring


class ColourFeatures:
    def vectors(self, images, pool="avg"):
        v = np.array([np.asarray(im.convert("RGB"), float).mean((0, 1)) + 1 for im in images])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def last(msgs, kind):
    return next(m for m in reversed(msgs) if m["type"] == kind)


def folder(tmp_path, n=6):
    """3n images: red, green, blue in turn (image i has kind i % 3), each shade a little different."""
    root = tmp_path / "photos"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3 * n):
        colour = np.array([(200, 20, 20), (20, 200, 20), (20, 20, 200)][i % 3]) + rng.integers(0, 40, 3)
        Image.new("RGB", (40, 30), tuple(int(c) for c in colour)).save(root / f"img{i:02d}.png")
    return root


def test_image_classes_grid_sort_suggest_and_odd_ones(tmp_path, monkeypatch):
    monkeypatch.setitem(api._MODELS, "embed", ColourFeatures())
    p = Project.create(tmp_path / "proj", ["red", "green", "blue"], images=folder(tmp_path), task="classify")
    msgs = []
    s = Session(p, msgs.append)
    s.handle({"type": "ready"})
    assert last(msgs, "project")["task"] == "classify" and last(msgs, "project")["formats"] == ["folders", "yolo", "csv"]

    s.handle({"type": "grid", "of": "images"})
    assert len(last(msgs, "grid")["cards"]) == 18 and last(msgs, "grid")["cards"][0]["cls"] is None
    s.handle({"type": "thumbs", "of": "images", "keys": [0, 1]})
    assert set(last(msgs, "thumbs")["src"]) == {0, 1}

    s.handle({"type": "sort", "of": "images"})
    s.job.join()
    g = last(msgs, "groups")
    assert g["k"] == 3 and sorted(sorted(x) for x in g["groups"])[0] == [0, 3, 6, 9, 12, 15]
    s.handle({"type": "sort", "of": "images", "k": 2})            # the slider: instant re-cut
    assert last(msgs, "groups")["k"] == 2

    s.handle({"type": "tag", "of": "images", "keys": [0, 1, 2], "cls": 0})
    s.handle({"type": "tag", "of": "images", "keys": [1, 2], "cls": None})
    s.handle({"type": "tag", "of": "images", "keys": [1], "cls": 1})
    s.handle({"type": "tag", "of": "images", "keys": [2], "cls": 2})
    assert [c["cls"] for c in last(msgs, "cards")["cards"]] == [2]
    s.handle({"type": "suggest_tags"})
    s.job.join()
    first = p.tags()
    assert 3 < len(first) < 18 and all(t["cls"] == i % 3 for i, t in first.items())   # the near ones first
    s.handle({"type": "accept_tags", "keys": [k for k in first if k > 2][:1]})
    assert sorted(p.statuses()).count(4) == 4
    for _ in range(6):                        # accept, suggest again: the named kinds grow until all are classed
        s.handle({"type": "accept_tags"})
        s.handle({"type": "suggest_tags"})
        s.job.join()
    tags = p.tags()                           # the most isolated shades may stay for a person to class
    assert len(tags) >= 12 and all(t["cls"] == i % 3 for i, t in tags.items())

    s.handle({"type": "tag", "of": "images", "keys": [3], "cls": 2})     # a red image called blue
    s.handle({"type": "accept_tags"})
    s.handle({"type": "odd", "of": "images"})
    s.job.join()
    assert last(msgs, "odd")["items"][0][:2] == [3, 0]
    s.handle({"type": "undo"})
    s.handle({"type": "export", "format": "folders"})
    s.job.join()
    assert last(msgs, "notify")["item"]["title"] == "Exported class folders dataset"
    assert not any(m["type"] == "notify" and m["item"]["level"] == "error" for m in msgs)


def test_parts_grid_relabels_a_part_everywhere(tmp_path):
    p = Project.create(tmp_path / "proj", ["part", "bolt"], images=folder(tmp_path, 1))
    p.put(0, 7, 0, (2, 2, 12, 12), "manual")
    p.put(1, 7, 0, (3, 3, 13, 13), "tracked")
    p.put(2, 8, 0, (0, 0, 30, 20), "manual")
    msgs = []
    s = Session(p, msgs.append)
    s.handle({"type": "grid", "of": "parts"})
    assert [(c["key"], c["item"]) for c in last(msgs, "grid")["cards"]] == [(7, 0), (8, 2)]
    s.handle({"type": "tag", "of": "parts", "keys": [7], "cls": 1})
    assert [b["cls"] for k in (0, 1, 2) for b in p.boxes(k)] == [1, 1, 0]
    s.handle({"type": "undo"})
    assert [b["cls"] for k in (0, 1) for b in p.boxes(k)] == [0, 0]


def test_brush_and_eraser_edit_an_outline(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    Image.new("RGB", (80, 60), "gray").save(root / "a.jpg")
    p = Project.create(tmp_path / "proj", ["part"], images=root, task="segment")
    msgs = []
    s = Session(p, msgs.append)
    x, y, png = M.crop_png(ring())
    s.handle({"type": "paint", "item": 0, "cls": 0, "x": x, "y": y, "png": png})
    b = last(msgs, "item")["boxes"][0]
    assert b["box"] == [20.0, 10.0, 61.0, 51.0] and b["mask"][:2] == [20, 10] and b["source"] == "manual"
    half = ring(); half[:, 40:] = False                            # the eraser took the right half
    x, y, png = M.crop_png(half)
    s.handle({"type": "paint", "item": 0, "obj": b["obj"], "cls": 0, "x": x, "y": y, "png": png})
    assert (p.mask(0, b["obj"]) == half).all() and p.boxes(0)[0]["box"][2] == 40.0
    buf = io.BytesIO()
    Image.new("RGBA", (3, 3), (0, 0, 0, 0)).save(buf, "PNG")        # everything erased: the part is deleted
    s.handle({"type": "paint", "item": 0, "obj": b["obj"], "cls": 0, "x": 0, "y": 0,
              "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
    assert p.boxes(0) == []
    s.handle({"type": "undo"})
    assert (p.mask(0, b["obj"]) == half).all()
    s.handle({"type": "settings", "parent": "", "task": "detect"})
    assert p.task == "detect" and last(msgs, "project")["task"] == "detect"
