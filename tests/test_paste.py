"""Copy and paste a part as in CVAT (Ctrl+C, Ctrl+V, then a click): the page sends where the copy landed - its box and,
in outline projects, its outline crop moved there - and the engine makes a new part of the copied class on that
frame, cut to the frame's edges; one undo step each."""
import numpy as np

from engine import masks as M
from engine.api import Session
from engine.project import Project
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


def session(p):
    out = []
    return Session(p, out.append), out


def moved(crop, dx, dy):
    """The page's move of a copied outline: its PNG crop placed dx, dy further on."""
    return [crop[0] + dx, crop[1] + dy, crop[2]]


def test_a_box_lands_where_it_was_placed_as_a_new_part(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 2, "box": [10, 12, 30, 40], "cls": 1})
    obj = p.boxes(2)[0]["obj"]
    s.handle({"type": "paste", "item": 9, "cls": CLASSES[1], "box": [5, 6, 25, 34]})
    s.handle({"type": "paste", "item": 2, "cls": CLASSES[1], "box": [30, 2, 50, 30]})    # the same frame: another part
    [b9] = p.boxes(9)
    assert b9["obj"] != obj and b9["box"] == [5, 6, 25, 34] and b9["cls"] == 1 and b9["source"] == "manual"
    assert len(p.boxes(2)) == 2 and p.boxes(2)[0]["box"] == [10, 12, 30, 40]
    assert {"type": "select", "item": 2, "obj": p.boxes(2)[1]["obj"]} in out
    s.handle({"type": "undo"})
    s.handle({"type": "undo"})
    assert p.boxes(9) == [] and [b["obj"] for b in p.boxes(2)] == [obj]


def test_a_box_over_the_edge_is_cut_and_one_off_the_frame_is_refused(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    w, h = p.image_size(0)
    s.handle({"type": "paste", "item": 3, "cls": CLASSES[0], "box": [w - 10, -5, w + 10, 15]})
    assert p.boxes(3)[0]["box"] == [w - 10, 0, w, 15]
    before = sum(m.get("type") == "status" for m in out)
    s.handle({"type": "paste", "item": 4, "cls": CLASSES[0], "box": [w + 5, 5, w + 25, 25]})
    s.handle({"type": "paste", "item": 4, "cls": "no such class", "box": [1, 1, 5, 5]})
    assert p.boxes(4) == [] and sum(m.get("type") == "status" for m in out) == before + 2   # "Saving..." ends
    texts = [m["text"] for m in out if m.get("type") == "toast"]
    assert any("over the picture" in t for t in texts) and any("no longer in this project" in t for t in texts)


def test_an_outline_lands_exactly_where_it_was_placed_and_is_cut_at_the_edge(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1, task="segment")
    s, _ = session(p)
    w, h = p.image_size(0)
    m = np.zeros((h, w), bool)
    m[8:30, 12:20] = True
    m[8:14, 20:40] = True                                                           # an L, not its box
    crop = M.crop_png(m)
    s.handle({"type": "paste", "item": 11, "cls": CLASSES[2], "box": [17, 11, 45, 33], "mask": moved(crop, 5, 3)})
    want = np.zeros_like(m)
    want[11:33, 17:25] = True
    want[11:17, 25:45] = True
    assert np.array_equal(p.mask(11, p.boxes(11)[0]["obj"]), want) and p.boxes(11)[0]["box"] == [17, 11, 45, 33]
    s.handle({"type": "paste", "item": 12, "cls": CLASSES[2], "box": [-8, 11, 20, 33], "mask": moved(crop, -20, 3)})
    got = p.mask(12, p.boxes(12)[0]["obj"])
    assert got.sum() == m[:, 20:].sum() and p.boxes(12)[0]["box"][0] == 0        # what hung over the left edge goes
