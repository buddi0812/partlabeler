"""Moving or resizing an outline by dragging it (its box's handles, or a press on the part): the outline keeps its own
shape - moved, or stretched into the new box - instead of being outlined again; one undo step."""
import numpy as np

from engine.api import Session
from engine.project import Project
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


def test_a_dragged_outline_keeps_its_shape_and_a_resized_one_stretches(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1, task="segment")
    s = Session(p, lambda m: None)
    w, h = p.image_size(0)
    m = np.zeros((h, w), bool)
    m[8:30, 12:20] = True
    m[8:14, 20:40] = True                                                           # an L, not its box
    p.put(3, 7, 2, (12, 8, 40, 30), mask=m)
    s.handle({"type": "box", "item": 3, "obj": 7, "cls": 2, "box": [17, 11, 45, 33], "fit": True})   # moved 5, 3
    assert np.array_equal(p.mask(3, 7), np.roll(np.roll(m, 3, 0), 5, 1)) and p._row(3, 7)["source"] == "manual"
    s.handle({"type": "box", "item": 3, "obj": 7, "cls": 2, "box": [17, 11, 59, 44], "fit": True})   # 1.5 times the size
    got = p.mask(3, 7)
    assert abs(int(got.sum()) - 2.25 * int(m.sum())) < 0.1 * 2.25 * m.sum() and p._row(3, 7)["box"] == [17, 11, 59, 44]
    assert got[11:20, 17:59].all(axis=0).sum() > 30 and not got[11 + 10:, 17 + 13:].any()   # still an L: corner empty
    s.handle({"type": "undo"})
    s.handle({"type": "undo"})
    assert np.array_equal(p.mask(3, 7), m)


def test_after_a_move_m_does_not_bring_back_the_click_outline_at_the_old_place(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1, task="segment")
    s = Session(p, lambda m: None)
    w, h = p.image_size(0)
    m = np.zeros((h, w), bool)
    m[8:20, 10:30] = True
    p.put(2, 5, 0, (10, 8, 30, 20), mask=m)
    s.outlines[(2, 5)] = ([m, m], 0)                                                # what a click left for M
    s.handle({"type": "box", "item": 2, "obj": 5, "cls": 0, "box": [20, 18, 40, 30], "fit": True})
    assert (2, 5) not in s.outlines
    s.handle({"type": "cycle", "item": 2, "obj": 5})                                # nothing to cycle: stays moved
    assert p._row(2, 5)["box"] == [20, 18, 40, 30]
