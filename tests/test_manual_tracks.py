"""Tracks by hand (CVAT's Propagate and tracks), through the session: propagate a part, move it on a later frame (a
keyframe) and the copies in between follow; the part ends where it leaves the picture; undo takes each step back."""
import numpy as np

from engine.api import Session
from engine.project import Project, _fit_mask
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


def box_at(p, item, obj):
    r = p._row(item, obj)
    return None if r is None else ([round(v, 2) for v in r["box"]], r["source"])


def test_propagate_then_keyframes_then_the_part_ends(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)          # 20 frames, one task
    s = Session(p, lambda m: None)
    p.set_reviewed(3)                                                               # a confirmed frame is never changed
    s.handle({"type": "box", "item": 0, "box": [10, 10, 20, 20], "cls": 0})
    obj = p.boxes(0)[0]["obj"]
    s.handle({"type": "propagate", "item": 0, "objs": [obj], "count": -1})
    assert box_at(p, 7, obj) == ([10, 10, 20, 20], "propagated") and box_at(p, 3, obj) is None
    assert len(p.items_with(obj)) == 19

    s.handle({"type": "box", "item": 10, "obj": obj, "box": [30, 20, 40, 30], "cls": 0})   # moved: a keyframe
    assert box_at(p, 10, obj) == ([30, 20, 40, 30], "manual")
    assert box_at(p, 5, obj) == ([20, 15, 30, 25], "propagated")                    # half way between the two
    assert box_at(p, 15, obj) == ([30, 20, 40, 30], "propagated")                   # after the last: where it was left

    s.handle({"type": "undo"})                                                      # one step: the move and the follow
    assert box_at(p, 5, obj) == ([10, 10, 20, 20], "propagated") and box_at(p, 10, obj)[1] == "propagated"
    s.handle({"type": "box", "item": 10, "obj": obj, "box": [30, 20, 40, 30], "cls": 0})

    s.handle({"type": "part_ends", "item": 14, "obj": obj})                         # it leaves the picture after 14
    assert box_at(p, 14, obj) is not None and all(box_at(p, k, obj) is None for k in range(15, 20))
    s.handle({"type": "undo"})
    assert box_at(p, 17, obj) is not None


def test_propagate_never_overwrites_a_persons_own_box(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s = Session(p, lambda m: None)
    s.handle({"type": "box", "item": 0, "box": [10, 10, 20, 20], "cls": 0})
    obj = p.boxes(0)[0]["obj"]
    p.put(4, obj, 0, [1, 1, 5, 5], "manual")                                       # placed by hand on frame 4
    s.handle({"type": "propagate", "item": 0, "count": 6})                          # no objs: every part on the frame
    assert box_at(p, 4, obj) == ([1, 1, 5, 5], "manual") and box_at(p, 6, obj)[1] == "propagated"
    assert box_at(p, 2, obj) == ([10, 10, 20, 20], "propagated")                    # a copy, as placed on frame 0
    assert box_at(p, 7, obj) is None                                                # only the next 6 frames


def test_an_outline_follows_scaled_into_the_new_box():
    m = np.zeros((40, 60), bool)
    m[10:20, 10:20] = True
    out = _fit_mask(m, [10, 10, 20, 20], [30, 5, 50, 25])                            # moved and twice the size
    ys, xs = np.nonzero(out)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (30, 49, 5, 24) and out.sum() == 400
