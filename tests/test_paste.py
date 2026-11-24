"""Copy a part (Ctrl+C) and paste it (Ctrl+V) on another frame at exactly the same place, through the session: the
copy is what the part looked like when copied (on_copy's clip), the same part on another frame of its task, a new
part on the same frame or in another task, outlines included; each paste is one undo step."""
import numpy as np

from engine.api import Session
from engine.project import Project
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


def row(p, item, obj):
    r = p._row(item, obj)
    return None if r is None else ([round(v, 2) for v in r["box"]], r["cls"], r["source"])


def session(p):
    out = []
    return Session(p, out.append), out


def copy(s, out, item, obj):
    s.handle({"type": "copy", "item": item, "obj": obj})
    return [m for m in out if m.get("type") == "clip"][-1]


def paste(s, item, clip):
    s.handle({"type": "paste", "item": item, "clip": clip})


def test_paste_on_another_frame_is_the_same_part_in_the_same_place(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 2, "box": [10, 12, 30, 40], "cls": 1})
    obj = p.boxes(2)[0]["obj"]
    paste(s, 9, copy(s, out, 2, obj))
    assert row(p, 9, obj) == ([10, 12, 30, 40], 1, "manual")
    assert {"type": "select", "item": 9, "obj": obj} in out
    p.put(9, obj, 1, [50, 50, 60, 60], "propagated")                                # tracked somewhere else: moved back
    paste(s, 9, copy(s, out, 2, obj))
    assert row(p, 9, obj) == ([10, 12, 30, 40], 1, "manual") and len(p.boxes(9)) == 1
    s.handle({"type": "undo"})
    assert row(p, 9, obj) == ([50, 50, 60, 60], 1, "propagated")


def test_paste_on_the_same_frame_or_another_task_is_a_new_part(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    p.add_source(video=video, every=1)
    other = p.ranges[p.sources[1]["id"]][0]
    s, out = session(p)
    s.handle({"type": "box", "item": 0, "box": [5, 5, 15, 25], "cls": 0})
    obj = p.boxes(0)[0]["obj"]
    paste(s, 0, copy(s, out, 0, obj))
    twin = [b for b in p.boxes(0) if b["obj"] != obj]
    assert len(twin) == 1 and twin[0]["box"] == [5, 5, 15, 25] and twin[0]["cls"] == 0
    paste(s, other, copy(s, out, 0, obj))
    there = p.boxes(other)
    assert len(there) == 1 and there[0]["obj"] not in (obj, twin[0]["obj"]) and there[0]["box"] == [5, 5, 15, 25]
    s.handle({"type": "undo"})
    s.handle({"type": "undo"})
    assert [b["obj"] for b in p.boxes(0)] == [obj] and p.boxes(other) == []


def test_pasting_between_propagated_copies_makes_a_keyframe_they_follow(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 0, "box": [10, 10, 20, 20], "cls": 0})
    obj = p.boxes(0)[0]["obj"]
    s.handle({"type": "propagate", "item": 0, "objs": [obj], "count": -1})
    s.handle({"type": "box", "item": 18, "obj": obj, "box": [30, 10, 40, 20], "cls": 0})   # a later keyframe
    paste(s, 10, copy(s, out, 0, obj))                  # back where frame 0 has it
    assert row(p, 10, obj) == ([10, 10, 20, 20], 0, "manual")
    assert row(p, 5, obj)[0] == [10, 10, 20, 20] and row(p, 14, obj)[0] == [20, 10, 30, 20]   # half way 10 -> 18
    s.handle({"type": "undo"})
    assert row(p, 10, obj)[2] == "propagated" and row(p, 9, obj)[0] == [20, 10, 30, 20]     # half way 0 -> 18


def test_an_outline_is_pasted_exactly(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1, task="segment")
    s, out = session(p)
    h, w = p.image_size(0)[::-1]
    m = np.zeros((h, w), bool)
    m[8:30, 12:20] = True
    m[8:14, 20:40] = True                                                           # an L, not its box
    p.put(3, 7, 2, (12, 8, 40, 30), mask=m)
    paste(s, 11, copy(s, out, 3, 7))
    assert np.array_equal(p.mask(11, 7), m) and row(p, 11, 7)[:2] == ([12, 8, 40, 30], 2)


def test_the_copy_is_the_place_it_had_when_copied(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 0, "box": [0, 10, 10, 20], "cls": 0})
    obj = p.boxes(0)[0]["obj"]
    s.handle({"type": "propagate", "item": 0, "objs": [obj], "count": -1})
    s.handle({"type": "box", "item": 18, "obj": obj, "box": [18, 10, 28, 20], "cls": 0})
    clip = copy(s, out, 4, obj)                                                     # a propagated copy, [4, 10, 14, 20]
    paste(s, 12, clip)                                                              # a keyframe: frame 4 moves now
    assert row(p, 4, obj)[0] != [4, 10, 14, 20]
    paste(s, 15, clip)                                                              # still where it was when copied
    assert row(p, 12, obj)[0] == row(p, 15, obj)[0] == [4, 10, 14, 20]
    s.handle({"type": "delete", "item": 4, "obj": obj})                             # gone from its frame: still pastes
    paste(s, 2, clip)
    assert row(p, 2, obj) == ([4, 10, 14, 20], 0, "manual")


def test_a_frame_of_another_size_or_a_removed_class_is_refused(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 1, "box": [1, 1, 9, 9], "cls": 0})
    clip = copy(s, out, 1, p.boxes(1)[0]["obj"])
    before = sum(m.get("type") == "status" for m in out)
    paste(s, 4, {**clip, "size": [clip["size"][0] * 2, clip["size"][1]]})
    paste(s, 5, {**clip, "cls": "no such class", "obj": 999})                      # a new part of it
    assert p.boxes(4) == p.boxes(5) == []
    toasts = [m["text"] for m in out if m.get("type") == "toast"]
    assert any("different size" in t for t in toasts) and any("no longer in this project" in t for t in toasts)
    assert sum(m.get("type") == "status" for m in out) == before + 2                # "Saving..." ends on each refusal


def test_the_same_part_keeps_the_class_it_has_now(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 2, "box": [10, 10, 20, 20], "cls": 0})
    obj = p.boxes(2)[0]["obj"]
    s.handle({"type": "propagate", "item": 2, "objs": [obj], "count": -1})
    clip = copy(s, out, 2, obj)
    p.set_class(obj, 1)                                                             # relabelled after copying
    paste(s, 9, clip)
    paste(s, 1, clip)                                                               # not on frame 1 yet: the nearest
    assert row(p, 9, obj) == ([10, 10, 20, 20], 1, "manual") and row(p, 1, obj)[1] == 1


def test_a_copy_from_before_a_restart_or_a_deletion_pastes_as_a_new_part(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 2, "box": [10, 10, 20, 20], "cls": 0})
    obj = p.boxes(2)[0]["obj"]
    s.handle({"type": "propagate", "item": 2, "objs": [obj], "count": -1})
    clip = copy(s, out, 2, obj)
    s.handle({"type": "delete_frames", "items": [0]})                               # frame places moved
    paste(s, 7, clip)
    assert len(p.boxes(7)) == 2                                                     # a new part, the old one untouched
    clip = copy(s, out, 1, obj)
    s2, out2 = session(Project(p.folder))                                           # the app restarted: ids may repeat
    s2.handle({"type": "paste", "item": 9, "clip": clip})
    assert len(s2.p.boxes(9)) == 2
    s2.handle({"type": "paste", "item": 9, "clip": copy(s2, out2, 1, obj)})         # copied again: the same part
    assert len(s2.p.boxes(9)) == 2 and s2.p._row(9, obj)["source"] == "manual"


def test_a_part_gone_from_its_task_pastes_as_a_new_part(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 1, "box": [1, 1, 9, 9], "cls": 0})
    obj = p.boxes(1)[0]["obj"]
    clip = copy(s, out, 1, obj)
    s.handle({"type": "delete", "item": 1, "obj": obj})
    paste(s, 4, clip)
    assert len(p.boxes(4)) == 1 and p.boxes(4)[0]["obj"] != obj and p.boxes(4)[0]["box"] == [1, 1, 9, 9]


def test_the_clip_carries_the_ticket_of_the_tab_that_asked(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1)
    s, out = session(p)
    s.handle({"type": "box", "item": 0, "box": [1, 1, 9, 9], "cls": 0})
    s.handle({"type": "copy", "item": 0, "obj": p.boxes(0)[0]["obj"], "ticket": "tab-a"})
    assert [m for m in out if m.get("type") == "clip"][-1]["ticket"] == "tab-a"
