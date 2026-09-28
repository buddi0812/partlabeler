"""Re-tracking after a fix (track with redo) and clearing parts after a frame (engine/api.py), with a stand-in
tracker that carries the start frame's boxes to every frame it is asked for."""
from engine import api
from engine.api import Session
from engine.project import Project
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


class FakeTracker:
    def track(self, project, start, count, on_item, should_stop, direction=1, masks=False):
        parts = {b["obj"]: (b["cls"], b["box"], 0.9) for b in project.boxes(start) if b["source"] != "suggested"}
        span = range(start + 1, start + count + 1) if direction > 0 else range(start - 1, start - count - 1, -1)
        for k in span:
            on_item(k, dict(parts))
        return len(span)


def classes(p, item):
    return sorted(b["cls"] for b in p.boxes(item))


def labeled(tmp_path, video, monkeypatch):
    monkeypatch.setitem(api._MODELS, "track", FakeTracker())
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=2)      # 10 frames
    s = Session(p, lambda m: None)
    s.call({"type": "box", "item": 0, "cls": 0, "box": [1, 1, 10, 10]})
    s.call({"type": "box", "item": 0, "cls": 1, "box": [20, 20, 30, 30]})
    s.call({"type": "track", "item": 0, "count": 5})                         # frames 1–5 get both parts
    return p, s


def test_retrack_carries_a_fix_on_and_keeps_peoples_work(tmp_path, video, monkeypatch):
    p, s = labeled(tmp_path, video, monkeypatch)
    s.call({"type": "box", "item": 3, "cls": 2, "box": [40, 5, 50, 15]})       # a person's own box on frame 3
    s.call({"type": "review", "item": 4})                                     # frame 4 confirmed
    s.call({"type": "delete", "item": 0, "obj": next(b["obj"] for b in p.boxes(0) if b["cls"] == 1)})
    s.call({"type": "track", "item": 0, "count": 2})
    assert classes(p, 2) == [0, 1]                                            # plain tracking leaves the stale part
    said = s.call({"type": "track", "item": 0, "count": -1, "redo": True})
    assert any("Re-tracked 9 frames ahead" in x for x in said)
    assert classes(p, 2) == [0] and classes(p, 3) == [0, 2]                   # stale part gone, own box kept
    assert classes(p, 4) == [0, 1] and classes(p, 9) == [0]                   # confirmed frame untouched; empty filled
    s.call({"type": "undo"})
    assert classes(p, 2) == [0, 1] and classes(p, 9) == []


def test_clear_parts_after_a_frame_by_class_range_and_confirmed(tmp_path, video, monkeypatch):
    p, s = labeled(tmp_path, video, monkeypatch)
    s.call({"type": "review", "item": 4})
    said = s.call({"type": "clear", "item": 1, "classes": [1], "count": -1})
    assert "Cleared 3 parts from 3 frames" in said[-1]
    assert classes(p, 1) == [0, 1] and classes(p, 2) == [0] and classes(p, 4) == [0, 1] and classes(p, 5) == [0]
    s.call({"type": "clear", "item": 1, "count": 1})                          # every class, the next frame only
    assert classes(p, 2) == [] and classes(p, 3) == [0]
    s.call({"type": "clear", "item": 1, "count": -1, "confirmed": True})
    assert classes(p, 4) == [] and all(classes(p, k) == [] for k in range(2, 10)) and classes(p, 1) == [0, 1]
    s.call({"type": "undo"})
    assert classes(p, 4) == [0, 1] and classes(p, 3) == [0]
    assert s.call({"type": "clear", "item": 9, "count": -1}) == ["No frames after this one"]


def test_confirmed_frames_steer_tracking_and_suggest_sees_every_class(tmp_path, video, monkeypatch):
    from engine.suggest import Suggester
    p, s = labeled(tmp_path, video, monkeypatch)
    obj = next(b["obj"] for b in p.boxes(0) if b["cls"] == 0)
    assert p.anchors(obj) == [0]
    s.call({"type": "review", "item": 3})                                     # a person checked the tracked frame 3
    assert p.anchors(obj) == [0, 3]
    s.call({"type": "box", "item": 7, "cls": 2, "box": [2, 2, 8, 8]})         # class 2 only on frame 7
    for k in range(4, 10):
        s.call({"type": "review", "item": k})
    refs = Suggester._refs(p, 1, max_refs=4)                                 # few refs: every class still in
    assert 7 in refs and len(refs) == 4
