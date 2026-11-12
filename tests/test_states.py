"""Parts that change look (engine/states.py): the settings text, per-frame states while tracking, and a class change
between states touching one frame only."""
import numpy as np
import pytest
from PIL import Image

from engine import api, states
from engine.api import Session
from engine.project import Project
from tests.test_retrack import FakeTracker

CLASSES = ["left_drl", "right_drl", "left_park", "right_park", "grill"]
LOOK = "BBDDBBDB"                                  # per image: the lamp bright or dim


def test_parse_and_text():
    assert states.parse("drl, park", CLASSES) == [["left_drl", "left_park"], ["right_drl", "right_park"]]
    assert states.parse("left_drl; left_park\n\n", CLASSES) == [["left_drl", "left_park"]]
    for bad in ("drl", "drl, fog", "grill, nope"):
        with pytest.raises(ValueError):
            states.parse(bad, CLASSES)
    groups = states.parse("drl, park\nleft_drl, grill", CLASSES)
    assert states.text(groups, CLASSES) == "drl, park\nleft_drl, grill"


def test_tracking_gives_each_frame_its_state(tmp_path, monkeypatch):
    monkeypatch.setitem(api._MODELS, "track", FakeTracker())
    (tmp_path / "imgs").mkdir()
    for k, look in enumerate(LOOK):
        img = np.full((48, 64, 3), 40, np.uint8)
        img[10:30, 10:30] = 250 if look == "B" else 120
        Image.fromarray(img).save(tmp_path / "imgs" / f"{k:02d}.png")
    p = Project.create(tmp_path / "proj", CLASSES, images=tmp_path / "imgs")
    s = Session(p, lambda m: None)
    s.call({"type": "settings", "parent": "", "states": "drl, park"})
    assert p.meta["states"] == [["left_drl", "left_park"], ["right_drl", "right_park"]]
    s.call({"type": "box", "item": 0, "cls": 0, "box": [10, 10, 30, 30]})     # bright: left_drl
    s.call({"type": "box", "item": 2, "cls": 2, "box": [10, 10, 30, 30]})     # dim: left_park, the person's fix
    s.call({"type": "track", "item": 2, "count": -1, "redo": True})
    got = "".join("D" if p.boxes(k)[0]["cls"] == 2 else "B" for k in range(3, 8))
    assert got == LOOK[3:]                                                    # back to drl when it brightens

    obj = p.boxes(4)[0]["obj"]
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 4})           # another state: this frame only
    assert [p.boxes(k)[0]["cls"] for k in (3, 4, 5)] == [2, 2, 0]
    s.call({"type": "set_class", "obj": obj, "cls": 4, "item": 4})           # another part: every frame
    assert {p.boxes(k)[0]["cls"] for k in range(2, 8)} == {4}


def lamp_project(tmp_path, monkeypatch):
    monkeypatch.setitem(api._MODELS, "track", FakeTracker())
    (tmp_path / "imgs").mkdir(parents=True)
    for k, look in enumerate(LOOK):
        img = np.full((48, 64, 3), 40, np.uint8)
        img[10:30, 10:30] = 250 if look == "B" else 120
        Image.fromarray(img).save(tmp_path / "imgs" / f"{k:02d}.png")
    p = Project.create(tmp_path / "proj", CLASSES, images=tmp_path / "imgs")
    return p, Session(p, lambda m: None)


def tracked_lamp(tmp_path, monkeypatch, states_=None):
    p, s = lamp_project(tmp_path, monkeypatch)
    if states_:
        p.set_meta(states=states_)
    s.call({"type": "box", "item": 0, "cls": 0, "box": [10, 10, 30, 30]})    # drawn bright: left_drl
    s.call({"type": "track", "item": 0, "count": -1})
    return p, s, p.boxes(0)[0]["obj"]


def offer(s):
    n = s.notes.items[0]
    return n["action"]["msg"] if n["title"] == "Did this part change state here?" else None


def test_a_relabel_that_looks_like_a_new_state_is_offered_and_one_click_makes_it(tmp_path, monkeypatch):
    p, s, obj = tracked_lamp(tmp_path, monkeypatch)
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 2, "guess_states": True})   # frame 2 is dim: park
    assert {p.boxes(k)[0]["cls"] for k in range(8)} == {2} and not p.meta.get("states")   # as always, until asked
    s.call(offer(s))                                                         # "Yes, it changed state here"
    assert [p.boxes(k)[0]["cls"] for k in range(4)] == [0, 0, 2, 0] and p.meta["states"] == [["left_drl", "left_park"]]
    s.call({"type": "undo"})                                                 # one step takes back both
    assert {p.boxes(k)[0]["cls"] for k in range(8)} == {0} and p.meta["states"] == []
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 2, "guess_states": True})
    s.call(offer(s))
    s.call({"type": "review", "item": 2})
    s.call({"type": "track", "item": 2, "count": -1, "redo": True})
    assert "".join("D" if p.boxes(k)[0]["cls"] == 2 else "B" for k in range(3, 8)) == LOOK[3:]


def test_the_offer_only_counts_right_after_its_change(tmp_path, monkeypatch):
    p, s, obj = tracked_lamp(tmp_path, monkeypatch)
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 2, "guess_states": True})
    button = offer(s)
    s.call({"type": "review", "item": 5})                                    # something else first
    said = s.call(button)
    assert any("Nothing changed" in x for x in said) and not p.meta.get("states")
    assert {p.boxes(k)[0]["cls"] for k in range(8)} == {2}


def test_no_offer_for_programs_same_look_sides_or_two_groups(tmp_path, monkeypatch):
    p, s, obj = tracked_lamp(tmp_path, monkeypatch)
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 2})           # a program: never asked
    assert offer(s) is None and {p.boxes(k)[0]["cls"] for k in range(8)} == {2}
    s.call({"type": "set_class", "obj": obj, "cls": 0, "item": 0})
    s.call({"type": "set_class", "obj": obj, "cls": 4, "item": 1, "guess_states": True})   # looks as where drawn
    assert offer(s) is None
    s.call({"type": "set_class", "obj": obj, "cls": 0, "item": 0})
    s.call({"type": "set_class", "obj": obj, "cls": 3, "item": 2, "guess_states": True})   # left_drl -> right_park
    assert offer(s) is None
    q, t, o2 = tracked_lamp(tmp_path / "two", monkeypatch, [["left_drl", "left_park"], ["right_park", "grill"]])
    t.call({"type": "set_class", "obj": o2, "cls": 4, "item": 2, "guess_states": True})   # would join two groups
    assert offer(t) is None
    r, u, o3 = tracked_lamp(tmp_path / "side", monkeypatch, [["right_park", "grill"]])
    u.call({"type": "set_class", "obj": o3, "cls": 4, "item": 2, "guess_states": True})   # grill's group: right side
    assert offer(u) is None


def test_a_new_state_can_join_a_group_and_undo_takes_back_only_that(tmp_path, monkeypatch):
    p, s, obj = tracked_lamp(tmp_path, monkeypatch, [["left_drl", "left_park"]])
    p.set_meta(states=[["left_drl", "left_park"], ["right_drl", "right_park"]])
    s.call({"type": "set_class", "obj": obj, "cls": 4, "item": 2, "guess_states": True})   # grill
    s.call(offer(s))
    assert p.meta["states"] == [["left_drl", "left_park", "grill"], ["right_drl", "right_park"]]
    assert p.boxes(3)[0]["cls"] == 0 and p.boxes(2)[0]["cls"] == 4
    s.call({"type": "undo"})
    assert p.meta["states"] == [["left_drl", "left_park"], ["right_drl", "right_park"]] and p.boxes(2)[0]["cls"] == 0


def test_groups_are_found_from_parts_labelled_with_two_classes(tmp_path, monkeypatch):
    p, s = lamp_project(tmp_path, monkeypatch)
    for k, c in ((0, 0), (1, 0), (2, 2), (3, 2)):                            # one part, drl twice, park twice
        p.put(k, 7, c, (10, 10, 30, 30))
    p.put(4, 8, 4, (10, 10, 30, 30))                                         # another part, one class
    assert states.found_groups(p) == [[0, 2]]
    p.put(5, 8, 1, (10, 10, 30, 30))                                         # one frame only: not enough
    assert states.found_groups(p) == [[0, 2]]
    assert states.merge([[1, 2], [3, 4], [2, 3], [5, 6]]) == [[1, 2, 3, 4], [5, 6]]


def test_an_old_offer_leaves_the_live_one_alone_and_a_running_job_waits(tmp_path, monkeypatch):
    p, s, obj = tracked_lamp(tmp_path, monkeypatch)
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 2, "guess_states": True})
    old = offer(s)
    s.call({"type": "undo"})
    s.call({"type": "set_class", "obj": obj, "cls": 2, "item": 2, "guess_states": True})
    live = offer(s)
    assert old["key"] != live["key"]
    said = s.call(old)                                                       # an older notice's button
    assert any("no longer open" in x for x in said) and not p.meta.get("states")
    s.running = True                                                          # a job is running
    assert any("A job is running" in x for x in s.call(live)) and not p.meta.get("states")
    s.running = False
    s.call(live)                                                              # the live one still works
    assert p.meta["states"] == [["left_drl", "left_park"]]
