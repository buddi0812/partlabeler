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
