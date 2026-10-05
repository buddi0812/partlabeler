"""Annotate by text (engine/textlabel.py): phrases, scopes, and the frame loop with a stand-in for SAM 3."""
import numpy as np
import pytest

from engine import textlabel
from engine.project import Project
from tests.test_project import video  # noqa: F401  (fixture)

CAR, WHEEL = [5.0, 5.0, 40.0, 30.0], [10.0, 20.0, 18.0, 28.0]   # the wheel sits inside the car


class FakeSam3:
    """detect_texts as engine.parent.ConceptFinder: a car and a wheel on every frame (masks: their boxes filled)."""
    def __init__(self):
        self.calls = 0

    def detect_texts(self, image, texts, threshold=0.5, masks=False):
        self.calls += 1
        W, H = image.size

        def hit(box, score):
            m = None
            if masks:
                m = np.zeros((H, W), bool)
                m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
            return box, score, m
        found = {"car": [hit(CAR, 0.9)], "wheel": [hit(WHEEL, 0.8)]}
        return {t: found.get(t, []) for t in texts}


def test_phrases_are_split_named_and_deduplicated():
    assert textlabel.parse("car, truck  cabin;wheel\n car") == [("car", "car"), ("truck cabin", "truck cabin"),
                                                                ("wheel", "wheel")]
    assert textlabel.parse([{"text": "wheels", "label": "wheel"}, "bus"]) == [("wheels", "wheel"), ("bus", "bus")]
    assert textlabel.parse("  ,; ") == []


def test_scopes(tmp_path, video):
    p = Project.create(tmp_path / "p", ["car"], video=video, every=2)
    p.add_source(video=video, every=4)                         # a second task: frames 10-14
    assert textlabel.scope_items(p, 3, "frame") == [3]
    assert textlabel.scope_items(p, 3, "rest") == list(range(3, 10))
    assert textlabel.scope_items(p, 12, "task") == list(range(10, 15))
    assert textlabel.scope_items(p, 3, "project") == list(range(15))
    with pytest.raises(ValueError, match="scope"):
        textlabel.scope_items(p, 3, "everything")


def test_annotating_frames_links_ids_keeps_nested_parts_and_never_doubles(tmp_path, video):
    p = Project.create(tmp_path / "p", ["car", "wheel"], video=video, every=2)
    p.put(2, 900, 0, [1, 1, 3, 3])                             # a person's part with an id the run must not reuse
    sam = FakeSam3()
    res = textlabel.annotate(p, [0, 1, 2, 3], [("car", 0), ("wheel", 1)], sam)
    assert res == {"frames": 4, "parts": 8, "per_class": {"car": 4, "wheel": 4}, "stopped": False}
    assert sam.calls == 4                                      # one image encoding per frame for every phrase
    cars = {k: [b for b in p.boxes(k) if b["cls"] == 0 and b["source"] == "imported"] for k in range(4)}
    assert len({c[0]["obj"] for c in cars.values()}) == 1      # the same car from frame to frame
    assert {b["obj"] for b in p.boxes(2)} >= {900} and len(p.boxes(2)) == 3
    wheel = next(b for b in p.boxes(0) if b["cls"] == 1)
    assert wheel["source"] == "imported" and wheel["score"] == 0.8 and wheel["box"] == WHEEL
    again = textlabel.annotate(p, [0, 1, 2, 3], [("car", 0), ("wheel", 1)], sam)
    assert again["parts"] == 0 and len(p.boxes(0)) == 2        # a second run adds nothing twice
    stopped = textlabel.annotate(p, [4, 5], [("car", 0)], sam, should_stop=lambda: True)
    assert stopped["stopped"] and stopped["frames"] == 0


def test_outline_projects_keep_sam3s_masks(tmp_path, video):
    p = Project.create(tmp_path / "p", ["car"], video=video, every=2, task="segment")
    textlabel.annotate(p, [0], [("car", 0)], FakeSam3(), outlines=True)
    obj = p.boxes(0)[0]["obj"]
    assert obj in p.masks(0)
