"""Teach from a project task and label its other tasks (engine/teach.py task_dataset, engine/transfer.py
label_project, the label_tasks job), with a stand-in detector."""
import json
from types import SimpleNamespace

import numpy as np

from engine import api, detector
from engine.api import Session
from engine.project import Project
from engine.teach import task_dataset
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


def two_tasks(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=2)        # task 1: 10 frames
    p.add_source(video=video, every=2)                                         # task 2: 10 more
    return p, [t["id"] for t in p.sources]


def test_task_dataset_writes_confirmed_frames_as_boxes(tmp_path, video):
    p, (t1, _) = two_tasks(tmp_path, video)
    p.put(0, 1, 0, [1, 1, 11, 11]); p.set_reviewed(0)
    p.set_reviewed(1)                                                          # confirmed empty: a negative
    p.put(2, 2, 1, [5, 5, 15, 15])                                             # not confirmed: left out
    out = tmp_path / "ds"
    assert task_dataset(p, t1, out) == {"frames": 2, "boxes": 1, "empty": 1}
    names = sorted(f.stem for f in (out / "labels").glob("*.txt"))
    assert names == [f"{p.sources[0]['name']}_f000000", f"{p.sources[0]['name']}_f000002"]
    assert (out / "labels" / f"{names[0]}.txt").read_text().split()[0] == "0"
    assert (out / "labels" / f"{names[1]}.txt").read_text() == ""
    assert (out / "classes.txt").read_text().split() == CLASSES
    assert json.loads((out / "spans.json").read_text()) == {p.sources[0]["name"]: [0, 18]}   # the whole task


def fake_run(tmp_path, monkeypatch, boxes):
    """A finished Teach run whose model finds `boxes` [(class, [x1, y1, x2, y2], score)] in every image."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "settings.json").write_text(json.dumps({"classes": CLASSES, "checkpoint": "m.pth", "size": "small",
                                                   "resolution": 640, "threshold": 0.5, "parent": None}))
    monkeypatch.setattr(detector, "load", lambda *a, **k: object())
    monkeypatch.setattr(detector, "predict", lambda model, img, thr: SimpleNamespace(
        xyxy=np.array([b for _, b, _ in boxes], float).reshape(-1, 4), class_id=np.array([c for c, _, _ in boxes]),
        confidence=np.array([s for _, _, s in boxes])))
    return run


def test_an_outline_model_labels_with_its_own_outlines(tmp_path, video, monkeypatch):
    p, (t1, t2) = two_tasks(tmp_path, video)
    run = fake_run(tmp_path, monkeypatch, [(0, [1, 1, 10, 10], 0.9)])
    W, H = p.image_size(p.ranges[t2][0])
    fake = detector.predict

    def with_mask(model, img, thr):
        d = fake(model, img, thr)
        d.mask = np.zeros((1, H, W), bool)
        d.mask[0, 2:6, 3:9] = True                                              # the model's own outline
        return d
    monkeypatch.setattr(detector, "predict", with_mask)
    sam = []
    label = __import__("engine.transfer", fromlist=["label_project"]).label_project
    label(p, run, [t2], outline=lambda item, boxes: sam.append(item) or [None] * len(boxes))
    a = p.ranges[t2][0]
    m = p.mask(a, p.boxes(a)[0]["obj"]).astype(bool)
    assert not sam and m.sum() == 24 and m[2:6, 3:9].all()                     # no SAM 3: the model's pixels


def test_label_tasks_fills_unchecked_frames_and_keeps_part_ids(tmp_path, video, monkeypatch):
    p, (t1, t2) = two_tasks(tmp_path, video)
    a, b = p.ranges[t2]
    p.put(a + 3, 99, 2, [30, 30, 40, 40]); p.set_reviewed(a + 5)               # a person's frame and a confirmed one
    run = fake_run(tmp_path, monkeypatch, [(0, [1, 1, 10, 10], 0.9), (1, [20, 20, 30, 30], 0.8),
                                           (1, [20, 20, 30, 31], 0.5)])        # the same part twice
    s = Session(p, lambda m: None)
    said = s.call({"type": "label_tasks", "run": str(run), "tasks": [t2]})
    assert "Labeled 8 frames with run" in said[-1] and "16 parts to check" in said[-1]
    got = p.boxes(a)
    assert [(x["cls"], x["source"], x["score"]) for x in got] == [(0, "imported", 0.9), (1, "imported", 0.8)]
    assert [x["obj"] for x in p.boxes(a + 1)] == [x["obj"] for x in got]      # the same parts along the frames
    assert [x["cls"] for x in p.boxes(a + 3)] == [2] and p.boxes(a + 5) == []  # people's frames untouched
    assert all(p.boxes(k) == [] for k in range(*p.ranges[t1]))                # other tasks too
    assert not set(range(a, b)) & set(p.flags())                             # sure, steady parts: nothing to check
    p.db.execute("UPDATE boxes SET score=0.5 WHERE item=? AND cls=0", (a + 1,))
    assert a + 1 in p.flags()                                                  # an unsure one is
    s.call({"type": "label_tasks", "run": str(run), "tasks": [t2]})           # again: replaced, not doubled
    assert len(p.boxes(a)) == 2
    s.call({"type": "undo"})
    s.call({"type": "undo"})
    assert p.boxes(a) == [] and [x["cls"] for x in p.boxes(a + 3)] == [2]



def test_confirm_the_tracked_frames_takes_the_pre_labelled_ones(tmp_path, video, monkeypatch):
    """Hawk-AI's "Pre-label with the last model" (2026-10-24): the model's sure parts rank as tracked (status 2), so the
    job menu's "Confirm the tracked frames" (canvas.js: status 2 or 3, not flagged) takes them; an unsure one is
    flagged and left for a person; the confirmed ones train the next run."""
    p, (_, t2) = two_tasks(tmp_path, video)
    a, b = p.ranges[t2]
    run = fake_run(tmp_path, monkeypatch, [(0, [1, 1, 10, 10], 0.9)])
    s = Session(p, lambda m: None)
    s.call({"type": "label_tasks", "run": str(run), "tasks": [t2]})
    p.db.execute("UPDATE boxes SET score=0.4 WHERE item=?", (a + 2,))           # one part the model was unsure of
    st, flagged = p.statuses(), set(p.flags())
    picked = [i for i in range(a, b) if st[i] in (2, 3) and i not in flagged]  # what the page sends
    assert picked == [i for i in range(a, b) if i != a + 2]
    s.call({"type": "review_many", "items": picked, "item": a})
    assert all(p.is_reviewed(i) for i in picked) and not p.is_reviewed(a + 2)
    assert task_dataset(p, t2, tmp_path / "ds")["frames"] == len(picked)
