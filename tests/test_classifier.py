"""engine/classifier.py: Teach for image classes (classify projects), with a tiny stand-in for DINOv3 so it runs
anywhere: the split, the proof, the ONNX export and its parity, and the run as GET /api/v1/models lists it."""
import json

import numpy as np
import torch
from PIL import Image

from engine import classifier
from engine.project import Project


class Tiny(torch.nn.Module):          # stands in for DINOv3: mean colour per channel as the "features"
    def forward(self, x):
        return x.mean(dim=(2, 3))


def test_split_keeps_some_of_every_class_back():
    labels = [0] * 10 + [1] * 3 + [2]
    tr, te = classifier.split(labels)
    assert sorted(tr + te) == list(range(14)) and {labels[i] for i in te} == {0, 1}
    assert 13 in tr                                       # a class with one image is only trained on


def test_a_classify_run_proves_and_exports_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "_backbone", lambda device: (Tiny(), [0.5] * 3, [0.25] * 3))
    imgs = tmp_path / "crops"
    imgs.mkdir()
    for i in range(12):                                     # red crops are "seated", blue ones "missing"
        c = (200 + i, 30, 30) if i % 2 == 0 else (30, 30, 200 + i)
        Image.fromarray(np.full((40, 60, 3), c, np.uint8)).save(imgs / f"c{i:02d}.jpg")
    p = Project.create(tmp_path / "states", ["clip_seated", "clip_missing"], images=imgs, task="classify")
    for k in range(12):
        p.set_tags([k], k % 2, source="manual")
        p.set_reviewed(k)
    rep = classifier.train(p, tmp_path / "_teach" / "run1", project="states")
    assert rep["passed"] and rep["knowledge_score"] == 100.0 and rep["held_out"]["per_class"]["clip_missing"]["recall"] == 1.0
    card = json.loads((tmp_path / "_teach" / "run1" / "export" / "model.json").read_text())
    assert card["task"] == "classify" and card["classes"] == ["clip_seated", "clip_missing"] and card["input"]["shape"] == [1, 3, 224, 224]
    assert card["parity"]["agree"] == card["parity"]["frames"] > 0 and card["parity"]["worst_prob_diff"] < 1e-4
    p.db.close()
