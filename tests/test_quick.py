"""Quick transfer (engine/quick.py): no training, the dataset's examples matched in new frames. A fake
matcher stands in for DINOv3 (no model weights)."""
import json

import numpy as np

import pytest
from fastapi.testclient import TestClient

from engine import hw
from engine.quick import SourceView, quick_transfer
from tests.test_project import video  # noqa: F401  (fixture)
from tests.test_teach import make_source
from ui import host_fastapi as host


class FakeSuggester:
    """Returns the first example's boxes, scaled to the frame: what matching would find on a perfect day."""

    def __init__(self):
        self.calls, self.thresholds = [], []

    def maxima(self, view, k, max_refs=30):
        return {"lamp": 0.5 + k / 100}

    def suggest(self, view, k, max_refs=30, thresholds=None):
        self.calls.append((k, view.statuses().count(4), view.image(k).size))
        self.thresholds.append(dict(thresholds))
        W, H = view.image(k).size
        W0, H0 = view.image(0).size
        return [(b["cls"], [b["box"][0] * W / W0, b["box"][1] * H / H0, b["box"][2] * W / W0, b["box"][3] * H / H0], 0.8)
                for b in view.boxes(0)]


def test_source_view_spreads_examples_and_reads_yolo_boxes(tmp_path):
    src = make_source(tmp_path / "src", n=20)
    view = SourceView(src, ["left_lamp", "right_lamp", "bolt"], refs=5)
    assert [r["name"] for r in view.refs] == [f"cam_f{k * 5:06d}" for k in (0, 4, 8, 12, 16)]
    assert view.boxes(0) == [{"cls": 0, "source": "manual", "box": [24.0, 48.0, 56.0, 72.0]},
                             {"cls": 1, "source": "manual", "box": [104.0, 48.0, 136.0, 72.0]}]
    a, b = view.add("frame A"), view.add("frame B")
    assert (a, b) == (5, 6) and view.statuses() == [4] * 5 + [0, 0] and view.targets == {5: None, 6: "frame B"}
    assert view.boxes(6) == [] and view.image(6) == "frame B"
    with pytest.raises(ValueError):
        (tmp_path / "empty" / "images").mkdir(parents=True)
        SourceView(tmp_path / "empty", ["a"])


def test_quick_transfer_writes_a_reviewable_run(tmp_path, video, monkeypatch):
    monkeypatch.setattr(hw, "free_gpu_memory", lambda: None)
    src, run, fake = make_source(tmp_path / "src"), tmp_path / "home" / "_teach" / "quick_src", FakeSuggester()
    r = quick_transfer(src, [video], run, refs=4, suggester=fake, fit=False)
    out = run / "labels" / "car_front"
    s = json.loads((out / "summary.json").read_text())
    assert (r["mode"], s["mode"], s["frames"], s["boxes"], s["every"]) == ("quick", "quick", 4, 8, 5)
    assert json.loads((run / "settings.json").read_text())["mode"] == "quick"
    assert (out / "labels" / "car_front_f000005.txt").read_text().startswith("0 0.25")
    assert [c[:2] for c in fake.calls] == [(8, 4), (9, 4), (10, 4), (11, 4)]         # 4 calibration frames first
    # lamps are in every source frame (presence capped at 0.98): the 2nd percentile of the 4 frames' best scores
    assert fake.thresholds[0] == {"lamp": pytest.approx(0.5406)} and r["sources"][str(video)]["thresholds"] == {"lamp": 0.541}

    from tests.test_host import app_client
    with app_client(tmp_path) as c:
        [listed] = c.get("/api/runs").json()
        assert listed["mode"] == "quick" and listed["outputs"][0]["summary"]["mode"] == "quick"
        assert c.post("/api/transfer", json={"run": "quick_src", "sources": [str(video)]}).status_code == 400
        assert c.post("/api/quick", json={"dataset": str(tmp_path / "nope"), "sources": [str(video)]}).status_code == 400
        assert c.post("/api/quick", json={"dataset": str(src), "sources": []}).status_code == 400


class FakeSeg:
    """A 'mask' that is the prompt box shrunk by 2 px on every side, or empty for a box starting at x=0."""

    def set_image(self, img):
        self.size = img.size

    def segment(self, box):
        W, H = self.size
        m = np.zeros((H, W), bool)
        if box[0] > 0:
            m[int(box[1]) + 2:int(box[3]) - 2, int(box[0]) + 2:int(box[2]) - 2] = True
        return m, 0.9


def test_tighten_fits_boxes_to_the_mask_and_keeps_the_match_otherwise():
    from PIL import Image
    from engine.quick import tighten
    img = Image.new("RGB", (100, 80))
    got = tighten(FakeSeg(), img, [(0, [10, 10, 50, 50], 0.8), (1, [0, 10, 40, 50], 0.7)])
    assert got == [(0, [12.0, 12.0, 48.0, 48.0], 0.8), (1, [0, 10, 40, 50], 0.7)]
