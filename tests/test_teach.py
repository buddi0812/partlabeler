"""Teach & Transfer logic and the CLI: fast, CPU only, no model weights (the detector is faked)."""
import json

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from typer.testing import CliRunner

from engine import detector, hw
from engine.cli import app
from engine.teach import (analyse, f1_sweep, from_crop, infer_sampling, parse_row, pick_threshold, prepare,
                          source_items, teach, time_blocks, to_crop)
from engine.transfer import frames_to_check, transfer
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)

runner = CliRunner()
BOXES = np.array([[24, 48, 56, 72], [104, 48, 136, 72]], float)


def make_source(root, n=20, every=5):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("left_lamp\nright_lamp\nbolt\n")
    for k in range(n):
        name = f"cam_f{k * every:06d}"
        Image.new("RGB", (160, 120), (k * 12 % 255, 90, 60)).save(root / "images" / f"{name}.jpg")
        (root / "labels" / f"{name}.txt").write_text("0 0.25 0.5 0.2 0.2\n1 0.75 0.5 0.2 0.2\n")
    return root


def fake_predict(model, images, threshold=0.5, batch=8):
    """The two labels of make_source, at the same relative place in any image."""
    def one(im):
        W, H = im.size
        xyxy = np.array([[0.15, 0.4, 0.35, 0.6], [0.65, 0.4, 0.85, 0.6]]) * [W, H, W, H]
        return sv.Detections(xyxy=xyxy, class_id=np.array([0, 1]), confidence=np.array([0.9, 0.8]))
    return [one(im) for im in images] if isinstance(images, list) else one(images)


@pytest.fixture
def fake_detector(monkeypatch):
    monkeypatch.setattr(detector, "load", lambda *a, **k: object())
    monkeypatch.setattr(detector, "predict", fake_predict)
    monkeypatch.setattr(hw, "free_gpu_memory", lambda: None)


def runs(flags):
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def test_time_blocks_are_whole_blocks_spread_over_the_video():
    flags = time_blocks(717, 0.2)
    assert 0.18 < sum(flags) / 717 < 0.22
    (a0, a1), (b0, b1) = runs(flags)                       # two separate blocks of neighbouring frames
    assert a1 < 717 / 2 < b0 and b0 - a1 > 100
    for n in (2, 5, 12, 40):
        assert any(time_blocks(n)) and not all(time_blocks(n))
    with pytest.raises(ValueError):
        time_blocks(10, 0)


def test_prepare_splits_by_time_without_leakage(tmp_path):
    src = make_source(tmp_path / "src", n=30)
    (src / "images" / "cam_f000003.jpg").write_bytes((src / "images" / "cam_f000000.jpg").read_bytes())  # no label
    rec = prepare(src, tmp_path / "ds", held_out=0.2)
    assert set(rec["train"]).isdisjoint(rec["valid"]) and len(rec["train"]) + len(rec["valid"]) == 30
    frames = [int(n[-6:]) for n in rec["valid"]]
    assert len(runs([k * 5 in frames for k in range(30)])) == 2 and min(frames) < 75 < max(frames)
    on_disk = {s: {p.stem for p in (tmp_path / "ds" / s / "images").iterdir()} for s in ("train", "valid")}
    assert on_disk == {"train": set(rec["train"]), "valid": set(rec["valid"])}
    assert (tmp_path / "ds/valid/labels" / f"{rec['valid'][0]}.txt").read_text().count("\n") == 2
    assert prepare(src, tmp_path / "ds", held_out=0.2) == rec                     # complete: reused


def test_items_in_numeric_frame_order(tmp_path):
    (tmp_path / "images").mkdir()
    for n in ("v_f10", "v_f5", "v_f100", "photo"):
        Image.new("RGB", (8, 8)).save(tmp_path / "images" / f"{n}.png")
    assert [it["name"] for it in source_items(tmp_path)] == ["photo", "v_f5", "v_f10", "v_f100"]


def test_yolo_crop_round_trip():
    W, H, crop = 1920, 1080, (300, 200, 1500, 1000)
    row = parse_row("3 0.400000 0.550000 0.100000 0.080000")
    inner = parse_row(to_crop(row, crop, W, H))
    assert inner[0] == 3 and inner[3] == pytest.approx(0.1 * W / 1200, abs=1e-6)
    back = parse_row(from_crop(inner, crop, W, H))
    assert back[0] == 3 and np.allclose(back[1:], row[1:], atol=1e-5)
    assert to_crop(parse_row("0 0.05 0.05 0.04 0.04"), crop, W, H) is None           # outside the parent: dropped
    edge = parse_row(to_crop(parse_row("0 0.16 0.5 0.04 0.1"), crop, W, H))          # straddles it: clipped
    assert edge[1] - edge[3] / 2 == pytest.approx(0, abs=1e-6)


def test_analysis_counts_and_warnings(tmp_path):
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("left_drl\nright_drl\nbolt\nnut\n")
    labels = ["0 0.3 0.5 0.2 0.2\n1 0.7 0.5 0.2 0.2\n2 0.5 0.8 0.01 0.1\n",      # bolt: tiny (6.4 px at 640)
              "0 0.3 0.5 0.2 0.2\n1 0.7 0.5 0.2 0.2\n1 0.7 0.5 0.2 0.2\n",      # exact duplicate
              "0 0.3 0.5 0.2 0.2\n1 0.7 0.5 0.2 0.2\n9 0.5 0.5 0.1 0.1\nabc\n",  # two invalid lines
              "0 0.3 0.5 0.2 0.2\n2 0.5 0.5 0.1 0.1\n",
              ""]                                                                  # checked background
    for k in range(6):
        Image.new("RGB", (160, 120)).save(root / "images" / f"line1_f{k * 10:06d}.jpg")
        if k < 5:
            (root / "labels" / f"line1_f{k * 10:06d}.txt").write_text(labels[k])
    a = analyse(root, resolution=640)
    assert (a["images"], a["labeled"], a["background"], a["unlabeled"], a["invalid_lines"]) == (6, 5, 1, 1, 2)
    pc = a["per_class"]
    assert [pc[c]["boxes"] for c in ("left_drl", "right_drl", "bolt", "nut")] == [4, 3, 2, 0]
    assert pc["right_drl"]["duplicates"] == 1 and pc["bolt"]["tiny"] == 1 and pc["left_drl"]["presence"] == 0.8
    assert pc["left_drl"]["width_px"] == [32.0, 32.0, 32.0]
    assert a["mirrored_pairs"] and a["sampling"] == {"every": 10, "offset": 0, "naming": "{stem}_f{frame:06d}",
                                                     "image_ext": ".jpg"}
    text = "\n".join(a["warnings"])
    for part in ("1 image(s) have no label file", "2 label line(s) are invalid", "cannot learn them: nut\n",
                 "fewer than 10 boxes: left_drl (4), right_drl (3), bolt (2)\n", "helps): bolt (1)\n",
                 "dropped for training: right_drl (1)"):
        assert part in text


def test_flip_only_without_left_right_pairs_and_batch_plan():
    assert "HorizontalFlip" not in detector.aug_config(["left_drl", "right_drl", "grill"])
    assert detector.aug_config(["bolt", "nut"])["HorizontalFlip"] == {"p": 0.5}
    assert "HorizontalFlip" in detector.aug_config(["left_bolt", "nut"])                  # no pair
    assert detector.aug_config(["bolt"], "off") == {}
    assert detector.aug_config(["bolt"], "strong")["HueSaturationValue"]["hue_shift_limit"] == 180
    with pytest.raises(ValueError):
        detector.aug_config(["bolt"], "extreme")
    assert detector.batch_plan("small", 640, 10.5) == (4, 4)                               # as S7 on the 3060
    assert detector.batch_plan("small", 640, None) == (2, 8)                               # CPU
    assert detector.batch_plan("nano", 320, 11.0, n_train=6) == (6, 2)
    with pytest.raises(ValueError):
        detector.check_resolution(600)


def test_threshold_from_f1_sweep():
    targets = [sv.Detections(xyxy=BOXES.copy(), class_id=np.array([0, 1]))] * 3
    pred = sv.Detections(xyxy=np.vstack([BOXES, [[0, 0, 10, 10]]]), class_id=np.array([0, 1, 0]),
                         confidence=np.array([0.8, 0.6, 0.3]))
    sweep = f1_sweep([pred] * 3, targets)
    assert sweep[0.2] == 0.8 and sweep[0.4] == 1.0 and sweep[0.6] == 1.0 and sweep[0.7] == pytest.approx(0.6667, abs=1e-4)
    assert pick_threshold(sweep) == 0.5                               # middle of the 0.35-0.6 plateau
    assert pick_threshold({0.2: 0.5, 0.3: 0.9, 0.4: 0.7}) == 0.3


def test_every_n_from_file_names():
    names = [f"cam_f{i:06d}" for i in (0, 5, 10, 20, 25, 30)]            # f000015 not labeled
    assert infer_sampling(names) == {"every": 5, "offset": 0, "naming": "{stem}_f{frame:06d}"}
    assert infer_sampling(["a_f000003", "a_f000008", "a_f000013"])["offset"] == 3
    assert infer_sampling(["A_f0000", "B_f0000", "A_f0010", "B_f0010"]) == {"every": 10, "offset": 0,
                                                                           "naming": "{stem}_f{frame:04d}"}
    assert infer_sampling(["IMG_001", "IMG_002"]) == {"every": None, "offset": None, "naming": "{name}"}


def test_frames_to_check():
    rows = [("f0", ["a", "b", "c", "c"]), ("f1", ["a", "b", "c", "c"]), ("f2", ["a", "b", "c"]),
            ("f3", ["a"]), ("f4", ["a", "c", "c", "c"])]
    got = frames_to_check(rows, {"a": 0.98, "b": 0.95, "c": 0.3})
    assert got == [{"image": "f3", "boxes": 1, "missing": ["b"]}, {"image": "f4", "boxes": 4, "missing": ["b"]}]
    assert frames_to_check([], {"a": 1.0}) == []


def test_teach_end_to_end_with_a_fake_detector(tmp_path, monkeypatch, fake_detector):
    calls = []

    def fake_train(dataset_dir, out_dir, size, epochs, resolution, aug, resume, progress, should_stop):
        calls.append(resume)
        (out_dir / "checkpoint_best_total.pth").write_bytes(b"x")
        return out_dir / "checkpoint_best_total.pth"

    monkeypatch.setattr(detector, "train", fake_train)
    src, run = make_source(tmp_path / "src"), tmp_path / "run"
    r = teach(src, run, epochs=1, resolution=320)
    assert r["passed"] and r["knowledge_score"] == 100.0 and r["held_out"]["mAP50"] == 1.0
    assert all(v["drop"] == 0 for v in r["held_out"]["stress"].values())
    s = json.loads((run / "settings.json").read_text())
    assert (s["classes"], s["every"], s["naming"], s["checkpoint"]) == (
        ["left_lamp", "right_lamp", "bolt"], 5, "{stem}_f{frame:06d}", "model/checkpoint_best_total.pth")
    assert s["presence"] == {"left_lamp": 1.0, "right_lamp": 1.0, "bolt": 0.0}
    assert "PASS" in (run / "report.md").read_text() and json.loads((run / "report.json").read_text())["passed"]
    teach(src, run, epochs=1, resolution=320)                            # complete: no second training
    assert calls == [None]
    (run / "model/trained.json").unlink()                                # interrupted run: resumes
    (run / "model/last.ckpt").write_bytes(b"x")
    teach(src, run, epochs=1, resolution=320)
    assert calls[-1].name == "last.ckpt" and not (run / "model/last.ckpt").exists()


def test_transfer_then_review_with_a_fake_detector(tmp_path, video, fake_detector):
    run = tmp_path / "run"
    run.mkdir()
    (run / "settings.json").write_text(json.dumps({
        "classes": CLASSES, "parent": None, "size": "small", "resolution": 640, "threshold": 0.5, "every": 5,
        "naming": "{stem}_f{frame:06d}", "image_ext": ".jpg", "checkpoint": "model/x.pth",
        "presence": {"left_drl": 1.0, "right_drl": 0.95, "grill": 0.2}}))
    r = transfer(run, [video], tmp_path / "auto")
    out = tmp_path / "auto" / "car_front"
    s = json.loads((out / "summary.json").read_text())
    assert (s["frames"], s["boxes"], s["every"], s["status"]) == (4, 8, 5, "done")
    assert s["boxes_per_class"] == {"left_drl": 4, "right_drl": 4, "grill": 0} and s["frames_to_check"] == []
    assert sorted(p.name for p in (out / "labels").iterdir())[1] == "car_front_f000005.txt"
    assert (out / "preview.mp4").stat().st_size > 0 and (out / "classes.txt").read_text().split() == CLASSES
    assert r["sources"][str(video)]["folder"] == str(out)
    res = runner.invoke(app, ["review", str(tmp_path / "proj"), "--video", str(video), "--labels", str(out / "labels"),
                              "--classes", str(out / "classes.txt")])
    assert res.exit_code == 0, res.output
    assert "4 frames (1 in 5), 8 boxes from 4 label files" in res.output


@pytest.mark.parametrize("cmd", ["app", "new", "export", "teach", "transfer", "review", "models", "doctor"])
def test_cli_help(cmd):
    res = runner.invoke(app, [cmd, "--help"])
    assert res.exit_code == 0 and "Usage" in res.output


def test_cli_new_and_export(tmp_path, video):
    classes = tmp_path / "classes.txt"
    classes.write_text("\n".join(CLASSES) + "\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "car_front_f000005.txt").write_text("1 0.5 0.5 0.25 0.5\n")
    proj = tmp_path / "proj"
    res = runner.invoke(app, ["new", str(proj), "--video", str(video), "--classes", str(classes), "--labels", str(labels)])
    assert res.exit_code == 0, res.output
    assert "4 frames, 3 classes" in res.output and "imported 1 boxes from 1 label files" in res.output
    res = runner.invoke(app, ["export", str(proj), "--out", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out/labels/car_front_f000005.txt").read_text().startswith("1 0.5")
    res = runner.invoke(app, ["export", str(proj), "--format", "coco"])
    assert res.exit_code == 0 and len(list((proj / "exports").glob("coco_*/annotations.json"))) == 1
    res = runner.invoke(app, ["new", str(tmp_path / "p2"), "--video", str(video), "--images", str(tmp_path),
                              "--classes", str(classes)])
    assert res.exit_code != 0
