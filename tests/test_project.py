import json

import av
import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

from engine.project import Project, read_classes

CLASSES = ["left_drl", "right_drl", "grill"]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "car_front.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=25)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(20):
            img = np.full((48, 64, 3), i * 10, np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


@pytest.fixture
def images(tmp_path):
    root = tmp_path / "photos"
    (root / "sub").mkdir(parents=True)
    for name in ("a.jpg", "b.png", "sub/c.jpg"):
        Image.new("RGB", (80, 60), "gray").save(root / name)
    return root


def test_video_project_samples_every_nth_frame(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=5)
    assert [it["frame"] for it in p.items] == [0, 5, 10, 15]
    assert p.items[1]["name"] == "car_front_f000005"
    assert Project(tmp_path / "proj").items == p.items          # reopens identically


def test_image_folder_lists_images_recursively(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES, images=images)
    assert [it["name"] for it in p.items] == ["a", "b", "sub/c"]


def test_statuses_and_tracked_never_overwrite_manual(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    p.put(0, 1, 0, (1, 2, 11, 12))                               # manual anchor
    p.put_tracked(1, {1: (0, (2, 2, 12, 12), 0.9)})
    p.put_tracked(0, {1: (0, (50, 50, 60, 60), 0.9)})            # must not replace the manual box
    assert p.boxes(0)[0]["box"] == [1, 2, 11, 12]
    p.put(2, 5, 2, (0, 0, 5, 5), source="suggested")
    p.set_reviewed(3)
    assert p.statuses() == [3, 2, 1, 4]
    p.put_tracked(1, {1: (0, None, None)})                       # object lost -> tracked box removed
    assert p.boxes(1) == []


def test_yolo_export_import_round_trip(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    p.put(0, 1, 0, (8, 6, 24, 18))
    p.put(1, 2, 2, (30.5, 10, 60, 40))
    p.put(2, 3, 1, (0, 0, 10, 10), source="suggested")           # not exported
    p.set_reviewed(3)                                            # exported as an empty (background) file
    res = p.export("yolo", tmp_path / "out")
    assert res == {"images": 3, "boxes": 2, "folder": str(tmp_path / "out")}
    assert (tmp_path / "out/labels/car_front_f000015.txt").read_text() == ""
    assert read_classes(tmp_path / "out/data.yaml") == CLASSES
    assert read_classes(tmp_path / "out/classes.txt") == CLASSES

    q = Project.create(tmp_path / "proj2", CLASSES, video=video)
    stats = q.import_yolo(tmp_path / "out/labels")
    assert stats == {"files_matched": 3, "files_unmatched": 0, "boxes": 2}
    got = {(b["cls"], tuple(round(v, 3) for v in b["box"])) for k in range(4) for b in q.boxes(k)}
    assert got == {(0, (8, 6, 24, 18)), (2, (30.5, 10, 60, 40))}


def test_import_matches_by_frame_number_when_names_differ(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "cam_1_station_x_f000010.txt").write_text("1 0.5 0.5 0.25 0.5\n")
    (labels / "unrelated_f000099.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert p.import_yolo(labels) == {"files_matched": 1, "files_unmatched": 1, "boxes": 1}
    assert p.boxes(2)[0]["box"] == [24.0, 12.0, 40.0, 36.0]


def test_coco_export_loads_in_pycocotools(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES, images=images)
    p.put(0, 1, 1, (10, 10, 30, 20))
    p.put(2, 2, 2, (0, 0, 80, 60))
    res = p.export("coco", tmp_path / "coco")
    assert res["images"] == 2 and res["boxes"] == 2 and (tmp_path / "coco/images/a.jpg").exists()
    coco = COCO(str(tmp_path / "coco/annotations.json"))
    assert coco.loadCats(2)[0]["name"] == "right_drl"
    assert coco.loadAnns(coco.getAnnIds(imgIds=[1]))[0]["bbox"] == [10, 10, 20, 10]


def test_create_refuses_existing_project(tmp_path, images):
    Project.create(tmp_path / "proj", CLASSES, images=images)
    with pytest.raises(FileExistsError):
        Project.create(tmp_path / "proj", CLASSES, images=images)


def test_every_export_format_writes_the_same_boxes(tmp_path, images):
    import xml.etree.ElementTree as ET
    p = Project.create(tmp_path / "proj", CLASSES, images=images)
    p.put(0, 1, 1, (10, 10, 30, 20))
    p.put(2, 2, 2, (0, 0, 90, 70))                               # clipped to the 80x60 image
    p.put(1, 3, 0, (5, 5, 9, 9), source="suggested")             # never exported
    for fmt in Project.FORMATS:
        res = p.export(fmt, tmp_path / fmt)
        assert (res["images"], res["boxes"]) == (2, 2), fmt
    cvat = ET.parse(tmp_path / "cvat/annotations.xml").getroot()
    boxes = [(b.get("label"), b.get("xbr")) for b in cvat.iter("box")]
    assert boxes == [("right_drl", "30.00"), ("grill", "80.00")]
    voc = ET.parse(tmp_path / "voc/Annotations/sub__c.xml").getroot()
    assert voc.find("object/name").text == "grill" and voc.find("object/bndbox/ymax").text == "60"
    assert (tmp_path / "voc/JPEGImages/a.jpg").exists()
    ls = json.loads((tmp_path / "labelstudio/tasks.json").read_text())
    v = ls[0]["annotations"][0]["result"][0]["value"]
    assert v["rectanglelabels"] == ["right_drl"] and round(v["x"], 3) == 12.5 and round(v["width"], 3) == 25.0
    assert (tmp_path / "coco/annotations.json").exists()
    with pytest.raises(ValueError):
        p.export("pascal", tmp_path / "x")


def test_snapshot_restore_round_trip(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    p.put(0, 1, 0, (1, 1, 5, 5))
    snap = p.snapshot([0, 1])
    p.put(0, 1, 2, (2, 2, 6, 6))
    p.put(1, 2, 1, (0, 0, 3, 3))
    p.set_reviewed(1)
    assert p.restore(snap) == [0, 1]
    assert [(b["cls"], b["box"]) for b in p.boxes(0)] == [(0, [1, 1, 5, 5])]
    assert p.boxes(1) == [] and not p.is_reviewed(1)


def test_flags_size_change_against_the_person_box_and_jumps(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    p.put(0, 1, 0, (10, 10, 20, 20))                              # person's box, area 100
    p.put_tracked(1, {1: (0, (10, 10, 21, 21), 0.9)})             # fine
    p.put_tracked(2, {1: (0, (10, 10, 30, 30), 0.9)})             # 4x the area -> check
    p.put_tracked(3, {1: (0, (40, 30, 51, 41), 0.9)})             # jumped far from frame 2 -> check
    assert p.flags() == [2, 3]
    p.set_reviewed(2)
    assert p.flags() == [3]


def test_mirrored_pairs():
    from engine.project import mirrored_pairs
    assert mirrored_pairs(["left_drl", "right_drl", "grill"])
    assert not mirrored_pairs(["left_drl", "grill"]) and not mirrored_pairs(["bolt", "nut"])


def test_box_overlap_helpers():
    from engine.parent import iou, overlap
    outer, inner = (0, 0, 10, 10), (2, 2, 4, 4)
    assert overlap(outer, inner) == pytest.approx(1.0) and iou(outer, inner) == pytest.approx(0.04)
    assert overlap((0, 0, 1, 1), (5, 5, 6, 6)) == 0
