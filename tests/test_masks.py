"""Outlines: mask storage, page transport, polygons for export, and outline / image-class projects end to end."""
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu

from engine import masks as M
from engine.project import Project, read_classes
from tests.test_project import CLASSES, images  # noqa: F401  (fixture)


def ring(h=60, w=80):
    m = np.zeros((h, w), bool)
    cv2.circle(m.view(np.uint8), (40, 30), 20, 1, -1)
    cv2.circle(m.view(np.uint8), (40, 30), 8, 0, -1)
    return m


def two_pieces(h=60, w=80):
    m = np.zeros((h, w), bool)
    m[5:20, 5:25] = True
    m[35:55, 50:75] = True
    return m


def test_rle_and_page_crops_round_trip():
    m = ring()
    assert (M.decode(M.encode(m)) == m).all()
    x, y, src = M.crop_png(m)
    assert (x, y) == (20, 10) and src.startswith("data:image/png;base64,")
    assert (M.from_png(src, x, y, *m.shape) == m).all()
    assert M.crop_png(np.zeros((4, 4), bool)) is None


def test_clean_fills_small_holes_and_drops_specks():
    m = np.zeros((40, 40), bool)
    m[5:30, 5:30] = True
    m[10:12, 10:12] = False                                      # a 4 px hole
    m[35, 35] = True                                             # a speck
    c = M.clean(m)
    assert c[10:12, 10:12].all() and not c[35, 35] and c[5:30, 5:30].all()


def test_polygons_keep_holes_and_pieces():
    assert [len(holes) for _, holes in M.polygons(ring())] == [1]
    assert len(M.polygons(two_pieces())) == 2
    assert M.iou(M.draw(M.polygons(ring()), (60, 80)), ring()) >= 0.98


def test_one_polygon_for_yolo_bridges_holes_and_pieces():
    for m in (ring(), two_pieces()):
        poly = M.single_polygon(m)
        drawn = np.zeros(m.shape, np.uint8)
        cv2.fillPoly(drawn, [np.round(poly).astype(np.int32)], 1)   # how Ultralytics draws training masks
        assert M.iou(drawn.astype(bool), m) >= 0.95


def test_coco_uses_rle_for_holes_and_polygons_otherwise():
    assert isinstance(M.coco_segmentation(ring()), dict)
    square = np.zeros((60, 80), bool); square[10:50, 10:70] = True
    seg = M.coco_segmentation(square)
    assert isinstance(seg, list) and M.iou(mu.decode(mu.merge(mu.frPyObjects(seg, 60, 80))).astype(bool), square) >= 0.95


def test_cvat_mask_runs_rebuild_the_mask():
    m = two_pieces()
    runs, left, top, w, h = M.cvat_rle(m)
    flat, v = [], 0
    for r in runs:
        flat += [v] * r
        v = 1 - v
    rebuilt = np.zeros_like(m)
    rebuilt[top:top + h, left:left + w] = np.array(flat, bool).reshape(h, w)
    assert (rebuilt == m).all()


def test_outline_project_exports_every_format(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES, images=images, task="segment")
    p.put(0, 1, 1, (0, 0, 1, 1), mask=ring())                    # the box comes from the mask
    assert p.boxes(0)[0]["box"] == [20.0, 10.0, 61.0, 51.0]
    p.put(1, 2, 2, (0, 0, 1, 1), mask=two_pieces())
    p.put(2, 3, 0, (5, 5, 9, 9))                                 # a box without an outline: left out
    for fmt in p.formats:
        res = p.export(fmt, tmp_path / fmt)
        assert (res["images"], res["boxes"], res["no_outline"]) == (3, 2, 1), fmt
    line = (tmp_path / "yolo/labels/a.txt").read_text().split()
    assert line[0] == "1" and len(line) > 7
    coco = json.loads((tmp_path / "coco/annotations.json").read_text())
    assert isinstance(coco["annotations"][0]["segmentation"], dict) and coco["annotations"][0]["area"] == int(ring().sum())
    seg = coco["annotations"][1]["segmentation"]                 # small pieces: polygons would lose too much
    rle = mu.frPyObjects(seg, 60, 80) if isinstance(seg, list) else {**seg, "counts": seg["counts"].encode()}
    assert M.iou(mu.decode(mu.merge(rle) if isinstance(rle, list) else rle).astype(bool), two_pieces()) >= 0.95
    cvat = ET.parse(tmp_path / "cvat/annotations.xml").getroot()
    assert [e.tag for e in cvat.iter() if e.tag in ("box", "polygon", "mask")] == ["mask", "mask"]
    seg = np.array(Image.open(tmp_path / "voc/SegmentationClass/b.png"))
    assert set(np.unique(seg)) == {0, 3} and (seg == 3).sum() == two_pieces().sum()
    assert "right_drl:" in (tmp_path / "voc/labelmap.txt").read_text()
    ls = json.loads((tmp_path / "labelstudio/tasks.json").read_text())
    assert [r["type"] for r in ls[1]["annotations"][0]["result"]] == ["polygonlabels", "polygonlabels"]
    assert "PolygonLabels" in (tmp_path / "labelstudio/labeling_config.xml").read_text()


def test_yolo_polygons_import_as_outlines(tmp_path, images):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels/a.txt").write_text("2 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n1 0.5 0.5 0.2 0.2\n")
    p = Project.create(tmp_path / "proj", CLASSES, images=images, task="segment")
    assert p.import_yolo(tmp_path / "labels")["boxes"] == 2
    masks = p.masks(0)
    assert len(masks) == 1 and M.decode(next(iter(masks.values()))).sum() > 0.2 * 80 * 60
    q = Project.create(tmp_path / "boxes", CLASSES, images=images)               # a box project keeps boxes only
    q.import_yolo(tmp_path / "labels")
    assert q.masks(0) == {} and q.boxes(0)[0]["box"] == [20.0, 15.0, 60.0, 45.0]


def test_image_class_project_exports_folders_split_and_csv(tmp_path, images):
    p = Project.create(tmp_path / "proj", ["ok", "scratch"], images=images, task="classify")
    p.set_tags([0, 2], 1)
    p.set_tags([1], 0, "suggested", [0.8])
    p.set_tags([0], 0, "suggested", [0.9])                       # never replaces a person's class
    assert p.statuses() == [4, 1, 4] and p.tags()[0]["cls"] == 1
    assert p.export("folders", tmp_path / "f")["images"] == 2
    assert sorted(f.name for f in (tmp_path / "f/scratch").iterdir()) == ["a.jpg", "sub__c.jpg"]
    res = p.export("yolo", tmp_path / "y")
    assert res["train"] + res["val"] == 2 and read_classes(tmp_path / "y/classes.txt") == ["ok", "scratch"]
    p.accept_tags([1])
    assert p.export("csv", tmp_path / "c")["images"] == 3
    assert (tmp_path / "c/labels.csv").read_text().splitlines()[0] == "path,class"
