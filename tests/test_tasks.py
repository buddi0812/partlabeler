"""Projects with several tasks (videos and image folders): numbering, video details, exports per task, linked
frames, lossless frames, and tracking kept inside one task."""
import json
import os

import av
import numpy as np
from PIL import Image

from engine.api import Session
from engine.project import Project, read_classes
from tests.test_project import CLASSES, images, video  # noqa: F401  (fixtures)


def make_video(path, n=12, shade=0):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=10)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(n):
            img = np.full((48, 64, 3), (i * 15 + shade) % 255, np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


def test_tasks_append_and_keep_item_numbers(tmp_path, video, images):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=5)
    assert [t["name"] for t in p.sources] == ["car_front"] and len(p.items) == 4
    info = p.sources[0]["info"]
    assert (info["width"], info["height"], info["kept"], info["codec"]) == (64, 48, 4, "h264") and info["fps"] == 25
    p.put(3, 1, 0, (1, 1, 5, 5))
    second = make_video(tmp_path / "car_front.mp4.copy.mp4", n=12)
    t = p.add_source(video=second, every=4)
    folder = p.add_source(images=images)
    assert [x["name"] for x in p.sources] == ["car_front", "car_front.mp4.copy", "photos"]
    assert p.ranges == {1: (0, 4), 2: (4, 7), 3: (7, 10)} and p.boxes(3)[0]["box"] == [1, 1, 5, 5]   # labels stay put
    assert p.items[4]["name"] == "car_front.mp4.copy_f000000" and p.items[7]["name"] == "photos/a"
    assert (tmp_path / "proj/frames/t2/f000008.jpg").exists() and t["frames"] == "frames/t2"         # numbered from 1
    assert folder["info"]["images"] == 3 and folder["info"]["width"] == 80
    assert Project(tmp_path / "proj").ranges == p.ranges                  # reopens identically
    again = p.add_source(video=second)                                     # same file twice: unique names
    assert again["name"] == "car_front.mp4.copy_2"


def test_a_project_made_before_tasks_opens_as_one_task(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=5)
    for f in (tmp_path / "proj/frames/t1").iterdir():                     # where such projects kept their frames
        f.rename(tmp_path / "proj/frames" / f.name)
    meta = json.loads((tmp_path / "proj/project.json").read_text())
    legacy = {k: meta[k] for k in ("name", "task", "classes", "parent", "kind", "source", "every")}
    (tmp_path / "proj/project.json").write_text(json.dumps(legacy))
    q = Project(tmp_path / "proj")
    assert [it["name"] for it in q.items] == [it["name"] for it in p.items] and q.sources[0]["frames"] == "frames"
    q.add_source(video=make_video(tmp_path / "other.mp4"), every=6)
    assert [t["name"] for t in Project(tmp_path / "proj").sources] == ["car_front", "other"] and len(q.items) == 6


def test_export_one_task_some_or_all_with_linked_frames(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=5)
    p.add_source(video=make_video(tmp_path / "line3.mp4"), every=6)
    p.put(0, 1, 0, (1, 1, 5, 5)); p.put(5, 2, 1, (2, 2, 9, 9))
    assert p.export("yolo", tmp_path / "all")["images"] == 2
    one = p.export("yolo", tmp_path / "one", tasks=["line3"])
    assert one["images"] == 1 and [f.name for f in (tmp_path / "one/images").iterdir()] == ["line3_f000006.jpg"]
    assert os.path.samefile(tmp_path / "one/images/line3_f000006.jpg", p.items[5]["path"])   # a link, not a copy
    assert p.export("coco", tmp_path / "c", tasks=[1])["images"] == 1
    try:
        p.export("yolo", tmp_path / "x", tasks=["nope"])
        raise AssertionError("an unknown task must be refused")
    except ValueError as e:
        assert "no task called nope" in str(e)


def test_lossless_frames_keep_the_decoded_pixels(tmp_path):
    src = make_video(tmp_path / "v.mp4", n=6, shade=7)
    p = Project.create(tmp_path / "proj", CLASSES, video=src, every=2, frame_format="webp")
    with av.open(str(src)) as c:
        decoded = [f.to_image() for k, f in enumerate(c.decode(video=0)) if k % 2 == 0]
    stored = [Image.open(it["path"]).convert("RGB") for it in p.items]
    assert [it["path"].suffix for it in p.items] == [".webp"] * 3
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(decoded, stored))


def test_tracking_stays_inside_the_task(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=5)
    p.add_source(video=make_video(tmp_path / "line3.mp4"), every=6)
    msgs = []
    s = Session(p, msgs.append)
    p.put(3, 1, 0, (1, 1, 5, 5))
    s.handle({"type": "track", "item": 3, "count": 20})                  # last frame of task 1
    assert msgs[-1] == {"type": "toast", "text": "No frames after this one in this task"}
    s.handle({"type": "ready"})
    proj = next(m for m in msgs if m["type"] == "project")
    assert [(t["name"], t["start"], t["end"]) for t in proj["tasks"]] == [("car_front", 0, 4), ("line3", 4, 6)]
    assert s._frame(5) == "frame 2 of line3"
    s.handle({"type": "add_task", "path": str(tmp_path / "missing.mp4")})
    assert "Not found" in msgs[-1]["text"]


def test_rename_subset_and_delete_a_task_keep_the_others_labels(tmp_path, video, images):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=5)            # items 0-3
    p.add_source(video=make_video(tmp_path / "line3.mp4"), every=6)                  # items 4-5
    p.add_source(images=images)                                                       # items 6-8
    p.put(1, 1, 0, (1, 1, 5, 5)); p.put(5, 2, 1, (2, 2, 9, 9)); p.put(7, 3, 2, (3, 3, 8, 8))
    p.set_reviewed(7)
    t = p.update_source(2, name="line 3!", subset="validation")
    assert (t["name"], t["subset"]) == ("line_3_", "validation") and p.items[4]["name"] == "line_3__f000000"
    res = p.remove_source(2)
    assert (res["items"], res["labels"]) == (2, 1) and not (tmp_path / "proj/frames/t2").exists()
    assert [x["name"] for x in p.sources] == ["car_front", "photos"] and len(p.items) == 7
    assert p.boxes(1)[0]["box"] == [1, 1, 5, 5]                                      # before: untouched
    assert p.boxes(5)[0]["box"] == [3, 3, 8, 8] and p.is_reviewed(5)                 # after: moved down by 2
    assert Project(tmp_path / "proj").ranges == {1: (0, 4), 3: (4, 7)}
    p.remove_source(1)                                                                # the first video's frames go,
    assert (tmp_path / "proj/frames").exists() and len(p.items) == 3                # not the folder of the others
    try:
        p.remove_source(3)
        raise AssertionError("the last task must stay")
    except ValueError:
        pass
