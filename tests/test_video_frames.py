"""Frames read straight from the video (frame storage "video"): no image files, exactly the pixels of a full decode,
and every feature (listing, exports, backups, task import and delete) works on them; stored frames convert to it
with their labels."""
import numpy as np
import pytest
from PIL import Image

from engine.project import Project, backup_project, extract_frames, restore_project
from tests.test_host import app_client, wait  # noqa: F401
from tests.test_tasks import make_video


def test_frames_read_from_the_video_are_the_decoded_pixels_and_work_everywhere(tmp_path):
    v = make_video(tmp_path / "a.mp4", n=40)
    p = Project.create(tmp_path / "proj", ["bolt"], task="segment")
    t = p.add_source(video=v, every=3, frames="video")
    link = p.folder / "frames" / "t1" / "video.mp4"
    assert t["format"] == "video" and t["info"]["kept"] == 14 and not list(p.folder.rglob("f*.*"))
    assert link.stat().st_nlink >= 2                                   # a hard link to the source: no extra space
    extract_frames(v, tmp_path / "ref", 3, "webp")
    ref = sorted((tmp_path / "ref").glob("f*.webp"))
    assert [it["name"] for it in p.items] == [f"a_{f.stem}" for f in ref]
    for k in (5, 0, 13, 7, 8, 12):                                     # back and forth: seeks, and reading ahead
        assert np.array_equal(np.asarray(p.image(k)), np.asarray(Image.open(ref[k]).convert("RGB")))
    assert p.image_size(3) == (64, 48) and p.thumbnail(3, 32).size == (32, 24)

    p.put(2, 1, 0, (5, 5, 20, 20))
    p.set_reviewed(2)
    p.export("yolo", tmp_path / "ex")
    assert [f.name for f in (tmp_path / "ex" / "images").iterdir()] == ["a_f000006.jpg"]

    backup_project(p.folder, tmp_path / "b.zip")
    q = Project(restore_project(tmp_path / "b.zip", tmp_path / "home"))
    assert len(q.items) == 14 and np.array_equal(np.asarray(q.image(4)), np.asarray(p.image(4)))
    assert q.boxes(2)[0]["box"] == [5, 5, 20, 20]
    r = Project.create(tmp_path / "other", ["bolt"], task="segment")
    assert r.add_tasks_from(tmp_path / "b.zip")["items"] == 14 and np.array_equal(np.asarray(r.image(9)), np.asarray(p.image(9)))

    p.add_source(video=v, every=10, frames="video", name="second")
    p.remove_source(1)
    assert v.exists() and not (p.folder / "frames" / "t1").exists() and len(p.items) == 4


def test_stored_frames_convert_to_the_video_keeping_labels(tmp_path):
    v = make_video(tmp_path / "a.mp4", n=30)
    p = Project.create(tmp_path / "proj", ["bolt"], frame_format="webp")
    p.add_source(video=v, every=4)
    names, before = [it["name"] for it in p.items], np.asarray(p.image(3))
    p.put(1, 1, 0, (1, 1, 9, 9))
    res = p.frames_to_video(1, tmp_path / "old")
    assert res["frames"] == len(names) == 8 and res["freed"] > 0
    assert [it["name"] for it in p.items] == names and p.boxes(1)[0]["box"] == [1, 1, 9, 9]
    assert np.array_equal(np.asarray(p.image(3)), before) and p.sources[0]["format"] == "video"
    assert not list(p.folder.rglob("f*.webp")) and len(list((tmp_path / "old" / "t1").glob("f*.webp"))) == 8
    with pytest.raises(ValueError):
        p.frames_to_video(1, tmp_path / "old")


def test_the_app_makes_video_frame_tasks_by_default_and_serves_their_frames(tmp_path):
    v = make_video(tmp_path / "a.mp4", n=20)
    with app_client(tmp_path) as c:
        assert c.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]}).status_code == 200
        job = c.post("/api/projects/line/tasks", json={"sources": [str(v)], "every": 5}).json()["job"]
        assert wait(c, job)["error"] is None
        t = c.get("/api/projects/line/tasks/1").json()
        assert t["format"] == "video" and t["frames"] == 4
        r = c.get("/items/line/2.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"


def test_the_app_switches_a_project_with_stored_frames_to_its_videos(tmp_path):
    v = make_video(tmp_path / "a.mp4", n=20)
    with app_client(tmp_path) as c:
        c.post("/api/projects", json={"name": "old", "labels": [{"name": "bolt"}], "frame_format": "jpg"})
        assert wait(c, c.post("/api/projects/old/tasks", json={"sources": [str(v)], "every": 5}).json()["job"])["error"] is None
        assert c.get("/api/projects/old/tasks/1").json()["format"] == "jpg"
        res = wait(c, c.post("/api/projects/old/frames-to-video").json()["job"])
        assert res["error"] is None and res["result"]["tasks"] == 1 and res["result"]["freed"] > 0
        assert c.get("/api/projects/old/tasks/1").json()["format"] == "video"
        assert c.post("/api/projects/old/frames-to-video").status_code == 400             # nothing left to switch
        assert (tmp_path / "home" / "_old_frames" / "old" / "t1").is_dir()
        assert [p["name"] for p in c.get("/api/projects").json()] == ["old"]                # the old files are no project
