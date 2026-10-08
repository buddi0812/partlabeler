"""The box's cycle recordings, browsed by station/camera (engine/recordings.py), and starting a project from
them through the existing host API (no real box or recordings API: a fake folder, as the app falls back to)."""
import shutil

import pytest

from engine import recordings as R
from tests.test_host import app_client, wait
from tests.test_project import video  # noqa: F401  (fixture)


@pytest.fixture
def rec_dir(tmp_path, video, monkeypatch):
    """<tmp>/recdir/station1/{cam1,cam2}, cam1 with a ROI and one marked cycle on its only clip."""
    d = tmp_path / "recdir"
    (d / "station1" / "cam1").mkdir(parents=True)
    (d / "station1" / "cam2").mkdir(parents=True)
    shutil.copy(video, d / "station1" / "cam1" / "clip1.mp4")
    shutil.copy(video, d / "station1" / "cam2" / "clip2.mp4")
    (d / "station1" / "recordings.json").write_text(
        '{"cameras": {"cam1": {"roi": [1, 2, 30, 20]}}, '
        '"recordings": {"clip1.mp4": {"cycles": [{"t_start_s": 0.1, "t_end_s": 0.5, "verdict": "pass"}]}}}')
    monkeypatch.setattr(R, "app_config", lambda: {"recordings_dir": str(d)})
    return d


def test_list_all_reads_the_fake_folder(rec_dir):
    d = R.list_all()
    assert [s["id"] for s in d["stations"]] == ["station1"]
    cams = {c["id"]: c["roi"] for c in d["stations"][0]["cameras"]}
    assert cams == {"cam1": [1, 2, 30, 20], "cam2": None}
    by_id = {r["id"]: r for r in d["recordings"]}
    assert by_id["station1/cam1/clip1.mp4"]["cycles"] == [{"t_start_s": 0.1, "t_end_s": 0.5, "verdict": "pass"}]
    assert by_id["station1/cam2/clip2.mp4"]["cycles"] == []
    for r in by_id.values():
        assert r["width"] and r["height"] and r["duration_s"] > 0 and r["codec"]

    narrowed = R.list_all(station_id="station1", camera_id="cam2")["recordings"]
    assert [r["id"] for r in narrowed] == ["station1/cam2/clip2.mp4"]


def test_resolve_finds_the_file_and_rejects_escaping_ids(rec_dir):
    p = R.resolve("station1/cam1/clip1.mp4")
    assert p is not None and p.is_file()
    assert R.resolve("station1/cam1/missing.mp4") is None
    assert R.resolve("../../station1/cam1/clip1.mp4") is None        # not 3 parts: rejected before touching disk
    assert R.resolve("station1/../../etc/clip1.mp4") is None         # resolves outside the recordings folder


def test_probe_and_thumbnail(rec_dir):
    p = R.resolve("station1/cam1/clip1.mp4")
    info = R.probe(p)
    assert info["width"] == 64 and info["height"] == 48 and info["duration_s"] > 0
    thumb = R.thumbnail(p, rec_dir.parent / "thumbs")
    assert thumb.exists() and thumb.stat().st_size > 0
    assert R.thumbnail(p, rec_dir.parent / "thumbs") == thumb         # same path again: cached, not regenerated


def test_host_lists_and_serves_a_recording_and_it_becomes_a_task(tmp_path, rec_dir):
    with app_client(tmp_path) as c:
        d = c.get("/api/recordings").json()
        assert d["stations"][0]["id"] == "station1"
        rid = next(r["id"] for r in d["recordings"] if r["camera_id"] == "cam1")

        thumb = c.get(f"/api/recordings/thumb?id={rid}")
        assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
        vid = c.get(f"/api/recordings/video?id={rid}")
        assert vid.status_code == 200 and len(vid.content) > 0

        assert c.get("/api/recordings/thumb?id=../../etc/passwd").status_code == 404

        path = next(r["path"] for r in d["recordings"] if r["id"] == rid)
        c.post("/api/projects", json={"name": "from_recordings", "task": "detect", "labels": [{"name": "part"}]})
        r = c.post("/api/projects/from_recordings/tasks", json={"sources": [path], "every": 1})
        assert r.status_code == 200
        wait(c, r.json()["job"])
        proj = c.get("/api/projects/from_recordings").json()
        assert len(proj["tasks"]) == 1
