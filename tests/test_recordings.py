"""The box's cycle recordings, browsed by station/camera (engine/recordings.py), and starting a project from
them through the existing host API (no real box or recordings API: a fake folder, as the app falls back to)."""
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

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


def test_adding_a_recording_as_a_task_with_a_camera_crop(tmp_path, rec_dir):
    """The Recordings tab's "Crop to camera ROI", end to end through the REST route: a source can be an object
    carrying camera_id/crop, and the stored frames come out already cropped (hawk-ai-internal §13.3)."""
    with app_client(tmp_path) as c:
        d = c.get("/api/recordings").json()
        path = next(r["path"] for r in d["recordings"] if r["camera_id"] == "cam1")
        c.post("/api/projects", json={"name": "cropped", "task": "detect", "labels": [{"name": "part"}], "frame_format": "jpg"})
        r = c.post("/api/projects/cropped/tasks",
                   json={"sources": [{"path": path, "camera_id": "cam1", "crop": [0.25, 0.0, 0.5, 1.0]}], "every": 1})
        wait(c, r.json()["job"])
        proj = c.get("/api/projects/cropped").json()
        t = proj["tasks"][0]
        assert t["camera_id"] == "cam1" and t["crop"] == [0.25, 0.0, 0.5, 1.0]
        assert (t["info"]["width"], t["info"]["height"]) == (32, 48)


def test_teach_passes_task_cameras_for_the_right_task_ids(monkeypatch, tmp_path, rec_dir):
    """A regression check for a real bug caught while building this: the host's /api/teach handler once looked
    up a task by id using Project.task_of(), which actually takes an ITEM index, not a task id - it must use
    the task's own id to find its camera_id/crop, matching whichever task ids the request asks to train on."""
    captured = {}

    def fake_teach(dataset_dir, run_dir, **kw):
        captured.update(kw)
        return {"status": "done", "knowledge_score": 0, "passed": True, "held_out": {}}

    from engine import teach as teach_mod
    monkeypatch.setattr(teach_mod, "teach", fake_teach)
    with app_client(tmp_path) as c:
        d = c.get("/api/recordings").json()
        path1 = next(r["path"] for r in d["recordings"] if r["camera_id"] == "cam1")
        path2 = next(r["path"] for r in d["recordings"] if r["camera_id"] == "cam2")
        c.post("/api/projects", json={"name": "p", "task": "detect", "labels": [{"name": "part"}], "frame_format": "jpg"})
        r = c.post("/api/projects/p/tasks", json={"sources": [path2], "every": 1})                          # task 1: no camera
        wait(c, r.json()["job"])
        r = c.post("/api/projects/p/tasks",
                   json={"sources": [{"path": path1, "camera_id": "cam1", "crop": [0.0, 0.0, 0.5, 0.5]}], "every": 1})
        wait(c, r.json()["job"])                                                                             # task 2: cam1
        r = c.post("/api/teach", json={"project": "p", "tasks": [2]})                                        # only task 2
        assert r.status_code == 200, r.text
        wait(c, r.json()["job"])
        assert set(captured["task_cameras"]) == {"clip1"}                                                    # not task 1's name
        assert captured["task_cameras"]["clip1"] == {"camera_id": "cam1", "crop": [0.0, 0.0, 0.5, 0.5]}
        assert captured["project"] == "p"                             # §12.3: a run's display name is its project


@pytest.fixture
def fake_recordings_api(monkeypatch):
    """A real local HTTP server standing in for the box's recordings API: records the last request's query
    params and headers, and replies with one recording whose "path" is itself (nothing on disk to serve)."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            seen["path"], seen["query"] = u.path, {k: v[0] for k, v in parse_qs(u.query).items()}
            seen["headers"] = dict(self.headers)
            body = json.dumps({"stations": [], "recordings": [{"id": "s1/c1/clip.mp4", "station_id": "s1",
                                                                 "camera_id": "c1", "path": "/dev/null"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen["url"] = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(R, "app_config", lambda: {"recordings_api": seen["url"], "recordings_token": "shh-secret"})
    yield seen
    server.shutdown()


def test_list_all_sends_the_token_and_from_to_before(fake_recordings_api):
    d = R.list_all(station_id="s1", camera_id="c1", limit=10, from_="2026-01-01T00:00:00", to="2026-01-02T00:00:00",
                    before="2026-01-03T00:00:00")
    assert d["recordings"][0]["id"] == "s1/c1/clip.mp4"
    assert fake_recordings_api["headers"]["X-Hawk-Token"] == "shh-secret"
    assert fake_recordings_api["query"] == {"station_id": "s1", "camera_id": "c1", "limit": "10",
                                             "from": "2026-01-01T00:00:00", "to": "2026-01-02T00:00:00",
                                             "before": "2026-01-03T00:00:00"}


def test_resolve_pages_past_the_default_limit_of_100(fake_recordings_api):
    assert R.resolve("s1/c1/clip.mp4") == R.Path("/dev/null")
    assert fake_recordings_api["query"]["limit"] == "500"              # not the list_all() default of 100


def test_list_all_without_a_token_sends_no_header(monkeypatch, fake_recordings_api):
    monkeypatch.setattr(R, "app_config", lambda: {"recordings_api": fake_recordings_api["url"]})   # no token key
    R.list_all()
    assert "X-Hawk-Token" not in fake_recordings_api["headers"]


@pytest.mark.parametrize("crop, ok", [
    ([0.1, 0.2, 0.5, 0.6], True), ([0, 0, 1, 1], True),               # the whole frame is a valid crop
    ([-0.01, 0.2, 0.5, 0.6], False),                                  # x < 0
    ([0.6, 0.2, 0.5, 0.6], False),                                    # x + w > 1
    ([0.1, 0.2, 0, 0.6], False),                                      # w == 0
    ([0.1, 0.2, 0.5], False), ("nope", False), (None, False),
    (["0.1", 0.2, 0.5, 0.6], False), ([True, 0.2, 0.5, 0.6], False),  # hawk refuses strings and bools too
])
def test_valid_crop_matches_hawks_own_import_rule(crop, ok):
    assert R.valid_crop(crop) == ok


def test_crop_rect_floors_the_start_and_ceils_the_end_clipped_to_the_frame():
    assert R.crop_rect([0.1, 0.2, 0.5, 0.6], 100, 50) == (10, 10, 60, 40)
    assert R.crop_rect([0.0, 0.0, 1.0, 1.0], 100, 50) == (0, 0, 100, 50)       # the whole frame, exactly
    assert R.crop_rect([0.333, 0.0, 0.334, 1.0], 10, 10) == (3, 0, 7, 10)      # floor(3.33)=3, ceil(6.67)=7


def test_crop_rect_never_returns_an_empty_cut_at_the_frame_edge():
    """x within valid_crop's 1e-6 slack can round up to the frame width via floor/ceil; hawk clamps x1 <= W-1
    (same for y) so the cut is always at least 1px, and so does this - caught by the app module's 200k-crop
    cross-check against hawk's own crop.py (2026-10-09)."""
    x1, y1, x2, y2 = R.crop_rect([1.0000005, 1.0000005, 0.0000001, 0.0000001], 10, 10)
    assert x1 == 9 and x2 == 10 and y1 == 9 and y2 == 10
