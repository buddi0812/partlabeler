"""The integration API (ui/v1.py) and its Python client (client/partlabeler_client.py), end to end (no GPU models: a
stand-in for SAM 3's text search)."""
import base64
import importlib.util
import io
import socket
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from engine import api
from tests.test_host import app_client, wait  # noqa: F401
from tests.test_project import video  # noqa: F401  (fixture)
from tests.test_textlabel import CAR, WHEEL, FakeSam3
from ui import host_fastapi as host

ROOT = Path(__file__).resolve().parents[1]


def jpeg(w=64, h=48) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def v1(tmp_path, monkeypatch):
    monkeypatch.setitem(api._MODELS, "concept", FakeSam3())
    with app_client(tmp_path) as c:
        token = c.post("/api/me/tokens", json={"label": "line"}).json()["token"]
        c.cookies.clear()
        c.headers["Authorization"] = f"Bearer {token}"
        yield c, token


def test_health_needs_no_token_and_the_rest_does(tmp_path):
    with app_client(tmp_path, signed_in=False) as c:
        h = c.get("/api/v1/health").json()
        assert h["status"] == "ok" and h["api"] == "v1"
        assert c.post("/api/v1/projects", json={"name": "x"}).status_code == 401


def test_annotate_one_image_without_storing_anything(v1):
    c, _ = v1
    r = c.post("/api/v1/annotate", json={"image": base64.b64encode(jpeg()).decode(), "objects": "car, wheel"}).json()
    assert r["width"] == 64 and [(p["label"], p["box"], p["score"]) for p in r["parts"]] == [("car", CAR, 0.9),
                                                                                             ("wheel", WHEEL, 0.8)]
    m = c.post("/api/v1/annotate", files={"file": ("f.jpg", jpeg(), "image/jpeg")},
               data={"objects": "car", "outlines": "true"}).json()
    assert [p["label"] for p in m["parts"]] == ["car"] and len(m["parts"][0]["outline"]) >= 4
    assert c.post("/api/v1/annotate", json={"image": "bm90IGFuIGltYWdl", "objects": ["car"]}).status_code == 400
    assert c.post("/api/v1/annotate", json={"image": base64.b64encode(jpeg()).decode(), "objects": " , "}).status_code == 400
    assert c.get("/api/v1/projects/nope").status_code == 404


def test_upload_annotate_by_text_read_and_export(v1, video):
    c, _ = v1
    assert c.post("/api/v1/projects", json={"name": "line", "type": "detect", "labels": ["car"]}).json() == {"name": "line"}
    job = c.post("/api/v1/projects/line/uploads", files={"file": ("cam0.mp4", video.read_bytes(), "video/mp4")},
                 data={"every": "2", "name": "cam0"}).json()
    done = wait(c, job["job"])
    assert done["error"] is None and c.get(f"/api/v1/jobs/{job['job']}").json()["finished"]
    proj = c.get("/api/v1/projects/line").json()
    assert proj["tasks"][0]["name"] == "cam0" and proj["tasks"][0]["frames"] == 10
    shots = c.post("/api/v1/projects/line/uploads",
                   files=[("files", (f"{k}.jpg", jpeg(), "image/jpeg")) for k in range(3)]).json()
    assert wait(c, shots["job"])["error"] is None and len(c.get("/api/v1/projects/line").json()["tasks"]) == 2

    c.post("/api/tools/confirm_frames", json={"project": "line", "items": [4]})
    r = c.post("/api/v1/projects/line/annotate-text", json={"objects": ["car", "wheel"], "scope": "rest", "task": 1,
                                                             "frame": 3}).json()
    assert r["busy"] is False and r["labels"] == ["car", "wheel"]               # "wheel" is a new label
    frames = {f["item"]: [p["label"] for p in f["parts"]]
              for f in c.get("/api/v1/projects/line/annotations?task=1").json()["frames"]}
    assert sorted(k for k, v in frames.items() if v) == [2, 3, 5, 6, 7, 8, 9]   # frame 3 on, the confirmed one left
    assert frames[2] == ["car", "wheel"] and frames[4] == []
    one = c.get("/api/v1/projects/line/annotations?item=2&outlines=false").json()["frames"][0]["parts"][0]
    assert one["source"] == "imported" and one["score"] == 0.9
    st = c.get("/api/v1/projects/line/status").json()
    assert st["busy"] is False
    bad = c.post("/api/v1/projects/line/annotate-text", json={"objects": ["car"], "scope": "galaxy", "item": 0})
    assert bad.status_code == 400 and "scope" in bad.json()["detail"]

    ex = c.post("/api/v1/projects/line/export", json={"format": "yolo", "tasks": [1]}).json()
    assert wait(c, ex["job"])["error"] is None
    z = c.get(f"/api/v1/projects/line/exports/{ex['job']}.zip")
    assert z.status_code == 200 and any(n.endswith(".txt") for n in zipfile.ZipFile(io.BytesIO(z.content)).namelist())
    assert c.get("/api/v1/projects/line/exports/nope.zip").status_code == 404
    names = [t["name"] for t in c.get("/api/v1/tools").json()["tools"]]
    assert "annotate_text" in names and "add_tasks" in names


def test_other_machines_reach_only_the_integration_api(v1):
    c, token = v1
    c.post("/api/v1/projects", json={"name": "far", "labels": ["car"]})
    auth = {"Authorization": f"Bearer {token}"}
    with TestClient(host.app, client=("10.1.2.3", 5000)) as far:
        assert far.get("/api/v1/health").status_code == 200
        assert far.get("/api/projects", headers=auth).status_code == 403                    # the app's own API
        assert far.get("/", headers=auth).status_code == 403
        assert far.post("/api/auth/login", json={"username": "a", "password": "b"}).status_code == 403
        assert far.get("/api/v1/projects/far", headers=auth).status_code == 200
        r = far.post("/api/v1/tools/add_tasks", json={"project": "far", "sources": ["C:/"]}, headers=auth)
        assert r.status_code == 403 and "uploads" in r.json()["detail"]
        assert "add_tasks" not in [t["name"] for t in far.get("/api/v1/tools", headers=auth).json()["tools"]]
        assert far.get("/api/v1/projects/far").status_code == 401                          # no token
    proxied = c.get("/api/projects", headers={"X-Forwarded-For": "203.0.113.9"})          # relayed by a local proxy
    assert proxied.status_code == 403


def _client_module():
    spec = importlib.util.spec_from_file_location("partlabeler_client", ROOT / "client" / "partlabeler_client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_python_client_against_a_running_server(v1, video, tmp_path):
    _, token = v1
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(host.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    t0 = time.time()
    while not server.started and time.time() - t0 < 10:
        time.sleep(0.05)
    try:
        PL = _client_module()
        pl = PL.PartLabeler(f"http://127.0.0.1:{port}", token=token, retries=0)
        assert pl.health()["status"] == "ok"
        (tmp_path / "f.jpg").write_bytes(jpeg())
        assert [p["label"] for p in pl.annotate(tmp_path / "f.jpg", "car, wheel")["parts"]] == ["car", "wheel"]
        pl.create_project("cli", labels=["car"])
        pl.wait(pl.upload("cli", video, every=5, name="v")["job"])
        pl.annotate_text("cli", ["car"], scope="task", task=1)
        assert pl.wait_idle("cli")["busy"] is False
        assert len(pl.annotations("cli", task=1)["frames"]) == 4
        out = pl.export("cli", "coco", path=tmp_path / "cli.zip")
        assert zipfile.is_zipfile(out["path"])
        with pytest.raises(PL.PartLabelerError) as e:
            pl.project("missing")
        assert e.value.status == 404
        assert PL.PartLabeler(f"http://127.0.0.1:{port}", token="plt_wrong", retries=0).health()["status"] == "ok"
        with pytest.raises(PL.PartLabelerError) as e:
            PL.PartLabeler(f"http://127.0.0.1:{port}", token="plt_wrong", retries=0).project("cli")
        assert e.value.status == 401
    finally:
        server.should_exit = True


def test_import_a_handed_over_project_once_per_source(v1):
    """Hawk-AI's SOP studio hands over frames and its boxes (app-015-plan (b)): one project per station + demo
    version, images as a task, boxes as parts to check; a second hand-over adds only the new images."""
    c, _ = v1
    b64 = base64.b64encode(jpeg()).decode()
    body = {"name": "Door trim", "source": {"station": "l2-s4", "demo_version": 3}, "classes": ["clip", "door panel"],
            "images": [{"name": "pin-1.jpg", "jpeg_b64": b64}, {"name": "run-2.jpg", "jpeg_b64": b64}],
            "boxes": [{"image": "pin-1.jpg", "class": "door panel", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}]}
    r = c.post("/api/v1/projects/import", json=body).json()
    assert (r["project_id"], r["images_added"], r["boxes_added"], r["task"]) == ("Door trim", 2, 1, 1)
    assert r["url"].endswith("/p/Door trim")
    p = host.project("Door trim")
    assert p.classes == ["clip", "door panel"] and sorted(p.statuses()) == [0, 2]      # the box: a part to check
    again = c.post("/api/v1/projects/import", json={**body, "name": "Renamed", "classes": ["clip", "hinge"],
                                                    "images": body["images"] + [{"name": "run-3.jpg", "jpeg_b64": b64}]}).json()
    assert (again["project_id"], again["images_added"], again["task"]) == ("Door trim", 1, 2)
    assert host.project("Door trim").classes == ["clip", "door panel", "hinge"]
    other = c.post("/api/v1/projects/import", json={**body, "source": {"station": "l2-s4", "demo_version": 4}}).json()
    assert other["project_id"] == "Door trim 2"
    bad = {**body, "source": {"station": "x", "demo_version": 1}, "boxes": [{**body["boxes"][0], "class": "nope"}]}
    assert c.post("/api/v1/projects/import", json=bad).status_code == 400
    assert not (host.home() / "Door trim 3").exists()                     # refused before anything is stored
    assert c.post("/api/v1/projects/import", json={**body, "images": [{"name": "a.jpg", "jpeg_b64": "bm90"}]}).status_code == 400


def test_import_with_hawk_apis_service_key_lands_in_the_default_projects_folder(tmp_path, monkeypatch):
    """hawk-api holds only the box's models_token: with it (local only) and no account, the hand-over goes into the
    app's default projects folder, the one every account sees unless it picked its own."""
    import engine.accounts as accounts_mod
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {"models_token": "right-token"})
    body = {"name": "Door trim", "source": {"station": "l2-s4", "demo_version": 3}, "classes": ["clip"],
            "images": [{"name": "pin-1.jpg", "jpeg_b64": base64.b64encode(jpeg()).decode()}], "boxes": []}
    with app_client(tmp_path, signed_in=False) as c:
        url = "/api/v1/projects/import"
        assert c.post(url, json=body).status_code == 403
        assert c.post(url, json=body, headers={"X-Hawk-Token": "wrong"}).status_code == 403
        assert c.post(url, json=body, headers={"X-Hawk-Token": "right-token", "X-Forwarded-For": "10.0.0.5"}).status_code == 403
        r = c.post(url, json=body, headers={"X-Hawk-Token": "right-token"})
        assert r.status_code == 200 and r.json()["images_added"] == 1
        assert (tmp_path / "home" / "Door trim" / "project.json").exists()
        assert c.get(url).status_code == 401                 # only that exact POST is open: a project named "import" isn't
        assert c.get("/api/v1/train/queue", headers={"X-Hawk-Token": "right-token"}).status_code == 200
        assert c.get("/api/v1/train/queue").status_code == 403
        assert c.post("/api/v1/projects/Door trim/train", json={}, headers={"X-Hawk-Token": "right-token"}).status_code == 400
        assert c.get("/api/v1/projects/Door trim").status_code == 401        # the rest still needs an account


def test_import_one_project_per_task_type_and_classify_takes_each_images_class(v1):
    c, _ = v1
    b64 = base64.b64encode(jpeg()).decode()
    src = {"station": "l2-s4", "demo_version": 3}
    det = c.post("/api/v1/projects/import", json={"name": "D4", "source": src, "classes": ["clip"],
                                                  "images": [{"name": "a.jpg", "jpeg_b64": b64}]}).json()
    seg = c.post("/api/v1/projects/import", json={"name": "D4 outlines", "type": "segment", "source": src, "classes": ["clip"],
                                                  "images": [{"name": "a.jpg", "jpeg_b64": b64}]}).json()
    assert det["project_id"] == "D4" and seg["project_id"] == "D4 outlines" and seg["images_added"] == 1
    assert host.project("D4 outlines").task == "segment"
    body = {"name": "D4 states", "type": "classify", "source": src, "classes": ["clip_seated", "clip_missing"],
            "images": [{"name": "a.jpg", "jpeg_b64": b64, "class": "clip_seated"}, {"name": "b.jpg", "jpeg_b64": b64, "class": "clip_missing"},
                       {"name": "c.jpg", "jpeg_b64": b64}]}
    cls = c.post("/api/v1/projects/import", json=body).json()
    assert (cls["project_id"], cls["images_added"], cls["classes_added"]) == ("D4 states", 3, 2)
    p = host.project("D4 states")
    assert p.task == "classify" and sorted(p.statuses()) == [0, 2, 2]      # two classes to check, one image without
    assert c.post("/api/v1/projects/import", json={**body, "boxes": [{"image": "a.jpg", "class": "clip_seated", "x": 0, "y": 0, "w": .5, "h": .5}]}).status_code == 400
    assert c.post("/api/v1/projects/import", json={**body, "images": [{"name": "d.jpg", "jpeg_b64": b64, "class": "nope"}]}).status_code == 400


def test_the_training_queue_waits_for_room_then_teaches_and_exports(v1, monkeypatch):
    """Setup flow v2: send a project to training; it waits while the GPU is full, then runs Teach and the export,
    one at a time, and answers to the box's service key too. (Teach and the export stubbed: no GPU here.)"""
    from engine import hw
    from ui import trainq
    c, _ = v1
    b64 = base64.b64encode(jpeg()).decode()
    body = {"name": "D4", "source": {"station": "l2-s4", "demo_version": 3}, "classes": ["clip"],
            "images": [{"name": "a.jpg", "jpeg_b64": b64}],
            "boxes": [{"image": "a.jpg", "class": "clip", "x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}]}
    assert c.post("/api/v1/projects/import", json=body).status_code == 200
    assert c.post("/api/v1/projects/D4/train", json={}).status_code == 400            # nothing confirmed yet
    host.project("D4").set_reviewed(0)
    free = {"gb": 0.5}
    monkeypatch.setattr(hw, "free_memory_gb", lambda: free["gb"])
    monkeypatch.setattr(trainq, "POLL_S", 0.2)
    taught = []

    def teach(b):
        taught.append(b)
        return {"run": b["name"], "job": host.start_job("teach", lambda progress, stop: (progress(1, 1, "epoch 1"),
                                                       {"passed": True, "knowledge_score": 91.0})[1])}
    monkeypatch.setattr(host, "teach", teach)
    monkeypatch.setattr(host, "export_model", lambda b: {"job": host.start_job("export", lambda progress, stop: {"path": "x"})})
    est = c.get("/api/v1/projects/D4/train/estimate").json()
    assert est["frames"] == 1 and est["size"] == "small" and est["minutes"][0] <= est["minutes"][1] and est["need_gb"] > 1
    r = c.post("/api/v1/projects/D4/train", json={"confirmed_only": True}).json()
    assert r["position"] == 0 and r["state"] == "queued" and r["minutes"] == est["minutes"]

    def until(ok, timeout=20):
        t0 = time.time()
        while not ok():
            assert time.time() - t0 < timeout, "timed out"
            time.sleep(0.1)
    until(lambda: c.get(f"/api/v1/train/runs/{r['run']}").json()["state"] == "waiting_for_room")
    w = c.get(f"/api/v1/train/runs/{r['run']}").json()
    assert w["free_gb"] == 0.5 and w["need_gb"] > 1 and not taught
    free["gb"] = 11.0                                                                # the box made room
    until(lambda: c.get(f"/api/v1/train/runs/{r['run']}").json()["state"] == "done")
    d = c.get(f"/api/v1/train/runs/{r['run']}").json()
    assert (d["passed"], d["knowledge_score"], d["export_id"], d["progress"]) == (True, 91.0, r["run"], 1.0)
    assert taught[0]["size"] == "small" and taught[0]["tasks"] == [1]
    assert any(x["run"] == r["run"] for x in c.get("/api/v1/train/queue").json()["runs"])
    assert c.get("/api/v1/train/runs/nope").status_code == 404
    free["gb"] = 0.5                                                                 # no room: it waits, then is cancelled
    r2 = c.post("/api/v1/projects/D4/train", json={}).json()
    until(lambda: c.get(f"/api/v1/train/runs/{r2['run']}").json()["state"] == "waiting_for_room")
    assert c.post(f"/api/v1/train/runs/{r2['run']}/cancel").json()["state"] == "cancelled"
    time.sleep(0.5)
    assert c.get(f"/api/v1/train/runs/{r2['run']}").json()["state"] == "cancelled" and len(taught) == 1


def test_import_hawk_recordings_as_video_tasks_once_and_only_from_its_media_folder(v1, tmp_path, monkeypatch):
    """Hawk-AI hands over its recordings as video tasks (Track works there): only videos in hawk_media_dir's
    recorded/ or golden/, each once per project, every Nth frame, cut to the camera's work area when asked."""
    c, _ = v1
    import av
    import engine.accounts as accounts_mod
    media = tmp_path / "media"
    (media / "recorded" / "cam1").mkdir(parents=True)
    clip = media / "recorded" / "cam1" / "r1.mp4"
    with av.open(str(clip), "w") as out:
        s = out.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(30):
            fr = av.VideoFrame(64, 48, "yuv420p")
            fr.pts = i
            for p in s.encode(fr):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    real = accounts_mod.app_config
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {**real(), "hawk_media_dir": str(media)})
    src = {"station": "pen", "demo_version": 0}
    body = {"name": "Pen cycles", "source": src, "classes": ["pen_tube"],
            "videos": [{"name": "r1", "task": "Cycle 1", "path": str(clip), "every": 3, "crop": [0, 0, 0.5, 1]}]}
    r = c.post("/api/v1/projects/import", json=body).json()
    assert (r["project_id"], r["videos_added"], r["images_added"]) == ("Pen cycles", 1, 0)
    t, = host.project("Pen cycles").sources
    assert (t["kind"], t["name"], t["every"], t["crop"], t["info"]["width"]) == ("video", "Cycle_1", 3, [0, 0, 0.5, 1], 32)   # task names: its own rule (no spaces)
    assert c.post("/api/v1/projects/import", json=body).json()["videos_added"] == 0         # never twice
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(clip.read_bytes())
    for bad in (str(outside), str(media / "recorded" / ".." / ".." / "elsewhere.mp4"), str(media / "recorded" / "cam1")):
        r = c.post("/api/v1/projects/import", json={**body, "videos": [{"name": "x", "path": bad}]})
        assert r.status_code == 400
    assert len(host.project("Pen cycles").sources) == 1



def test_prelabel_labels_the_new_recordings_with_a_finished_run_key_only(tmp_path, monkeypatch, video):
    """Hawk-AI's "Pre-label with the last model" (2026-10-24): with hawk-api's key, the tasks with no confirmed frame get
    the run's model labels (to check); a project with a part the model never learned is refused (422)."""
    import json as _json
    import time as _time
    import supervision as sv
    import engine.accounts as accounts_mod
    from engine import detector
    from engine.project import Project
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {"models_token": "right-token"})
    monkeypatch.setattr(detector, "load", lambda *a, **k: object())
    monkeypatch.setattr(detector, "predict", lambda model, img, thr: sv.Detections(
        xyxy=np.array([[1, 1, 10, 10]], float), class_id=np.array([0]), confidence=np.array([0.9])))
    with app_client(tmp_path, signed_in=False) as c:
        p = Project.create(host.home() / "Pen bench cycles", ["pen"], video=video, every=2)   # task 1: done before
        p.add_source(video=video, every=2)                                                 # task 2: a new recording
        old, new = (t["id"] for t in p.sources)
        p.set_reviewed(p.ranges[old][0])
        run = host.runs_dir() / "Pen bench cycles_1"
        run.mkdir(parents=True)
        (run / "settings.json").write_text(_json.dumps({"classes": ["pen"], "checkpoint": "m.pth", "size": "small",
                                                        "resolution": 640, "threshold": 0.5, "parent": None}))
        url, key = "/api/v1/projects/Pen bench cycles/prelabel", {"X-Hawk-Token": "right-token"}
        assert c.post(url, json={"run": run.name}).status_code == 403
        r = c.post(url, json={"run": run.name}, headers=key)
        assert r.status_code == 200 and r.json() == {"tasks": 1, "state": "started"}, r.text
        q = host.project("Pen bench cycles")
        t0 = _time.time()
        while _time.time() - t0 < 30 and not all(q.boxes(k) for k in range(*q.ranges[new])):
            _time.sleep(0.1)
        assert all(q.boxes(k) and all(x["source"] == "imported" for x in q.boxes(k)) for k in range(*q.ranges[new]))
        assert not any(q.boxes(k) for k in range(*q.ranges[old]))                          # a task with a confirmed frame
        (run / "settings.json").write_text(_json.dumps({"classes": ["cap"], "checkpoint": "m.pth", "size": "small",
                                                        "resolution": 640}))
        assert c.post(url, json={"run": run.name}, headers=key).status_code == 422        # it never learned "pen"
        assert c.post(url, json={"run": "nope"}, headers=key).status_code == 404
