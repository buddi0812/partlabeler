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
