"""The start screen's JSON API and the annotator's WebSocket, through FastAPI's test client (no models)."""
import time

import pytest
from fastapi.testclient import TestClient

from engine.project import Project
from tests.test_project import CLASSES, images, video  # noqa: F401  (fixtures)
from ui import host_fastapi as host


@pytest.fixture
def client(tmp_path):
    host.state.update(home=tmp_path / "home", sessions={}, clients={}, jobs={})
    (tmp_path / "home").mkdir()
    with TestClient(host.app) as c:
        yield c


def wait(client, jid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["finished"]:
            return j
        time.sleep(0.1)
    raise TimeoutError(jid)


def test_create_list_open_and_annotate(client, tmp_path, video):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "car_front_f000005.txt").write_text("2 0.5 0.5 0.25 0.5\n")
    r = client.post("/api/projects", json={"name": "line 2", "video": str(video), "every": 5,
                                            "classes": "left_drl\nright_drl\ngrill", "labels": str(labels),
                                            "parent": "engine block"})
    j = wait(client, r.json()["job"])
    assert j["error"] is None and j["result"]["imported"]["boxes"] == 1
    [p] = client.get("/api/projects").json()
    assert (p["name"], p["items"], p["labeled"], p["classes"], p["parent"]) == ("line 2", 4, 1, CLASSES, "engine block")
    assert client.post("/api/projects", json={"name": "line 2", "video": str(video), "classes": "a"}).status_code == 400

    assert client.get("/p/line 2").status_code == 200 and client.get("/p/nope").status_code == 404
    assert client.get("/items/line 2/1.jpg").headers["content-type"] == "image/jpeg"
    with client.websocket_connect("/ws/line 2") as ws:
        ws.send_json({"type": "ready"})
        got = {}
        while "item" not in got:
            m = ws.receive_json()
            got[m["type"]] = m
        assert got["project"]["parent"] == "engine block" and got["item"]["src"] == "/items/line 2/0.jpg"
        ws.send_json({"type": "box", "item": 2, "cls": 0, "box": [1, 1, 20, 20]})
        while (m := ws.receive_json())["type"] != "item":
            pass
        assert m["boxes"][0]["cls"] == 0
    assert Project(tmp_path / "home" / "line 2").boxes(2)[0]["source"] == "manual"


def test_create_needs_classes_and_an_existing_source(client, tmp_path):
    assert client.post("/api/projects", json={"name": "x", "video": str(tmp_path / "none.mp4"), "classes": "a"}).status_code == 400
    assert "class names" in client.post("/api/projects", json={"name": "x", "images": str(tmp_path)}).json()["detail"]


def test_browse_and_classes_file(client, tmp_path, images):
    top = client.get("/api/browse").json()
    assert top["path"] == "" and top["dirs"]
    res = client.get("/api/browse", params={"path": str(images), "kind": "images"}).json()
    assert res["dirs"] == ["sub"] and [f["name"] for f in res["files"]] == ["a.jpg", "b.png"] and res["images_here"] == 2
    (tmp_path / "classes.txt").write_text("bolt\nnut\n")
    assert client.get("/api/classes", params={"path": str(tmp_path / "classes.txt")}).json()["classes"] == ["bolt", "nut"]
    assert client.get("/api/browse", params={"path": str(tmp_path / "missing")}).status_code == 400
