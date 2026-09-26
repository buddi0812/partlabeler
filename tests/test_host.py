"""The start screen's JSON API and the annotator's WebSocket, through FastAPI's test client (no models)."""
import time

import pytest
from fastapi.testclient import TestClient

from engine.project import Project
from tests.test_project import CLASSES, images, video  # noqa: F401  (fixtures)
from ui import host_fastapi as host


@pytest.fixture
def client(tmp_path):
    host.state.update(home=tmp_path / "home", sessions={}, clients={}, jobs={}, notes_home=None)
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


def test_trash_restore_and_notifications(client, tmp_path, video):
    r = client.post("/api/projects", json={"name": "line 3", "video": str(video), "classes": "a\nb"})
    wait(client, r.json()["job"])
    notes = client.get("/api/notifications").json()
    assert notes["unread"] == 1 and notes["items"][0]["title"] == "Created project line 3"
    assert notes["items"][0]["action"] == {"type": "open", "project": "line 3", "label": "Open project"}

    entry = client.post("/api/projects/line 3/trash").json()["entry"]
    assert [p["name"] for p in client.get("/api/projects").json()] == []
    assert [t["name"] for t in client.get("/api/trash").json()] == ["line 3"]
    moved = client.get("/api/notifications").json()["items"][0]
    assert moved["level"] == "warning" and moved["action"] == {"type": "restore", "entry": entry, "label": "Restore"}

    assert client.post(f"/api/trash/{entry}/restore").json() == {"name": "line 3"}
    assert [p["name"] for p in client.get("/api/projects").json()] == ["line 3"] and client.get("/api/trash").json() == []
    assert client.post(f"/api/trash/{entry}/restore").status_code == 404

    client.post("/api/notifications/read")
    assert client.get("/api/notifications").json()["unread"] == 0
    client.post("/api/notifications/clear")
    assert client.get("/api/notifications").json()["items"] == []
    # history is stored per projects folder and survives a restart
    assert (tmp_path / "home" / ".partlabeler" / "notifications.json").exists()


def test_open_folder_refuses_paths_outside_the_projects_folder(client, tmp_path):
    assert client.post("/api/open-folder", json={"path": str(tmp_path)}).status_code == 400
    assert client.post("/api/open-folder", json={"path": str(tmp_path / "home" / "missing")}).status_code == 400


def test_image_class_project_without_classes_and_grid_thumbnails(client, images):
    r = client.post("/api/projects", json={"name": "sorting", "images": str(images), "task": "classify", "classes": ""})
    assert wait(client, r.json()["job"])["error"] is None
    [p] = client.get("/api/projects").json()
    assert (p["task"], p["classes"], p["items"]) == ("classify", [], 3)
    with client.websocket_connect("/ws/sorting") as ws:
        ws.send_json({"type": "add_class", "name": "ok", "of": "images", "keys": [0, 2]})
        ws.send_json({"type": "grid", "of": "images"})
        while (m := ws.receive_json())["type"] != "grid":
            pass
    assert [c["cls"] for c in m["cards"]] == [0, None, 0] and m["cards"][1]["src"] == "/thumbs/sorting/images/1.jpg"
    t = client.get("/thumbs/sorting/images/1.jpg")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    assert client.get("/thumbs/sorting/images/9.jpg").status_code == 404
    bad = client.post("/api/projects", json={"name": "x", "images": str(images), "task": "nope", "classes": "a"})
    assert bad.status_code == 400


def test_update_check_is_cached_and_update_waits_for_jobs(client, monkeypatch):
    from engine import update
    calls = []
    monkeypatch.setattr(update, "check", lambda: calls.append(1) or {"available": True, "changes": ["new"], "can_update": True})
    assert client.get("/api/update").json()["available"] and client.get("/api/update").json()["changes"] == ["new"]
    assert len(calls) == 1                                               # GitHub is asked at most every 6 hours
    monkeypatch.setenv("PARTLABELER_NO_UPDATE_CHECK", "1")
    assert client.get("/api/update").json() == {"available": False, "off": True}
    host.state["jobs"]["x"] = {"finished": False}
    assert client.post("/api/update").status_code == 409                # never mid-job
    assert "version" in client.get("/api/info").json()
