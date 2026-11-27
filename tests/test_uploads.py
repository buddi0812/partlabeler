"""A teammate on another computer (team server) adds tasks from their own computer: the page uploads the video or the
image folder in pieces into the project (uploads/<id>/), the task is made from the upload, and deleting the task
deletes it. The server's own folders stay out of sight from other computers (no file browser, no reading by path)."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_host import TEST_LOGIN, app_client, wait
from tests.test_project import images, video  # noqa: F401  (fixtures)
from ui import host_fastapi as host

FAR = ("10.1.2.3", 5000)


def send(c, uid, path, part, name=None):
    data = Path(path).read_bytes()
    for off in range(0, max(1, len(data)), part):
        r = c.put(f"/api/projects/line/uploads/{uid}/{name or Path(path).name}?offset={off}", content=data[off:off + part])
        assert r.status_code == 200, r.text


def test_a_teammate_adds_a_video_and_an_image_folder_from_their_computer(tmp_path, monkeypatch, video, images):
    with app_client(tmp_path) as here:
        assert here.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]}).status_code == 200
        assert '"remote": false' in here.get("/").text
        monkeypatch.setitem(host.state, "team", True)
        monkeypatch.setattr(host, "UPLOAD_PART", 1000)                               # many pieces, even for test files
        with TestClient(host.app, client=FAR) as far:
            assert far.post("/api/auth/login", json=TEST_LOGIN).status_code == 200
            assert '"remote": true' in far.get("/").text                             # the page uploads instead
            assert far.get("/api/browse").status_code == 403                         # the server's folders: not listed
            assert far.get("/api/classes", params={"path": str(images)}).status_code == 403

            v = far.post("/api/projects/line/uploads", json={"kind": "video", "name": video.name}).json()
            send(far, v["id"], video, 1000)
            pics = far.post("/api/projects/line/uploads", json={"kind": "images", "name": "photos"}).json()
            for f in ("a.jpg", "b.png"):                                               # the page sends the top folder's pictures
                send(far, pics["id"], images / f, 1000)
            job = far.post("/api/projects/line/tasks", json={"sources": [{"upload": v["id"]}, {"upload": pics["id"]}],
                                                              "every": 1}).json()["job"]
            res = wait(far, job)
            assert res["error"] is None, res
            first, second = res["result"]["tasks"]
            up = tmp_path / "home" / "line" / "uploads"
            assert (up / v["id"] / video.name).read_bytes() == video.read_bytes()       # whole, in order
            assert far.get(f"/api/projects/line/tasks/{first}").json()["frames"] == 20
            t = far.get(f"/api/projects/line/tasks/{second}").json()
            assert t["frames"] == 2 and t["name"] == "photos"
            assert far.delete(f"/api/projects/line/tasks/{second}").status_code == 200
            assert not (up / pics["id"]).exists() and (up / v["id"]).exists()          # the task's upload went with it


def test_upload_pieces_are_checked(tmp_path, video):
    with app_client(tmp_path) as c:
        c.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        uid = c.post("/api/projects/line/uploads", json={"kind": "video", "name": "clip.mp4"}).json()["id"]
        url = f"/api/projects/line/uploads/{uid}"
        assert c.put(f"{url}/clip.mp4?offset=0", content=b"abc").json() == {"size": 3}
        assert c.put(f"{url}/clip.mp4?offset=0", content=b"abc").status_code == 409     # sent twice: refused, not doubled
        assert c.put(f"{url}/clip.mp4?offset=3", content=b"de").json() == {"size": 5}
        assert c.put(f"{url}/notes.txt?offset=0", content=b"x").status_code == 400      # not a video
        assert c.put(f"{url}/other.mp4?offset=0", content=b"x").status_code == 400      # one video per upload
        assert c.put(f"/api/projects/line/uploads/{'0' * 32}/clip.mp4?offset=0", content=b"x").status_code == 404
        assert c.put(f"/api/projects/line/uploads/..%2F..%2Fx/clip.mp4", content=b"x").status_code in (404, 405)
        r = c.put(f"{url}/clip.mp4?offset=5", content=b"x" * (host.UPLOAD_PART + 1))
        assert r.status_code == 413 and (tmp_path / "home" / "line" / "uploads" / uid / "clip.mp4").stat().st_size == 5
        pics = c.post("/api/projects/line/uploads", json={"kind": "images", "name": "../../evil"}).json()["id"]
        assert c.put(f"/api/projects/line/uploads/{pics}/..%5C..%5Cx.jpg?offset=0", content=b"x").status_code == 200
        assert [p.name for p in (tmp_path / "home" / "line" / "uploads" / pics).rglob("*.jpg")] == ["x.jpg"]
        assert (tmp_path / "home" / "line" / "uploads" / pics / "evil").is_dir()          # names stay inside the upload


def test_paths_on_the_server_are_refused_from_another_computer(tmp_path, monkeypatch, video, images):
    with app_client(tmp_path) as here:
        here.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        (tmp_path / "classes.txt").write_text("secret\n")
        monkeypatch.setitem(host.state, "team", True)
        with TestClient(host.app, client=FAR) as far:
            far.post("/api/auth/login", json=TEST_LOGIN)
            for method, url, body in (
                    ("post", "/api/projects/line/tasks", {"sources": [str(video)]}),
                    ("post", "/api/projects/line/tasks", {"sources": [{"path": str(video)}]}),
                    ("post", "/api/projects", {"name": "x", "classes_file": str(tmp_path / "classes.txt")}),
                    ("post", "/api/projects", {"name": "y", "classes": "a", "video": str(video)}),
                    ("post", "/api/projects/line/annotations", {"labels": str(images), "task": 1}),
                    ("post", "/api/projects/line/tasks/import", {"path": str(tmp_path / "b.zip")}),
                    ("post", "/api/backups/restore", {"path": str(tmp_path / "b.zip")}),
                    ("post", "/api/tools/add_tasks", {"project": "line", "sources": [str(video)]})):
                r = getattr(far, method)(url, json=body)
                assert r.status_code == 403, (url, body, r.status_code, r.text)
            assert not (tmp_path / "home" / "x").exists()
            monkeypatch.setattr("engine.recordings.resolve", lambda rid: video if rid == "st/cam/a.mp4" else None)
            job = far.post("/api/projects/line/tasks", json={"sources": [{"recording": "st/cam/a.mp4"}]}).json()["job"]
            assert wait(far, job)["error"] is None                                  # a recording, named by its id
        assert wait(here, here.post("/api/projects/line/tasks", json={"sources": [str(images)]}).json()["job"])["error"] is None


def test_an_uploaded_folder_task_survives_a_backup_and_a_moved_project(tmp_path, images):
    from engine.project import Project, backup_project, restore_project
    with app_client(tmp_path) as c:
        c.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        uid = c.post("/api/projects/line/uploads", json={"kind": "images", "name": "photos"}).json()["id"]
        for f in ("a.jpg", "b.png"):
            send(c, uid, images / f, 1 << 20)
        assert wait(c, c.post("/api/projects/line/tasks", json={"sources": [{"upload": uid}]}).json()["job"])["error"] is None
        host.state["sessions"].clear()
    folder = tmp_path / "home" / "line"
    backup_project(folder, tmp_path / "line.zip")
    restored = Project(restore_project(tmp_path / "line.zip", tmp_path / "other"))
    assert len(restored.items) == 2 and all(Path(it["path"]).is_file() for it in restored.items)
    import shutil
    moved = Path(shutil.copytree(folder, tmp_path / "moved"))                      # e.g. a new projects folder: the
    meta = (moved / "project.json").read_text(encoding="utf-8")                    # old place is gone
    (moved / "project.json").write_text(meta.replace(str(folder).replace("\\", "\\\\"), str(tmp_path / "gone").replace("\\", "\\\\")), encoding="utf-8")
    assert str(tmp_path / "gone").replace("\\", "\\\\") in (moved / "project.json").read_text(encoding="utf-8")
    p = Project(moved)
    assert len(p.items) == 2 and Path(p.sources[0]["source"]).is_relative_to(moved)


def test_uploads_nobody_uses_go_after_a_day_and_long_names_keep_their_type(tmp_path, monkeypatch, images):
    import os
    with app_client(tmp_path) as c:
        c.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        used = c.post("/api/projects/line/uploads", json={"kind": "images", "name": "photos"}).json()["id"]
        send(c, used, images / "a.jpg", 1 << 20)
        assert wait(c, c.post("/api/projects/line/tasks", json={"sources": [{"upload": used}]}).json()["job"])["error"] is None
        left = c.post("/api/projects/line/uploads", json={"kind": "video", "name": "half.mp4"}).json()["id"]
        up = tmp_path / "home" / "line" / "uploads"
        for f in [*up.rglob("*"), *up.iterdir()]:
            os.utime(f, (0, 0))                                                         # long ago
        c.post("/api/projects/line/uploads", json={"kind": "video", "name": "new.mp4"})
        assert (up / used).exists() and not (up / left).exists()
    assert host._file_name("x" * 300 + ".jpeg").endswith(".jpeg") and len(host._file_name("x" * 300 + ".jpeg")) == 120


def test_teach_transfer_and_the_annotator_connection_refuse_server_paths_from_another_computer(tmp_path, monkeypatch, video):
    with app_client(tmp_path) as here:
        here.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        assert wait(here, here.post("/api/projects/line/tasks", json={"sources": [str(video)]}).json()["job"])["error"] is None
        monkeypatch.setitem(host.state, "team", True)
        with TestClient(host.app, client=FAR) as far:
            far.post("/api/auth/login", json=TEST_LOGIN)
            for url, body in (("/api/teach", {"dataset": str(tmp_path)}),
                              ("/api/quick", {"dataset": str(tmp_path), "sources": [str(video)]}),
                              ("/api/review", {"output": str(tmp_path / "a" / "b"), "source": str(video)})):
                assert far.post(url, json=body).status_code == 403, url
            with far.websocket_connect("/ws/line") as sock:
                sock.send_json({"type": "add_task", "path": str(video)})
                while (m := sock.receive_json())["type"] != "toast":
                    pass
                assert "own computer" in m["text"]
            assert len(host.project("line").sources) == 1                          # nothing added


def test_an_uploaded_folder_called_uploads_is_found_after_a_move(tmp_path):
    from engine.project import Project
    from PIL import Image
    p = Project.create(tmp_path / "proj", ["bolt"])
    pics = tmp_path / "proj" / "uploads" / ("a" * 32) / "uploads"
    pics.mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(pics / "x.jpg")
    p.add_source(images=pics, every=1)
    p.db.close()
    import shutil
    moved = Path(shutil.copytree(tmp_path / "proj", tmp_path / "moved"))
    meta = (moved / "project.json").read_text(encoding="utf-8")
    old = str(tmp_path / "proj").replace("\\", "\\\\")
    (moved / "project.json").write_text(meta.replace(old, str(tmp_path / "gone").replace("\\", "\\\\")), encoding="utf-8")
    q = Project(moved)
    assert Path(q.sources[0]["source"]) == moved / "uploads" / ("a" * 32) / "uploads" and len(q.items) == 1


def test_deleting_a_task_keeps_an_upload_another_task_still_reads_and_a_blank_path_is_refused(tmp_path, images):
    with app_client(tmp_path) as c:
        c.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        uid = c.post("/api/projects/line/uploads", json={"kind": "images", "name": "photos"}).json()["id"]
        send(c, uid, images / "a.jpg", 1 << 20)
        res = wait(c, c.post("/api/projects/line/tasks", json={"sources": [{"upload": uid}, {"upload": uid}]}).json()["job"])
        first, second = res["result"]["tasks"]
        assert c.delete(f"/api/projects/line/tasks/{first}").status_code == 200
        assert (tmp_path / "home" / "line" / "uploads" / uid).exists()                 # the other task still reads it
        assert c.get(f"/api/projects/line/tasks/{second}").json()["frames"] == 1
        assert c.post("/api/projects/line/tasks", json={"sources": [" "]}).status_code == 400   # not "this folder"
        assert c.post("/api/projects/line/tasks", json={"sources": [{"path": ""}]}).status_code == 400


def test_another_computer_is_not_told_the_servers_folders(tmp_path, monkeypatch):
    with app_client(tmp_path) as here:
        host.note("success", "Exported line", "", None, host.folder_action(tmp_path / "out"))
        assert here.get("/api/info").json()["home"] and any(n.get("action") for n in here.get("/api/notifications").json()["items"])
        monkeypatch.setitem(host.state, "team", True)
        with TestClient(host.app, client=FAR) as far:
            far.post("/api/auth/login", json=TEST_LOGIN)
            assert far.get("/api/info").json()["home"] == ""
            items = far.get("/api/notifications").json()["items"]
            assert items and not any((n.get("action") or {}).get("type") == "folder" for n in items)
            assert str(tmp_path) not in json.dumps(items)


def test_server_paths_are_taken_out_of_what_another_computer_reads():
    cases = {
        r"Exported 120 images to D:\PartLabeler\projects\line\export_yolo": "Exported 120 images to (on the server)",
        "Saved in /srv/partlabeler/projects/line/backup.zip.": "Saved in (on the server)",
        r"Copied from \\box\share\x.zip ok": "Copied from (on the server) ok",
        r"Source: `C:\Users\x\_teach\run\source`": "Source: `(on the server)`",
        "See https://example.com/a/b and 3/4 done": "See https://example.com/a/b and 3/4 done",
    }
    assert {t: host.scrub(t) for t in cases} == cases
    m = host.seen_from_afar({"type": "notify", "item": {"title": "Exported", "detail": r"in D:\x\y",
                                                         "action": {"type": "folder", "path": r"D:\x\y"}}})
    assert m["item"]["action"] is None and "D:" not in json.dumps(m)


def test_the_annotator_connection_of_another_computer_gets_no_server_paths(tmp_path, monkeypatch, video):
    with app_client(tmp_path) as here:
        here.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        assert wait(here, here.post("/api/projects/line/tasks", json={"sources": [str(video)], "every": 5}).json()["job"])["error"] is None
        monkeypatch.setitem(host.state, "team", True)
        with TestClient(host.app, client=FAR) as far:
            far.post("/api/auth/login", json=TEST_LOGIN)
            with far.websocket_connect("/ws/line") as sock:
                sock.send_json({"type": "ready", "item": 0})
                host.note("success", "Exported line", f"in {tmp_path / 'out'}", "line", host.folder_action(tmp_path / "out"))
                seen = []
                while not any(m.get("type") == "notify" for m in seen):
                    seen.append(sock.receive_json())
            assert str(tmp_path) not in json.dumps([m for m in seen if m.get("type") in ("notify", "notifications")])
            assert far.get("/api/me").json()["home"] == ""


def test_a_teammate_on_another_computer_keeps_getting_updates_after_propagate(tmp_path, monkeypatch, video):
    """seen_from_afar once took the "items" (frame numbers) of item_changed for notifications and crashed the page's
    update loop: after Propagate a remote teammate's page heard nothing until a reload."""
    assert host.seen_from_afar({"type": "item_changed", "items": [1, 2]}) == {"type": "item_changed", "items": [1, 2]}
    with app_client(tmp_path) as here:
        here.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        assert wait(here, here.post("/api/projects/line/tasks", json={"sources": [str(video)], "every": 1}).json()["job"])["error"] is None
        monkeypatch.setitem(host.state, "team", True)
        with TestClient(host.app, client=FAR) as far:
            far.post("/api/auth/login", json=TEST_LOGIN)
            with far.websocket_connect("/ws/line") as sock:
                def until(kind, **match):
                    while True:
                        m = sock.receive_json()
                        if m.get("type") == kind and all(m.get(k) == v for k, v in match.items()):
                            return m
                sock.send_json({"type": "ready", "item": 0})
                sock.send_json({"type": "box", "item": 0, "cls": 0, "box": [5, 5, 20, 20]})
                until("item", item=0)
                sock.send_json({"type": "propagate", "item": 0, "count": 5})
                assert until("item_changed")["items"] == [1, 2, 3, 4, 5]
                sock.send_json({"type": "goto", "item": 3})                          # the page still hears back
                assert len(until("item", item=3)["boxes"]) == 1
