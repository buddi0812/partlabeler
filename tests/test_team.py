"""Team server: other machines use the whole app (signed in), accounts and this computer's own things stay local,
sign-in has an attempt limit, undo is per person, and a teammate's job waits for the running one."""
import threading
import time

from fastapi.testclient import TestClient

from engine.api import ACTOR, Session
from engine.project import Project
from tests.test_host import TEST_LOGIN, app_client  # noqa: F401
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)
from ui import host_fastapi as host

FAR = ("10.1.2.3", 5000)


def test_team_server_lets_signed_in_teammates_in_and_keeps_this_computer_local(tmp_path, monkeypatch):
    with app_client(tmp_path) as here:
        assert here.post("/api/auth/signup", json={"username": "ben", "password": "ben-pass-1"}).status_code == 200
        with TestClient(host.app, client=FAR) as far:
            assert far.get("/", follow_redirects=False).status_code == 403          # not a team server: API only
            monkeypatch.setitem(host.state, "team", True)
            assert far.get("/", follow_redirects=False).headers["location"].startswith("/login")
            assert far.post("/api/auth/signup", json={"username": "eve", "password": "eve-pass-1"}).status_code == 403
            assert far.post("/api/auth/login", json={"username": "ben", "password": "ben-pass-1"}).status_code == 200
            assert far.get("/api/projects").status_code == 200                      # the app, as ben
            assert far.get("/api/me").json()["user"]["username"] == "ben"
            for method, path in (("post", "/api/open-folder"), ("post", "/api/update"), ("post", "/api/assistant/key"),
                                 ("post", "/api/me/projects-folder")):
                assert getattr(far, method)(path, json={}).status_code == 403, path
        assert here.post("/api/auth/signup", json={"username": "cara", "password": "cara-pass-1"}).status_code == 200


def test_sign_in_pauses_after_too_many_wrong_passwords(tmp_path):
    with app_client(tmp_path, signed_in=False) as c:
        c.post("/api/auth/signup", json=TEST_LOGIN)
        wrong = {**TEST_LOGIN, "password": "not-it-at-all"}
        assert [c.post("/api/auth/login", json=wrong).status_code for _ in range(host.LOGIN_TRIES)] == [401] * host.LOGIN_TRIES
        assert c.post("/api/auth/login", json=TEST_LOGIN).status_code == 429             # even the right one, for now
        host.state["failed_logins"].clear()
        assert c.post("/api/auth/login", json=TEST_LOGIN).status_code == 200


def as_(who, fn):
    token = ACTOR.set(who)
    try:
        return fn()
    finally:
        ACTOR.reset(token)


def test_undo_takes_back_only_your_own_change(tmp_path, video):
    p = Project.create(tmp_path / "p", CLASSES, video=video)
    s = Session(p, lambda m: None)
    as_("ana", lambda: s.handle({"type": "box", "item": 0, "cls": 0, "box": [1, 1, 5, 5]}))
    as_("ben", lambda: s.handle({"type": "box", "item": 1, "cls": 0, "box": [2, 2, 6, 6]}))
    as_("ana", lambda: s.handle({"type": "undo"}))                    # ana's box goes, ben's (made later) stays
    assert p.boxes(0) == [] and len(p.boxes(1)) == 1


def test_a_teammates_job_waits_for_the_running_one(tmp_path, video):
    p = Project.create(tmp_path / "p", CLASSES, video=video)
    msgs, order, gate = [], [], threading.Event()
    s = Session(p, msgs.append)
    as_("ana", lambda: s._run("First", lambda: (gate.wait(5), order.append("first"))))
    as_("ana", lambda: s._run("Again", lambda: order.append("again")))                    # her own: refused
    assert any(m.get("text", "").startswith("Busy") for m in msgs)
    ben = threading.Thread(target=lambda: as_("ben", lambda: s._run("Second", lambda: order.append("second"))))
    ben.start()
    time.sleep(0.3)
    assert order == [] and any("starts when it ends" in m.get("text", "") for m in msgs)   # waiting, not refused
    gate.set()
    ben.join(5)
    s.job.join(5)
    assert order == ["first", "second"]
