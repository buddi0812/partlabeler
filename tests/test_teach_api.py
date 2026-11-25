"""Teach from a project's tasks through the app (the start screen's Teach form): it needs confirmed frames and says so
in plain words, one run folder never gets two jobs at once, and only a trained detector run offers Export for
production."""
import json

from tests.test_host import app_client, wait  # noqa: F401
from tests.test_project import video  # noqa: F401  (fixture)
from ui import host_fastapi as host


def test_teach_needs_confirmed_frames_and_a_busy_run_takes_no_second_job(tmp_path, video):
    with app_client(tmp_path) as c:
        c.post("/api/projects", json={"name": "line", "labels": [{"name": "bolt"}]})
        assert wait(c, c.post("/api/projects/line/tasks", json={"sources": [str(video)], "every": 5}).json()["job"])["error"] is None
        r = c.post("/api/teach", json={"project": "line", "tasks": [1], "name": "v1"})
        assert r.status_code == 400 and "confirm the frames" in r.json()["detail"] and str(tmp_path) not in r.json()["detail"]
        p = host.session("line").p                                                  # now it has two
        for k in range(*p.ranges[1])[:2]:
            p.set_reviewed(k)
        host.BUSY_RUNS.add(host._run_key("v1"))
        try:
            assert c.post("/api/teach", json={"project": "line", "tasks": [1], "name": "v1"}).status_code == 409
            run = host.runs_dir() / "v1"
            run.mkdir(parents=True)
            (run / "settings.json").write_text(json.dumps({"checkpoint": "model/best.pth", "size": "nano"}))
            assert c.post("/api/export-model", json={"run": "v1"}).status_code == 409
        finally:
            host.BUSY_RUNS.discard(host._run_key("v1"))


def test_only_a_trained_detector_run_can_be_exported(tmp_path):
    with app_client(tmp_path) as c:
        for name, settings in (("boxes", {"checkpoint": "m.pth"}), ("quick", {"mode": "quick"}), ("classes", {"kind": "classify"})):
            (host.runs_dir() / name).mkdir(parents=True)
            (host.runs_dir() / name / "settings.json").write_text(json.dumps(settings))
        runs = {r["name"]: r for r in c.get("/api/runs").json()}
        assert {n: r["exportable"] for n, r in runs.items()} == {"boxes": True, "quick": False, "classes": False}


def test_the_command_line_teach_rounds_its_resolution_to_the_model_size():
    import inspect
    from engine import cli
    assert inspect.signature(cli.teach).parameters["resolution"].default.default is None


def test_a_finished_job_that_never_reported_progress_says_done(tmp_path):
    with app_client(tmp_path) as c:
        jid = host.start_job("export", lambda progress, should_stop: {"ok": 1})
        assert wait(c, jid)["text"] == "Done"
