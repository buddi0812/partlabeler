"""Exporting a Teach run for production (ui/host_fastapi.py export_model / list_exports) and the matching MCP
tools: the job/validation plumbing, with engine.detector.export_onnx faked (it needs a real trained model and
ONNX Runtime, covered separately by production use, not by this fast unit test)."""
import json

from engine import detector
from tests.test_host import app_client, wait


def fake_export_onnx(run_dir, out_dir=None, batch=1):
    out_dir = (out_dir or run_dir / "export")
    out_dir.mkdir(parents=True, exist_ok=True)
    card = {"model": "model.onnx", "classes": ["bolt"], "threshold": 0.5, "resolution": 640,
            "parity": {"frames": 10, "boxes": 10, "missing": 0, "worst_px": 1.2}}
    (out_dir / "model.json").write_text(json.dumps(card), encoding="utf-8")
    return card | {"path": str(out_dir / "model.onnx")}


def make_run(home, name="run_1"):
    run = home / "_teach" / name
    run.mkdir(parents=True)
    (run / "settings.json").write_text(json.dumps({"checkpoint": "best.pth"}), encoding="utf-8")
    return run


def test_export_model_rejects_a_run_with_no_checkpoint(tmp_path):
    with app_client(tmp_path) as c:
        r = c.post("/api/export-model", json={"run": "nope"})
        assert r.status_code == 400


def test_export_model_runs_the_job_and_writes_model_json(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "export_onnx", fake_export_onnx)
    with app_client(tmp_path) as c:
        make_run(tmp_path / "home")
        r = c.post("/api/export-model", json={"run": "run_1", "batch": 2})
        assert r.status_code == 200
        j = wait(c, r.json()["job"])
        assert j["error"] is None and j["result"]["classes"] == ["bolt"]
        assert (tmp_path / "home" / "_teach" / "run_1" / "export" / "model.json").exists()

        exports = c.get("/api/exports").json()
        assert exports == [{"run": "run_1", "path": str((tmp_path / "home" / "_teach" / "run_1" / "export").resolve()),
                            "created": exports[0]["created"], "classes": ["bolt"], "threshold": 0.5,
                            "resolution": 640, "parity": {"frames": 10, "boxes": 10, "missing": 0, "worst_px": 1.2}}]


def test_export_model_tool_and_list_exports_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "export_onnx", fake_export_onnx)
    from engine.accounts import Accounts
    from ui import host_fastapi as host
    from ui import mcp
    host.state.update(home=tmp_path / "home", sessions={}, clients={}, jobs={}, notes={},
                      accounts=Accounts(tmp_path / "accounts.json"))
    (tmp_path / "home").mkdir(exist_ok=True)
    make_run(tmp_path / "home")
    res, _ = mcp.call("export_model", {"run": "run_1", "wait_s": 10})
    assert res["finished"] and res["error"] is None

    out, _ = mcp.call("list_exports", {})
    assert out["exports"][0]["run"] == "run_1" and out["exports"][0]["classes"] == ["bolt"]
