"""GET /api/v1/models (hawk-ai-internal §12.3): the one /api/v1 route gated by a service key (X-Hawk-Token /
app.json "models_token") instead of an account token, local-only even with the right key, listing every export
for a same-box production system's deploy picker. Also Rivet's list_stations/deploy_link tools (ui/mcp.py)."""
import json

import engine.accounts as accounts_mod
import pytest

from tests.test_host import app_client
from ui import host_fastapi as host
from ui import mcp


def make_export(home, run, project=None, crop=None, passed=True, score=90.0):
    d = home / "_teach" / run
    (d / "export").mkdir(parents=True)
    (d / "export" / "model.onnx").write_bytes(b"x")
    (d / "export" / "model.json").write_text(json.dumps(
        {"model": "model.onnx", "classes": ["bolt"], "threshold": 0.4, "resolution": 640,
         "crop": crop, "parent": None}), encoding="utf-8")
    (d / "settings.json").write_text(json.dumps({"project": project, "source": str(d / "source")}), encoding="utf-8")
    (d / "report.json").write_text(json.dumps(
        {"passed": passed, "knowledge_score": score,
         "prepare": {"train_frames": 10, "held_out_frames": 3}}), encoding="utf-8")
    return d


def test_models_for_deploy_merges_model_settings_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "home", lambda: tmp_path / "home")
    make_export(tmp_path / "home", "run_1", project="yard-cams", crop={"cam1": [0.1, 0.1, 0.5, 0.5]})
    make_export(tmp_path / "home", "run_2")                                   # no project: falls back to source name
    out = host.models_for_deploy()
    by_id = {m["id"]: m for m in out}
    r1 = by_id["run_1"]
    assert r1["name"] == "yard-cams" and r1["crop"] == {"cam1": [0.1, 0.1, 0.5, 0.5]} and r1["classes"] == ["bolt"]
    assert r1["passed"] is True and r1["knowledge_score"] == 90.0
    assert r1["frames"] == {"train": 10, "held_out": 3}
    assert r1["onnx_path"] == str((tmp_path / "home" / "_teach" / "run_1" / "export" / "model.onnx").resolve())
    assert by_id["run_2"]["name"] == "source"                                 # Path(".../run_2/source").name


def test_api_v1_models_needs_the_service_key_not_an_account_token(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {"models_token": "right-token"})
    with app_client(tmp_path, signed_in=False) as c:
        make_export(tmp_path / "home", "run_1", project="yard-cams")
        assert c.get("/api/v1/models").status_code == 403                                 # no header at all
        assert c.get("/api/v1/models", headers={"X-Hawk-Token": "wrong"}).status_code == 403
        r = c.get("/api/v1/models", headers={"X-Hawk-Token": "right-token"})
        assert r.status_code == 200 and r.json()["models"][0]["id"] == "run_1"
        # a Bearer account token (the usual /api/v1 auth) must not substitute for the service key
        assert c.get("/api/v1/models", headers={"Authorization": "Bearer whatever-account-token"}).status_code == 403


def test_api_v1_models_compares_the_token_in_constant_time(tmp_path, monkeypatch):
    """Hawk-AI orchestrator security review (2026-10-09): a plain `!=` on a secret invites timing attacks even
    loopback-only; hmac.compare_digest is the fix hawk-api's own token checks already use."""
    import hmac as hmac_mod
    calls = []
    real = hmac_mod.compare_digest
    monkeypatch.setattr(hmac_mod, "compare_digest", lambda a, b: (calls.append((a, b)), real(a, b))[1])
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {"models_token": "right-token"})
    with app_client(tmp_path, signed_in=False) as c:
        make_export(tmp_path / "home", "run_1")
        assert c.get("/api/v1/models", headers={"X-Hawk-Token": "wrong"}).status_code == 403
        assert c.get("/api/v1/models", headers={"X-Hawk-Token": "right-token"}).status_code == 200
    assert calls == [("wrong", "right-token"), ("right-token", "right-token")]


def test_api_v1_models_refuses_even_a_valid_key_from_another_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {"models_token": "right-token"})
    with app_client(tmp_path, signed_in=False) as c:
        make_export(tmp_path / "home", "run_1")
        r = c.get("/api/v1/models", headers={"X-Hawk-Token": "right-token", "X-Forwarded-For": "10.0.0.5"})
        assert r.status_code == 403


def test_api_v1_models_refuses_when_no_token_is_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {})                           # no "models_token" set
    with app_client(tmp_path, signed_in=False) as c:
        make_export(tmp_path / "home", "run_1")
        assert c.get("/api/v1/models", headers={"X-Hawk-Token": ""}).status_code == 403


def test_list_stations_tool_wraps_recordings(monkeypatch, tmp_path):
    from engine import recordings
    monkeypatch.setattr(recordings, "app_config", lambda: {"recordings_dir": str(tmp_path / "rec")})
    (tmp_path / "rec" / "line1" / "cam1").mkdir(parents=True)
    out, _ = mcp.call("list_stations", {})
    assert out["stations"] == [{"id": "line1", "name": "line1"}]


def test_deploy_link_builds_the_dashboard_url(monkeypatch):
    monkeypatch.setattr(accounts_mod, "app_config", lambda: {})                           # default dashboard_url
    out, _ = mcp.call("deploy_link", {"export": "run_1", "station": "line 1"})
    assert out["url"] == "http://127.0.0.1:8080/models?deploy=run_1&station=line%201"

    monkeypatch.setattr(accounts_mod, "app_config", lambda: {"dashboard_url": "http://127.0.0.1:9999/"})
    out, _ = mcp.call("deploy_link", {"export": "run_1", "station": "line1"})
    assert out["url"] == "http://127.0.0.1:9999/models?deploy=run_1&station=line1"
