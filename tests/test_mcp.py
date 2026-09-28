"""The API for programs (ui/mcp.py): API tokens, the MCP endpoint and the plain JSON tools, end to end (no GPU models:
box parts in a detection project, and a stand-in for SAM 3's text search)."""
import base64
import json

import pytest

from engine import api
from tests.test_host import app_client, wait  # noqa: F401
from tests.test_project import video  # noqa: F401  (fixture)


class FakeConcept:
    def detect(self, image, text=None, boxes=None, threshold=0.5):
        return [([10.0, 10.0, 30.0, 30.0], 0.9), ([40.0, 5.0, 60.0, 25.0], 0.7)]


@pytest.fixture
def agent(tmp_path, video):
    with app_client(tmp_path) as c:
        token = c.post("/api/me/tokens", json={"label": "Claude Code"}).json()["token"]
        c.cookies.clear()                                  # from here on only the token counts
        c.headers["Authorization"] = f"Bearer {token}"
        yield c, token


def rpc(c, method, params=None, id=1):
    r = c.post("/mcp", json={"jsonrpc": "2.0", "id": id, "method": method, **({"params": params} if params else {})})
    assert r.status_code == 200, r.text
    return r.json()["result"]


def tool(c, name, /, **args):
    res = rpc(c, "tools/call", {"name": name, "arguments": args})
    data = json.loads(res["content"][0]["text"]) if not res["isError"] else res["content"][0]["text"]
    return res, data


def test_tokens_protect_mcp_and_rest(tmp_path):
    with app_client(tmp_path) as c:
        made = c.post("/api/me/tokens", json={"label": "opencode"}).json()
        assert made["token"].startswith("plt_") and [t["label"] for t in made["tokens"]] == ["opencode"]
        stored = (tmp_path / "accounts.json").read_text(encoding="utf-8")
        assert made["token"] not in stored                  # only its hash is kept
        c.cookies.clear()
        assert c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
        bad = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"Authorization": "Bearer plt_nope"})
        assert bad.status_code == 401 and "revoked" in bad.json()["detail"]
        auth = {"Authorization": f"Bearer {made['token']}"}
        assert c.get("/api/projects", headers=auth).status_code == 200          # the REST API takes it too
        assert c.get("/mcp", headers=auth).status_code == 405
        evil = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={**auth, "Origin": "https://evil.example"})
        assert evil.status_code == 403
        assert c.delete(f"/api/me/tokens/{made['tokens'][0]['id']}", headers=auth).json() == []
        assert c.get("/api/projects", headers=auth).status_code == 401


def test_mcp_handshake_and_tool_list(agent):
    c, _ = agent
    init = rpc(c, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert init["protocolVersion"] == "2025-06-18" and "tools" in init["capabilities"] and "task + frame" in init["instructions"]
    assert c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
    names = [t["name"] for t in rpc(c, "tools/list")["tools"]]
    assert len(names) == 29 and {"get_frame", "add_part", "track", "export", "frames_to_video"} <= set(names)
    assert c.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "nope"}).json()["error"]["code"] == -32601


def test_an_agent_annotates_a_video_end_to_end(agent, tmp_path, video, monkeypatch):
    c, _ = agent
    res, d = tool(c, "create_project", name="line", type="detect", labels=["bolt", {"name": "nut", "color": "#00ff00"}])
    assert d == {"name": "line"}
    res, job = tool(c, "add_tasks", project="line", sources=[str(video)], every=5, wait_s=60)
    assert job["finished"] and job["error"] is None and job["result"]["tasks"] == [1]
    _, proj = tool(c, "get_project", project="line")
    t = proj["tasks"][0]
    assert proj["type"] == "detect" and t["frames"] == 4 and t["start_item"] == 0 and t["video"]["width"] == 64

    res, frame = tool(c, "get_frame", project="line", task=1, frame=2)
    assert frame["item"] == 1 and frame["width"] == 64 and frame["parts"] == [] and res["content"][1]["type"] == "image"
    assert base64.b64decode(res["content"][1]["data"])[:2] == b"\xff\xd8"                  # a JPEG

    _, added = tool(c, "add_part", project="line", item=1, label="Bolt", box=[5, 5, 20, 25])
    part = added["part"]
    assert part["label"] == "bolt" and part["box"] == [5, 5, 20, 25] and part["source"] == "manual"
    res, err = tool(c, "add_part", project="line", item=1, label="washer", box=[1, 1, 5, 5])
    assert res["isError"] and "labels: bolt, nut" in err                                  # the model can correct itself
    res, err = tool(c, "add_part", project="line", label="bolt", box=[1, 1, 5, 5])
    assert res["isError"] and "task and frame" in err

    _, edited = tool(c, "edit_part", project="line", item=1, part=part["id"], label="nut")
    assert edited["part"]["label"] == "nut"
    _, ann = tool(c, "get_annotations", project="line", task=1)
    assert [(f["item"], [p["label"] for p in f["parts"]]) for f in ann["frames"]] == [(1, ["nut"])]
    tool(c, "undo", project="line")
    assert tool(c, "get_annotations", project="line", item=1)[1]["frames"][0]["parts"][0]["label"] == "bolt"

    monkeypatch.setitem(api._MODELS, "concept", FakeConcept())
    _, found = tool(c, "find_parts", project="line", item=2, text="hex bolt", label="bolt")
    assert [p["source"] for p in found["found"]] == ["suggested", "suggested"] and found["busy"] is False
    _, acc = tool(c, "accept_suggestions", project="line", item=2)
    assert [p["source"] for p in acc["parts"]] == ["manual", "manual"]

    _, conf = tool(c, "confirm_frames", project="line", items=[1, 2])
    assert conf["frames"] == 2
    _, st = tool(c, "status", project="line")
    assert st["tasks"][0]["by_status"]["confirmed"] == 2 and st["busy"] is False

    _, dele = tool(c, "delete_parts", project="line", item=2, parts=[found["found"][0]["id"]])
    assert len(dele["parts"]) == 1
    _, ex = tool(c, "export", project="line", format="yolo", confirmed_only=True, wait_s=60)
    assert ex["finished"] and ex["error"] is None and ex["result"]["images"] == 2
    _, bk = tool(c, "backup", project="line", task=1, wait_s=60)
    assert bk["result"]["file"].endswith(".zip")

    rest = c.post("/api/tools/get_frame", json={"project": "line", "item": 2, "max_side": 32}).json()
    assert rest["image"].startswith("data:image/jpeg;base64,") and len(rest["parts"]) == 1
    assert c.post("/api/tools/get_frame", json={"project": "line"}).status_code == 400
    assert c.post("/api/tools/nope", json={}).status_code == 404
    assert len(c.get("/api/tools").json()["tools"]) == 29
