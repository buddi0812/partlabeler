"""Rivet driving PartLabeler through its own tools on a local model (ui/rivet_tools.py): the tool-calling loop,
the confirm-before-destructive-tool-calls flow, and the action log. The local model itself is faked with an
httpx MockTransport (no real llama.cpp needed) so this runs fast and offline."""
import json

import httpx
import pytest

from engine.accounts import Accounts
from ui import host_fastapi as host
from ui import mcp, rivet_tools
from tests.test_project import video  # noqa: F401  (fixture)


class FakeLLM:
    """Replays one OpenAI-style assistant message per /v1/chat/completions call, in order."""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.bodies = []
        self.auth = []                                     # the Authorization header seen on each call, in order

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.auth.append(request.headers.get("authorization"))
        if request.url.path.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "fake-7b"}]})
        if request.url.path.endswith("/v1/chat/completions"):
            self.bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": self.replies.pop(0)}]})
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))


def tool_call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    host.state.update(home=tmp_path / "home", sessions={}, clients={}, jobs={}, notes={},
                      accounts=Accounts(tmp_path / "accounts.json"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setattr(rivet_tools, "base_url", lambda: "http://fake-llm")
    rivet_tools.PENDING.clear()
    return tmp_path / "home"


def test_reply_runs_a_safe_tool_then_answers(home):
    mcp.call("create_project", {"name": "demo", "type": "detect", "labels": ["bolt"]})
    llm = FakeLLM(
        {"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "list_projects")]},
        {"role": "assistant", "content": "You have one project: demo."},
    )
    events = list(rivet_tools.reply([{"role": "user", "text": "what projects do I have?"}], client=llm.client()))
    assert {"model": "fake-7b"} in events
    assert {"text": "You have one project: demo."} in events
    assert {"done": True} in events
    # the model saw the real tool result, not a guess
    assert '"demo"' in llm.bodies[1]["messages"][-1]["content"]


def test_reply_pauses_a_destructive_call_for_confirmation(home, video):
    mcp.call("create_project", {"name": "demo", "type": "detect", "labels": ["bolt"]})
    mcp.call("add_tasks", {"project": "demo", "sources": [str(video), str(video)], "every": 5, "wait_s": 30})
    assert len(mcp.call("get_project", {"project": "demo"})[0]["tasks"]) == 2   # a project keeps at least one task

    llm = FakeLLM({"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "delete_task", project="demo", task=1, confirm=True)]})
    events = list(rivet_tools.reply([{"role": "user", "text": "delete task 1"}], actor="tester", client=llm.client()))
    confirm = next(e["confirm"] for e in events if "confirm" in e)
    assert confirm["actions"][0]["tool"] == "delete_task" and "task 1" in confirm["actions"][0]["summary"]
    assert len(rivet_tools.PENDING) == 1
    # not actually run yet
    assert len(mcp.call("get_project", {"project": "demo"})[0]["tasks"]) == 2

    llm2 = FakeLLM({"role": "assistant", "content": "Deleted task 1."})
    events = list(rivet_tools.resume(confirm["id"], actor="tester", client=llm2.client()))
    assert {"text": "Deleted task 1."} in events
    assert len(mcp.call("get_project", {"project": "demo"})[0]["tasks"]) == 1
    assert rivet_tools.PENDING == {}

    log = (home / ".partlabeler" / "assistant-actions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(log[-1])
    assert entry["tool"] == "delete_task" and entry["confirmed"] is True and entry["actor"] == "tester"


def test_cancel_drops_the_pending_call_without_running_it(home, video):
    mcp.call("create_project", {"name": "demo", "type": "detect", "labels": ["bolt"]})
    mcp.call("add_tasks", {"project": "demo", "sources": [str(video)], "every": 5, "wait_s": 30})
    llm = FakeLLM({"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "delete_task", project="demo", task=1)]})
    events = list(rivet_tools.reply([{"role": "user", "text": "delete task 1"}], client=llm.client()))
    confirm = next(e["confirm"] for e in events if "confirm" in e)

    assert rivet_tools.cancel(confirm["id"]) is True
    assert rivet_tools.PENDING == {}
    assert len(mcp.call("get_project", {"project": "demo"})[0]["tasks"]) == 1
    entry = json.loads((home / ".partlabeler" / "assistant-actions.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["confirmed"] is False and "declined" in entry["result"]


def test_no_local_model_configured_yields_nothing_to_run(home, monkeypatch):
    monkeypatch.setattr(rivet_tools, "base_url", lambda: None)
    events = list(rivet_tools.reply([{"role": "user", "text": "hi"}]))
    assert events == [{"error": "no local model configured"}]


def test_unreachable_local_model_falls_back_offline(home):
    def handler(request):
        raise httpx.ConnectError("refused")
    events = list(rivet_tools.reply([{"role": "user", "text": "hi"}], client=httpx.Client(transport=httpx.MockTransport(handler))))
    assert events[0] == {"offline": True, "reason": "local model not reachable"}
    assert events[-1] == {"done": True}


def test_an_assistant_llm_key_is_sent_as_a_bearer_token_on_every_call(home, monkeypatch):
    """A Hawk-AI box runs llama-server with --api-key: without this, model_id() (GET /v1/models) 401s, Rivet
    reports "not reachable", and the local assistant never works on a real box."""
    monkeypatch.setattr(rivet_tools.assistant, "local_key", lambda: "box-key-123")
    llm = FakeLLM({"role": "assistant", "content": "hi there"})
    events = list(rivet_tools.reply([{"role": "user", "text": "hi"}], client=llm.client()))
    assert {"text": "hi there"} in events
    assert llm.auth == ["Bearer box-key-123", "Bearer box-key-123"]     # GET /v1/models, then POST /v1/chat/completions


def test_no_assistant_llm_key_sends_no_authorization_header(home, monkeypatch):
    """Older boxes (and any local server with no key) behave exactly as before "assistant_llm_key" existed."""
    monkeypatch.setattr(rivet_tools.assistant, "local_key", lambda: None)
    llm = FakeLLM({"role": "assistant", "content": "hi there"})
    events = list(rivet_tools.reply([{"role": "user", "text": "hi"}], client=llm.client()))
    assert {"text": "hi there"} in events
    assert llm.auth == [None, None]
