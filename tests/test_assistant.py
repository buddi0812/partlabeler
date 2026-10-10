"""Rivet, the helper (engine/assistant.py): offline answers, model fallback, what Gemini is sent. No network."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from engine import assistant
from ui import host_fastapi as host


@pytest.fixture
def no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(assistant, "CONFIG", tmp_path / "cfg" / "assistant.json")
    assistant._preferred.update(model=None, until=0)


def sse(*texts):
    return "".join(f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': t}]}}]})}\n\n" for t in texts)


def test_offline_answers_come_from_the_guide(no_key):
    assert assistant.offline_answer("how can i export?").startswith("**Export the dataset**")
    assert assistant.offline_answer("explain me how teach and transfer works").startswith("**Teach & Transfer: how it works**")
    assert "I couldn't find that" in assistant.offline_answer("zzqx")
    events = list(assistant.reply([{"role": "user", "text": "where is the trash?"}]))
    assert events[0] == {"offline": True, "reason": "no_key"} and "Trash" in events[1]["text"] and events[-1] == {"done": True}
    assert list(assistant.reply([]))[0]["error"]


def test_the_screen_description_carries_no_names():
    ctx = {"page": "annotator", "kind": "video", "frame": 12, "frames": 400, "confirmed": 3, "to_check": 2,
           "classes": 5, "tool": "box", "busy": True, "name": "secret_line", "project": "secret_line",
           "class_names": ["secret_part"], "frames_extra": "ignore me"}
    text = assistant.describe(ctx)
    assert text.startswith("the annotator with a video, frame 12 of 400, 3 confirmed, 2 to check")
    assert "secret" not in text and "ignore" not in text and "draw box" in text
    assert assistant.describe({"page": "home", "projects": 4, "teach": False}) == \
        "the start screen, 4 projects listed, Teach & Transfer is not installed"
    prompt = assistant.system_prompt(ctx, ["export", "rm -rf", "shortcuts"])
    assert "- export:" in prompt and "rm -rf" not in prompt and "secret" not in prompt


def test_busy_model_falls_back_to_the_next_and_is_preferred_later(no_key):
    seen = []

    def handler(req: httpx.Request):
        model = req.url.path.split("/models/")[1].split(":")[0]
        seen.append((model, req.headers.get("x-goog-api-key"), dict(req.url.params), json.loads(req.content)))
        if model == assistant.MODELS[0]:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, text=sse("Click **Export**", " in the top bar.\n\n[[do:export]]"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    msgs = [{"role": "assistant", "text": "hello"}, {"role": "user", "text": "how do I export?"}]
    events = list(assistant.reply(msgs, {"page": "home"}, ["export"], key="AIza-test", client=client))
    assert [e for e in events if "model" in e] == [{"model": assistant.MODELS[1]}]
    assert "".join(e.get("text", "") for e in events) == "Click **Export** in the top bar.\n\n[[do:export]]"
    model, key, params, body = seen[-1]
    assert key == "AIza-test" and "key" not in params and params["alt"] == "sse"     # the key travels in a header
    assert [c["role"] for c in body["contents"]] == ["user"]                        # starts with the user's turn
    seen.clear()
    list(assistant.reply(msgs, key="AIza-test", client=client))
    assert [s[0] for s in seen] == [assistant.MODELS[1]]                              # the working model goes first


def test_no_model_answers_so_the_guide_does(no_key):
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(429, json={"error": {"message": "quota"}})))
    events = list(assistant.reply([{"role": "user", "text": "how can i export"}], key="AIza-test", client=client))
    assert events[0]["offline"] and "Export the dataset" in events[1]["text"]


def test_refused_key_is_reported(no_key):
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403, json={"error": {"message": "bad key"}})))
    events = list(assistant.reply([{"role": "user", "text": "hi"}], key="AIza-test", client=client))
    assert "refused the API key" in events[-1]["error"]


def test_key_file_and_http_endpoints(no_key, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        assistant.save_key("not a key")
    assistant.save_key("A" * 39, check=False)
    assert assistant.api_key() == ("A" * 39, "file") and assistant.info()["ready"]
    assistant.save_key("")
    assert assistant.api_key() == (None, "none")

    from tests.test_host import app_client
    with app_client(tmp_path) as c:
        info = c.get("/api/assistant").json()
        assert info["ready"] is False and info["tips"] and "api_key" not in json.dumps(info)
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "text": "how can I export?"}], "context": {"page": "home"}})
        lines = [json.loads(x) for x in r.text.splitlines()]
        assert r.headers["content-type"].startswith("application/x-ndjson")
        assert lines[0]["offline"] and "Export" in lines[1]["text"] and lines[-1] == {"done": True}
        assert c.post("/api/assistant/key", json={"key": "nope"}).status_code == 400


def test_box_managed_hides_the_local_models_name(no_key, tmp_path, monkeypatch):
    """app.json "box_managed": true (a Hawk-AI box, or any product PartLabeler is embedded in): the box's
    operator is not PartLabeler's own developer, so the local model's internal name is not their business."""
    import engine.accounts as accounts_mod
    from ui import rivet_tools
    from tests.test_host import app_client
    monkeypatch.setattr(assistant, "local_url", lambda: "http://fake-llm")
    monkeypatch.setattr(rivet_tools, "model_id", lambda url, client=None: "qwen3.5-4b-q4km")
    with app_client(tmp_path) as c:
        info = c.get("/api/assistant").json()
        assert info["ready"] is True and info["local_model"] == "qwen3.5-4b-q4km" and info["box_managed"] is False

        monkeypatch.setattr(accounts_mod, "app_config", lambda: {"box_managed": True})
        info = c.get("/api/assistant").json()
        assert info["ready"] is True and info["local_model"] is None and info["box_managed"] is True


@pytest.fixture
def no_google_network(monkeypatch):
    """Fails the test (not just the call) the instant anything tries to reach Google's API, through every
    httpx entry point the code uses: plain httpx.get (assistant.save_key) and Client.get/post/stream (a client
    built with no arguments, as assistant.reply() does when none is passed in)."""
    def guard(url):
        if "googleapis.com" in str(url):
            raise AssertionError(f"tried to reach Google: {url}")
    real_get = httpx.get
    monkeypatch.setattr(httpx, "get", lambda url, *a, **k: (guard(url), real_get(url, *a, **k))[1])
    for name in ("get", "post"):
        real = getattr(httpx.Client, name)
        monkeypatch.setattr(httpx.Client, name, lambda self, url, *a, _real=real, **k: (guard(url), _real(self, url, *a, **k))[1])
    real_stream = httpx.Client.stream
    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url, *a, **k: (guard(url), real_stream(self, method, url, *a, **k))[1])


def _session(tmp_path):
    from engine.api import Session
    from engine.project import Project
    from tests.test_project import CLASSES
    img = tmp_path / "imgs"
    img.mkdir()
    from PIL import Image
    Image.new("RGB", (32, 24)).save(img / "a.jpg")
    msgs = []
    return Session(Project.create(tmp_path / "proj", CLASSES, images=img), msgs.append), msgs


def test_cloud_disabled_never_reaches_google_from_any_entry_point(no_key, no_google_network, tmp_path, monkeypatch):
    """app.json "assistant_cloud": false, "assistant_llm" unset (a box that must never reach the internet for
    this): every entry point - REST info/chat/key, and the websocket's on_assistant* (engine/api.py, used by
    the main app too, not just notebooks) - says so, never Gemini. 2026-10-08 review: on_assistant* did not
    check this at all, so a raw websocket message could reach Gemini even with the REST routes guarded."""
    from engine import assistant
    monkeypatch.setattr(assistant, "local_url", lambda: None)
    monkeypatch.setattr(assistant, "cloud_disabled", lambda: True)
    from tests.test_host import app_client
    with app_client(tmp_path) as c:
        info = c.get("/api/assistant").json()
        assert info == {"name": "Rivet", "ready": False, "source": "none", "local_model": None,
                        "can_save": False, "tips": info["tips"], "pros": info["pros"], "box_managed": False}
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "text": "hi"}], "context": {"page": "home"}})
        assert [json.loads(x) for x in r.text.splitlines()] == [{"error": assistant.NOT_INSTALLED}]
        assert c.post("/api/assistant/key", json={"key": "AIza-whatever"}).status_code == 400

    s, msgs = _session(tmp_path)
    s.handle({"type": "assistant_info"})
    assert msgs[-1] == {"type": "assistant_info", "name": "Rivet", "ready": False, "source": "none",
                        "local_model": None, "can_save": False, "tips": [], "pros": "", "box_managed": False}
    s.handle({"type": "assistant", "id": "q1", "messages": [{"role": "user", "text": "hi"}]})
    assert msgs[-1] == {"type": "assistant", "id": "q1", "error": assistant.NOT_INSTALLED}
    s.handle({"type": "assistant_key", "key": "AIza-whatever"})
    assert msgs[-1]["source"] == "none" and "key_error" in msgs[-1]


def test_local_model_unreachable_never_falls_back_to_google(no_key, no_google_network, tmp_path, monkeypatch):
    """"assistant_llm" set but nothing answers there (e.g. the box's model is still starting): still no Gemini,
    from any entry point - it must not silently fall back just because the local model didn't answer."""
    from engine import assistant
    monkeypatch.setattr(assistant, "local_url", lambda: "http://127.0.0.1:1")   # nothing listens on port 1
    monkeypatch.setattr(assistant, "cloud_disabled", lambda: False)
    from tests.test_host import app_client
    with app_client(tmp_path) as c:
        info = c.get("/api/assistant").json()
        assert info["source"] == "local" and info["ready"] is False and info["local_model"] is None
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "text": "hi"}], "context": {"page": "home"}})
        events = [json.loads(x) for x in r.text.splitlines()]
        assert events[0] == {"offline": True, "reason": "local model not reachable"} and events[-1] == {"done": True}

    s, msgs = _session(tmp_path)
    s.handle({"type": "assistant_info"})
    assert msgs[-1]["source"] == "local" and msgs[-1]["ready"] is False


def test_notebook_widgets_ask_through_the_session(no_key, tmp_path):
    import time

    from engine.api import Session
    from engine.project import Project
    from tests.test_project import CLASSES
    img = tmp_path / "imgs"
    img.mkdir()
    from PIL import Image
    Image.new("RGB", (32, 24)).save(img / "a.jpg")
    msgs = []
    s = Session(Project.create(tmp_path / "proj", CLASSES, images=img), msgs.append)
    s.handle({"type": "assistant_info"})
    assert msgs[-1]["type"] == "assistant_info" and msgs[-1]["ready"] is False
    s.handle({"type": "assistant", "id": "q1", "messages": [{"role": "user", "text": "how do I undo?"}], "context": {"page": "notebook"}})
    t0 = time.time()
    while not any(m.get("done") for m in msgs) and time.time() - t0 < 5:
        time.sleep(0.02)
    got = [m for m in msgs if m["type"] == "assistant"]
    assert all(m["id"] == "q1" for m in got) and got[0]["offline"] and got[-1]["done"]
    s.handle({"type": "assistant_key", "key": "bad"})
    assert "doesn't look like" in msgs[-1]["key_error"]


def test_tips_are_fixed_text_and_repeated_first_questions_are_cached(no_key):
    info = assistant.info()
    assert all(t["more"].startswith(f"**{t['guide']}**") and len(t["more"]) > 80 for t in info["tips"])
    assert "Apache-2.0" in info["pros"] and "advantage" not in assistant.system_prompt({}, [])
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, text=sse("Press **T**."))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assistant._cache.clear()
    ask = lambda q: "".join(e.get("text", "") for e in assistant.reply(
        [{"role": "user", "text": q}], {"page": "annotator", "kind": "video", "frame": 3}, ["export"], key="AIza-test", client=client))
    assert ask("How do I track?") == ask("how do i track") == "Press **T**." and len(calls) == 1
    follow_up = [{"role": "user", "text": "How do I track?"}, {"role": "assistant", "text": "Press T."}, {"role": "user", "text": "and back?"}]
    list(assistant.reply(follow_up, key="AIza-test", client=client))
    assert len(calls) == 2                                                            # follow-ups are never cached
