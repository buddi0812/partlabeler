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

    host.state.update(home=tmp_path / "home", sessions={}, clients={}, jobs={}, notes_home=None)
    (tmp_path / "home").mkdir()
    with TestClient(host.app) as c:
        info = c.get("/api/assistant").json()
        assert info["ready"] is False and info["tips"] and "api_key" not in json.dumps(info)
        r = c.post("/api/assistant/chat", json={"messages": [{"role": "user", "text": "how can I export?"}], "context": {"page": "home"}})
        lines = [json.loads(x) for x in r.text.splitlines()]
        assert r.headers["content-type"].startswith("application/x-ndjson")
        assert lines[0]["offline"] and "Export" in lines[1]["text"] and lines[-1] == {"done": True}
        assert c.post("/api/assistant/key", json={"key": "nope"}).status_code == 400


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
