"""Rivet driving PartLabeler through its own tools (ui/mcp.py), on a local OpenAI-compatible tool-calling model
(llama.cpp, or any such server, given in app.json "assistant_llm"). This is a separate, local-only path used
INSTEAD of engine/assistant.py's Gemini chat when configured and reachable; Gemini users and everyone without
"assistant_llm" set are unaffected. app.json "assistant_llm_key" (optional) is sent as `Authorization: Bearer
<key>` on every call to that server (e.g. a Hawk-AI box's llama-server runs with --api-key); absent, no header
is sent, as before "assistant_llm_key" existed. Never logged.

The model id is never hardcoded: read from the server's own GET /v1/models at each chat, so whatever model tier
is running (a small 4B/16k-context model and up) needs no PartLabeler change. The tool list and prompt are
sized for a small model: a short, fixed subset of PartLabeler's tools (~2.5k tokens of schema), not all of them.

Destructive tools (delete, confirm, train, export) never run from the model's own tool call: reply() pauses and
yields {"confirm": {...}} instead; resume() runs them only after the person clicks Confirm (assistant_confirm in
ui/host_fastapi.py). Every tool call actually run is appended to <projects>/.partlabeler/assistant-actions.jsonl.

ponytail: tool-call arguments are read from one whole (non-streaming) model reply, not assembled from streamed
deltas - simpler, and the text answer is still sent to the browser in one piece rather than token-by-token.
Upgrade to streamed deltas if the box's answers feel too slow to start.
"""
import json
import time
import uuid

import httpx

from engine import assistant
from ui import mcp

H = None                              # the host module (ui/host_fastapi.py), set by mount()

TOOL_NAMES = ("list_projects", "get_project", "create_project", "add_tasks", "set_labels", "delete_task",
             "clear_parts", "confirm_frames", "get_job", "teach", "list_runs", "label_tasks", "export_model",
             "list_exports", "list_stations", "deploy_link", "undo")
DESTRUCTIVE = {"delete_task", "clear_parts", "set_labels", "confirm_frames", "teach", "label_tasks", "export_model"}
MAX_TOOL_TURNS = 6                    # tool-calling turns per message: a bound against loops
PENDING: dict = {}                    # confirm id -> {"messages", "calls", "ts", "actor"}
PENDING_TTL_S, PENDING_MAX = 900, 50


def mount(host) -> None:
    global H
    H = host


# engine/assistant.py's local_url()/cloud_disabled()/backend() are the one shared decision every caller (this
# module, ui/host_fastapi.py's REST routes, engine/api.py's websocket handlers) must use, so they can't drift.
def base_url() -> str | None:
    return assistant.local_url()


def _auth_headers() -> dict:
    """Authorization for base_url()'s server, when app.json "assistant_llm_key" (or the env var) is set - e.g. a
    Hawk-AI box's llama-server runs with --api-key, so every call needs a Bearer token or the server 401s and
    Rivet looks unreachable. Absent (older boxes, or a key-less local server): no header, as before."""
    key = assistant.local_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def model_id(url: str, client: httpx.Client | None = None) -> str | None:
    """The one model name the server currently serves, or None if it can't be reached."""
    try:
        r = (client or httpx).get(f"{url.rstrip('/')}/v1/models", timeout=5, headers=_auth_headers())
        r.raise_for_status()
        data = r.json().get("data") or []
        return data[0]["id"] if data else None
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None


def _tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": s}}
            for n, (_, d, s) in mcp.TOOLS.items() if n in TOOL_NAMES]


def _system_prompt() -> str:
    return ("You operate PartLabeler, an industrial dataset annotation tool, only through the tools you were "
            "given, as the signed-in person. Look before you act: list_projects and get_project first. Keep "
            "replies short and concrete. Deleting, confirming frames, training and exporting are never run by "
            "you directly - the person confirms them in the app, and you only see the result afterwards. If a "
            "tool call fails, say so plainly and stop instead of guessing. deploy_link does not deploy anything "
            "itself: give the person its returned url as plain text in your reply so they can open it and approve "
            "on the box's own screen.")


def _summarise(name: str, args: dict) -> str:
    try:
        p = args.get("project", "?")
        if name == "create_project":
            return f"Create project \"{args.get('name', '?')}\""
        if name == "add_tasks":
            n = len(args.get("sources") or [])
            return f"Add {n} task{'s' if n != 1 else ''} to \"{p}\""
        if name == "set_labels":
            names = [x.get("name", x) if isinstance(x, dict) else x for x in (args.get("labels") or [])]
            return f"Replace the labels of \"{p}\" with: {', '.join(names) or '(none)'}"
        if name == "delete_task":
            return f"Delete task {args.get('task', '?')} of \"{p}\" (cannot be undone)"
        if name == "clear_parts":
            labels = ", ".join(args.get("labels") or []) or "all labels"
            return f"Remove {labels} from frames of \"{p}\""
        if name == "confirm_frames":
            n = len(args.get("items") or [])
            return f"{'Confirm' if args.get('confirmed', True) else 'Un-confirm'} {n} frame{'s' if n != 1 else ''} in \"{p}\""
        if name == "teach":
            return f"Train a detector on \"{p}\" (takes a while, uses the GPU)"
        if name == "label_tasks":
            return f"Label task(s) {args.get('tasks')} of \"{p}\" with Teach run \"{args.get('run', '?')}\""
        if name == "export_model":
            return f"Export Teach run \"{args.get('run', '?')}\" for production"
    except Exception:
        pass
    return f"{name} {json.dumps(args)}"


def _log(entry: dict) -> None:
    path = H.home() / ".partlabeler" / "assistant-actions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _gc_pending() -> None:
    now = time.time()
    for k in [k for k, v in PENDING.items() if now - v["ts"] > PENDING_TTL_S]:
        PENDING.pop(k, None)
    while len(PENDING) > PENDING_MAX:
        PENDING.pop(next(iter(PENDING)), None)


def _chat_once(url: str, model: str, messages: list[dict], tools: list[dict], client: httpx.Client | None = None) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0.2}
    if tools:
        body["tools"], body["tool_choice"] = tools, "auto"
    r = (client or httpx).post(f"{url.rstrip('/')}/v1/chat/completions", json=body, timeout=120, headers=_auth_headers())
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _run_tool(call: dict, actor, confirmed: bool) -> str:
    name, args = call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")
    try:
        out, _img = mcp.call(name, args)
        result = json.dumps(out)[:4000]
    except Exception as e:
        result = f"error: {mcp.error_text(e)}"
    _log({"ts": time.time(), "actor": actor, "tool": name, "args": args, "result": result[:500], "confirmed": confirmed})
    return result


def reply(messages, ctx=None, actor=None, client: httpx.Client | None = None):
    """Yield the same event shapes as engine.assistant.reply ({"model"}/{"offline"}, {"text"}, {"done"}), plus
    {"confirm": {"id", "actions": [{"tool", "args", "summary"}]}} when a destructive tool call needs a person's OK."""
    url = base_url()
    if not url:
        yield {"error": "no local model configured"}
        return
    model = model_id(url, client)
    if not model:
        yield {"offline": True, "reason": "local model not reachable"}
        yield {"text": "The local model is not answering right now. Try again in a moment."}
        yield {"done": True}
        return
    convo = [{"role": "system", "content": _system_prompt()}]
    convo += [{"role": "user" if m.get("role") == "user" else "assistant", "content": m.get("text", "")} for m in (messages or []) if m.get("text")]
    tools = _tool_schemas()
    yield {"model": model}
    own = client is None
    client = client or httpx.Client()
    try:
        for _ in range(MAX_TOOL_TURNS):
            try:
                msg = _chat_once(url, model, convo, tools, client)
            except httpx.HTTPError as e:
                yield {"offline": True, "reason": str(e)}
                yield {"text": "The local model could not be reached."}
                yield {"done": True}
                return
            calls = msg.get("tool_calls") or []
            if not calls:
                yield {"text": msg.get("content") or "(no answer)"}
                yield {"done": True}
                return
            convo.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            blocked = [c for c in calls if c["function"]["name"] in DESTRUCTIVE]
            if blocked:
                cid = uuid.uuid4().hex[:12]
                _gc_pending()
                PENDING[cid] = {"messages": convo, "calls": calls, "ts": time.time(), "actor": actor}
                yield {"confirm": {"id": cid, "actions": [
                    {"tool": c["function"]["name"], "args": json.loads(c["function"]["arguments"] or "{}"),
                     "summary": _summarise(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))}
                    for c in calls]}}
                return
            for c in calls:
                result = _run_tool(c, actor, confirmed=False)
                convo.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        yield {"text": "That took more steps than I can do in one go. Ask again to continue from here."}
        yield {"done": True}
    finally:
        if own:
            client.close()


def resume(call_id: str, actor=None, client: httpx.Client | None = None):
    """Run a confirmed tool call (or calls) and let the model phrase the result. Yields reply()'s event shapes."""
    pending = PENDING.pop(call_id, None)
    if not pending:
        yield {"error": "That request is no longer pending (it may have expired)."}
        return
    convo = pending["messages"]
    for c in pending["calls"]:
        result = _run_tool(c, actor if actor is not None else pending.get("actor"), confirmed=True)
        convo.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    url = base_url()
    model = model_id(url, client) if url else None
    if not url or not model:
        yield {"text": "Done."}
        yield {"done": True}
        return
    yield {"model": model}
    own = client is None
    client = client or httpx.Client()
    try:
        msg = _chat_once(url, model, convo, [], client)
        yield {"text": msg.get("content") or "Done."}
    except httpx.HTTPError:
        yield {"text": "Done."}
    finally:
        if own:
            client.close()
    yield {"done": True}


def cancel(call_id: str, actor=None) -> bool:
    pending = PENDING.pop(call_id, None)
    if not pending:
        return False
    for c in pending["calls"]:
        _log({"ts": time.time(), "actor": actor if actor is not None else pending.get("actor"),
             "tool": c["function"]["name"], "args": json.loads(c["function"]["arguments"] or "{}"),
             "result": "declined by the person", "confirmed": False})
    return True
