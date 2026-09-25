"""The annotator's message flow, end to end, without any model (box tool, relabel, delete, review, export)."""
from engine.api import Session
from engine.project import Project, read_classes
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)


def last(msgs, kind):
    return next(m for m in reversed(msgs) if m["type"] == kind)


def test_box_relabel_delete_review_export(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    msgs = []
    s = Session(p, msgs.append)

    s.handle({"type": "ready"})
    assert last(msgs, "project")["classes"] == CLASSES and last(msgs, "project")["count"] == 4
    assert last(msgs, "item")["item"] == 0 and last(msgs, "status")["statuses"] == [0, 0, 0, 0]

    s.handle({"type": "box", "item": 1, "cls": 0, "box": [5, 5, 20, 15]})
    obj = last(msgs, "item")["boxes"][0]["obj"]
    assert last(msgs, "item")["boxes"][0]["source"] == "manual"
    s.handle({"type": "box", "item": 1, "cls": 2, "box": [6, 6, 22, 16], "obj": obj})   # redraw keeps the class
    assert [(b["cls"], b["box"]) for b in last(msgs, "item")["boxes"]] == [(0, [6, 6, 22, 16])]

    s.handle({"type": "set_class", "obj": obj, "cls": 1, "item": 1})
    assert last(msgs, "item")["boxes"][0]["cls"] == 1
    s.handle({"type": "box", "item": 2, "cls": 2, "box": [0, 0, 10, 10]})
    s.handle({"type": "delete", "item": 2, "obj": last(msgs, "item")["boxes"][0]["obj"]})
    assert last(msgs, "item")["boxes"] == []

    s.handle({"type": "review", "item": 1})
    assert last(msgs, "status")["statuses"] == [0, 4, 0, 0]

    s.handle({"type": "export", "format": "yolo", "reviewed_only": True})
    s.job.join()
    note = last(msgs, "notify")["item"]
    assert note["title"] == "Exported YOLO dataset" and note["detail"].startswith("1 images, 1 boxes")
    assert note["action"]["type"] == "folder"
    out = next((p.folder / "exports").glob("yolo_*"))
    assert read_classes(out / "classes.txt") == CLASSES
    assert (out / "labels/car_front_f000005.txt").read_text().startswith("1 ")
    assert not any(m["type"] == "error" for m in msgs)


def test_unknown_message_reports_an_error(tmp_path, video):
    msgs = []
    Session(Project.create(tmp_path / "proj", CLASSES, video=video), msgs.append).handle({"type": "nope"})
    assert msgs[-1]["type"] == "notify" and msgs[-1]["item"]["level"] == "error"


def test_undo_restores_each_step_and_settings_persist(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    msgs = []
    s = Session(p, msgs.append)
    s.handle({"type": "box", "item": 0, "cls": 0, "box": [1, 1, 9, 9]})
    obj = p.boxes(0)[0]["obj"]
    s.handle({"type": "set_class", "obj": obj, "cls": 2, "item": 0})
    s.handle({"type": "review", "item": 0})
    s.handle({"type": "undo"})
    assert not p.is_reviewed(0) and last(msgs, "item_changed")["items"] == [0]
    s.handle({"type": "undo"})
    assert p.boxes(0)[0]["cls"] == 0
    s.handle({"type": "undo"})
    assert p.boxes(0) == []
    s.handle({"type": "undo"})
    assert last(msgs, "toast")["text"] == "Nothing to undo"

    s.handle({"type": "settings", "parent": " engine block "})
    assert Project(p.folder).meta["parent"] == "engine block" and last(msgs, "project")["parent"] == "engine block"
    s.handle({"type": "settings", "parent": ""})
    assert Project(p.folder).meta["parent"] is None
    s.handle({"type": "find_all", "item": 0, "obj": 999})       # nothing selected: a hint, no job
    assert "Select a box first" in last(msgs, "toast")["text"]
    s.handle({"type": "track", "item": 0, "count": 5, "direction": -1})
    assert last(msgs, "toast")["text"] == "No frames before this one"
    assert not any(m["type"] == "error" for m in msgs)


def test_notifications_record_events_merge_repeats_and_persist(tmp_path, video):
    from engine.notify import Notifications
    p = Project.create(tmp_path / "proj", CLASSES, video=video)
    msgs = []
    notes = Notifications(tmp_path / "notes.json", on_change=msgs.append)
    s = Session(p, msgs.append, notes=notes)
    s.handle({"type": "ready"})
    assert last(msgs, "notifications") == {"type": "notifications", "items": [], "unread": 0}
    s.handle({"type": "box", "item": 0, "cls": 0, "box": [1, 1, 9, 9]})
    for item in (0, 1, 2):
        s.handle({"type": "review", "item": item})
    confirm = last(msgs, "notify")
    assert confirm["item"]["title"] == "Confirmed 3 frames" and confirm["item"]["count"] == 3 and confirm["unread"] == 1
    s.handle({"type": "delete", "item": 0, "obj": p.boxes(0)[0]["obj"]})
    assert last(msgs, "notify")["item"]["title"] == "Deleted a left_drl box on frame 1"
    assert [n["title"] for n in notes.snapshot()["items"]] == ["Deleted a left_drl box on frame 1", "Confirmed 3 frames"]

    reopened = Notifications(tmp_path / "notes.json")             # survives a restart
    assert reopened.snapshot()["unread"] == 2
    s.handle({"type": "notifications_read"})
    assert last(msgs, "notifications")["unread"] == 0 and Notifications(tmp_path / "notes.json").snapshot()["unread"] == 0
    s.handle({"type": "nope"})
    assert last(msgs, "notify")["item"]["level"] == "error"
    s.handle({"type": "notifications_clear"})
    assert last(msgs, "notifications")["items"] == []


def test_notebook_outbox_holds_thread_messages_until_a_poll():
    """Colab drops widget messages sent from threads: the widget host queues them and flushes on each poll."""
    import threading

    from ui.host_widget import Outbox
    sent = []
    out = Outbox(sent.append)
    t = threading.Thread(target=lambda: [out({"n": i}) for i in range(3)])
    t.start()
    t.join()
    assert sent == []                                            # waiting for the page to ask
    out({"n": "main"})                                           # a main-thread send flushes first, keeps order
    assert sent == [{"n": 0}, {"n": 1}, {"n": 2}, {"n": "main"}]
    threading.Thread(target=lambda: out({"n": 4})).start()
    import time
    time.sleep(0.05)
    out.flush()
    assert sent[-1] == {"n": 4}


def test_colab_run_keeps_the_cell_running_until_the_annotator_sits_idle(tmp_path, video, monkeypatch):
    """Colab only counts a running cell as activity: run() loops while the annotator is used and returns after
    idle_minutes without a message from it (automatic polls do not count)."""
    import sys
    import types
    from contextlib import contextmanager

    import ui.host_widget as hw_
    clock = {"t": 1000.0}
    monkeypatch.setattr(hw_.time, "time", lambda: clock["t"])
    monkeypatch.setattr(hw_.time, "sleep", lambda s: None)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    w = hw_.Annotator(Project.create(tmp_path / "proj", CLASSES, video=video))
    monkeypatch.setattr(w, "send", lambda *a, **k: None)
    w.outbox = hw_.Outbox(lambda m: None)
    calls = []

    def poll(n):                                     # each poll call: one minute passes
        calls.append(clock["t"])
        clock["t"] += 60
        if len(calls) == 3:
            w._on_msg(w, {"type": "goto", "item": 1}, [])   # the user does something at minute 3
        else:
            w._on_msg(w, {"type": "poll"}, [])              # automatic polls keep coming

    @contextmanager
    def ui_events():
        yield poll
    monkeypatch.setitem(sys.modules, "jupyter_ui_poll", types.SimpleNamespace(ui_events=ui_events))
    monkeypatch.setattr("IPython.display.display", lambda *a, **k: None)
    w.run(idle_minutes=10)
    assert len(calls) == 3 + 10                     # 10 idle minutes after the last real action, then it returns
