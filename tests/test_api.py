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
