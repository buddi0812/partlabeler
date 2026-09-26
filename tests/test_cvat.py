"""CVAT-style projects: an empty project, tasks with subsets and options, jobs with stage/state, label
colours, exports split by subset, per job, without images, and annotations uploaded into a task."""
import json
import time

from PIL import Image

from engine.project import Project, read_classes
from tests.test_project import CLASSES, images  # noqa: F401  (fixture)
from tests.test_tasks import make_video


def test_empty_project_then_tasks_with_jobs(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES, colors=["#ff0000", None, "#00ff00"])
    assert p.sources == [] and p.items == [] and p.meta["created"]
    assert [l["color"] for l in p.labels()] == ["#ff0000", None, "#00ff00"]
    v = p.add_source(video=make_video(tmp_path / "v.mp4", n=20), every=2, start=4, stop=15, quality=80,
                     subset="train", segment_size=2)
    assert [it["frame"] for it in p.items] == [4, 6, 8, 10, 12, 14] and v["subset"] == "train"
    assert [(j["start"], j["end"]) for j in p.jobs()] == [(0, 2), (2, 4), (4, 6)] and [j["id"] for j in p.jobs()] == [1, 2, 3]
    f = p.add_source(images=images, sorting="natural")
    assert p.jobs(f["id"])[0]["id"] == 4 and p.jobs(f["id"])[0]["frames"] == 3
    assert {j["state"] for j in p.jobs()} == {"new"}
    time.sleep(0.01)
    p.put(3, 1, 0, (1, 1, 5, 5))                                          # work starts in job 2
    assert [j["state"] for j in p.jobs()][:3] == ["new", "in progress", "new"] and p.jobs()[1]["labeled"] == 1
    j = p.set_job(2, stage="validation")                                  # a new stage without a state: new
    assert (j["stage"], j["state"]) == ("validation", "new")
    p.set_job(1, stage="acceptance", state="completed")
    st = Project.task_status(p.jobs(v["id"]))
    assert st == {"status": "annotation", "done": 1, "review": 1, "annotating": 1, "total": 3}
    assert Project(tmp_path / "proj").jobs()[0]["stage"] == "acceptance"     # saved


def test_label_constructor_renames_recolours_and_deletes(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES, images=images)
    p.put(0, 1, 0, (1, 1, 5, 5)); p.put(0, 2, 1, (2, 2, 6, 6)); p.put(1, 3, 2, (3, 3, 7, 7))
    res = p.set_labels([{"name": "grille", "color": "#123456", "from": 2}, {"name": "left_drl", "from": 0},
                        {"name": "bolt", "color": None, "from": None}])      # right_drl deleted, grill renamed first
    assert p.classes == ["grille", "left_drl", "bolt"] and res["removed"] == 1
    assert [b["cls"] for k in (0, 1) for b in p.boxes(k)] == [1, 0]
    for bad in ([{"name": ""}], [{"name": "a"}, {"name": "a"}], [{"name": "a", "color": "red"}]):
        try:
            p.set_labels(bad)
            raise AssertionError("bad labels must be refused")
        except ValueError:
            pass


def test_export_splits_by_subset_and_by_job(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES)
    p.add_source(video=make_video(tmp_path / "a.mp4", n=10), every=5, subset="Train")       # items 0-1
    p.add_source(video=make_video(tmp_path / "b.mp4", n=10), every=5, subset="Validation")  # items 2-3
    p.add_source(images=images)                                                               # items 4-6, no subset
    for k in range(7):
        p.put(k, k + 1, 0, (1, 1, 5, 5))
    res = p.export("yolo", tmp_path / "y")
    assert res["images"] == 7
    assert sorted(d.name for d in (tmp_path / "y/images").iterdir()) == ["Train", "Validation", "default"]
    yaml = (tmp_path / "y/data.yaml").read_text()
    assert "train: images/Train" in yaml and "val: images/Validation" in yaml and read_classes(tmp_path / "y/data.yaml") == CLASSES
    p.export("coco", tmp_path / "c")
    assert sorted(f.name for f in (tmp_path / "c/annotations").iterdir()) == ["instances_Train.json", "instances_Validation.json", "instances_default.json"]
    one = p.export("yolo", tmp_path / "t", tasks=["b"])                     # one task: flat, as before
    assert one["images"] == 2 and (tmp_path / "t/images/b_f000005.jpg").exists()
    job = p.jobs()[0]
    assert p.export("yolo", tmp_path / "j", jobs=[job["id"]])["images"] == 2
    bare = p.export("coco", tmp_path / "n", save_images=False)
    data = json.loads((tmp_path / "n/annotations/instances_Train.json").read_text())
    assert not (tmp_path / "n/images").exists() and data["images"][0]["file_name"].endswith("f000000.jpg")
    cvat = (tmp_path / "x").mkdir() or p.export("cvat", tmp_path / "x")
    assert 'subset="Validation"' in (tmp_path / "x/annotations.xml").read_text() and bare["images"] == 7


def test_upload_annotations_into_a_task_replace_or_append(tmp_path):
    p = Project.create(tmp_path / "proj", CLASSES)
    p.add_source(video=make_video(tmp_path / "a.mp4", n=10), every=5)
    p.add_source(video=make_video(tmp_path / "b.mp4", n=10), every=5)
    p.put(2, 1, 0, (1, 1, 5, 5))
    labels = tmp_path / "labels"; labels.mkdir()
    (labels / "b_f000005.txt").write_text("2 0.5 0.5 0.25 0.5\n")
    (labels / "a_f000000.txt").write_text("1 0.5 0.5 0.25 0.5\n")             # another task: not touched
    b = range(*p.ranges[2])                                                    # task b (tasks count from 1)
    assert p.import_yolo(labels, items=b)["boxes"] == 1 and p.boxes(0) == []
    assert len(p.boxes(2)) == 1 and p.boxes(3)[0]["cls"] == 2               # append kept item 2's box
    p.import_yolo(labels, items=b, replace=True)
    assert p.boxes(2) == [] and len(p.boxes(3)) == 1                         # replace cleared the task first


def test_natural_sorting_of_an_image_folder(tmp_path):
    root = tmp_path / "shots"; root.mkdir()
    for n in ("img10", "img2", "img1"):
        Image.new("RGB", (8, 8)).save(root / f"{n}.png")
    p = Project.create(tmp_path / "proj", CLASSES, images=root, sorting="natural")
    assert [it["name"] for it in p.items] == ["img1", "img2", "img10"]


def test_backup_and_restore_keep_everything(tmp_path, images):
    from engine.project import backup_project, restore_project
    p = Project.create(tmp_path / "proj", CLASSES, colors=["#ff0000", None, None])
    p.add_source(video=make_video(tmp_path / "a.mp4", n=10), every=5, subset="train")
    p.add_source(images=images)
    p.put(1, 1, 0, (1, 1, 5, 5)); p.put(3, 2, 2, (2, 2, 6, 6)); p.set_job(1, stage="validation")
    info = backup_project(tmp_path / "proj", tmp_path / "b.zip")
    assert info["items"] == 5
    dest = restore_project(tmp_path / "b.zip", tmp_path / "home")
    q = Project(dest)
    assert dest.name == "proj" and [it["name"] for it in q.items] == [it["name"] for it in p.items]
    assert q.boxes(3)[0]["box"] == [2, 2, 6, 6] and q.jobs()[0]["stage"] == "validation" and q.labels()[0]["color"] == "#ff0000"
    assert str(dest) in q.sources[1]["source"]                                 # pictures now inside the project
    assert restore_project(tmp_path / "b.zip", tmp_path / "home").name == "proj_2"
