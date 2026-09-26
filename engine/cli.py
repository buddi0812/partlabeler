"""partlabeler command line (also `python -m engine.cli`).

    partlabeler app                                   the annotator start page in the browser
    partlabeler new PROJECT --video V --classes F     a project from a video (or --images DIR)
    partlabeler add PROJECT --video V                 another video (or --images DIR) as a new task
    partlabeler backup PROJECT [--task T] | restore ZIP   a project (or some tasks) as one zip, and back
    partlabeler import-task PROJECT ZIP               add a backup's tasks, with their progress, to a project
    partlabeler account list | reset NAME | remove NAME   accounts of the local app (forgotten password: reset)
    partlabeler export PROJECT --format coco          write the dataset
    partlabeler teach DATASET --run RUN               learn a labeled source (Teach & Transfer)
    partlabeler transfer RUN VIDEO... --out DIR       label similar videos / image folders like the source
    partlabeler review PROJECT --video V --labels DIR/labels --classes F    check transferred labels
    partlabeler update [--check]                      get the latest version from GitHub (projects are kept)
    partlabeler models | doctor

Heavy modules (torch, rfdetr, SAM 3) load inside the commands, so --help stays instant.
"""
import json
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import click
import typer

FORMATS = ("yolo", "coco", "cvat", "voc", "labelstudio", "folders", "csv")   # folders, csv: image classes
SIZES = ("nano", "small", "medium")

app = typer.Typer(help="PartLabeler: annotated datasets of machine parts from videos and image folders.",
                  no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)


@contextmanager
def _bar():
    """A rich progress bar; yields progress(done, total, text) for the engine."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as bar:
        task = bar.add_task("Starting", total=None)
        yield lambda done, total, text="": bar.update(task, completed=done, total=total or None, description=text)


def _classes(path: Path) -> list[str]:
    from engine.project import read_classes
    names = read_classes(path)
    if not names:
        raise typer.BadParameter(f"{path} lists no classes", param_hint="--classes")
    return names


@app.command("app")
def app_cmd(home: Path = typer.Option(Path("projects"), help="Folder holding the projects."),
            port: int = typer.Option(8765, help="Local port."),
            browser: bool = typer.Option(True, "--browser/--no-browser", help="Open the page in a browser.")):
    """Start the annotator (start page, projects, Teach & Transfer) at http://127.0.0.1:PORT."""
    from engine.update import finish_pending
    finish_pending(log=typer.echo)                             # packages an update from the start screen left for now
    from ui.host_fastapi import serve
    serve(home, port, browser)


@app.command()
def update(home: Path = typer.Option(None, help="Projects folder whose labels are backed up first (default: projects/ next to the app)."),
           check: bool = typer.Option(False, "--check", help="Only say whether a newer version exists.")):
    """Update PartLabeler from GitHub. Projects, labels, exports and models are kept; labels are backed up first."""
    from engine import update as up
    home = home or up.APP / "projects"
    try:
        res = up.check()
    except Exception as e:
        typer.echo(f"Could not reach GitHub: {type(e).__name__}: {e}")
        raise typer.Exit(1)
    typer.echo(f"This copy: {res['current']}   newest on GitHub: {res['latest']} ({res['date'][:10]})")
    if not res["available"]:
        typer.echo(res["reason"] or "PartLabeler is up to date.")
        raise typer.Exit(0 if res["can_update"] else 1)
    for line in res["changes"]:
        typer.echo(f"  - {line}")
    if not res["can_update"]:
        typer.echo(res["reason"])
        raise typer.Exit(1)
    if check:
        typer.echo("Run `partlabeler update` (or update_windows.bat) to update.")
        return
    try:
        out = up.update(home, log=typer.echo)
    except Exception as e:
        typer.echo(f"Not updated: {e}")
        raise typer.Exit(1)
    typer.echo(f"Updated {out['from']} -> {out['to']}. Python packages: {out['packages']}.")
    if out["backup"]:
        typer.echo(f"Labels backed up to {out['backup']}")
    typer.echo("Start PartLabeler again to use the new version.")


@app.command()
def new(project: Path = typer.Argument(..., help="New project folder."),
        video: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Source video."),
        images: Path | None = typer.Option(None, exists=True, file_okay=False, help="Source image folder."),
        classes: Path = typer.Option(..., exists=True, dir_okay=False, help="classes.txt or data.yaml."),
        every: int = typer.Option(5, min=1, help="Video: keep 1 frame in N."),
        task: str = typer.Option("detect", click_type=click.Choice(["detect", "segment", "classify"]),
                                 help="Project type, fixed once made: detect (object detection, boxes), segment (segmentation, outlines) or classify (classification, one class per image)."),
        frames: str = typer.Option("jpg", click_type=click.Choice(["jpg", "webp"]),
                                   help="Video frames: jpg (compact, visually lossless) or webp (lossless, about 2x the space)."),
        labels: Path | None = typer.Option(None, exists=True, file_okay=False, help="YOLO labels to import.")):
    """Create a project from a video or an image folder (more can be added: `partlabeler add`), optionally with existing YOLO labels."""
    from engine.project import Project
    if (video is None) == (images is None):
        raise typer.BadParameter("give exactly one of --video or --images")
    names = _classes(classes)
    with _bar() as progress:
        p = Project.create(project, names, video=video, images=images, every=every, progress=progress, task=task,
                           frame_format=frames)
    typer.echo(f"{p.folder}: {len(p.items)} {'frames' if video else 'images'}, {len(names)} classes")
    if labels:
        st = p.import_yolo(labels)
        typer.echo(f"imported {st['boxes']} boxes from {st['files_matched']} label files"
                   + (f" ({st['files_unmatched']} matched no frame or image)" if st["files_unmatched"] else ""))


@app.command()
def add(project: Path = typer.Argument(..., exists=True, file_okay=False, help="Project folder."),
        video: Path | None = typer.Option(None, exists=True, dir_okay=False, help="A video to add as a task."),
        images: Path | None = typer.Option(None, exists=True, file_okay=False, help="An image folder to add as a task."),
        every: int = typer.Option(5, min=1, help="Video: keep 1 frame in N."),
        subset: str = typer.Option("", help="Subset (Train, Validation, Test or any word): its own folders in project exports."),
        segment_size: int = typer.Option(0, min=0, help="Frames per job (0: one job for the task)."),
        start: int | None = typer.Option(None, help="Video: first frame."), stop: int | None = typer.Option(None, help="Video: last frame."),
        quality: int = typer.Option(95, min=5, max=100, help="Video: JPEG quality of the stored frames."),
        lossless: bool = typer.Option(False, "--lossless", help="Video: store frames losslessly (exact pixels, about 2x space)."),
        name: str | None = typer.Option(None, help="Task name (default: the file or folder name).")):
    """Add a video or an image folder to a project as a new task (as CVAT's Create a new task)."""
    from engine.api import describe
    from engine.project import Project
    if (video is None) == (images is None):
        raise typer.BadParameter("give exactly one of --video or --images")
    p = Project(project)
    with _bar() as progress:
        t = p.add_source(video=video, images=images, every=every, progress=progress, subset=subset, name=name,
                         segment_size=segment_size, start=start, stop=stop, quality=quality, lossless=lossless or None)
    a, b = p.ranges[t["id"]]
    typer.echo(f"added task {t['name']}: {b - a} {'frames' if video else 'images'} ({describe(t)}); "
               f"the project has {len(p.sources)} tasks, {len(p.items)} frames/images")


@app.command()
def backup(project: Path = typer.Argument(..., exists=True, file_okay=False, help="Project folder."),
           task: list[str] = typer.Option(None, "--task", help="Only this task (name), as CVAT's Backup task; repeat for several."),
           out: Path | None = typer.Option(None, help="Zip file (default: <projects>/_backups/project_<name>_backup_<time>.zip).")):
    """Back up a project into one zip (settings, labels, frames and folder pictures), as CVAT's Backup project."""
    from engine.project import Project, backup_project
    ids = None
    if task:
        p = Project(project)
        by_name = {t["name"]: t["id"] for t in p.sources}
        p.db.close()
        if missing := [t for t in task if t not in by_name]:
            raise typer.BadParameter(f"no task {', '.join(missing)}; tasks: {', '.join(by_name)}", param_hint="--task")
        ids = [by_name[t] for t in task]
    kind = f"task_{project.name}_{'_'.join(task)}" if task else f"project_{project.name}"
    out = out or project.parent / "_backups" / f"{kind}_backup_{time.strftime('%Y_%m_%d_%H_%M_%S')}.zip"
    res = backup_project(project, out, tasks=ids)
    typer.echo(f"backed up {res['items']} frames or images, {res['size'] / 2**20:.0f} MB -> {res['file']}")


@app.command()
def restore(backup_zip: Path = typer.Argument(..., exists=True, dir_okay=False, help="A project backup zip."),
            home: Path = typer.Option(Path("projects"), help="Projects folder to restore into."),
            name: str | None = typer.Option(None, help="New project name (default: the backup's, made unique).")):
    """Create a project from a backup zip (CVAT's Create from backup)."""
    from engine.project import restore_project
    typer.echo(f"restored -> {restore_project(backup_zip, home, name)}")


@app.command("import-task")
def import_task(project: Path = typer.Argument(..., exists=True, file_okay=False, help="Project folder."),
                backup_zip: Path = typer.Argument(..., exists=True, dir_okay=False, help="A task or project backup zip.")):
    """Add the tasks of a backup zip to a project with their frames, labels, confirmed frames and jobs."""
    from engine.project import Project
    res = Project(project).add_tasks_from(backup_zip)
    typer.echo(f"added {', '.join(res['tasks'])}: {res['items']} frames or images"
               + (f"; new labels: {', '.join(res['labels_added'])}" if res["labels_added"] else ""))


account_app = typer.Typer(help="Accounts of the local app: each keeps its settings and projects folder.", no_args_is_help=True)
app.add_typer(account_app, name="account")


@account_app.command("list")
def account_list():
    """The accounts on this computer."""
    from engine.accounts import Accounts
    for u in Accounts().users() or [{"username": "(none yet: the app asks for one on first start)", "name": ""}]:
        typer.echo(f"{u['username']}  {u['name'] if u['name'] != u['username'] else ''}".rstrip())


@account_app.command("reset")
def account_reset(username: str = typer.Argument(..., help="The account's username.")):
    """Set a new password (for a forgotten one); signs the account out everywhere."""
    from engine.accounts import Accounts
    acc = Accounts()
    if not any(u["username"].lower() == username.lower() for u in acc.users()):
        raise typer.BadParameter(f"no account called {username}")
    password = typer.prompt("New password", hide_input=True, confirmation_prompt=True)
    try:
        acc.set_password(username, password)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    typer.echo(f"New password set for {username}.")


@account_app.command("remove")
def account_remove(username: str = typer.Argument(..., help="The account's username.")):
    """Remove an account and its settings; its projects stay where they are."""
    from engine.accounts import Accounts
    try:
        Accounts().delete(username)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0]))
    typer.echo(f"Removed {username}.")


@app.command()
def export(project: Path = typer.Argument(..., exists=True, file_okay=False, help="Project folder."),
           fmt: str = typer.Option("yolo", "--format", click_type=click.Choice(FORMATS), help="Dataset format."),
           reviewed_only: bool = typer.Option(False, "--reviewed-only", help="Only frames/images marked reviewed."),
           task: list[str] = typer.Option(None, "--task", help="Only this task (name); repeat for several. Default: all."),
           job: list[int] = typer.Option(None, "--job", help="Only this job (number); repeat for several."),
           no_images: bool = typer.Option(False, "--no-images", help="Label files only (CVAT's Save images off)."),
           out: Path | None = typer.Option(None, help="Output folder (default PROJECT/exports/<format>_<time>).")):
    """Write the project's labels as a dataset: every task, or the ones named with --task."""
    from engine.project import Project
    out = out or project / "exports" / f"{fmt}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        res = Project(project).export(fmt, out, reviewed_only, tasks=task or None, jobs=job or None, save_images=not no_images)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--task" if task else "--format")
    what = f", {res['boxes']} boxes" if "boxes" in res else ""
    typer.echo(f"exported {res['images']} images{what} -> {res.get('folder') or res.get('file')}")
    if res.get("no_outline"):
        typer.echo(f"left out {res['no_outline']} boxes without an outline (open the project and use 'Outline boxes')")


@app.command()
def teach(dataset: Path = typer.Argument(..., exists=True, file_okay=False,
                                         help="Labeled source: images/, labels/, classes.txt or data.yaml."),
          run: Path = typer.Option(..., help="Run folder (model, settings, report); rerun to resume."),
          parent: str | None = typer.Option(None, help="Object the parts sit on, e.g. 'engine block'; crops to it."),
          size: str = typer.Option("small", click_type=click.Choice(SIZES), help="RF-DETR size."),
          epochs: int = typer.Option(30, min=1),
          resolution: int = typer.Option(640, help="Model input size, a multiple of 32."),
          held_out: float = typer.Option(0.2, min=0.05, max=0.5, help="Share of frames held out (time blocks)."),
          aug: str = typer.Option("strong", click_type=click.Choice(("strong", "medium", "off")))):
    """Learn a labeled source and prove on held-out frames that the model reproduces its labels."""
    from engine.teach import teach as run_teach
    with _bar() as progress:
        r = run_teach(dataset, run, parent, size, epochs, resolution, held_out, aug, progress=progress)
    if r["status"] != "done":
        typer.echo(f"stopped during {r['step']}; run the same command again to resume")
        raise typer.Exit(1)
    ho = r["held_out"]
    typer.echo(f"knowledge score {r['knowledge_score']:.1f}/100: held-out mAP50 {ho['mAP50']:.3f}, mAP50-95 "
               f"{ho['mAP50_95']:.3f}, recall {ho['recall']:.3f}, precision {ho['precision']:.3f}, threshold {ho['threshold']:.2f}")
    typer.echo("colour stress, mAP50 drop in points: " + ", ".join(f"{k} {v['drop']:.1f}" for k, v in ho["stress"].items()))
    typer.echo(f"{'PASS' if r['passed'] else 'FAIL'} ({len(r['warnings'])} warnings) -> {Path(run) / 'report.md'}")


@app.command()
def transfer(run: Path = typer.Argument(..., exists=True, file_okay=False, help="Run folder made by teach."),
             sources: list[Path] = typer.Argument(..., exists=True, help="Videos and/or image folders."),
             out: Path = typer.Option(..., help="Output folder; one subfolder per source."),
             every: int | None = typer.Option(None, min=1, help="Label 1 frame in N (default: as the source)."),
             threshold: float | None = typer.Option(None, min=0.0, max=1.0, help="Confidence (default: from teach)."),
             parent: str | None = typer.Option(None, help="Override the run's parent object ('' = none)."),
             preview: bool = typer.Option(True, "--preview/--no-preview", help="Write preview.mp4 per video."),
             max_frames: int | None = typer.Option(None, min=1, help="Only the first N frames/images of each source."),
             tracks: bool = typer.Option(True, "--tracks/--no-tracks",
                                         help="Check videos along tracks: list label flips, brief misses and one-frame boxes to check.")):
    """Label other videos or image folders in the same format as the taught source."""
    from engine.transfer import transfer as run_transfer
    with _bar() as progress:
        r = run_transfer(run, sources, out, every, threshold, parent, preview, max_frames, tracks=tracks, progress=progress)
    for src, s in r["sources"].items():
        typer.echo(f"{Path(src).name}: {s['frames']} frames, {s['boxes']} boxes (median {s['boxes_per_frame_median']:g}"
                   f"/frame), {s['empty_frames']} empty, {len(s['frames_to_check'])} to check -> {s['folder']}")
        if s.get("track_check"):
            c = s["track_check"]
            typer.echo(f"  track check: {c['label_flip']} label flips, {c['possible_miss']} possible misses, "
                       f"{c['lone_box']} one-frame boxes (in the frames to check; labels unchanged)")


@app.command()
def quick(dataset: Path = typer.Argument(..., exists=True, file_okay=False,
                                        help="Labeled dataset folder (images/, labels/, classes.txt or data.yaml)."),
          sources: list[Path] = typer.Argument(..., exists=True, help="Videos and/or image folders."),
          run: Path = typer.Option(..., help="Output run folder; labels go to <run>/labels/<source>."),
          parent: str | None = typer.Option(None, help="What the parts sit on; matching runs inside it."),
          every: int | None = typer.Option(None, min=1, help="Label 1 frame in N (default: as the source)."),
          max_frames: int | None = typer.Option(None, min=1, help="Only the first N frames/images of each source."),
          preview: bool = typer.Option(True, "--preview/--no-preview", help="Write preview.mp4 per video.")):
    """Label similar videos or folders without training, by matching the dataset's examples (a quick preview:
    expect about 70% of parts; check every frame, or use teach + transfer for accurate labels)."""
    from engine.quick import quick_transfer
    with _bar() as progress:
        r = quick_transfer(dataset, sources, run, parent=parent, every=every, preview=preview, max_frames=max_frames,
                           progress=progress)
    for src, s in r["sources"].items():
        typer.echo(f"{Path(src).name}: {s['frames']} frames, {s['boxes']} boxes (no training: check every frame) -> {s['folder']}")


@app.command()
def review(project: Path = typer.Argument(..., help="New project folder."),
           video: Path = typer.Option(..., exists=True, dir_okay=False, help="The transferred video."),
           labels: Path = typer.Option(..., exists=True, file_okay=False, help="Its transfer output labels/ folder."),
           classes: Path = typer.Option(..., exists=True, dir_okay=False, help="classes.txt or data.yaml."),
           every: int | None = typer.Option(None, min=1, help="Default: from the transfer's summary.json, else 5.")):
    """Make a project from a transferred video with its automatic labels imported, to check in the annotator."""
    from engine.project import Project
    summary = labels.parent / "summary.json"
    every = every or (json.loads(summary.read_text(encoding="utf-8")).get("every") if summary.exists() else None) or 5
    with _bar() as progress:
        p = Project.create(project, _classes(classes), video=video, every=every, progress=progress)
    st = p.import_yolo(labels)
    typer.echo(f"{p.folder}: {len(p.items)} frames (1 in {every}), {st['boxes']} boxes from {st['files_matched']} "
               f"label files" + (f", {st['files_unmatched']} unmatched" if st["files_unmatched"] else ""))
    if os.environ.get("COLAB_RELEASE_TAG"):                  # no web app on Colab: the notebook shows it
        typer.echo(f"open it in the notebook's Step 6 with PROJECT_NAME = {p.folder.name}")
    else:
        typer.echo(f"open it with: partlabeler app --home {p.folder.parent}")


@app.command()
def models():
    """Download and verify the annotation models (SAM 3, DINOv3)."""
    from engine.models import DINOV3, SAM3, fetch
    for spec in (SAM3, *DINOV3.values()):
        typer.echo(f"{spec['repo_id']} -> {fetch(spec)}")
    typer.echo("all models downloaded and verified")


@app.command()
def doctor():
    """Versions, device, free disk and which model weights are cached."""
    import torch
    from engine import hw
    ok = lambda b: "ok" if b else "MISSING"
    typer.echo(f"python   {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    typer.echo(f"torch    {torch.__version__}, CUDA {torch.version.cuda or '-'}, available: {torch.cuda.is_available()}")
    typer.echo(f"device   {hw.describe()}")
    for drive in dict.fromkeys(Path(p).anchor for p in (Path.cwd(), Path.home())):
        typer.echo(f"disk     {shutil.disk_usage(drive).free / 2**30:.0f} GB free on {drive}")
    from huggingface_hub import snapshot_download
    from engine.models import _ALLOW, DINOV3, SAM3
    for spec in (SAM3, *DINOV3.values()):
        try:
            snapshot_download(spec["repo_id"], revision=spec["revision"], allow_patterns=_ALLOW, local_files_only=True)
            cached = True
        except Exception:
            cached = False
        typer.echo(f"weights  {spec['repo_id']}: {ok(cached)}" + ("" if cached else " (run: partlabeler models)"))
    try:
        import importlib.metadata as md
        from rfdetr.assets.model_weights import get_model_cache_dir
        from engine.detector import SIZES as RF
        typer.echo(f"rfdetr   {md.version('rfdetr')} importable; pretrained weights: " + ", ".join(
            f"{s} {'cached' if (Path(get_model_cache_dir()) / f'rf-detr-{s}.pth').exists() else 'not cached'}" for s in RF)
                   + " (downloaded on first teach)")
    except Exception as e:
        typer.echo(f"rfdetr   MISSING ({e.__class__.__name__}: {e}); Teach & Transfer needs: pip install -e .[teach]")


if __name__ == "__main__":
    app()
