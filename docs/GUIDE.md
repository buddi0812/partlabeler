# PartLabeler user guide

PartLabeler builds labeled datasets of parts (any product, any part list) from videos and image folders:
boxes (object detection), exact outlines (instance segmentation) or one class per image (classification).
The result is the dataset itself: YOLO, COCO, CVAT, Pascal VOC or Label Studio files, or class folders, that you
train your own model on. Labeling trains nothing: you click, the models outline, track, group and suggest, and
you confirm.
Everything runs on your own computer. This guide is also what Rivet, the in-app helper, answers from.

## Start the app

Double-click `run_windows.bat` (Windows), run `./run.sh` (Linux / macOS) or `python -m engine.cli app`.
The start screen opens in your browser at http://127.0.0.1:8765. Keep the black console window open while
you work: closing it stops the app. Your work is saved as you go, so you can close the browser at any time.
Projects are saved in the `projects` folder next to the app (start with `--home <folder>` to use another), or in
the folder your account chose in Settings.

## Accounts and signing in

The first time the app opens it asks you to **create your account**: a username, your name (optional) and a
password of at least 8 characters. After that it shows **Sign in**; **Create an account** there adds another
person, for example a teammate who shares the computer. **Keep me signed in** keeps you signed in for 30 days in
that browser; without it you are signed out after 12 hours.

Accounts live on this computer only (in `~/.partlabeler/accounts.json`, passwords stored as secure hashes).
Each account has its own settings: theme and colours, helpers, defaults and its **projects folder**. Projects
themselves are plain folders: accounts that use the same projects folder see the same projects. An account is
not encryption: anyone who can open this computer's files can open the projects.

**Forgot your password?** Passwords can't be emailed. Anyone at this computer can set a new one from a command
prompt in the PartLabeler folder: `.venv\Scripts\python -m engine.cli account reset USERNAME`
(`account list` shows the accounts, `account remove USERNAME` deletes one; its projects stay).

The account menu is the round button with your initials at the top right of every page: it shows who is signed
in and has **Settings** and **Sign out**. Notebooks (Jupyter, Colab) have no sign-in: the notebook is already
yours, and the annotator there uses the default settings.

## Settings

Open **Settings** from the account menu. Every change is saved at once ("Saved" flashes next to the title) and
belongs to your account.

- **Account**: your name (shown in the top bar), **Change password** (signs out your other browsers),
  **Delete this account** (asks for the password; your projects stay), **Sign out**.
- **Appearance**: **Theme** Light, Dark or **Match my computer** (follows Windows' light or dark mode);
  **Colour palette** for buttons, links and highlights (Teal, Blue, Indigo, Violet, Rose, Orange, Green,
  Graphite; your label colours stay as they are); **Animations** on or off.
- **Helpers and messages**: **Rivet, the helper** on or off (from the next page you open), **Rivet's tips**,
  the **First-steps hint** over the frame in the annotator, **Pop-up messages** (off: only problems pop up; the
  bell keeps everything) and **Look for updates** (off: only when you click Check for updates).
- **Annotating and export**: **Frames per track** (how far T and R track; the annotator also remembers the
  number you type there), **Brush size** for outlines (the annotator remembers the slider too), the **Export
  format** picked first in export dialogs and whether **Confirmed frames only** starts ticked.
- **New tasks**: the starting **Frame step**, **Image quality**, **Lossless frames** and **Segment size** of
  Create a new task.
- **Projects folder**: where your projects are kept. Type a full path (or **Browse…**) and click **Use this
  folder**. With **Move everything in the current folder there** ticked, your projects, trash, backups and Teach
  runs move there first (a progress bar shows it; keep the app open until it is done). Other accounts that use
  the old folder then no longer see what moved. **Back to the default folder** fills in the app's folder.
- **Unfinished work as a zip**: how to back up a project or a task, and **Import a zip…** as a new project or
  into one of your projects (see Backup and restore).

## Projects, tasks and jobs (as in CVAT)

PartLabeler is organised like CVAT:

- A **project** has a type (object detection, segmentation or classification) and its **labels** (names and
  colours). It holds tasks.
- A **task** is one video or one image folder, with its own settings (frame step, start and stop frame, image
  quality, a **subset** such as Train, Validation or Test). Its frames are split into jobs.
- A **job** is a range of a task's frames (all of them unless a segment size was set), with a **stage**
  (annotation, validation, acceptance) and a **state** (new, in progress, rejected, completed). The annotator
  always opens one job.

The top bar of every page has **Projects**, **Tasks** (every task of every project) and **Jobs** (every job).

## The start screen (Projects)

The start screen lists your projects as cards with a preview, the project type, how many tasks, labels and
frames, and how many are confirmed. **Search…** filters them, the sort menu orders them by updated date, name or
created date, and **+** offers **Create a new project** and **Create from backup…**. Each card's ⋯ menu has
Open, Export dataset, Backup project, Show in folder and Move to trash. The start screen also has Teach &
Transfer, the notifications (the bell, or the N key) and the Trash, where moved projects can be restored.

## Create a project

1. On the start screen, find **Create a new project** (or **+ → Create a new project**) and type a name.
2. Pick the **Project type**: **Object detection**, **Segmentation** or **Classification** (see Project types
   below). It is fixed for the project, and so are its export formats.
3. Add the **Labels**: **Add label**, then type a name and pick its colour. **Raw** shows them as JSON to edit or
   paste; **Load from classes.txt / data.yaml…** adds the names from a file. Names starting with `left_` /
   `right_` are treated as mirror twins. Classification projects may start with no labels: you name them while
   sorting.
4. Optional, under **Advanced configuration**: the default way this project's video frames are stored.
5. **Submit & Open** creates it and opens its page; **Submit & Continue** creates it and stays, for the next one.

## The project page

The project page shows the project's type, when it was made and how many jobs are done, its **Labels** and its
**Tasks**.

- **Edit labels** opens the label constructor: rename a label, change its colour, add labels or remove them.
  Removing a label deletes its annotations too (you are asked first); renaming and recolouring keep them.
- **Tasks** are listed with a preview, `#number: name`, when they were made and last changed, the subset, the
  video's details (resolution, frame rate, length, codec, file size, frames kept) and CVAT's progress line:
  *done · on review · annotating · total* jobs, with how many frames are confirmed. Tasks with subsets are
  grouped under each subset. **Search…** and the sort menu (ID, name, updated date, subset) find them.
- **+ Create a new task** adds one video or image folder; **Create multi tasks** adds several, one task each.
- Each task's **Open** goes to the task page; its ⋯ menu has **Upload annotations**, **Export task dataset**
  and **Delete**.
- **Actions** (top right): **Export dataset**, **Backup project**, **Show in folder**, **Move to trash**.

## Create a task

On the project page, click **+ Create a new task** (or **Create multi tasks**).

- **Name** (empty: the file or folder name; for several tasks `{{file_name}}` and `{{index}}` fill in each).
- **Subset**: Train, Test, Validation or any word (optional). Exporting the whole project puts each subset in
  its own folders; tasks without one go to `default`.
- The project's labels are used.
- **Select files**: **Add a video…** or **Add an image folder…** (or type a path and press Enter).
- **Advanced configuration**: **Image quality** (JPEG quality of a video's stored frames, 95 by default) or
  **Lossless frames** (exact pixels, about twice the space); **Frame step** (keep every Nth frame); **Start
  frame** and **Stop frame**; **Segment size** (frames per job; empty: one job for the task); **Sorting method**
  for an image folder (lexicographical or natural, so `img2` comes before `img10`).
- **Submit & Open** makes the task and opens its page; **Submit & Continue** stays for the next task.

From the command line: `python -m engine.cli add projects/<name> --video line3.mp4 --subset Train
--segment-size 200` (see `--help` for the rest). On Colab, list several paths in Step 6's SOURCE.

**Frames and disk space.** A video's frames are stored when the task is made: JPEG (visually lossless at quality
95) or lossless WebP (exact decoded pixels, about twice the space). Truly lossless images are always larger than
JPEG, so that is the trade. Exports never copy the stored frames: they link to them (a hard link on the same
drive), so an export takes almost no extra space and its images are identical to what you labeled. If you edit
exported images in place, edit a copy, since they share the file with the project.

## The task page and its jobs

The task page shows the task's preview, its name (click it to rename), when it was made, its **Subset** (change it
right there), the progress line, and a table of its media: source, resolution, frame rate, duration, codec,
file size, frames in the video, frame step, start and stop frame, frames kept and how they are stored (image
folders: resolution, number of images, size, sorting).

**Jobs** lists the task's jobs: `Job #number`, its **Stage** and **State** (change them in the menus), the frame
range, frame count, how many frames are confirmed and when it last changed. **Open** (or the job's number) opens
the annotator on that job. A new job counts as *in progress* as soon as something in it changes. The task's
progress counts a job as *done* when it is in the acceptance stage and completed, and *on review* while in
validation. **Actions** has Upload annotations, Export task dataset and Delete.

**Upload annotations** (task or job) reads a YOLO labels folder (boxes, or polygons in a segmentation project),
matched to frames by file name or frame number. **Replace** removes the existing labels there first; **Append**
adds to them.

## Project types: object detection, segmentation, classification

Chosen once, when the project is made: a project holds one kind of label, and its exports follow it.

- **Object detection**: a box around every part. The default.
- **Segmentation** (instance segmentation): the exact shape of every part, as a mask ("outlines" in this guide). Everything that makes a box
  makes an outline here: clicking, drawing a box (SAM 3 outlines the part inside it), tracking, Suggest and Find
  similar. The brush and eraser fix edges. Exports hold polygons or masks.
- **Classification**: one class per image, labeled on a grid of pictures with smart grouping ("image classes"
  in this guide). Good for sorting a messy folder (for example ok / scratch / dent, or one folder per part type).

Boxes imported into a segmentation project (existing YOLO labels, Teach & Transfer results) become outlines with
**Outline boxes** (… **on every frame**) under Find parts. The annotator shows the type next to the project name.

## The annotator screen

The annotator shows one job. Top bar: ← Projects, the project name and type, the **job menu** (`Job #n` and its
state: **Finish the job** (marks it accepted and completed), its **Stage** and **State**, ◀ ▶ for the previous
and next job, a list of every job, **Open the task** and **Back to the project**), the frame number within the
job, chips that say whether
this frame is *Confirmed* or *To check*, the save status (*All changes saved*), the bell and **?**
(keyboard shortcuts) and ⤢ (full screen; on Colab it fills the window height, press F11 too for a
true full screen). Below it: frame buttons, the **Track** controls (videos), **Undo**, **Export** and
**Confirm frame**. The image is in the middle, the side panel on the right has Tool, Class, Boxes on this
frame, Find parts, Export dataset and Settings. The timeline at the bottom shows every frame; click it to jump.

Timeline colors: green = confirmed, teal = has your boxes, blue = tracked or imported, purple = suggestions,
grey = no boxes, an amber mark = to check. Box styles: solid = yours, dashed = tracked or imported,
dotted = suggestions.

## Label a part (click to outline)

1. Pick the class first: click it in the Class list or press its number key (1–9, 0 for the tenth,
   [ and ] to step through longer lists).
2. Click the part in the image. SAM 3 outlines it and draws a tight box. It picks the smallest outline;
   press **M** for the next larger one (for example the whole assembly instead of one bolt).
3. To fix a box: Ctrl+click selects it, then Shift+click grows it and Alt+click shrinks it.
4. Picking a class while a box is selected re-labels that box. **Del** deletes the selected box, **Esc**
   deselects, **Ctrl+Z** undoes (up to 50 steps).

## Zoom in and out

Hold **Ctrl** and turn the mouse wheel (or pinch on a touchpad) to zoom in or out at the pointer, up to 800%.
**+** and **−** zoom from the keyboard, **Z** shows the whole frame again, and the buttons at the bottom right of
the picture do the same. When zoomed in, the wheel moves the view (Shift+wheel sideways), and so does dragging
with **Space** held or with the middle mouse button. The zoom stays when you go to the next frame, so one small
part can be checked through a whole video; from about 3 screen pixels per image pixel on, the pixels show
sharp, for exact edge work.

## Draw a box by hand

Press **B** (or choose **Draw box** under Tool) and drag. With a box selected, dragging redraws that box.
Press **C** to go back to click-to-outline. Hand-drawn boxes are useful for parts SAM cannot separate.

## Track through a video

Label every part on one frame, then press **T** (or **Ahead ▶**) to track those boxes ahead by the number of
frames in the box next to it (20 by default); **Shift+T** tracks to the end. **R** / **◀ Back** tracks
backwards and **Shift+R** tracks to the start. **X** or **Stop** stops a running job. Tracking uses SAM 3 and
keeps the style of the boxes you drew. In a segmentation project tracking starts from each part's outline, so
what you fixed by hand (a notch erased, an edge brushed) is what SAM 3 follows into the next frames; after
fixing an outline, track again from that frame to redo the frames after it. Long videos are tracked in chunks, and if the GPU runs out of memory
the chunk size is halved automatically.

## Frames to check

After tracking, frames where a box jumped, changed size a lot or went missing get an amber *To check* mark.
**Next to check** (Shift+→) jumps to the next flagged or unconfirmed frame. Fix what is wrong, then confirm.

## Confirm frames

**Confirm frame** (Enter) marks the frame as checked and moves to the next one. Confirmed frames turn green
on the timeline. Confirming a frame with no boxes records it as checked background (an empty label file on
export). Click **Unconfirm frame** to undo a confirmation.

## Outlines (segmentation)

In a segmentation project the side panel's Tool section has these tools:

1. **Click to outline** (C): click a part and SAM 3 outlines it; **M** steps to the next larger outline, Shift+click
   adds to it and Alt+click takes away from it, as for boxes.
2. **Draw box** (B): drag a box around a part; SAM 3 outlines the part inside it.
3. **Brush** (P): paint onto the selected outline (Ctrl+click an outline to select it). With nothing selected,
   the brush paints a new part of the current class.
4. **Eraser** (E): erase from the selected outline. Erasing all of it deletes the part.
5. **Smart brush** (Shift+P): paint roughly over what is missing. The outline model (SAM 3) looks at the part and
   the stroke, and only what it sees as the part is added: the paint stops at the part's edge even when the
   brush spills over it. With nothing selected it starts a new part.
6. **Smart eraser** (Shift+E): erase roughly over what does not belong (a spill onto the background or a
   neighbour). Only what the outline model sees as not the part is removed; the part's own edge stays.

The plain brush and eraser paint exactly where you drag; use them where the outline model gets an edge wrong
(for example on blurry frames). A smart stroke that changes nothing says so. **,** and **.** make the brush
smaller and larger (or use the Brush size slider). Tracking (T) carries the
outlines through the video, and Suggest, Find similar and Accept all work as for boxes, with outlines.
**Outline boxes** outlines every box on the frame that has none yet (imported boxes, Teach & Transfer results); **…on every frame** does the whole project.

Export: YOLO gets segmentation polygons (one line per part, pieces and holes joined as Ultralytics expects),
COCO gets polygons, or RLE masks for parts with holes or that are too small for a faithful polygon, CVAT gets
polygons or masks, Pascal VOC gets SegmentationClass and SegmentationObject PNGs, and Label Studio gets
polygons (holes are not kept there). Boxes without an outline are left out and counted in the notification.

## Image classes and smart sorting

A classification project opens as a grid of pictures instead of one frame at a time.

- **Select** pictures: click one, Shift+click a range, Ctrl+click to add or remove, Ctrl+A for everything shown.
  Then press a class number (1–9, 0) or click the class in the side panel. Del clears the class.
- **New class…** (under Class): adds a class; the selected pictures get it straight away.
- **Group by look** (G): puts look-alike pictures together (DINOv3 features, nothing leaves the computer).
  The **Groups** slider under Sort gives fewer or more groups; it starts at the best fit. Each group has
  **Select all** and **Give all a class…**, so naming a group labels all of it at once (pictures you already
  classed yourself keep their class).
- **Suggest classes** (S): once at least two classes have pictures, it suggests a class for the others. It only
  suggests where a picture looks like pictures of that class; the rest stay without a class. Check the dotted
  cards (**Suggested** lists the least sure first), **Accept suggestions** (Y), then suggest again: each round
  reaches further.
- **Check labels**: lists pictures whose class disagrees with their look-alikes (likely mistakes) under
  **To check**, with the class they look like on the badge.
- **Duplicates** lists near-identical pictures after grouping; **Unsure** lists pictures that fit no group well.
- Double-click a picture (or press Space) for a closer look; arrows step, numbers give the class.

On the example dataset's part pictures (3,000, 14 classes) the first grouping came close to the real classes,
suggestions from 3 examples per class were right 88% of the time, and Check labels found 88 of 90 wrong labels.

Export: **Class folders** (one folder per class), **YOLO** (Ultralytics classification: train/ and val/ with a
folder per class; neighbouring images go to the same side) or **CSV list** (path, class; no copies). Suggested
classes nobody accepted are never exported.

## Sort parts: name or fix classes in bulk

In an object detection or segmentation project, **Sort parts** (G, under Find parts) shows every part as a picture (a tracked part
once), with the same grid tools: Group by look, the Groups slider, Check labels, selecting and class numbers.
Giving a part a class changes it on every frame it appears in. Double-click a part to go to its frame.
This allows "label now, name later": outline or box every part as one class (for example `part`), add the real
classes, then group the parts and name each group. Check labels finds parts whose class disagrees with their
look-alikes. **Back to frames** (Esc) returns to the annotator.

## Find parts: Suggest, Find similar, Accept all

- **Suggest** (S) proposes boxes that look like parts you already labeled on other frames or images
  (DINOv3 matching). Label a few examples first. Works in image folders and videos.
- **Find similar** (F): select one box first (Ctrl+click), then F finds more copies of that part on the
  same frame (SAM 3).
- Suggestions are dotted boxes. **Accept all** (Y) keeps them all; select one and press Del to drop it.
  Suggestions you never accept are left out of the export.

## Export the dataset

The project page (**Actions → Export dataset**), a task's ⋯ menu (**Export task dataset**) and the annotator
(**Export** in the top bar, then **This job**, **This task** or **Whole project**) all export. The dialog asks the
**Export format** (YOLO, COCO, CVAT, Pascal VOC or Label Studio; classification projects: class folders, YOLO or
CSV), **Save images** (off: the label files only), **Confirmed frames only** and a **Custom name**. With several
subsets in the export, each gets its own folders: YOLO `images/<subset>/` and `labels/<subset>/` (with
data.yaml's train, val and test set from subsets named so), COCO `annotations/instances_<subset>.json`, Pascal VOC
`ImageSets/Main/<subset>.txt`, CVAT a subset on each image. The export goes into the project's `exports/`
folder, named like `project_<name>_dataset_<time>_yolo`. In the annotator, pick the format, tick
**Confirmed frames only** if you want just the frames you checked, and click **Export**. Segmentation projects export
outlines (see Outlines). The files go into the project's `exports/` folder, in a new sub-folder
named after the format and time. A notification offers **Open folder**. Export as often as you like;
nothing is overwritten. From the command line: `python -m engine.cli export projects/<name> --format coco`.

## Project settings and the parent object

In the annotator, open **Settings** at the bottom of the side panel (these belong to the project; your own
preferences are on the Settings page of the account menu). The **parent object** is the thing the
parts sit on (for example "engine block" or "circuit board"). When it is set, Suggest searches only inside
it, which helps when the camera or distance changes. Leave it empty to search the whole image. The Settings
section also shows the device (GPU or CPU) and GPU memory in use.

## Teach & Transfer: how it works

Teach & Transfer is for when one video is already labeled and you want similar videos labeled the same way.
It is optional and is the only part of PartLabeler that trains a model.

1. **Teach**: pick a labeled dataset folder (YOLO format: `images/`, `labels/` and `classes.txt` or
   `data.yaml`). PartLabeler analyses the labels, holds back about 20% of the frames in time blocks, and
   trains an RF-DETR detector on your computer (Nano is fastest, Medium is the most accurate; 30 epochs by
   default). This can take from minutes to about an hour.
2. **Prove**: the trained model labels the held-back frames and is scored against your labels (mAP50, recall
   and a colour stress test). Targets: held-out mAP50 at least 0.90, recall at least 0.90, and at most a
   10-point drop when colours change. The report says whether the targets are met.
3. **Transfer**: choose the finished Teach run, add the videos or image folders to label and click
   **Start labeling**. Each output folder looks exactly like the source dataset (same classes, same file
   naming), plus a preview video and a list of frames to check. With **Check along tracks (videos)** ticked
   (the default), that list also names frames where a part's label flips (for example left to right), a part
   vanishes for a frame or two, or a box shows up on one frame only. Labels are never changed automatically.
4. **Review**: click **Review in annotator** next to an output. It opens as a normal project with the boxes
   imported, so you can check them, fix them, confirm and export.

**Quick transfer (no training)**: in the Transfer form, set **Label with** to *No training: match a labeled
dataset (quick preview)*, pick the labeled dataset folder, add the videos or folders and click **Start
labeling**. About 30 of the dataset's labeled images become examples that are matched in every new frame (no
model is trained, so it starts in seconds and runs at 1 to 2 seconds a frame). It is a rough first pass: on the
example dataset it found about 70% of the parts on frames like its examples and fewer on another colour, with
extra boxes to delete, so check every frame in the annotator (**Review in annotator**). For accurate labels,
use Teach. From the command line: `python -m engine.cli quick <dataset> <video> --run <folder>`.

A parent object can be given in Teach too; the model then looks only inside it. If a run stops, starting it
again resumes from the last epoch. If Teach & Transfer is greyed out, it was not installed: run the
installer again without `-NoTeach`.

## Notifications

The bell (or **N**) opens the notification history, shared by the start screen and every open project:
projects created, frames confirmed, boxes deleted, tracking finished, exports and errors. Items have
shortcuts such as **Go to frame**, **Open folder**, **Open project** or **Restore**. **Problems** shows only
warnings and errors. **Mark all as read** clears the badge, **Clear all** empties the list.

## Backup and restore

**Backup project** (the project's ⋯ menu or Actions) writes one zip into the `_backups` folder: the project's
settings, labels with their colours, tasks, jobs and all annotations, the stored video frames and the pictures of
its image folders, so it opens complete on another computer. **+ → Create from backup…** on the start screen
turns such a zip back into a project (with a new name if the old one is taken). Command line:
`python -m engine.cli backup projects/<name>` and `python -m engine.cli restore <zip>`.

**Backup task** (a task's ⋯ menu, or Actions on the task page) does the same for one task: its stored frames or
pictures, labels, confirmed frames, subset and jobs with their stage and state, so unfinished work carries on
elsewhere. The zip goes to `_backups` too. To continue it:

- in an existing project: **Import task…** on the project page (next to Create multi tasks). The project must
  be of the same type; labels are matched by name and any the project lacks are added. A whole project's backup
  works here too: all its tasks are added.
- as a project of its own: **+ → Create from backup…** on the start screen, or **Import a zip…** in Settings.

Command line: `python -m engine.cli backup projects/<name> --task <task name>` and
`python -m engine.cli import-task projects/<name> <zip>`.

## Trash and restore

On the start screen, a project's ⋯ menu → **Move to trash…** moves it (frames, labels and exports) into
`projects/_trash/`. Nothing is deleted. Restore it from the notification or from the **Trash** list below
the projects. If a project with the same name exists, the restored one gets `_2` added to its name.

## Keyboard shortcuts

Click: new part · M: next larger outline · Shift/Alt+click: grow/shrink the selected part · Ctrl+click:
select · B: draw box · C: click mode · 1–9, 0, [ ]: class · Del: delete · Esc: deselect · Ctrl+Z: undo ·
← → (A / D): previous / next frame · Shift+→: next to check · Enter: confirm · T / Shift+T: track ahead /
to the end · R / Shift+R: track back / to the start · X: stop · S: suggest · F: find similar · Y: accept all ·
N: notifications · ?: all shortcuts. Ctrl+wheel / + / −: zoom · Z: whole frame · Space+drag or wheel: move when zoomed. Outlines: P brush · E eraser ·
Shift+P smart brush · Shift+E smart eraser · , and . brush size. G: sort parts (grid: group
by look). Grid: 1–9 and 0 give the selected pictures a class · Ctrl+A select all shown · Del clear · Space or
double-click: closer look · Esc: clear the selection, then back to frames.

## Notebooks and Google Colab

The annotator also runs inside Jupyter, VS Code or Colab notebooks:
`from ui.host_widget import Annotator` then `Annotator("projects/line2")`. On Colab, open
`PartLabeler_Colab.ipynb`, choose *Runtime > Change runtime type > T4 GPU* and run the cells from top to
bottom. The annotator stays inside the notebook, as Colab's free tier requires. On Colab, Step 6 keeps its cell
running while you annotate: Colab disconnects a runtime about 30 minutes after the last cell finished and does
not count clicks in the annotator, but a running cell keeps it awake. The cell stops by itself 25 minutes after
your last action; press its stop button to run another step (the annotator keeps working).

## Install and requirements

Windows: double-click `install_windows.bat` (installs Python 3.12 via uv, the right PyTorch for your NVIDIA
driver, and about 4 GB of models). Linux / macOS: `bash install.sh`. Options: `-Cpu`, `-NoModels`,
`-NoTeach`, `-Dev`. `python -m engine.cli doctor` checks the setup. An NVIDIA GPU is recommended (developed
on an RTX 3060 12 GB; all models together use about 4 GB). CPU works but is slow.

## Update PartLabeler

The start screen's top bar always has an update button: **Check for updates**, **Up to date**, or
**Update available** when GitHub has a newer version (it checks at most every 6 hours). Click it to see what is
new, then **Update now**. When it says **restart to finish**, close the black PartLabeler window and start
`run_windows.bat` again. Other ways: double-click `update_windows.bat` (with PartLabeler closed), or run
`python -m engine.cli update` (`--check` only looks). In the annotator, Settings shows the version and a
**Check for updates** link. On Colab, Step 3 of the notebook gets the newest version every session.

An update never touches your work: projects, frames, labels, exports, Teach runs, the models and Rivet's key stay
as they are. Before it changes anything it copies every project's labels (project.json and labels.sqlite) to
`projects/.partlabeler/backups/` (the last 5 updates are kept). It works for a git clone and for an unzipped
copy. It refuses, changing nothing, when app files were edited on this computer or when a job (tracking,
Teach, Transfer) is running. If the update needs new Python packages, they are installed at the next start; if
that fails, run the installer again (it is safe to rerun and keeps your projects).

## Troubleshooting

- *Out of GPU memory*: close other programs that use the GPU (games, other notebooks) and try again.
  Tracking already halves its chunk size on its own.
- *Connection lost*: the console window was closed or the app crashed. Your work is saved: start the app
  again and reload the page.
- *A project shows an error on the start screen*: its source video or folder was moved. Move it back.
- *Suggest finds nothing*: label a few examples of that class first, or clear the parent object setting.
- *The page looks old after an update*: reload it (F5).
- *Forgot your password*: see Accounts and signing in (`account reset`).
- *My projects are gone after signing in*: your account may use another projects folder. Check Settings →
  Projects folder, or sign in with the account that made them.

## Why PartLabeler

- SAM 3 click-to-box and video tracking are built in; no separate model server to set up.
- It runs on your own machine and is free (Apache-2.0). Labeling sends no images or labels anywhere.
- Native Windows install with one double-click: no Docker, no WSL.
- It runs inside notebooks, including Google Colab's free T4, where web-UI tools are not allowed.
- Teach & Transfer learns one labeled video and labels similar ones on your own GPU, and proves its accuracy
  on held-out frames first (0.936 held-out mAP50 on the example dataset).
- It exports all five common formats: YOLO, COCO, CVAT XML, Pascal VOC and Label Studio, for boxes and outlines.
- Boxes, outlines (segmentation) and image classes in one tool, with smart grouping that sorts a messy folder of
  pictures, and a check that finds labels which disagree with their look-alikes.
- Tracked frames that look wrong are flagged for you, so you check the frames that need it.
- Where others are stronger: team workflows (roles, review queues, single sign-on) in CVAT, Label Studio and
  Roboflow, and more annotation types (keypoints, skeletons, rotated boxes, 3D) in CVAT and X-AnyLabeling.
  PartLabeler is built for one job: labeling parts in video and pictures, fast, on your own machine.

## Privacy and the helper

Labeling, tracking, suggestions, Teach & Transfer and export all run locally. Accounts and settings stay on
this computer. The update check asks GitHub
for the newest version number (nothing about you or your projects is sent; set PARTLABELER_NO_UPDATE_CHECK=1 to
check only when you click the button). Rivet, the helper robot, is the only feature that sends anything you type: it sends your typed question, the chat so far and which screen
you are on (for example "annotator, frame 12 of 400") to Google Gemini. It never sends images, labels,
project names or class names. Without a Gemini API key, Rivet answers from this guide offline.
