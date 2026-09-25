# PartLabeler user guide

PartLabeler builds labeled datasets of parts (any product, any part list) from videos and image folders.
The result is the dataset itself: YOLO, COCO, CVAT, Pascal VOC or Label Studio files you can train your own
detector on. Labeling trains nothing: you click, the models outline, track and suggest, and you confirm.
Everything runs on your own computer. This guide is also what Rivet, the in-app helper, answers from.

## Start the app

Double-click `run_windows.bat` (Windows), run `./run.sh` (Linux / macOS) or `python -m engine.cli app`.
The start screen opens in your browser at http://127.0.0.1:8765. Keep the black console window open while
you work: closing it stops the app. Your work is saved as you go, so you can close the browser at any time.
Projects are saved in the `projects` folder next to the app (start with `--home <folder>` to use another).

## The start screen

The start screen lists your projects (newest first) with how many frames are confirmed. From here you can:
open a project, create a new one, run Teach & Transfer, see notifications (the bell, or the N key) and
restore projects from the Trash. Each project card has a ⋯ menu with Open, Show in folder and Move to trash.

## Create a project

1. On the start screen, find **New project** and type a name (for example `gearbox_line2`).
2. Choose **Video** or **Image folder**, then click **Browse…** to pick the file or folder.
3. For a video, set **Keep every Nth frame** (5 keeps one frame in five; use 1 for every frame).
4. Type the class names, one per line (for example `bolt`, `bracket`, `connector`), or click
   **Load from classes.txt / data.yaml…**. Names starting with `left_` / `right_` are treated as mirror twins.
5. Optional, under **More options**: import existing YOLO labels to review or finish, and set a
   **parent object** (see Settings).
6. Click **Create project**. A progress bar shows the frames being prepared, then the annotator opens.

## The annotator screen

Top bar: ← Projects (back to the start screen), the project name, the frame number, chips that say whether
this frame is *Confirmed* or *To check*, the save status (*All changes saved*), the bell and **?**
(keyboard shortcuts). Below it: frame buttons, the **Track** controls (videos), **Undo**, **Export** and
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

## Draw a box by hand

Press **B** (or choose **Draw box** under Tool) and drag. With a box selected, dragging redraws that box.
Press **C** to go back to click-to-outline. Hand-drawn boxes are useful for parts SAM cannot separate.

## Track through a video

Label every part on one frame, then press **T** (or **Ahead ▶**) to track those boxes ahead by the number of
frames in the box next to it (20 by default); **Shift+T** tracks to the end. **R** / **◀ Back** tracks
backwards and **Shift+R** tracks to the start. **X** or **Stop** stops a running job. Tracking uses SAM 3 and
keeps the style of the boxes you drew. Long videos are tracked in chunks, and if the GPU runs out of memory
the chunk size is halved automatically.

## Frames to check

After tracking, frames where a box jumped, changed size a lot or went missing get an amber *To check* mark.
**Next to check** (Shift+→) jumps to the next flagged or unconfirmed frame. Fix what is wrong, then confirm.

## Confirm frames

**Confirm frame** (Enter) marks the frame as checked and moves to the next one. Confirmed frames turn green
on the timeline. Confirming a frame with no boxes records it as checked background (an empty label file on
export). Click **Unconfirm frame** to undo a confirmation.

## Find parts: Suggest, Find similar, Accept all

- **Suggest** (S) proposes boxes that look like parts you already labeled on other frames or images
  (DINOv3 matching). Label a few examples first. Works in image folders and videos.
- **Find similar** (F): select one box first (Ctrl+click), then F finds more copies of that part on the
  same frame (SAM 3).
- Suggestions are dotted boxes. **Accept all** (Y) keeps them all; select one and press Del to drop it.
  Suggestions you never accept are left out of the export.

## Export the dataset

Click **Export** in the top bar: it jumps to the Export dataset controls in the side panel. Pick the format
(YOLO, COCO, CVAT, Pascal VOC or Label Studio), tick **Confirmed frames only** if you want just the frames
you checked, and click **Export**. The files go into the project's `exports/` folder, in a new sub-folder
named after the format and time. A notification offers **Open folder**. Export as often as you like;
nothing is overwritten. From the command line: `python -m engine.cli export projects/<name> --format coco`.

## Settings and the parent object

In the annotator, open **Settings** at the bottom of the side panel. The **parent object** is the thing the
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

## Trash and restore

On the start screen, a project's ⋯ menu → **Move to trash…** moves it (frames, labels and exports) into
`projects/_trash/`. Nothing is deleted. Restore it from the notification or from the **Trash** list below
the projects. If a project with the same name exists, the restored one gets `_2` added to its name.

## Keyboard shortcuts

Click: new part · M: next larger outline · Shift/Alt+click: grow/shrink the selected part · Ctrl+click:
select · B: draw box · C: click mode · 1–9, 0, [ ]: class · Del: delete · Esc: deselect · Ctrl+Z: undo ·
← → (A / D): previous / next frame · Shift+→: next to check · Enter: confirm · T / Shift+T: track ahead /
to the end · R / Shift+R: track back / to the start · X: stop · S: suggest · F: find similar · Y: accept all ·
N: notifications · ?: all shortcuts.

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

## Troubleshooting

- *Out of GPU memory*: close other programs that use the GPU (games, other notebooks) and try again.
  Tracking already halves its chunk size on its own.
- *Connection lost*: the console window was closed or the app crashed. Your work is saved: start the app
  again and reload the page.
- *A project shows an error on the start screen*: its source video or folder was moved. Move it back.
- *Suggest finds nothing*: label a few examples of that class first, or clear the parent object setting.
- *The page looks old after an update*: reload it (F5).

## Why PartLabeler

- SAM 3 click-to-box and video tracking are built in; no separate model server to set up.
- It runs on your own machine and is free (Apache-2.0). Labeling sends no images or labels anywhere.
- Native Windows install with one double-click: no Docker, no WSL.
- It runs inside notebooks, including Google Colab's free T4, where web-UI tools are not allowed.
- Teach & Transfer learns one labeled video and labels similar ones on your own GPU, and proves its accuracy
  on held-out frames first (0.936 held-out mAP50 on the example dataset).
- It exports all five common formats: YOLO, COCO, CVAT XML, Pascal VOC and Label Studio.
- Tracked frames that look wrong are flagged for you, so you check the frames that need it.
- Where others are stronger: team workflows (roles, review queues, single sign-on) in CVAT, Label Studio and
  Roboflow, and many more annotation types (polygons, skeletons, 3D) in CVAT and X-AnyLabeling.
  PartLabeler is built for one job: boxes on parts in video, fast, on your own machine.

## Privacy and the helper

Labeling, tracking, suggestions, Teach & Transfer and export all run locally. Rivet, the helper robot,
is the only feature that uses the internet: it sends your typed question, the chat so far and which screen
you are on (for example "annotator, frame 12 of 400") to Google Gemini. It never sends images, labels,
project names or class names. Without a Gemini API key, Rivet answers from this guide offline.
