# PartLabeler

https://github.com/user-attachments/assets/87e7f75c-94e3-48b0-a1d5-27f3808e5875

<sub>56-second promo · every UI shot is a real recording of the app on public
<a href="https://www.pexels.com/license/">Pexels</a> footage (Video Kickstarter, Distill) ·
<a href="videos/partlabeler-promo/renders/video.mp4">full-quality MP4</a></sub>

Build labeled datasets of industrial parts (any product, any part list) from videos and image folders.
What you get out is the dataset: YOLO, COCO, CVAT, Pascal VOC or Label Studio, ready to train your own
detector on. Labeling itself trains nothing. You click, the models outline, track and suggest, and you confirm.

| Mode | What it does |
|---|---|
| **Video** | Click or box each part on one frame. The boxes are tracked ahead or back (SAM 3) in the style you drew them. Frames where a box jumped, changed size or went missing are marked *to check*. You fix and confirm, then export. |
| **Image folder** | The same screen, one image at a time. **Suggest** proposes boxes that look like parts you already labeled (DINOv3 matching). **Find similar** finds more copies of a selected part in the same image (SAM 3). |
| **Teach & Transfer** | For a video that is already labeled: it learns those labels by training a detector (RF-DETR) on your machine and proves the result on held-out frames. It then labels other similar videos or image folders in the same format. Check the results in the annotator. |

The worked example throughout is a car-front lamp dataset. Nothing in the tool is specific to cars.
Design and decisions: [docs/PLAN.md](docs/PLAN.md). Measured results: [spikes/REPORT.md](spikes/REPORT.md).

## Install

**Windows (NVIDIA GPU or CPU):** clone or unzip the repo, then double-click `install_windows.bat`. It
installs uv and Python 3.12 if needed, creates `.venv`, picks the PyTorch build that matches your NVIDIA
driver (CUDA 13.2 / 13.0 / 12.6, or CPU), installs PartLabeler and downloads the models. Start with
`run_windows.bat`.

**Linux / macOS:**

```bash
git clone https://github.com/buddi0812/partlabeler.git
cd partlabeler
bash install.sh        # then start with ./run.sh
```

**Google Colab:** open `PartLabeler_Colab.ipynb` in Colab, choose *Runtime > Change runtime type > T4 GPU*
and run the steps from top to bottom.

Installer options (Windows `-Name` / Linux `--name`): `-Cpu` CPU-only PyTorch · `-NoModels` skip the model
download (later: `python -m engine.models`) · `-NoTeach` skip Teach & Transfer · `-Dev` add pytest,
playwright and jupyterlab · `-Cuda cu130` choose the PyTorch build yourself. Running it again is safe.

Models download on first install (about 4 GB; SAM 3 from a checksum-verified mirror and DINOv3).
`python -m engine.cli doctor` checks the setup.

## Use

Start the app (`run_windows.bat`, `./run.sh` or `python -m engine.cli app`). It opens a start screen at
http://127.0.0.1:8765 where you can:

- **create a project**: name it, pick a video (every Nth frame is kept) or an image folder, and type the
  class names or load them from `classes.txt` / `data.yaml`. You can also import existing YOLO labels to
  review or finish, and set a *parent object*, the thing the parts sit on (for example "engine block").
  Suggestions then search inside it, which helps when the camera or distance changes;
- **open a project** to annotate;
- **Teach & Transfer**: pick a labeled dataset folder (`images/`, `labels/`, `classes.txt`) and start
  teaching. When it finishes, add the videos or folders to label and start labeling. "Review in annotator"
  opens each result as a project.

Projects live in `projects/` (change it with `--home`). Everything is saved as you go.

### Annotator keys

| Key | Does |
|---|---|
| Click | new part of the current class (the smallest outline SAM finds; **M** for the next larger) |
| Shift / Alt + click | grow / shrink the selected part |
| Ctrl + click | select a box · **Del** deletes it · **Esc** deselects |
| B, then drag | draw a box by hand (redraws the selected box) · **C** back to clicking |
| 1–9, 0, [ ] | pick the class; also re-labels the selected box |
| T / Shift+T | track this frame's boxes ahead N frames / to the end |
| R / Shift+R | track back N frames / to the start |
| X | stop the running job |
| S · F · Y | suggest · find similar to the selected box · accept all suggestions |
| ← → (A / D) · Shift+→ | previous / next frame · next frame to check |
| Enter | confirm the frame and go on |
| Ctrl+Z | undo (last 50 steps) |

Solid boxes are yours, dashed ones came from tracking or import, and dotted ones are suggestions. On the
timeline, green means confirmed, teal means your boxes, blue means tracked or imported, purple means
suggestions, and an amber mark means *to check*. Export (side panel) writes to the project's `exports/`
folder. Frames you confirmed with no boxes become empty label files (checked background), and suggestions
you never accepted are left out.

### Command line

```bash
python -m engine.cli new projects/line2 --video line2.mp4 --classes classes.txt --every 5
python -m engine.cli export projects/line2 --format cvat --reviewed-only
python -m engine.cli teach data/line2_labeled --run projects/_teach/line2_v1 --parent "engine block"
python -m engine.cli transfer projects/_teach/line2_v1 line3.mp4 line4.mp4 --out projects/_teach/line2_v1/labels
python -m engine.cli --help
```

### In a notebook (Jupyter, VS Code, Colab)

```python
from ui.host_widget import Annotator
Annotator("projects/line2")          # the annotator, inside the notebook
```

On Colab, open [PartLabeler_Colab.ipynb](PartLabeler_Colab.ipynb). The annotator runs inside the notebook,
as Colab's free tier requires, and the T4 GPU uses fp16 automatically.

## How it compares

Checked against each tool's own documentation, pricing pages and repositories on 25 Sep 2026. "—" means the docs
describe no such feature, not that we proved it impossible. Corrections welcome.

| | PartLabeler | CVAT (self-hosted) | Label Studio (open source) | X-AnyLabeling | Roboflow Annotate |
|---|---|---|---|---|---|
| SAM 3 click + video tracking, built in | ✓ | SAM 1 via Nuclio add-on; SAM 2 tracker in Enterprise/Online [1][2] | via ML backend server [5][6] | SAM 3 via server add-on; SAM 2 video local [7][8] | SAM 3 built in (cloud); video propagation unclear [9][10] |
| Runs inside a notebook (Colab-friendly) | ✓ anywidget | — | — (no official widget) | — (desktop app) | — |
| Native Windows install, no Docker | ✓ | Docker Compose, Windows via WSL2 [11] | ✓ pip [12] | ✓ pip / app [13] | cloud |
| Learns a labeled video, labels similar ones | ✓ Teach & Transfer, local GPU | bring your own model (Nuclio) | write your own backend [16] | ✓ Ultralytics training, local [17] | ✓ train + Label Assist, cloud credits [18] |
| Exports YOLO · COCO · CVAT XML · VOC · Label Studio | 5 / 5 | 4 / 5 [19] | 4 / 5 [20] | 3 / 5 [21] | 3 / 5 [22] |
| Free, data stays on your machine | ✓ | ✓ (MIT) | ✓ (Apache-2.0) | ✓ (GPL-3.0) | paid tiers, cloud; free-plan data is public [14] |

Where the others are clearly stronger: team workflows (roles, review queues, SSO) in CVAT, Label Studio and Roboflow;
many more annotation types (polygons, skeletons, 3D, audio) in CVAT and X-AnyLabeling; years of maturity and
compliance options (CVAT v2.76, Label Studio Enterprise). PartLabeler is built for one job: boxes on industrial
parts in video, fast, on your own machine.

<details><summary>Sources</summary>

1. docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/ · github.com/cvat-ai/cvat/tree/develop/serverless/pytorch/facebookresearch
2. docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/
3. cvat.ai/resources/changelog/sam-3-image-segmentation
4. cvat.ai/resources/changelog/sam2-ai-agent-tracking
5. github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples · github.com/HumanSignal/label-studio/issues/9026
6. labelstud.io/guide/ml_tutorials/segment_anything_2_video
7. github.com/CVHub520/X-AnyLabeling/blob/main/examples/grounding/sam3/README.md
8. xanylabeling.com/examples/interactive_video_object_segmentation/sam3
9. blog.roboflow.com/sam3/
10. blog.roboflow.com/video-annotation/
11. docs.cvat.ai/docs/administration/community/basics/installation/
12. labelstud.io/guide/install
13. github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md
14. roboflow.com/pricing
15. cvat.ai/resources/changelog/announcing-cvat-ai-agents
16. labelstud.io/guide/ml
17. xanylabeling.com/examples/training/ultralytics
18. docs.roboflow.com/datasets/annotate/annotate/ai-labeling
19. docs.cvat.ai/docs/dataset_management/formats/
20. labelstud.io/guide/export
21. github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md
22. roboflow.com/formats

Colab's free tier disallows "bypassing the notebook UI to interact primarily via a web UI"
(research.google.com/colaboratory/faq.html), which is why PartLabeler's notebook mode is a widget.
</details>

## Hardware

| Machine | Precision | Notes |
|---|---|---|
| NVIDIA RTX 30xx / 40xx / 50xx | bf16 | developed on an RTX 3060 12 GB; all models together peak at about 4 GB |
| Colab T4 / older NVIDIA | fp16 | checked in fp16 on the 3060; speed on a T4 not measured yet |
| CPU only | fp32 | runs, but expect it to be slow (not benchmarked yet) |

Set `PARTLABELER_DEVICE` (`cuda` / `cpu`) or `PARTLABELER_DTYPE` (`bf16` / `fp16` / `fp32`) to override.
If the GPU runs out of memory while tracking, the chunk size is halved automatically.

## Tests

```bash
python -m pytest tests -q
```

The tests need no model weights. `spikes/` holds the research scripts behind the design decisions.

## Licenses of the models

SAM 3: SAM License (commercial use allowed, some uses prohibited). DINOv3: DINOv3 License. RF-DETR: Apache-2.0.
Check that these fit your use before shipping a dataset or model built with them.
