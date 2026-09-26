# PartLabeler: design for labeling unusual machine parts in video by correcting predictions

## Context
You want to annotate video frames of **unusual machine parts** without drawing every frame by hand. The work goes like this: you label a few **reference frames**, the tool predicts the **next frame**, you correct it, and the corrected frame becomes another reference, so each prediction gets better.

- **Two input types:** the tool also annotates a **folder of unrelated images** (no video order) using the same review-and-learn loop.
- **Fully automatic mode, "Teach & Transfer":** you give it a source video plus a folder of its manually labeled frames. It learns from them, proves it can re-create those labels on the source, then labels **other similar videos** with no one in the loop. Example: lamps labeled on a blue car → it labels lamps on green, white and black cars.
- **Where it's built:** first on this PC (RTX 3060 12 GB). Later it is optimized for **Colab (top priority)**, then small GPUs and CPU.
- **Your request:** the most reliable design, based on thorough research (3 parallel research passes, 2026-09-25).

> **Core principle (added 2026-09-25, overrides anything below):** the product is the **annotated dataset**
> that you train your own models on. The interactive modes (video and image folder) work with **no model
> training**: SAM 3 click, track and correct, plus training-free suggestions (DINOv3 matching, SAM 3
> "box one, get all"). The in-tool detector ("Boost") is **opt-in and off by default**, never started on
> its own. Quality is reported as annotation output: frames labeled, corrections needed, exports.
> Teach & Transfer stays a separate mode, because learning from an existing labeled source is what it is for.
>
> **Scope (added 2026-09-25):** a general-purpose industrial annotation tool for any product and part list.
> The car-lamp example data is one example application. Nothing is car-specific: the parent-object crop is an
> optional per-project setting, and left/right handling applies only when class names form left_/right_ pairs.

> **Status v0.1 (2026-09-25), what is built vs this plan.** Measured results: `spikes/REPORT.md`.
> - Built: projects from a video or an image folder; click-to-box (SAM 3, smallest outline first, M for
>   larger); box tool; tracking ahead **and back** with the drawn box's style copied, auto-halving chunks when
>   the GPU runs out of memory; DINOv3 suggestions (optional parent-object crop); **find similar** (SAM 3
>   exemplar within a frame, size/shape filtered, left/right twins resolved); undo (50 steps); review flags;
>   export to YOLO, COCO, CVAT 1.1, Pascal VOC and Label Studio; a start screen with a file browser;
>   Teach & Transfer as engine modules, CLI and start-screen panel; install scripts; a Colab notebook.
> - Changed by measurement: **no Laya / openjev quality gate.** Zero-shot, Laya Vision scored 0.51–0.54 AUROC
>   on bad boxes, text Laya 0.23, openjev 0.8B 0.55–0.72, openjev 4B 0.67–0.76 (9 GB). A plain rule does
>   better: area vs the person's box (0.91 AUROC) plus centre jumps; that is what flags frames now (S4c).
>   Self-training in Transfer was rejected (it made transfer worse, S7). No conformal auto-accept, score
>   combiner or autopilot yet: every frame is still confirmed by a person.
> - Not built yet: masks in exports (boxes only), in-tool Boost (opt-in), EdgeTAM/CPU presets, image-order
>   strategies beyond filename order, a Playwright UI test.

---

## What the research found (feasibility)

| Question | Verdict | What that means for the design |
|---|---|---|
| Does SAM 3 run here? | 🟢 through **Hugging Face `transformers` ≥5** (plain PyTorch, no Triton, works on Windows, CPU and GPU, selectable dtype) · 🟠 through Meta's official repo (needs triton-windows, a numpy conflict, decord has no Windows wheel, bf16 is hard-coded so it breaks on the T4) | **Use the Hugging Face SAM 3 classes.** |
| SAM 3.1? | 🟠 Not in `transformers`. Official repo only; a user reports running out of memory on a 24 GB card. | SAM **3** is the default. SAM 3.1 may come later as an optional speed-up. |
| Tracking + corrections + re-tracking | 🟢 `Sam3TrackerVideoModel`: add points, boxes or masks on **any** frame, then `propagate_in_video_iterator(start_frame_idx, max_frame_num_to_track)` | This drives the loop and the autopilot step size. |
| Boxes from reference frames finding parts in **new** frames | 🔴 **SAM 3 only uses an example box on the frame it was drawn on.** Confirmed in the source code and in open GitHub issues. Workarounds score poorly on industrial parts (CAD-Prompted SAM3: 0.17–0.42 PQ). | **Finding new or lost parts needs its own component** (see D2). |
| Example box on the **same** frame | 🟢 One box finds all similar parts in that frame (82 AP vs 61 AP for a text prompt on the ODinW benchmark) | A **"box one, get all"** tool for you. |
| VRAM | 🟠 A 121-frame session peaked at about 8.8 GiB, the state grows with every frame, and closing a session leaks memory (issue #305) | Process **in chunks** (~150–200 frames), offload the state to CPU, tear sessions down fully, and watch VRAM. |
| Colab T4 | 🟠 No bf16 support, so use fp16 through HF (quality not verified; to be tested). Slow, roughly 1 frame per second or less. Option: 560 px input. | Phase B needs its own test and preset. |
| **Colab free tier and web UIs** | 🔴 Colab's rules ban working "primarily via a web UI" instead of the notebook, and a Gradio share link counts. Gradio is also slow per click (300–500 ms). | **No Gradio.** The annotation canvas runs **inside the notebook** (anywidget) on Colab and as a local web page on your PC. |
| Existing tools (CVAT, Label Studio, X-AnyLabeling, SAMannot, Roboflow) | 🔴 None supports example boxes + corrections + re-tracking on video under a license and runtime Colab can use | **We build our own thin app.** SAMannot's "lock-and-refine" frames and Meta's SAM 2 demo serve as design references. |
| **Laya / Laya Vision** | 🔴 as the part-type classifier (no evidence it can tell niche parts apart; a week-old experimental project; weights CC BY-NC-SA) · 🟢 as a **calibrated yes/no quality check** (ECE 0.041, `agent.calibrate()` on your own corrections) | Laya Vision becomes the **quality gate**. Text-only Laya is **not used**: it can't see images, and its package name clashes with Laya Vision's. |
| Recognizing the part type from a few examples | 🟢 **Matching DINOv3/DINOv2 embeddings against known examples**: 44.6 mean AP on BOP industrial datasets (SAM + DINOv3), and 85.7% accuracy on look-alike industrial parts with 10 examples per class and no retraining | This is the **main classifier**, and it also powers the search for new parts. |
| Detector fine-tuned on corrected frames | 🟢 **RF-DETR-Seg N/S/M (Apache-2.0)**, strong on few-shot benchmarks (YOLO is AGPL) | "Boost": joins as a second source of detections once there's enough data. |
| Reading video | 🟢 PyAV (seeks to the exact frame) · OpenCV can land on the wrong frame · decord is dead | PyAV decodes once into chunked JPEG folders. |

---

## Architecture: engine separate from UI, three model roles

```
                ┌───────────── ENGINE (pure Python, no UI code) ─────────────┐
 video ─PyAV──► │ FrameStore: chunked JPEG cache (150–200 frames per chunk)   │
                │                                                             │
                │ D1 TRACKER   HF Sam3TrackerVideoModel (SAM 3)               │
                │   • references + corrections = masks/points on any frame    │
                │   • chunked sessions; best references re-added to each new  │
                │     chunk; state offloaded to CPU; full teardown per chunk  │
                │                                                             │
                │ D2 DISCOVERY (new, lost or re-appearing parts)              │
                │   • DINOv3 patch-similarity to each class's examples →      │
                │     peaks → SAM 3 point/box prompt → mask                   │
                │   • runs every K frames, or when a track is lost / score    │
                │     drops                                                   │
                │   • + Boost: RF-DETR-Seg trained on confirmed frames (v1.5) │
                │                                                             │
                │ D3 VERIFY & SCORE                                           │
                │   • class: DINOv3 crop embedding → nearest confirmed        │
                │     examples (+ linear probe once a class has 20+ examples) │
                │   • quality gate: Laya Vision yes/no questions ("one whole  │
                │     part?", "cut off?", "blurred?"), recalibrated on your   │
                │     corrections                                             │
                │   • combiner: logistic regression over [SAM score, class    │
                │     probability, top-1/top-2 margin, novelty distance, Laya │
                │     answer, same class across frames, mask-area jump]       │
                │   • conformal threshold → auto-accept keeps wrong labels    │
                │     under X% (default 5%)                                   │
                │                                                             │
                │ LOOP  predict → review → learn; autopilot; picks which      │
                │       frames you check                                      │
                │ STORE SQLite project + COCO-RLE masks; resume anywhere      │
                │ EXPORT COCO, YOLO-seg/det, CVAT XML (Datumaro), LS, VOC     │
                └───────────────▲──────────────────────────────▲──────────────┘
                                │ JSON msgs (clicks, boxes,     │
                                │ keys ↔ RLE masks, scores)    │
            ┌───────────────────┴────────┐        ┌────────────┴─────────────┐
            │ Local host: FastAPI +      │        │ Colab host: anywidget    │
            │ WebSocket; one-click       │        │ inside the notebook      │
            │ launcher opens browser     │        │ (policy-compliant)       │
            └───────────────┬────────────┘        └────────────┬─────────────┘
                            └──── SAME canvas.js (ES module, Konva.js) ────┘
                 masks drawn in the browser · instant hotkeys · only clicks and RLE go over the wire
```

### Projects and sources: videos and image folders
A **project** holds its part classes, the example bank of confirmed crops, the score combiner, Laya's calibration, and the Boost model. You can add **any mix of sources** to one project:
- **Video**: frames sampled from a video file.
- **Image folder**: unordered images in `.jpg`, `.png`, `.bmp`, `.tif` or `.webp`, with an option to include subfolders.

Every source feeds the same example bank, so parts labeled in images help predictions on videos, and the other way round.

| | **Video source** | **Image-folder source** |
|---|---|---|
| How the next item is predicted | D1 tracker (memory frames) + D2 discovery + D3 | **No tracker** (images aren't related). D2 discovery + **"box one, get all"** + Boost + D3 |
| What "next" means | The next frame in time (autopilot step 1/5/10/25) | **Chosen by the tool** (see ordering below) |
| What a confirmed item teaches the model | tracker memory + example bank + calibration + Boost | example bank + calibration + Boost |
| Specific handling | chunked sessions, carrying references between chunks | EXIF rotation fixed; mixed image sizes; optional **near-duplicate removal** (DINOv3 cosine > 0.97); image features cached per image |
| Bulk option | Autopilot | **"Predict all remaining"**: every image is labeled and colour-banded, then you review only the flagged ones |

**Image order** (a setting):
- **Auto (default):** during cold start, the images most similar to your confirmed examples come first, so early predictions are good and you build examples quickly. After that, the most uncertain images come first, balanced for variety (k-center), so every correction teaches the model the most.
- **Other choices:** filename order, random.

Boost (RF-DETR-Seg) matters more for image folders, since there's no tracker to lean on. It turns on after **40 confirmed items** from any source.

### Mode 3: Teach & Transfer (fully automatic)

Your idea, made concrete. It runs as a guided wizard in the UI, or as CLI commands (`partlabeler teach …`, `partlabeler transfer …`), which also run in Colab cells.

```
 ① IMPORT         source video + labels folder (COCO / YOLO-det/seg / CVAT / LabelMe / VOC /
                  Label Studio, via Datumaro)
                  → match each labeled image to its frame in the video: by frame number in the
                    filename, else by perceptual hash + DINOv3 against decoded frames; rescale
                    labels if the images were resized
 ② ANALYSE        report: classes, counts, sizes, where parts sit in the frame, which classes
                  appear together, which frames were labeled vs skipped, label-quality warnings
                  (tiny boxes, overlapping duplicates, frames missing a class)
                  → optional "parent object" question: what are the parts on? (e.g. "car").
                    Checked with SAM 3 text search: does the "car" mask contain ≥90% of the
                    labeled parts?
 ③ LEARN          split the labeled frames by time: 80% train, 20% held out (blocks of
                  neighbouring frames, so the test isn't just memory)
                  → fine-tune RF-DETR-Seg with heavy appearance augmentation (colour-hue rotation
                    through all 360°, grayscale, brightness/contrast, blur, JPEG, scale,
                    perspective)
                  → targeted synthetic data: recolour the PARENT object (e.g. the car body) to
                    random colours while keeping the labeled parts as they are ("same car, other
                    colours")
                  → example bank = DINOv3 embeddings of every labeled crop plus hue-shifted
                    copies (colour-robust matching)
                  → Laya Vision calibrated on the source crops
 ④ PROVE          re-annotate the whole source video automatically with the full pipeline
                  (below) and compare with your labels:
                  → **Knowledge score** = results on the HELD-OUT frames: mask/box mAP50,
                    mAP50-95, recall and precision per class
                  → **Colour stress test** = the same held-out frames with hue rotated,
                    grayscale and inverted body colour: measures the "blue car → other colours"
                    ability directly
                  → if a target is missed, suggestions follow (more labeled frames, turn on the
                    parent constraint, a bigger model, longer training), and a one-click retrain
                  → targets (adjustable): held-out mask mAP50 ≥ 0.90, recall per class ≥ 0.90,
                    colour-stress drop ≤ 10 points
                  → the final model is then retrained on 100% of the labeled frames
 ⑤ TRANSFER       for each new video (a batch queue):
                  RF-DETR-Seg finds parts (optionally only inside the parent-object mask)
                  → SAM 3 cleans up each mask from the detected box
                  → Sam3TrackerVideo spreads confident detections forward and backward, filling
                    missed frames and smoothing jitter
                  → D3 checks each result (DINOv3 class match + Laya gate + combiner,
                    conformal threshold)
                  → **self-training** (optional, 1–2 rounds): high-confidence results from the
                    new videos become extra training data → retrain → rerun. This adapts the
                    model to the new colours and lighting.
                  → out-of-distribution alarm: frames whose DINOv3 features are far from
                    anything in training are flagged
 ⑥ OUTPUT         for each video: an annotated folder (frames + labels in your chosen format),
                  a preview video with the labels drawn on, and a QA report (confidence per
                  frame, flagged segments, class counts, alarms).
                  Flagged frames can be opened in the interactive annotator with one click;
                  your fixes go back into the next training round.
```

**On "100% knowledge acquired":** a perfect score on the frames the model trained on is easy, but it only proves memory, not that the model will work on a green car. So the tool measures on **held-out** frames plus the **colour stress test**. Those are the honest signals of whether it will transfer. You get a clear go / no-go against your targets instead of a claimed 100%.

**Why this is reliable:**
- A fine-tuned detector (RF-DETR-Seg, Apache-2.0) is the proven way to reproduce labels from tens to hundreds of examples.
- Colour augmentation plus parent recolouring target exactly the kind of difference you described.
- SAM 3 is strong on common objects such as "car" with text alone, so limiting the search to inside the car cuts false detections on other colours.
- Self-training plus tracking fills gaps and adapts the model to new videos without people.

### The loop (unchanged from your idea, with more reliable parts)
1. **Set up:** upload the video, name the part classes, and label 3–5 **reference frames**. You click or box one part and SAM 3 outlines it; the "box one, get all" tool finds the rest in that frame.
2. **Predict** the next frame (or the next K): D1 tracks the parts it already knows, D2 finds new or lost ones, D3 labels and scores everything.
3. **Review:** each frame shows green, amber or red.
   - Keys: `A` accept · click to add or remove mask area · drag to redo a box · `1-9` change class · `Del` delete · `→` accept and go to the next frame.
   - Autopilot jumps you to the frames most likely to be wrong.
4. **Learn:** each confirmed frame updates several parts:
   - it becomes a tracker memory frame;
   - its crops join the class example sets, and rejected proposals become "background";
   - Laya's calibration is refit after 30 or more corrections;
   - the score combiner and the auto-accept threshold are refit every 50 corrections;
   - RF-DETR-Seg retrains in the background every N confirmed frames (Boost; this uses the same `detector.py` as Teach & Transfer).
5. **Autopilot:** after enough frames in a row with no edits, the step grows 1 → 5 → 10 → 25 frames. Any risky frame drops it back to 1.
6. **Cold start:** until each class has 5–10 confirmed examples, every frame goes to review.

---

## This machine (checked 2026-09-25)
- **CPU:** Ryzen 7 3700X (8 cores / 16 threads).
- **RAM:** 16 GB.
- **GPU:** **RTX 3060 12 GB** (compute capability 8.6, so bf16 works), driver 596.49 (CUDA 13.2).
- **Disk:** 189 GB free on C:.
- **OS:** Windows 11 Home.
- **Already installed:** torch 2.11+cu130 (CUDA working), ultralytics, OpenCV, Git, Node 24, ffmpeg.
- **Missing:** transformers, uv, and the `claude` CLI.
- **Python environment:** a new **Python 3.12 venv made with `uv`**, the same version Colab uses. The global Python 3.10 and 3.14 stay untouched.

## Stack
| Layer | Choice | License |
|---|---|---|
| Tracker + "box one, get all" | `transformers` ≥5: `Sam3TrackerVideoModel`, `Sam3Model` (weights `facebook/sam3`, gated) | SAM License (commercial OK; bans military/ITAR use) |
| Embeddings | DINOv3 ViT-S+/B (gated, "Built with DINOv3" attribution) · DINOv2 fallback (Apache) | DINOv3 License |
| Quality gate | Laya Vision `thaitea/laya-vision` (install from git, **never next to PyPI `laya`**) | Code Apache 2.0 · weights **CC BY-NC-SA 4.0** (can be swapped out if you go commercial) |
| Score combining | scikit-learn (logistic regression / isotonic), our own split-conformal code (~50 lines) | BSD |
| Trained detector (**core for Teach & Transfer**, also Boost) | RF-DETR-Seg N/S/M (`rfdetr`), trained on the 3060 (Colab T4 in Phase B) | Apache-2.0 |
| Augmentation | albumentations (hue/colour/grayscale/blur/JPEG/geometry) + our own parent-object recolouring (~80 lines) | MIT |
| Evaluation | pycocotools / faster-coco-eval (mask and box mAP, per-class precision and recall) | BSD/Apache |
| Label import | Datumaro (COCO, YOLO, CVAT, LabelMe, VOC, Label Studio) · imagehash (matching labeled images to video frames) | MIT/BSD |
| Video / storage / export | PyAV, SQLite, pycocotools (RLE), supervision (COCO/YOLO), Datumaro (CVAT XML) | BSD/MIT |
| CLI and job queue | typer (CLI) · a simple SQLite-backed job queue with checkpoints (so jobs survive a Colab 12 h disconnect) | MIT |
| UI | canvas.js (plain ES module + Konva.js) · FastAPI + uvicorn + WebSocket (local) · anywidget (Colab/Jupyter) | MIT/BSD |
| Tests | pytest, playwright (UI smoke test) | Apache/MIT |
| Phase B fallback | `EdgeTamVideoModel` (HF, same tracker API, 22× faster than SAM 2) for CPU and weak GPUs; SAM 3 at 560 px on the T4 | Apache-2.0 |

## Settings (defaults in bold)
| Setting | Choices |
|---|---|
| Device / precision | **Auto** → bf16 on the 3060 · fp16 on the T4 · fp32 on CPU |
| Tracker | **SAM 3** · *(Phase B)* EdgeTAM · *(optional)* SAM 3.1 via the official repo |
| Input resolution | **1008** · 840 · 560 |
| Frame sampling (video) | **every 5th frame** · target fps · scene change · all frames |
| Image folder | include subfolders **on** · near-duplicate removal **on (0.97)** · order **Auto** / filename / random · "Predict all remaining" button |
| Chunk size | **180** frames |
| Memory frames per chunk | **8** |
| Step size | 1 · 5 · 10 · **Autopilot** |
| Discovery | **every 10 frames + when a track is lost** · every frame · off |
| Auto-accept wrong-label cap | **5%** (conformal) · 1–10% slider |
| Laya quality gate | **on** · off · only on uncertain crops |
| Boost (RF-DETR) | **off** (opt-in; S6 showed 15 confirmed images reach mAP50 0.889 in a 3–5 min run if you turn it on) |
| Teach & Transfer | model size N / **S** / M · held-out split **20% (time blocks)** · targets **mAP50 ≥ 0.90, recall ≥ 0.90, colour-stress drop ≤ 10** · augmentation **strong** / medium / off · parent object *(optional text, e.g. "car")* · parent recolouring **on** · search inside parent only **on if a parent is given** · self-training rounds 0 / **1** / 2 · tracker smoothing **on** · output FPS **every frame** / every Nth · preview video **on** |
| Output | **masks + boxes** · boxes only · polygon tolerance |
| Export | **COCO** · YOLO-seg · YOLO-det · CVAT XML 1.1 · Label Studio · VOC; per source or the whole project; option to **keep the original filenames and folder layout** (for image folders) |

---

## Step 0: Development tools (runs first)
Project folder: the repository root.

| Tool | How it gets installed |
|---|---|
| **find-skills** | `npx skills add https://github.com/vercel-labs/skills --skill find-skills -a claude-code -g`. Then I search for `pytorch`, `computer vision`, `segment anything`, `huggingface transformers`, `fastapi`, `anywidget / jupyter widget`, `canvas / konva`, `python testing`, `playwright`. **You approve each skill before I install it.** |
| **claude-mem** + **ponytail** | The `claude` CLI isn't installed, so I use the `update-config` skill to add `extraKnownMarketplaces` (`thedotmack/claude-mem`, `DietrichGebert/ponytail`) and `enabledPlugins` (`claude-mem@thedotmack`, `ponytail@ponytail`) to `~/.claude/settings.json`, then restart. Fallback: `npm i -g @anthropic-ai/claude-code`, then `claude plugin install …`. |
| **graphify** | Already installed. Run `/graphify` after each milestone. |

## Step 1: Short tests that remove the biggest risks before building (scripts in `spikes/`, results written to `spikes/REPORT.md`)
You first request access on Hugging Face to `facebook/sam3` and DINOv3, then run `hf auth login` once. **Please also provide:**
- 1–2 short clips of your real parts;
- one folder of about 30 photos of them;
- for Teach & Transfer, **one source video + its manually labeled frames folder** (any common format), plus **1–2 "similar" target videos** (e.g. the other-colour cars).

Each test runs on these.

| # | Test | Passes if |
|---|---|---|
| S1 | `Sam3TrackerVideoModel` on the 3060: 3 reference masks, 180-frame chunk, bf16; one correction click at frame 90; re-track | It works; peak VRAM under 10 GB; VRAM returns to baseline after teardown; speed recorded |
| S2 | Carrying references into the next chunk: re-add the best references as mask inputs in a fresh session | Mask IoU vs a single-session run ≥0.9 on the overlapping frames |
| S3 | Discovery: DINOv3 similarity peaks → SAM 3 prompt, on frames where parts enter the view | Finds ≥80% of the parts you've labelled on sample frames; false proposals get caught by D3 |
| S4 | Laya Vision in the same environment: version conflicts, speed on the 3060, sensible quality-gate answers on 20 crops | Installs cleanly, under 100 ms per crop, answers make sense |
| S5 | canvas.js inside anywidget (local Jupyter) and inside FastAPI: click → mask round trip | Under 150 ms locally with the model loaded; hotkeys respond instantly |
| S6 | Image-folder mode on **30 random images of your parts**: label 5 → predict 25 using discovery + "box one, get all" | Finds ≥70% of the parts in the 25 before any Boost training, and the number of flagged images drops as you confirm more |
| S7 | Teach & Transfer on your sample: import + frame matching; RF-DETR-Seg S fine-tuned on the 3060 with colour augmentation; held-out and colour-stress scores; run on 1 target video | Frame matching ≥99% correct; training fits in 12 GB and finishes in under about 1.5 h; held-out mAP50 and colour-stress scores recorded; on target-video frames you spot-check, recall ≥0.8 before self-training |

**If a test fails:**
- S1/S2 fail → try the official SAM 3 repo with triton-windows, then WSL2.
- S3/S6 fail → make Boost (RF-DETR) the main way of finding new parts, starting earlier (after 15 items). In image mode, the "box one, get all" tool takes over until then.
- S4 fails → Laya becomes an optional plugin and the combiner runs without it.
- S7 misses targets → switch to model size M, turn on parent-object search and recolouring, add a SAM 3 LoRA fine-tune (images, 12 GB rank 8) as a second model, and/or ask for more varied labeled frames.

---

## Project layout
```
partlabeler/
  engine/
    sources.py       # Source interface: VideoSource (PyAV) and ImageFolderSource (scan, EXIF fix, near-duplicate removal)
    frames.py        # PyAV decode → chunked JPEG cache, sampling
    ordering.py      # what to show next: time order (video) or Auto/similar/uncertain/k-center (images)
    tracker.py       # Sam3TrackerVideo sessions, chunking, carrying references, teardown, VRAM watchdog
    discovery.py     # DINOv3 similarity → prompts → SAM 3 masks; "box one, get all" (Sam3Model)
    embed.py         # DINOv3/DINOv2 crop + patch embeddings (cached)
    classify.py      # per-class example bank, nearest-neighbour + linear probe, background class
    laya_gate.py     # Laya Vision quality questions, calibration refit
    score.py         # combiner (logistic regression), conformal threshold, frame risk ranking
    loop.py          # predict → review → learn, autopilot, cold start
    detector.py      # RF-DETR-Seg train / predict / checkpoints (used by Boost and by Teach & Transfer)
    importers.py     # Datumaro label import, matching labeled images to video frames (hash + DINOv3), rescaling
    analyse.py       # dataset analysis report, label-quality warnings, parent-containment check
    augment.py       # colour/appearance augmentation, parent-object recolouring
    evaluate.py      # held-out and colour-stress evaluation, Knowledge-score report
    teach.py         # orchestrates ①–④: import → analyse → split → train → prove → final retrain
    transfer.py      # ⑤–⑥: detect → SAM 3 cleanup → tracker smoothing → verify → self-training → export + QA report
    jobs.py          # background job queue, progress, checkpoint and resume
    cli.py           # `partlabeler teach|transfer|export` (typer)
    store.py         # SQLite schema, RLE masks, resume
    export.py        # COCO, YOLO, CVAT (Datumaro), Label Studio, VOC
    hw.py            # device and dtype detection, presets
    api.py           # one message protocol (JSON) used by both hosts
  ui/
    canvas.js        # shared ES module: image, mask layers, boxes, clicks, hotkeys, frame strip
    host_fastapi.py  # local web host (WebSocket)
    host_widget.py   # anywidget host (Colab/Jupyter)
  spikes/  tests/
  run_windows.bat  pyproject.toml  README.md
  PartLabeler_Colab.ipynb   # Phase B
```

## Build order
1. **Step 0 → Step 1 tests** (go / no-go).
2. **M1 (usable):** project store with sources (video + image folder), tracker, canvas, FastAPI host, manual loop (predict next → correct), "box one, get all", COCO/YOLO export.
3. **M2 (the learning loop):** discovery, classification, Laya gate, score combiner, autopilot, cold start, risk-ranked review, **image ordering + "Predict all remaining"**.
4. **M3 (stable):** long videos, chunk carry-over, VRAM watchdog, crash-safe autosave and resume, `run_windows.bat`, full exports.
5. **M4 (Teach & Transfer):** detector.py, importers, analysis, augmentation + recolouring, evaluation (Knowledge score + colour stress), transfer batch queue, tracker smoothing, self-training, QA report + preview video, wizard UI + CLI, and "open flagged frames in the annotator". Boost then comes almost for free because it uses the same detector.py.
6. **Phase B:** Colab (anywidget, fp16/560 px preset, Drive persistence), then small GPUs and CPU (EdgeTAM, DINOv2-S, Laya only on uncertain crops), plus a benchmark table per preset.
7. **Label types (added 2026-09-26, after research):** outlines (instance segmentation: SAM 3 masks kept as COCO RLE, brush/eraser, "Outline boxes", polygons or masks in all five exports; Teach & Transfer stays detection and SAM 3 outlines its boxes) and image classes (a grid with DINOv3 grouping, prototype/logistic suggestions that leave unfamiliar images alone, a wrong-label check, near-duplicates; class-folder, Ultralytics-classify and CSV exports), plus "Sort parts" for box and outline projects. S10 in `spikes/REPORT.md`. Not in v1: text-prompted zero-shot classes (SigLIP 2), multi-label images, a local VLM to name groups, polygon vertex editing, masks for Teach & Transfer's own model (RF-DETR-Seg).

`/graphify` runs after each milestone; claude-mem tracks progress between sessions.

## Verification
1. **Step 1 tests:** pass or fail against the targets above, recorded in `spikes/REPORT.md`.
2. **Unit tests:** exporters round-trip; RLE; loop state machine (a confirmed frame updates memory, examples and calibration); autopilot grow and shrink; conformal threshold keeps the wrong-label rate at or below the target on synthetic data; chunk carry-over.
3. **End-to-end on the 3060** with your clip: 2–3 classes, 3 reference frames, 100 frames.
   - Edits per frame should **fall** over time (the app shows a chart).
   - Auto-accepted frames spot-checked: wrong-label rate at or below 5%.
   - VRAM under 10 GB throughout.
4. **End-to-end with an image folder:** 100 mixed photos, 5 labeled by hand.
   - Flagged images per 10 reviewed should fall over time.
   - Near-duplicates get skipped.
   - "Predict all remaining" followed by reviewing only the flagged images meets the ≤5% wrong-label target on a spot check.
   - The export keeps the original filenames.
   - A mixed project (1 video + 1 folder) shares its example bank: labels from the folder improve the first predictions on the video.
5. **End-to-end Teach & Transfer:** your source video + labels.
   - The Knowledge report shows held-out and colour-stress scores against the targets.
   - The re-annotated source is shown next to your manual labels, with a per-frame diff viewer.
   - Transfer to 2 target videos (other colours) produces annotated folders, preview videos and QA reports.
   - Spot-checking 30 random target frames per video: recall and precision ≥ 0.9 after self-training, and everything wrong was flagged by the QA report.
   - Unit tests: frame matching with resized or renamed images, label import round-trip per format, time-block split has no leakage, recolouring leaves labeled parts untouched.
6. **Stability:** a 5-minute video, the app killed mid-run and resumed with no lost work; 3 chunk changes with no memory leak. A Teach & Transfer job killed mid-training resumes from its checkpoint.
7. **UI:** a Playwright smoke test (click → mask, hotkeys, export, wizard), plus a manual check in the built-in browser.
8. **Checking the exports:** open the COCO export in `pycocotools` and CVAT; run a quick RF-DETR training on the export.
9. **(Phase B)** the Colab T4 notebook runs end to end inside the notebook UI.

## Sources (main ones)
- SAM 3 and SAM 3.1: [HF Sam3TrackerVideo](https://huggingface.co/docs/transformers/model_doc/sam3_tracker_video) · [HF Sam3Video](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_video) · [SAM 3 repo](https://github.com/facebookresearch/sam3) · [exemplars same-frame only (#183)](https://github.com/facebookresearch/sam3/issues/183) · [VRAM leak (#305)](https://github.com/facebookresearch/sam3/issues/305) · [OOM on long videos (#169)](https://github.com/facebookresearch/sam3/issues/169) · [Windows Triton (#252)](https://github.com/facebookresearch/sam3/issues/252) · [SAM 3 paper](https://arxiv.org/html/2511.16719v1) · [CAD-Prompted SAM3](https://arxiv.org/html/2602.20551v2)
- Embedding matching: [SAM+DINOv3 on BOP](https://arxiv.org/html/2604.26404) · [CALIPER](https://arxiv.org/abs/2609.17820)
- Laya: [Laya Vision docs](https://r33drichards.github.io/laya-vision/) · [Laya](https://github.com/NandhaKishorM/laya)
- Confidence thresholds: [Conformal Labeling](https://arxiv.org/abs/2510.14581) · [Colander](https://arxiv.org/abs/2404.16188)
- Colab and UI: [Colab FAQ (web-UI rule)](https://research.google.com/colaboratory/faq.html) · [anywidget](https://anywidget.dev/blog/introducing-anywidget/)
- Detectors, trackers and reference tools: [RF-DETR](https://github.com/roboflow/rf-detr) · [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) · [LIT-LoRA](https://github.com/YoungXinyu1802/LIT-LoRA) · [SAMannot](https://arxiv.org/abs/2601.11301)
- Dev tools: [find-skills](https://github.com/vercel-labs/skills) · [claude-mem](https://github.com/thedotmack/claude-mem) · [ponytail](https://github.com/dietrichgebert/ponytail)
