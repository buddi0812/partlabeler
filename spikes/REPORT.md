# Step 1 test results

Machine: RTX 3060 12 GB, Ryzen 7 3700X, Windows 11 · Python 3.12.14 · torch 2.14.0+cu130 · transformers 5.17.0.
Raw numbers: `spikes/out/results.jsonl`. Sample clip: Hugging Face `sam2-fixtures/bedroom.mp4` (960×540, 180 frames)
until real part footage arrives.

## S1 — SAM 3 tracking (`s1_tracker.py`) — PASS
| Measure | Result | Target |
|---|---|---|
| Frames tracked | 180 / 180 | > 90% |
| Peak GPU memory | 5.9 GB | < 10 GB |
| Memory left after session teardown | 62 MB | < 256 MB (issue #305 leak not seen) |
| Tracking speed (bf16) | 1.9 frames/s | — |
| Model load | 2.3 s | — |

State on GPU vs CPU and fp16 vs bf16 made no speed difference: the cost is compute at 1008 px. Lower
input resolution is the lever for bulk runs (test in M3/Phase B).

## S2 — carrying references between chunks (`s2_chunks.py`) — PASS
Two 90-frame chunks vs one 180-frame session, mask IoU on the second chunk:

| Seeding of chunk B | Mean IoU | Worst frame |
|---|---|---|
| Last frame of chunk A only | 0.969 | 0.593 |
| Last frame + 2 earlier reference frames | **0.986** | 0.649 |

Decision: seed each new chunk with the previous chunk's last frame plus up to 2 earlier confirmed references.

## S5 — annotation canvas (`ui/canvas.js`, `ui/host_fastapi.py`, `ui/host_widget.py`, `s5_widget.ipynb`) — PASS
One ES module, two hosts. SAM 3 encodes the image once (~400 ms); each click runs only the prompt
encoder + mask decoder (~11 ms on the GPU). Round trip = click → mask drawn in the browser.

| Host | Round trip after warm-up | First click | Target |
|---|---|---|---|
| Local page (FastAPI + WebSocket) | 71–87 ms, later 20 ms | 228 ms | < 150 ms |
| JupyterLab notebook (anywidget) | 60–65 ms (median 61) | 185 ms | < 150 ms |

One positive click on the girl's dress segmented the whole child (score 0.92). The notebook host is the
Colab path (Colab forbids web-UI-first use; an in-notebook widget is allowed).

## S3 + S6 — finding parts without training (`s3_discovery.py`, `engine/embed.py`) — S6 marginal pass, S3 fail
30 random labeled White frames as an unordered folder: 5 references, 25 predicted. DINOv3 ViT-S/16
(timm) patch features of the reference boxes → similarity peaks in each car crop (threshold calibrated
leave-one-out on the references) → box of the reference median size, or a SAM 3 click mask. Left/right
twins share one appearance model; side from position. 0.9 s per image.

| | Found (recall) | Precision |
|---|---|---|
| All parts, IoU ≥ 0.3, reference-size box | **0.72** | 0.73 |
| All parts, IoU ≥ 0.5, reference-size box | 0.58 | 0.59 |
| Same with SAM click boxes (IoU ≥ 0.3 / 0.5) | 0.68 / 0.45 | 0.69 / 0.45 |
| Grille, skid, DRLs, head lamps (IoU ≥ 0.3) | 0.92–1.00 | high |
| Plate, parking lamp | 0.81–0.82 | medium |
| Roof rack / roof garnish (~20 px, look alike) | 0.33 / 0.58 | many swaps and extras |
| Sunroof | 0 (threshold calibrated too high) | — |

SAM click boxes are worse: a click often returns a sub-part or the whole panel, not the labeled part.
Plan fallback applies: switch to the trained detector (Boost) early; learning curve below decides when.

## S7 — Teach & Transfer on the example car data (`s7_prepare.py`, `s7_train.py`, `s7_eval.py`, `s7_transfer.py`) — PASS on the source, partial on transfer
Source: White, station 3, 717 frames labeled by the booth YOLO (every 5th frame, 7,613 boxes, 14 of 17
classes present, ~253 short gaps = YOLO misses). Targets: Black and Green, **station 2**
(different camera position, background and lighting), no labels.

Pipeline: SAM 3 text prompt "car" → crop (100% of labeled parts inside, 298/298) → RF-DETR Small, 640 px,
no horizontal flip (classes are the car's left/right), colour/brightness/blur/noise/affine augmentation,
time-block split (574 train / 143 held out), 30 epochs in 39.5 min on the RTX 3060 (4.4 GB).

| Measure | Result | Target |
|---|---|---|
| Held-out agreement with booth YOLO, mAP50 | **0.936** | ≥ 0.90 |
| Held-out mAP50-95 | 0.830 | — |
| Best confidence cut-off (micro F1 vs YOLO) | 0.45 (F1 0.962) | — |
| Classes with recall ≥ 0.97 | 11 of 14 | — |
| Weakest class | `brand_logo_fr` recall 0.18 (the YOLO labels the logo in only 22% of frames) | — |
| Colour stress (grayscale / black body / green body) | −0.2 / −1.6 / −2.4 points mAP50 | ≤ 10 points |
| Station 2, boxes per frame (no labels) | black 5.6, green 8.5 (white labels: 10.8) | — |

Station 2 visual check: lamps (DRL, head lamps) found in ~90–95% of frames on both cars; green also gets
grille, skid, sunroof, roof racks, parking lamp. Black car misses the grille and skid in darker frames
(dark plastic on a dark car under station-2 lighting). Scores against the source are agreement with the
booth YOLO, not accuracy.

Output (`s7_transfer.py`, baseline model, cut-off 0.45): `data/example/auto_labels/<video>/` with images/,
labels/ (YOLO, same 17 classes), classes.txt, data.yaml, preview.mp4, summary.json.
Black 687 frames / 3,802 boxes; Green 402 frames / 3,300 boxes.

### Self-training round (`s7_selftrain.py`) — rejected
Pseudo-labels from the baseline at conf ≥ 0.6 on 1,089 station-2 frames (+219 grille/skid/roof-rack boxes
carried over from neighbouring frames), fine-tuned 8 epochs from the baseline (33 min).

| | Baseline | Self-trained |
|---|---|---|
| White held-out mAP50 / mAP50-95 | 0.936 / 0.830 | 0.935 / 0.841 |
| Black: boxes per frame, grille, skid, roof rack presence | 5.6, 0.35, 0.23, 0.12 | 4.5, 0.28, 0.05, 0 |
| Green: boxes per frame, roof rack presence | 8.5, 0.75 | 6.8, 0 |

Confirmation bias: parts found below the pseudo-label cut-off became negatives, so the model learned to miss
them at station 2. Decision: keep the baseline. For a new station, the reliable fix is a few human-corrected
frames from that station (the M1 review loop) or labels from that station's own YOLO — not unsupervised
pseudo-labels. Consensus/temporal voting pseudo-labels are an option to revisit in M4.

## S4b — off-the-shelf checkers, zero-shot (`s4b_checkers.py`) — not reliable
User direction: no training; use open decision models (openjev, Laya) to help. Held-out White car
crops, 14 part types, ~10 instances each: real part crop vs the same box shifted off the part (drift), and
real crop with its own statement vs a different part's statement. AUROC 1.0 = perfect, 0.5 = chance.

| Checker | vs drift | vs wrong part | ms / check |
|---|---|---|---|
| openjev `qwen3.5-0.8b-nli-v2s-long` (image NLI, MIT, rev 058a6c2) | 0.72 | 0.55 | 119 |
| openjev `qwen3.5-4b-nli-v2` (same rev; 9 GB in bf16; ships no image-processor config, the 0.8B one is used — same Qwen3.5 processor) | 0.76 | 0.67 | 258 |
| Laya Vision (`thaitea/laya-vision` rev 8b318c9) | 0.54 | 0.51 | 140 |

openjev's code was reviewed before use (transformers only, no remote code; the one `torch.load` is in an
unused helper). None is dependable as a box checker on these parts; the 4B is better but needs 9 GB next to
SAM 3 (does not fit a 12 GB card or a T4 alongside the annotator's models).

## S4c — flagging tracked boxes that went wrong (`s4c_laya_flags.py`) — plain numbers win, Laya fails
Six start frames of the White video, source labels as the person's boxes, tracked 30 frames ahead each:
1,870 tracked boxes, 1,794 with a source label of the same class, 135 of those wrong (IoU < 0.5).

| Signal | AUROC |
|---|---|
| Area vs the person's box for that part | **0.913** |
| Tracker's own confidence (1 − score) | 0.877 |
| Centre moved since the previous frame | 0.702 |
| Area vs the previous frame | 0.602 |
| Laya (PyPI `laya` 0.3.20, text model, one yes/no question over the numbers as JSON, zero-shot, 47 ms/box) | 0.227 (worse than chance) |

Old app rule (jump > 0.5 diagonal or area ×2 frame-to-frame): flagged 9 boxes, caught 5% of wrong ones.
New rule in `Project.flags` (area vs nearest person box outside ×1.5, or centre moved > 0.3 diagonal): flags
6% of boxes, catches 57% of wrong ones, 2 of 3 flags real. Adding the tracker's score gave nothing more.
Tuned on one video, so the thresholds are defaults, not truths.

## M1 tracking check (`m1_track_check.py`) — track ahead from one frame vs source labels
Box style copied from the person's box onto SAM's outline (median IoU 0.705 → 0.74–0.93).

| Start frame | Parts on it | Next 60 frames: tracked boxes matching a label (IoU ≥ 0.5) | Labels covered | Median IoU | Speed |
|---|---|---|---|---|---|
| 400 | 12 | 98.8% | 95.2% | 0.929 | 0.55 frames/s |
| 100 | 8 | 82.7% | 80.4% | 0.845 | 0.76 frames/s |
| 0 | 2 | 61.7% | — | 0.736 | 1.59 frames/s |

## Find similar (`ConceptFinder.find_like`, key F) — SAM 3 exemplar search within one frame
White frame 500, one roof rail box (23 × 22 px) as the example: the second rail scores only 0.44 and is
2.9× bigger (perspective), so the defaults are score ≥ 0.3, size ⅓–3× and shape within 2×, with hits that
mostly cover or sit inside a better hit dropped. One left DRL box found the deleted right DRL (0.56), labeled
`right_drl` from where each twin's confirmed boxes sit. ~0.5 s per search on the 3060.

## M1 annotator — end-to-end check in the browser (`ui/`, `engine/api.py`, `engine/project.py`)
Example white-car project (717 frames). Verified by driving the page: timeline jump, quick keyboard stepping,
track 20 frames ahead (12 parts), click-to-box on the small brand logo, confirm, YOLO export.

- First version of click-to-box outlined the whole car for a click on the logo: SAM's single "best" outline
  is the whole object. Fixed: the first click uses the smallest of SAM's three nested outlines, **M** steps
  larger. SAM's own score for small outlines can be ~0.01, so it is not used to filter.
- Export matches the source format: tracked right_drl on f002015 `0.3200 0.6501 0.0726 0.0544` vs source
  `0.3201 0.6503 0.0719 0.0535`.
- Image-folder suggestions through the session, 5 labeled → 25 suggested, full frame: 67% of parts found
  (IoU ≥ 0.3), 176 extra boxes over 25 images, 0.3 s per image (S3 with car crops: 72%).
- Tests: 9 passing (project store, YOLO/COCO round trips, session message flow without models).

Not yet in M1: SAM 3 "box one, get all" for repeated parts; parent-object crop for suggestions.

## S4 — Laya Vision quality gate (`s4_laya.py`) — FAIL on answer quality
Pinned: code commit `568feee` (reviewed: safetensors only, no remote code, no subprocess), model revision `8b318c9`.

| Measure | Result | Target |
|---|---|---|
| Install | clean (vendored, reviewed source) | clean |
| Speed | 95 ms/crop median (3 questions) | < 100 ms |
| GPU memory | 0.8 GB | — |
| Consistency | two near-identical consecutive frames of a child: "a person" 84% vs 19% | sensible answers |

Zero-shot answers flip on near-identical inputs, and a "blurred/unclear" option attracts most answers.
Decision (plan fallback): Laya Vision is an optional quality gate, **off by default**; the confidence combiner
runs without it. Revisit after `agent.calibrate()` on ≥ 30 of the user's own corrections (M2).

## fp16 path (what Colab's T4 will use), checked on the 3060 with `PARTLABELER_DTYPE=fp16`
Click-to-outline, tracking 5 frames × 6 parts (no NaN scores), find similar and suggestions all run in fp16;
peak 3.96 GB allocated with every model loaded. Speed and quality on a real T4 still to be measured in Colab.

## S8 — track clean-up vs track check in Transfer (`s8_track_check.py`, `engine/tracks.py`) — check only
Question: Transfer labels each frame on its own; would linking detections into tracks and fixing single frames
help? Measured on the example dataset's 143 held-out frames (the taught RF-DETR-S at its chosen threshold 0.45,
scored against the source labels at IoU 0.5).

| Along tracks | F1 | Note |
|---|---|---|
| nothing changed (raw detections) | 0.962 | 73 false boxes, 47 missed |
| fill gaps of up to 2 sampled frames | 0.947 | 51 boxes added, 2 of them right |
| vote on the class (left/right flips) | 0.962 | no flips to fix with this detector |
| centred smoothing | 0.962 | no change at IoU 0.5 |
| drop low-score boxes seen on one frame | 0.962 | 5 false and 3 true boxes removed |

A part missing for a frame or two was usually really not labeled there (out of view or hidden), so filling is
wrong, and the other fixes change nothing measurable. Decision: never change labels; list inconsistent frames
instead (label flip, possible miss, lone low-score box). The check lists 24% of the frames (34 of 143); 71% of
those hold a detector error (52% of all frames do), together 38% of all errors (45 of 120): a modest but real
review aid. Plain box linking (greedy IoU with a velocity prediction), no new dependency; Roboflow's `trackers`
(ByteTrack) is the upgrade if a project has many identical parts per frame.

## S9 — quick transfer, no training (`s9_quick_transfer.py`, `engine/quick.py`) — a rough preview
Up to 30 labeled images spread over the source are the examples; each new frame is matched with the annotator's
Suggest (DINOv3), then SAM 3 fits each matched box to the part.

| Test | F1 at IoU 0.5 | Found (recall) | Speed |
|---|---|---|---|
| held-out frames of the source video, matched boxes | 0.64 | 65% | 0.4 s/frame |
| same, boxes fitted by SAM 3 (default) | 0.69 | 70% | 1.4 s/frame |
| taught RF-DETR-S on the same frames, for comparison | 0.96 | 97% | |
| another colour of the product (80 frames), against the taught detector's labels | 0.48 | 57% | 2.1 s/frame |

On another colour the thresholds the examples set for themselves found nothing (the examples all come from
one video and resemble each other far more than a part in another colour resembles them), so each new video is
calibrated first: per class, the score reached in about as many of 24 sampled frames as the part appears in the
source. The last row compares against the taught detector's labels, not hand labels: lamps agree in count, while
quick transfer puts roof and bumper parts in many more frames. The source has those in 93-95% of its frames and
the taught detector found the roof rack in only 12 of the 80, so part of the disagreement is the reference
missing parts. Real accuracy on another colour needs hand labels to measure. Verdict: a quick first pass to
review, not a replacement for Teach.

## S10 — smart sorting: grouping, suggestions, wrong labels (`s10_sorting.py`, `engine/sort.py`) — adopted
3,000 part pictures cut from the example dataset (14 classes; left/right twins are mirror images, so scores are also
given with each pair merged), DINOv3 features on an RTX 3060. Grouping: Ward clustering after PCA to 128 dims,
number of groups picked by the simplified silhouette. Suggestions: class prototypes below 5 examples, logistic
regression from 5. Wrong labels: 3% of labels flipped at random, found by leave-one-out neighbour votes.

| DINOv3, pooling | Speed | Groups (true 14 / 11) | Grouping NMI, twins merged | Right, 1 / 3 / 10 examples a class | Wrong labels found |
|---|---|---|---|---|---|
| small, CLS token | 470/s | 22 | 0.80 | 81% / 87% / 92% | 88 of 90, 87% of flags real |
| small, patch mean | 500/s | 17 | 0.85 | 83% / 86% / 92% | 88 of 90, 85% |
| base, CLS token | 250/s | 6 | 0.81 | 80% / 86% / 92% | 88 of 90, 88% |
| **base, patch mean** (adopted on GPU; small on CPU) | 250/s | 12 | 0.85 | 83% / 88% / 93% | 88 of 90, 88% |

The patch mean beats the CLS token here, against the literature's preference for CLS in instance retrieval; the
automatic number of groups swings from 6 to 22 between variants, hence the slider (re-cutting the tree is
instant). With twins merged, suggestions from 3 examples are right 93-94% of the time.

Suggestions for kinds nobody has named yet: with half the classes named, every picture of the other half got a
wrong suggestion. Now a class is only suggested where the picture is as close to one of its examples as the
closest 90% of neighbour pairs in the collection are to each other:

| Examples per named class | Named kinds suggested correctly | Unnamed kinds given a (wrong) suggestion |
|---|---|---|
| 1 | 46% | 1% |
| 3 | 59% | 2% |
| 10 | 72% | 5% |

Each round of accepting and suggesting again reaches further. Caveat: crops of one video's parts are easier to
group than a real messy folder; whole-image folders were only tried by hand (240 part pictures: 19 groups in
6 s including model loading, 2 sets of near-duplicates).
