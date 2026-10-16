# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Confirmed (2026-10-17): the annotator serves three groups equally.

- **Dedicated annotators**: team members who label for hours on a laptop or desktop, keyboard-first, who want speed and density.
- **Occasional operators**: line or inspection staff who label now and then (for example on a production box running PartLabeler for them) and need clear, guided controls.
- **Learners**: people new to annotation who are learning the craft on the tool itself; every action must be discoverable and explain itself without a manual.

## Product Purpose

PartLabeler builds labeled datasets of parts (any product, any part list) from videos and image folders: boxes (object detection), exact outlines (instance segmentation) or one class per image (classification). The output is the dataset itself (YOLO, COCO, CVAT, Pascal VOC, Label Studio, class folders, CSV). Labeling trains nothing: the person clicks, the models outline, track, group and suggest, and the person confirms. Success is a correct, confirmed dataset produced quickly.

## Positioning

SAM 3 click-to-outline and video tracking are built in, with no separate model server; it runs locally, free (Apache-2.0), with a one-double-click Windows install and no Docker; the same annotator runs inside Jupyter and Google Colab notebooks; Teach & Transfer proves a trained detector on held-out frames before it labels anything.

## Operating Context

- Annotating needs a laptop or desktop: below 900 px wide the annotator shows a notice instead of the canvas (width only, never a device check; notebooks are never gated).
- One screen shows one job (a range of one task's frames), as in CVAT: frame canvas, tools, classes, the parts on the frame, find/track actions, and a timeline of every frame's status.
- Long sessions driven by keyboard shortcuts (documented in docs/GUIDE.md and the in-app `?` list); people are trained on them.
- Hosts: the local FastAPI app (one page per job, REST helpers, accounts and per-account preferences) and a notebook widget (anywidget, incl. Colab: no REST, iframe sizing, no browser full screen).
- Team server mode: several people annotate the same project at once; edits appear live in every open tab; undo is per person.
- Embedded mode (`box_managed`): developer-only UI is hidden and no screen shows the computer's filesystem.

## Capabilities and Constraints

- Project types are fixed at creation: detect (boxes), segment (outlines stored as masks; box = mask box), classify (one class per image, on a grid).
- One ES module, `ui/canvas.js`, renders the annotator for both hosts; `engine/api.py` is the one message protocol.
- Theme tokens (`--c-*`, `ui/theme.css`) are chosen per account: light, dark or follow the computer, eight accent palettes, animations off. The annotator must follow them; notebooks fall back to the light values.
- Every change saves immediately and is undoable (Ctrl+Z), except operations stated as permanent.
- General-purpose industrial tool: no product- or car-specific defaults in the UI.
- Rivet, the in-app helper, answers from docs/GUIDE.md: the guide must stay accurate when the UI changes.
- Decided 2026-10-17: deleting a frame is permanent (the frame and its labels leave the project; a person confirms first).
- Undecided: new label types (points/keypoints, polylines, rotated boxes) would change every export format and Teach; not part of the current label model.

## Brand Commitments

- Name: PartLabeler. The helper is Rivet, a small robot.
- Voice: plain English, short sentences, the person's own words for things ("parts", "frames", "confirm"); no jargon, never a raw technical id or error dump shown to a person.
- Per-account appearance (theme and accent) is the person's choice and is honored everywhere.

## Evidence on Hand

- docs/GUIDE.md: the full user guide (features, shortcuts, workflows).
- samples/pcb_panel.mp4: a public sample video (a panel of circuit boards, from Pexels) for demonstrations.
- No testimonials, customer names, usage numbers or benchmarks beyond those already in docs/GUIDE.md may be invented. Customer footage is private and never appears in screenshots, docs or promotion.

## Product Principles

1. The output is the annotated dataset; models assist, the person decides and confirms.
2. Fast for experts, obvious for newcomers: every shortcut has a visible control, and every control names its shortcut.
3. Nothing is lost silently: work saves as it happens, undo is always near, and anything permanent says so before it happens.
4. Local and private by default: images and labels never leave the computer.
5. One annotator everywhere: the browser app and notebooks share the same screen and the same behavior.

## Accessibility & Inclusion

Keyboard-first operation with visible focus; every control labeled for assistive technology; reduced motion respected (the computer's setting and the account's "animations off"); light and dark themes both first-class. (Inferred from the existing implementation; no stricter standard has been set.)
