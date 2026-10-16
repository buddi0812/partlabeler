---
version: 1
slug: "ui-canvas-js"
primary_target: "ui/canvas.js"
related_targets: []
---

# Annotator (ui/canvas.js)

Scope: the labeling screen of one job, in the web app and in notebooks. Mode: Operate. Audience: dedicated annotators, occasional operators and learners, equally (PRODUCT.md). Task: pick a tool and a class, label parts on a frame, track, check flagged frames, confirm or delete frames, export. Constraints: one ES module for both hosts; account theme tokens (light/dark, accent) honored; every existing shortcut keeps working; annotating needs 900 px or more (notebooks never gated); delete frame is permanent and confirmed first.

## Direction contract

THESIS: A fixed map an annotator learns once: tools on a labeled left rail, the frame in the middle, an inspector on the right, the job's frames as one status ribbon below. It refuses the incumbent's mixed button row above the frame and its single long scrolling side column.

OWN-WORLD: PartLabeler's account theme (`--c-*`: light or dark, chosen accent). Ink-dark selection on the rail, white panels on a cool grey ground, near-black frame stage, status colours with jobs (confirmed green, tracked blue, flagged amber, delete red). One authored 1.75px stroke icon set. Segoe UI Variable, tabular numerals for every count.

STORY: The person sees every tool's name, key and how-to in place, labels, tracks, jumps to flagged frames, confirms; experts never leave the keyboard, learners never need the manual.

FIRST VIEWPORT: Header: back, project, type, job menu, frame N of M, To check chip, saved state, bell, Rivet, help, full screen. Left rail 212px: AI assisted, Manual, View groups, rows of icon, name, key (narrower screens step down: names without keys to 1100px, then icons). Centre: contextual options bar (tool, instructions, mode switch, brush size, current class) over the dark stage. Right inspector 320px, tabs Classes, Parts, Find, Export, Settings. Bottom: transport with Delete frame and the primary Confirm frame at right; the status ribbon with job bracket, raised flag ticks and a numbered playhead (the signature move: the whole job readable and scrubbable at a glance).

FORM: Rail and inspector: index 1 of the 7-entry resonance-ordered list, dealt second in the roll 3, 1, 5 (index 3, the editor dock, led); seed key 71a77254 (reproduce: concept-seed --scope surface --mode operate --from 71a77254 --candidate-count 7); locked in by the person on the decision page.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved

- New label types (points/keypoints, polylines, rotated boxes) change every export and Teach: separate decision.
