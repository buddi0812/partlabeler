---
format: 1920x1080
duration: 56s
message: "Turn raw factory video into a training dataset in minutes: free, local, and it runs in Colab."
arc: Hook → Value claim → Demo cycle (click, find similar, track) → Export + Teach & Transfer → Fair comparison → Install CTA
audience: ML and vision engineers, and their teammates, who label industrial parts for detector training
music: none
mode: collaborative
style_preset: blue-professional
---

# PartLabeler promo — storyboard

Silent video (no narration, no music): every frame's copy is on-screen text, listed under `onscreen:` as
cue-separated phrases; reveals pace to those cues. All product visuals are real recordings of the app on
public Pexels footage (see `capture/extracted/asset-descriptions.md`). Competitor facts: as of Sep 2026,
sourced (README lists the sources).

## Decisions

- **Format:** 1920×1080, ~56 s, no voiceover, no music, no captions (so no caption keep-out band).
- **Timing:** 01 0–5 · 02 5–10 · 03 10–17 · 04 17–25 · 05 25–32 · 06 32–40 · 07 40–49 · 08 49–56 = 56 s.
- **Spine:** the teal selection bracket. It frames the hook's last word, becomes the UI's box in 03–05,
  the teal checks in the comparison, and the terminal pill in the CTA. One accent, always meaning "labeled".
- **Brand (captured from the running app):** canvas #F3F5F7, panels #FFFFFF, ink #15202B, muted #5B6874,
  line #D7DDE3, accent teal #0A7C78 (soft #E1F0EF), ok #2F9E5B; type: Space Grotesk (display, numbers,
  chrome) + Inter (body) + mono for commands; card radius 14px, pill 100px, no shadows (frame.md).
- **Bans:** no fake product UI (every UI shot is a real recording); no competitor logos (names in text only);
  no glow or neon; no claims beyond the sourced table; no slideshow (each feature frame keeps the same
  annotator surface and hands off leftward); no screensaver motion (every move reveals a cue).
- **Held frame:** 04's "67 found in 0.7 s" holds still for ~1.5 s after the stat lands.
- **Seams:** leftward throughout; zoom-through only at the two section turns (02, 07).
- **Truthfulness:** UI recordings, counts and timings are real (this machine, RTX 3060); mAP50 0.94 is the
  measured held-out score on a real factory dataset (spikes/REPORT.md); comparison cells from vendor docs.

## Locked (sketch sheet v1 confirmed 2026-09-25)

Layout, hierarchy and copy of all 8 frames as drawn in `storyboard.html` v1. Timing change after lock:
04 grows to 8 s (the recording's fill lands at 4.5 s and the stat needs its hold), 07 shrinks to 9 s. Total stays 56 s.

## Video direction

- **Palette (frame.md roles):** canvas #F3F5F7 is every frame's ground except 01 (dark b-roll); panels
  #FFFFFF; ink #15202B for headlines; muted #5B6874 for body; teal #0A7C78 is the ONLY accent and always
  means "labeled" (bracket, checks, stats, pills, logo mark); teal-soft #E1F0EF for the PartLabeler column
  and tinted cards; amber #E09A2C appears once (05's to-check flag). No gradients, no glow, no shadows
  (hairline 1.5px borders at 20% teal, 14px cards, 100px pills).
- **Type:** Inter (pre-bundled: 400 · 700 · 900 only) for display 700/900 and body 400; JetBrains Mono for commands. Minimum
  on-screen size ~28px at 1920 (the README GIF shrinks everything); headlines 72-100px.
- **Motion grammar:** one camera, smooth long-tail settles (`power3` out), no overshoot, no bounce. Every
  reveal lands on its on-screen cue (silent film: the copy IS the voiceover), spread across each frame;
  nothing but the first cue enters at t=0. UI recordings play at 1x inside a floating white app window
  (14px radius, hairline border) that never tilts in 3D; videos are hoisted with fixed geometry, so every camera move
  on footage is baked into the clip (03 zoom, 04 pull-back, 05 spotlight) and frames never animate a video.
- **Rhythm:** 01 fast (two slams); 02 medium; 03-05 steady demo pace on one surface, leftward pushes;
  **held beats:** 04 after "67" lands (~1.5 s still), 07 after the last row (~1.5 s), 08 final lockup holds
  to the end (~2 s). Everything else moves on a cue.
- **Seams:** leftward throughout; the teal bracket is the thread (01 -> 02 underline, the UI's own boxes in
  03-05, the teal column in 07, the logo mark in 08).
- **Negative list:** no fake UI (all UI is the real recordings/stills), no competitor logos, no glow/bloom,
  no purple-blue gradients, no floating bokeh, no lazy breathing, no slow back-half pans or pushes, no
  `repeat`/`yoyo`, no randomness. Both failure modes banned: slideshow (front-load then freeze) and
  screensaver (things drifting for no reason).

## Frame 1 — Weeks of boxes

- scene: blurred factory b-roll; two blunt lines about hand-labeling land one after another
- voiceover: ""
- onscreen: "Your detector needs thousands of labeled frames." | "Drawing every box by hand takes weeks."
- duration: 5s
- transition_in: cut
- status: animated
- src: compositions/frames/01-hook.html
- type: hook
- persuasion: Pain validation
- beat: frustration
- blueprint: kinetic-type-beats
- asset_candidates: assets/hook_broll.jpg — still from the public conveyor clip (carton bundle over rollers), already darkened + blurred, full-bleed background

- focal: assets/hook_broll.jpg
- roles: hook_broll.jpg = background (full-bleed <img>, already darkened + blurred; add a 25% ink overlay for legibility; no pan, no push)
- sfx: none

Adapt (kinetic-type-beats, Hook escalation): keep the two-beat escalation landing on a payoff element; the payoff is the teal bracket snapping around "weeks.", not a logo.
Scene 1 (0.0-2.2s): full-bleed still b-roll (hook_broll.jpg), darkened + blurred, present from t=0; line 1 "Your detector needs thousands of labeled frames." arrives by per-word staggered reveal (`dynamic-content-sequencing`), left-aligned on the rule-of-thirds line, white display type ~96px. Nothing else on screen.
Scene 2 (2.2-3.6s): line 2 "Drawing every box by hand takes weeks." reveals the same way beneath it in teal-soft body-display ~64px.
Scene 3 (3.6-5.0s): the teal bracket draws around "weeks." (`svg-path-draw`, 4px teal stroke, square corners); then everything holds still until the zoom-through out.

narrativeRole: name the pain every labeling engineer knows, in their words, before any product appears.
keyMessage: hand-labeling video is the bottleneck.

## Frame 2 — PartLabeler

- scene: the type clears to the wordmark; the value claim lands under it; the real start screen rises behind
- voiceover: ""
- onscreen: "PartLabeler" | "Raw factory video → training dataset." | "In minutes. Free. Local."
- duration: 5s
- transition_in: zoom-through
- status: animated
- src: compositions/frames/02-intro.html
- type: product_intro
- persuasion: Friction reduction
- beat: relief + curiosity
- blueprint: kinetic-type-beats
- asset_candidates: assets/home.png — the real start screen (projects, new project, Teach & Transfer)

- focal: assets/home.png
- roles: home.png = supporting (floating white app window, right 55% of frame)
- sfx: none

Adapt (kinetic-type-beats, Product_Intro namedrop): keep the resolve-on-the-brand-name signature; the name lands first, the promise builds under it, the real start screen arrives last as proof.
Scene 1 (0.0-1.2s): canvas #F3F5F7; eyebrow "OPEN SOURCE · FOR INDUSTRIAL PARTS" then the wordmark "PartLabeler" (h1 ~104px, ink) arrive left, upper third; a 4px teal underline draws under the wordmark (the bracket from 01 collapsing into it). Asymmetric 45/55 layout.
Scene 2 (1.2-2.8s): "Raw factory video → training dataset." reveals word by word under the wordmark (~58px, ink, the arrow in teal).
Scene 3 (2.8-3.8s): three pills "In minutes" · "Free" · "Local" pop in left to right (`spring-pop-entrance`, smooth, no overshoot).
Scene 4 (3.8-5.0s): the real start screen slides in from the right edge inside a white app window (right 55%, slightly overlapping the canvas edge) and settles; hold.

narrativeRole: state the promise (the message) by beat 2; everything after is evidence.
keyMessage: this tool turns video into datasets fast.

## Frame 3 — One click, one box

- scene: the real annotator; the cursor glides to a bolt, one click → teal outline + box; a speed chip pops
- voiceover: ""
- onscreen: "Click a part." | "SAM 3 outlines it and boxes it." | "0.08 s per click"
- duration: 7s
- transition_in: crossfade
- status: animated
- src: compositions/frames/03-click.html
- type: feature_showcase
- persuasion: Show-don't-tell proof
- beat: ease
- blueprint: cursor-ui-demo
- asset_candidates: assets/clip03_click.mp4 — [video] real UI recording, 7.0 s, camera zoom onto the bolt already baked in (zoom in 2.3-3.4 s, back out 4.2-5.0 s); assets/click_mask.png — still of the masked bolt (fallback only)

- focal: assets/clip03_click.mp4
- roles: clip03_click.mp4 = cutout: approved video, data-start 0, data-duration 7, geometry x 80 y 140 width 1200 height 675 (fit cover); click_mask.png = unused

Hoisted-video contract: an approved video is declared in the frame as <video data-frame-video="approved" ...> with numeric data-frame-video-x/y/width/height; the assembler moves it to the host root, where it draws ABOVE all frame content with that fixed geometry. So: never place text or shapes where the video box is, never animate the video element (all camera moves are already baked into the clip), and draw the white app-window card (14px radius, 1.5px hairline) 8px larger than the box, behind it.
- sfx: none

Adapt (cursor-ui-demo, Key_Feature): the recording supplies the cursor and the UI's live response; keep the signature of the camera chasing the interaction with ONE zoom-to-target onto the bolt as it gets clicked. No reconstructed UI.
Scene 1 (0.0-2.3s): the app window (left 66%, 16:9, white hairline frame) plays the recording; the cursor glides toward the bottom-center bolt. Right column: eyebrow "STEP 1" and "Click a part." (h2 ~60px) reveal as the cursor starts moving.
Scene 2 (2.3-3.4s): as the click lands (media ~2.7 s, i.e. frame ~2.3 s) the clip itself zooms onto the bolt (baked in; nothing to animate); the teal mask and box appear in the recording. "SAM 3 outlines it and boxes it." reveals in the right column.
Scene 3 (3.4-5.0s): the "0.08 s" count-up card (`count-up` block, teal numeral ~90px, label "per click, after the model loads") pops in under the copy; the clip eases back out as the recording finds the other two bolts (~4.6 s).
Scene 4 (5.0-7.0s): the recording accepts the suggestions (all three solid); caption under the window "Real recording · public footage (Pexels)"; hold.

narrativeRole: first proof of speed: the basic act (making a box) costs one click.
keyMessage: boxes take a click, not a drag-and-adjust.

## Frame 4 — One example, 67 found

- scene: circuit-board panel; one chip pair clicked, F pressed, the panel fills with boxes; the clip slides aside to a hero stat
- voiceover: ""
- onscreen: "Box one." | "Find the rest." | "67 found in 0.7 s"
- duration: 8s
- transition_in: push-slide LEFT
- status: animated
- src: compositions/frames/04-find-similar.html
- type: feature_showcase
- persuasion: Statistical proof
- beat: awe
- blueprint: video-text-pivot
- asset_candidates: assets/clip04_pcb.mp4 — [video] real UI recording, 8.0 s: opens tight on the clicked chip pair, the click lands ~2.3 s, the panel fills with 67 boxes ~3.9 s while the camera pulls back to the full panel (baked in), last frame held from 6.4 s; assets/pcb_found.png — still (fallback only)

- focal: assets/clip04_pcb.mp4
- roles: clip04_pcb.mp4 = cutout: approved video, data-start 0, data-duration 8, geometry x 80 y 150 width 1120 height 630 (fit cover); pcb_found.png = unused

Hoisted-video contract: an approved video is declared in the frame as <video data-frame-video="approved" ...> with numeric data-frame-video-x/y/width/height; the assembler moves it to the host root, where it draws ABOVE all frame content with that fixed geometry. So: never place text or shapes where the video box is, never animate the video element (all camera moves are already baked into the clip), and draw the white app-window card (14px radius, 1.5px hairline) 8px larger than the box, behind it.
- sfx: none

Adapt (video-text-pivot): keep the one weight transfer from footage to hero stat; the video box is fixed (hoisted), so the transfer is the baked camera PULL-BACK inside the clip landing at the same moment the stat pops in beside it.
Scene 1 (0.0-2.4s): the app window (left, fixed) plays tight on the panel; the cursor glides to a chip pair and clicks (~2.3 s). Headline "Box one." reveals under the window, left.
Scene 2 (2.4-3.9s): the teal box appears on the clicked chip; "Find the rest." reveals after "Box one." in teal.
Scene 3 (3.9-5.3s): the panel fills with 67 boxes in the recording (media ~4.5 s); as they land the clip pulls back (baked) while "67" pops in on the right via the `count-up` block (0 -> 67, teal numeral ~220px), label "found from one example" below it, then "in 0.7 s · press F".
Scene 4 (5.3-8.0s): held beat: everything still; the recording (or pcb_found.png) holds the filled panel.

narrativeRole: the "wow": repeated parts are labeled from a single example.
keyMessage: one example labels the whole frame.

## Frame 5 — It follows the part

- scene: the carton bundle box follows the bundle across the conveyor; the timeline fills blue; amber marks flash as "to check"
- voiceover: ""
- onscreen: "Track forward or back." | "Frames that drift are flagged for you."
- duration: 7s
- transition_in: push-slide LEFT
- status: animated
- src: compositions/frames/05-track.html
- type: feature_showcase
- persuasion: Feature-to-benefit translation
- beat: control
- blueprint: device-surface-showcase
- asset_candidates: assets/clip05_track.mp4 — [video] real UI recording, 7.0 s: the carton_bundle box tracked frame by frame, and from 1.6 to 4.6 s everything except the timeline strip is dimmed (spotlight baked in); assets/tracked.png — end still (fallback only)

- focal: assets/clip05_track.mp4
- roles: clip05_track.mp4 = cutout: approved video, data-start 0, data-duration 7, geometry x 80 y 140 width 1200 height 675 (fit cover); tracked.png = unused

Hoisted-video contract: an approved video is declared in the frame as <video data-frame-video="approved" ...> with numeric data-frame-video-x/y/width/height; the assembler moves it to the host root, where it draws ABOVE all frame content with that fixed geometry. So: never place text or shapes where the video box is, never animate the video element (all camera moves are already baked into the clip), and draw the white app-window card (14px radius, 1.5px hairline) 8px larger than the box, behind it.
- sfx: none

Adapt (device-surface-showcase, static-tour): the held surface is the app window (fixed, hoisted); its screens advance by the recording's own frame stepping; the timeline spotlight is baked into the clip. Keep the element-level screen cycling (from the recording) + staggered side-headline reveal.
Scene 1 (0.0-1.6s): app window (left 66%) plays; the carton bundle enters with its purple box. Right column: eyebrow "STEP 3", "Track forward or back." (h2) reveals.
Scene 2 (1.6-4.6s): the box follows the bundle frame after frame (the recording); the timeline strip inside the recording is spotlighted (baked into the clip: the rest dims) as it shows the tracked range in blue with amber marks; the right column shows nothing new in this window, it is the recording's beat.
Scene 3 (4.6-7.0s): spotlight releases; "Frames that drift are flagged for you." reveals with a small amber square before "to check"; caption under the window "Real recording · forward + back, 40 frames in 23 s"; hold.

narrativeRole: scale from one frame to the whole video, with a safety net (flags) so speed doesn't cost trust.
keyMessage: label once, it carries through the video, and you only check what it flags.

## Frame 6 — Your format, your next videos

- scene: five format pills assemble; then a three-step strip: one labeled video → Teach → similar videos labeled, with the held-out score
- voiceover: ""
- onscreen: "Export YOLO · COCO · CVAT · Pascal VOC · Label Studio" | "Teach & Transfer:" | "learn one labeled video, label the next ones in the same format" | "held-out mAP50 0.94"
- duration: 8s
- transition_in: crossfade
- status: animated
- src: compositions/frames/06-export-transfer.html
- type: benefit_highlight
- persuasion: Value stacking
- beat: power
- blueprint: grid-card-assemble
- asset_candidates:

- focal: none (typography and cards)
- roles: none
- sfx: none

Adapt (grid-card-assemble, Key_Feature grid -> field-to-payoff): the pills assemble as the breadth field, then the three-card strip is the payoff; keep the staggered assemble-into-slot signature.
Scene 1 (0.0-2.2s): canvas; eyebrow "YOUR FORMAT" then five pills "YOLO" · "COCO" · "CVAT" · "Pascal VOC" · "Label Studio" assemble left to right into one row (`center-outward-expansion`, direct-into-slot form, smooth stagger), upper third.
Scene 2 (2.2-4.6s): eyebrow "TEACH & TRANSFER · YOUR NEXT VIDEOS"; three tinted cards reveal one by one left to right with teal arrows drawing between them (`svg-path-draw`): "One labeled video / yours, any YOLO dataset" -> "Teach / trains on your GPU, proves on held-out frames" -> "Similar videos, labeled / same classes, same format" (last card teal-soft). Triptych, full-width strip.
Scene 3 (4.6-6.4s): "0.94" counts up (`count-up` block, teal ~110px) with "held-out mAP50 on a real factory dataset" beside it, lower third.
Scene 4 (6.4-8.0s): hold.

narrativeRole: the dataset leaves in whatever format the training stack wants, and the work compounds: labeled videos teach the next ones.
keyMessage: output fits any pipeline and each labeled video speeds up the next.

## Frame 7 — Fair comparison

- scene: a clean comparison table builds row by row; PartLabeler's column fills with teal checks; a footnote keeps it honest
- voiceover: ""
- onscreen: "How it compares" | rows (below) | "As of Sep 2026 · sources in the README · CVAT, Label Studio and Roboflow lead on team workflows and annotation types"
- duration: 9s
- transition_in: zoom-through
- status: animated
- src: compositions/frames/07-compare.html
- type: social_proof
- persuasion: Negative contrast
- beat: confidence
- blueprint: grid-card-assemble
- asset_candidates:

Columns: PartLabeler · CVAT (self-hosted) · Label Studio (open source) · X-AnyLabeling · Roboflow Annotate.
Rows (cell text exactly as sourced):

| Row | PartLabeler | CVAT | Label Studio | X-AnyLabeling | Roboflow |
|---|---|---|---|---|---|
| SAM 3 click + video tracking, built in | ✓ | add-on server | ML backend | server add-on | ✓ (cloud) |
| Runs inside a notebook (Colab-friendly) | ✓ | — | — | — | — |
| Native Windows install, no Docker | ✓ | Docker / WSL2 | ✓ | ✓ | cloud |
| Learns your labeled video, labels the next | ✓ local | bring your model | write a backend | ✓ local | ✓ cloud credits |
| Exports YOLO · COCO · CVAT · VOC · LS | 5 / 5 | 4 / 5 | 4 / 5 | 3 / 5 | 3 / 5 |
| Free, data stays on your machine | ✓ | ✓ | ✓ | ✓ | paid · cloud |

- focal: none (table)
- roles: none
- sfx: none

Adapt (grid-card-assemble, Benefits vertical-list BUILD): rows are the list items, each entering with a mask-wipe of its text and a check draw-in in the PartLabeler column (`success-check` block for ✓ cells, smooth, no ring-pop overshoot). Keep ~1 row per second accumulation.
Scene 1 (0.0-1.2s): canvas; "How it compares" (h2) and the header row (PartLabeler column pre-tinted teal-soft; "CVAT (self-hosted)" · "Label Studio (open source)" · "X-AnyLabeling" · "Roboflow" in muted) arrive together.
Scene 2 (1.2-7.2s): six rows build top to bottom, one per ~1.0 s; each row's label wipes in from the left, then its cells fade in left to right with PartLabeler's cell last and emphasized (teal check draws / "5 / 5" in bold teal). Exact cell text as in the table above, no additions. Table spans the frame width (4-96%), rows ~72px tall, text >= 28px.
Scene 3 (7.2-9.0s): the footnote "As of Sep 2026 · sources in the README · CVAT, Label Studio and Roboflow lead on team workflows and annotation types." fades in bottom-left (~26px muted); held beat.

narrativeRole: answer "why not the tool we already use?" with checkable facts, not adjectives.
keyMessage: PartLabeler bundles what the others need add-ons, Docker or the cloud for.

## Frame 8 — Install in one click

- scene: headline demotes; a terminal pill types the install command; Colab badge and the wordmark lock up; credit line
- voiceover: ""
- onscreen: "One-click install on Windows · bash install.sh on Linux · runs in Colab" | "install_windows.bat" | "PartLabeler" | "Footage: Pexels (Video Kickstarter, Distill)"
- duration: 7s
- transition_in: crossfade
- status: animated
- src: compositions/frames/08-cta.html
- type: cta
- persuasion: Risk reversal
- beat: urgency-to-act
- blueprint: prompt-type-submit-generate
- asset_candidates:

- focal: none (terminal block + wordmark)
- roles: none
- sfx: none

Adapt (prompt-type-submit-generate, CTA install-command end card): keep the terminal pill springing in and the command typing with a blinking cursor held to the end; the end card resolves to the logo lockup.
Scene 1 (0.0-1.2s): canvas; "Try it on your own parts." (h1 ~84px, ink) reveals left, upper third.
Scene 2 (1.2-3.2s): a dark ink terminal pill springs in under it (`code-terminal-run` block, restyled: ink #15202B surface, mono ~40px, teal caret); it types "install_windows.bat" character by character.
Scene 3 (3.2-4.6s): "Linux: bash install.sh · Colab: open PartLabeler_Colab.ipynb" reveals under the pill (~34px muted, commands in ink mono).
Scene 4 (4.6-7.0s): the teal bracket draws as a small square logo mark and the wordmark "PartLabeler" (h1 ~96px) locks up beside it, lower-left; "Footage: Pexels · Video Kickstarter, Distill" fades in bottom-left (~22px muted); the caret blink (a finite sequence of discrete steps, not a repeat) is the only motion; holds to the end (final frame: no exit).

narrativeRole: remove the last objection (setup effort) and tell the teammate exactly what to run.
keyMessage: it takes one command to try.

## Final (rendered 2026-09-25)

`renders/video.mp4`: 56.0 s, 1920×1080, 30 fps, H.264, silent, 12.8 MB. Cut points as planned: 01 0–5 · 02 5–10 ·
03 10–17 · 04 17–25 · 05 25–32 · 06 32–40 · 07 40–49 · 08 49–56 (transitions overlap the next frame by 0.4–0.5 s).
Build deltas from the sketch: hoisted videos carry baked camera moves (03 zoom, 04 pull-back, 05 spotlight) and
follow their frames' transitions via root tweens in `index.html`; frame 01 uses a still b-roll; frame 02's start
screen is cropped to the UI; slow background pushes removed from 02, 06, 07. README: an inline player from a
GitHub attachment (web encode: x264 CRF 20, 4.3 MB); the full-quality render stays in `renders/video.mp4`.
Re-render: `npx hyperframes render --quality high --output renders/video.mp4`.
