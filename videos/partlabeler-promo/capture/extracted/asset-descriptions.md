# Asset descriptions — PartLabeler promo

Source: the PartLabeler app itself, running locally on a demo home (`C:\PartLabelerDemo\projects`) with two
public Pexels clips. Every box, mask and count below was produced by the real engine (no mock-ups).
No private footage anywhere in this inventory.

## App recordings (real UI, Chrome screencast, 1920x1080, 30 fps)

- `assets/rec_click_find.mp4` (10.0 s) — conveyor frame: cursor glides to a bolt, one click → teal SAM 3 mask + red `bolt` box ("Outline 1 of 3" toast); F → two more bolts appear as dotted suggestions (0.89, 0.74; toast "2 more like this"); Y accepts; Enter confirms and steps on.
- `assets/rec_track.mp4` (8.4 s) — conveyor frames 25→47: a purple `carton_bundle` box follows the bundle as it enters and crosses the rollers; the timeline strip below shows the tracked range (blue) and "to check" marks (amber).
- `assets/rec_pcb.mp4` (7.0 s) — circuit-board panel: one click on a chip pair, F → 67 more chip pairs boxed across the in-focus rows; the side list fills with suggestions (0.80 … 0.70); toast "67 more like this".

## Baked clips for the build (camera moves inside the footage; hoisted videos have fixed geometry)

- `assets/clip03_click.mp4` (7.0 s) [video] — from rec_click_find 0.4-7.4 s: glide, click, zoom onto the bolt 2.3-3.4 s, back out 4.2-5.0 s, other bolts found, accepted.
- `assets/clip04_pcb.mp4` (8.0 s) [video] — from rec_pcb 0.6-7.0 s + held last frame: tight on the chip pair, click ~2.3 s, 67 boxes fill ~3.9 s as the camera pulls back to the full panel.
- `assets/clip05_track.mp4` (7.0 s) [video] — from rec_track 0.6-7.6 s: box follows the carton bundle; 1.6-4.6 s everything except the timeline strip dims.
- `assets/hook_broll.jpg` — still from the public conveyor clip at 3.0 s (bundle over rollers), darkened + blurred, 1920x1080.

## App stills (1920x1080 PNG)

- `assets/home.png` — start screen: project list (pcb_panel, conveyor_line), New project form, Teach & Transfer panel.
- `assets/click_mask.png` — bolt clicked: teal mask + box, toast "Outline 1 of 3".
- `assets/find_similar.png` — the same frame after F: three bolts (one solid, two dotted suggestions).
- `assets/accepted.png` — suggestions accepted (all solid).
- `assets/tracked.png` — end of the tracking pass, carton bundle boxed, timeline visible.
- `assets/pcb_found.png` — 68 chip-pair boxes on the panel; side list; "67 more like this".
- `screenshots/full-page.png` — the start screen as captured by `hyperframes capture` (full page).

## Public footage (Pexels license: free commercial use, editing allowed, no attribution required — credited anyway)

- `assets/pexels-2376982-pcb-panel.mp4` (11 s, 1920x1080) — "A close-up video of a circuit board components", Video Kickstarter, pexels.com/video/2376982. Macro of a panel of identical PCB units, focus pull. Hook b-roll.
- `assets/pexels-852388-conveyor.mp4` (7.4 s, 1920x1080) — "Assembly Line", Distill, pexels.com/video/852388. Rollers and bolts; a carton bundle slides across. Hook b-roll.

## Facts the story may use (measured; see spikes/REPORT.md and the competitor research in BRIEF notes)

- Click → outline: 0.08 s per click after the model is loaded (RTX 3060).
- Find similar: 67 more chip pairs from one example in 0.7 s; bolts 2 of 2.
- Tracking: 40 frames in 23 s on the demo clip, forward and back; a size/jump rule flags frames to check.
- Teach & Transfer on a real factory dataset: held-out mAP50 0.94 after training on one labeled video; labels similar videos in the source's format.
- Exports: YOLO, COCO, CVAT 1.1, Pascal VOC, Label Studio.
- Runs as a notebook widget (Colab free-tier compliant), native Windows install (no Docker), free and local.
