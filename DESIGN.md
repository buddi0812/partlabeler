---
name: PartLabeler
description: Annotator for industrial part datasets; a fixed map of tools, frame, inspector and job ribbon.
colors:
  accent: "#0a7c78"
  accent-ink: "#075e5b"
  accent-soft: "#e1f0ef"
  ground: "#eef1f4"
  panel: "#ffffff"
  panel-2: "#f7f9fa"
  ink: "#15202b"
  muted: "#56636f"
  line: "#d5dbe1"
  line-2: "#e6eaee"
  hover: "#f3f7f8"
  stage: "#0d1217"
  status-empty: "#dfe4e9"
  status-suggested: "#b8a8e6"
  status-tracked: "#6aa6f2"
  status-yours: "#27a79b"
  status-confirmed: "#2f9e5b"
  flag: "#e09a2c"
  bad: "#c2412f"
  bad-ink: "#9b2f1f"
  bad-soft: "#fde8e5"
  bad-line: "#efb9ae"
  ok-soft: "#dcf1e4"
  ok-ink: "#1f7a44"
  warn-soft: "#fbecd3"
  warn-ink: "#8a5304"
  dark-ground: "#11171b"
  dark-panel: "#182025"
  dark-ink: "#e2e8eb"
  dark-line: "#2d3940"
typography:
  title:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "15px"
    fontWeight: 600
  dialog-title:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "17px"
    fontWeight: 600
  body:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  control:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "13.5px"
    fontWeight: 400
  section-label:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "13px"
    fontWeight: 600
  small:
    fontFamily: "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  key:
    fontFamily: "ui-monospace, Cascadia Mono, Consolas, monospace"
    fontSize: "11.5px"
    lineHeight: 1.2
rounded:
  swatch: "3px"
  key: "5px"
  segment: "6px"
  control: "7px"
  row: "8px"
  card: "9px"
  popover: "10px"
  dialog: "12px"
  pill: "99px"
spacing:
  xxs: "2px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "12px"
  xl: "14px"
  section: "18px"
components:
  button:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "5px 11px"
  button-hover:
    backgroundColor: "{colors.hover}"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.panel}"
    rounded: "{rounded.control}"
    padding: "5px 11px"
  button-primary-hover:
    backgroundColor: "{colors.accent-ink}"
  button-danger:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.bad-ink}"
    rounded: "{rounded.control}"
  button-danger-hover:
    backgroundColor: "{colors.bad-soft}"
  button-solid-danger:
    backgroundColor: "{colors.bad}"
    textColor: "{colors.panel}"
    rounded: "{rounded.control}"
  rail-tool:
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.row}"
    padding: "6px 8px"
  rail-tool-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.panel}"
  list-row:
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "5px 7px"
  list-row-selected:
    backgroundColor: "{colors.accent-soft}"
  segmented-track:
    backgroundColor: "{colors.ground}"
    rounded: "{rounded.row}"
    padding: "2px"
  segmented-option-active:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.segment}"
    padding: "3px 10px"
  inspector-tab:
    textColor: "{colors.muted}"
    typography: "{typography.section-label}"
    padding: "11px 9px 10px"
  inspector-tab-selected:
    textColor: "{colors.accent-ink}"
  chip:
    backgroundColor: "{colors.accent-soft}"
    textColor: "{colors.accent-ink}"
    rounded: "{rounded.pill}"
    padding: "2px 9px"
  input:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "4px 7px"
  toast:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.panel}"
    rounded: "{rounded.row}"
    padding: "9px 12px"
  status-ribbon:
    backgroundColor: "{colors.status-empty}"
    height: "40px"
  dialog:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.dialog}"
---

# Design System: PartLabeler

Evidence: the annotator `ui/canvas.js` (CSS string, `ICON_PATHS`/`ic()`, `TOOLS`, markup, `drawStrip()`) and `ui/theme.css`. The start screen (`ui/home.js`), `ui/pages.js` and the login page were not redesigned with this system and are outside this record's evidence; do not infer their rules from it, nor this file's rules from them.

## Overview

**Creative North Star: "The Fixed Map"**

One screen an annotator learns once and never has to search again: tools on a labeled left rail, the frame on a near-black stage in the middle, an inspector on the right, and the whole job as one status ribbon along the bottom. Every control sits in the same place for every project type; only its content changes with the label type.

The material is quiet and dense: white panels on a cool grey ground separated by 1px hairlines, no decoration, one accent from the person's own account. Colour carries meaning only: ink for the chosen tool, accent for the chosen class and the primary action, and a fixed status scale that tells the state of every frame at a glance. Counts are tabular so numbers never jitter while tracking runs.

**Key Characteristics:**
- Rail 212px, inspector 320px, stage fills the rest; steps down by container width, never by device.
- Each tool shows its name, key and how-to in place; the full keyboard map lives in the help dialog.
- One authored stroke icon family; no glyph icons.
- Status ribbon with job bracket, raised flag ticks and a numbered playhead.
- Follows the account theme (light, dark, system; eight accents) live, canvases included.

## Colors

A cool-neutral workbench with one account-chosen accent and a fixed, meaning-only status scale.

### Primary
- **Account Accent** (`accent`, default teal): primary buttons (Confirm frame, Export, Finish the job), selected class/part row border, selected inspector tab underline, focus ring, progress bar, range inputs. Per account via `--c-accent` (teal, blue, indigo, violet, rose, orange, green, graphite; light and dark values in `ui/theme.css`).
- **Accent Ink** (`accent-ink`): primary hover, back link, selected tab text, type chip text.
- **Accent Wash** (`accent-soft`): selected list rows, type chip, text selection.

### Neutral
- **Cool Ground** (`ground`): the app background, segmented-control track, count pills.
- **Panel White** (`panel`) / **Panel Mist** (`panel-2`): header, rail, options bar, inspector, dock, dialogs; `panel-2` for key caps.
- **Workbench Ink** (`ink`): text, the selected rail tool, tooltips, toasts, ribbon playhead.
- **Muted Slate** (`muted`): secondary text, group headings, how-to line.
- **Hairline** (`line`, `line-2`): every border and divider; `line-2` for help-grid rows.
- **Stage Black** (`stage`): the frame stage, card and peek backgrounds. Fixed in both themes: the image is always judged against near-black.
- Dark theme (`dark-*` and the rest of `ui/theme.css`'s dark block) replaces every neutral; modules fall back to the light values when the theme file is absent (notebooks).

### Status scale (frame states, ribbon and legend)
- **st0 Empty** (`status-empty`; themed): no boxes/outlines.
- **st1 Suggested** (`status-suggested`, lavender): suggestions only.
- **st2 Tracked** (`status-tracked`, blue): tracked or imported.
- **st3 Yours** (`status-yours`, teal): has the person's own parts.
- **st4 Confirmed** (`status-confirmed`, green): confirmed frame.
- **Flag** (`flag`, amber): to check; raised ticks on the ribbon, "Looks like" badges.
- **Bad** (`bad` family): delete and error only; danger buttons, error toasts, lost-save state, error banner.
- **OK / Warn washes** (`ok-*`, `warn-*`): reviewed and to-check chips, completed job state, info banner, warning toast (`warn-ink` as fill).

### Named Rules
**The Meaning-Only Colour Rule.** Colour on this screen always states something: chosen (ink/accent), frame state (st0-st4, flag) or destruction (bad). Nothing is coloured for decoration.

**The Fixed Status Rule.** st1-st4 and flag keep the same hues in light and dark so a frame's state reads the same for every account; only st0 follows the theme.

## Typography

**Body Font:** Segoe UI Variable Text (with Segoe UI, system-ui, -apple-system, sans-serif)
**Label/Mono Font:** ui-monospace, Cascadia Mono, Consolas (key caps only)

**Character:** One Windows-native UI family at small, tight sizes; weight 600 is the only emphasis.

### Hierarchy
- **Dialog title** (600, 16-17px): help and delete dialogs.
- **Title** (600, 15px): project name in the header, grid title.
- **Body** (400, 14px, 1.45): base text.
- **Control** (400, 13.5px): buttons, inputs, rail tools, list rows.
- **Section label** (600, 13px): inspector section headings, tabs; how-to line at 400/13px in muted.
- **Small** (12-12.5px): rail group headings (600, muted), chips, footer, notes, tooltip; 11.5px for count pills and job state.
- **Key** (mono 11.5px, 2px bottom border): keyboard keys in the rail, hint and help dialog.

### Named Rules
**The Tabular Count Rule.** Every number that changes (frame N of M, counts, zoom %, brush size, track frames, tab counts, ribbon tooltip) uses `font-variant-numeric: tabular-nums`.

## Layout

Grid of four fixed rows: header, banner, body, dock. A hidden banner never moves the frame into another row. Body is three columns: rail (`--rail-w` 212px), work (options bar 44px over the stage), inspector (`--insp-w` 320px). The dock holds the transport row over the 40px status ribbon and a footer with counts and the legend. Classify projects (and Sort parts) swap rail and stage for the card grid beside the inspector.

Steps are container queries on the annotator itself (`container-type: inline-size`), not media queries:
- **<=1360px:** rail 184px, inspector 280px; rail keys move into tooltips (names stay); secondary transport labels collapse to icons.
- **<=1100px:** rail 56px, icons only, group headings become 28px rules; remaining transport labels and the Track label hide. Delete and Confirm stay on the row at the right.
- **<=980px:** inspector moves under the frame (min 220px, 38%); options bar wraps and the how-to takes its own line.
- **<900px viewport (web host only):** a plain "needs a bigger screen" notice replaces the annotator; notebooks are never gated.

Rhythm: 14px horizontal padding on every bar; gaps 2/4/6/8/12/14px; inspector sections 18px apart.

## Elevation & Depth

Flat. Panels are separated by hairlines and tone, not shadow. Shadows appear only on things that float above the map.

### Shadow Vocabulary
- **Pressed segment** (`0 1px 2px rgba(21,32,43,.14)`): the selected option in a segmented control.
- **Popover** (`0 14px 34px rgba(21,32,43,.18)`): job menu and After this frame menu.
- **Floating dark** (`0 6px 16px rgba(0,0,0,.2)` tooltip, `0 6px 18px rgba(0,0,0,.22)` toast, `0 10px 30px rgba(0,0,0,.3)` first-run hint).
- **Backdrop** (`rgba(21,32,43,.45)`; peek `rgba(13,18,23,.78)`).

### Named Rules
**The Float-Only Shadow Rule.** A shadow means "above the screen": popovers, toasts, tooltips, the hint. Docked panels never get one.

## Shapes

Gently rounded, tighter for smaller things: 3px swatches, 5px keys, 6-7px controls and rows, 8px rail tools and toasts, 9-10px cards and popovers, 12px dialogs, full pills for chips, filters and the job menu summary. Borders are 1px hairlines; selected grid cards use a 2px accent border plus a 2px ring; suggested cards a dashed border.

Icons: one authored family in `ICON_PATHS`, drawn on a 24px grid with 1.75px round-capped, round-joined strokes, no fill, `currentColor`, rendered at 18px (14-15px inline, 22px in the delete dialog). New icons are added to that set, drawn the same way.

## Components

### Buttons
- **Shape:** gently rounded (7px), 1px hairline, 13.5px.
- **Default:** panel fill, ink text; hover keeps the fill near-white (`hover`) and turns the border accent.
- **Primary:** accent fill, on-accent text, 600; hover to accent ink. One per region (Confirm frame, Export, Finish the job, Clear).
- **Danger:** outline in bad-line with bad-ink text, bad wash on hover (Delete frame, Stop). **Solid danger** (bad fill, white) only inside the confirm dialog.
- **Icon / Ghost:** 5px 7px padding; ghost is borderless until hover. Every icon-only button has `aria-label` and a `title` with its key.
- **Focus:** 2px accent outline, 2px offset, everywhere.

### Rail tool row
Grid of icon (20px), name, key cap; 6px 8px, 8px radius, transparent at rest, `hover` wash. The active tool inverts to ink fill with panel text (the key cap follows). Grouped under muted headings: AI assisted, Manual, View. Segment-only tools hide in box projects.

### Options bar and segmented modes
44px bar above the stage: tool name (600), segmented mode switch, brush size, the one-line how-to (muted, ellipsis so the frame never jumps), current class at the right. Segmented control: ground-tinted track with 2px padding; options are borderless 12.5px muted text; the selected one sits on panel with ink 600 text and the pressed shadow.

### Inspector tabs
Classes, Parts (with count pill), Find, Export, Settings. 13px 600 muted text; selected tab is accent-ink with a 2px inset accent underline. Parts and Find hide on the grid.

### Class and part rows
Grid of 12px colour swatch, key/id (muted, tabular), name (ellipsis), count. 5px 7px, 7px radius; selected = accent wash with accent border. Part rows carry a muted borderless delete-x that turns bad on hover.

### Transport
Groups left to right: step/play/speed/next-to-check; Track (back, frame count, ahead, stop, After this frame menu); Undo; then Delete frame and primary Confirm frame pinned right (`margin-left: auto`) even when the row wraps.

### Status ribbon (signature)
A 40px canvas, 6px radius: one cell per frame filled from st0-st4; flag ticks rise above the band in amber; the job bracket prints the first and last frame number in muted; the playhead is a 2px ink line under a numbered ink pill. Hover shows a tooltip "Frame N · state · to check"; click/drag scrubs. Redraws on theme and accent change.

### Toasts
Bottom-right stack, max 400px, ink fill with panel text, 8px radius; error = bad fill, warning = warn-ink fill, both white text. Action buttons are transparent with a currentColor border.

### Confirm dialog (destructive)
Native `<dialog>`, 440px, 12px radius, 20px 22px padding; title 17px with the alert icon in bad; muted body states exactly what goes. Buttons right-aligned: **Keep** (autofocused, default) then **Delete for good** (solid danger).

### Chips
Pills, 12px 600: project type (accent wash), reviewed (ok wash), to check (warn wash). The job menu summary is a bordered pill whose border turns accent on hover/open.

## Do's and Don'ts

### Do:
- **Do** read every colour from the module tokens (`--accent`, `--ink`, `--st0`...) mapped to `--c-*` with the light value as fallback, so notebooks and accounts both work.
- **Do** redraw any canvas when `data-theme`/`data-accent` or the system colour scheme changes.
- **Do** put a tool's name, key and how-to on screen where it is used, and list every key in the help dialog.
- **Do** make anything permanent a modal confirm with Keep autofocused; everything else saves at once and is undoable.
- **Do** step the layout by container width (1360 / 1100 / 980) and keep Delete and Confirm on the transport row at every step.
- **Do** use tabular numerals for every changing number.

### Don't:
- **Don't** use unicode glyphs as icons; draw them in the 24px, 1.75px round-stroke set.
- **Don't** add shadows to docked panels or bars.
- **Don't** use the status hues (st1-st4, flag) or bad for anything but frame state and destruction.
- **Don't** gate the annotator by device or touch; width only, and never in a notebook.
- **Don't** change the stage from near-black in either theme.
