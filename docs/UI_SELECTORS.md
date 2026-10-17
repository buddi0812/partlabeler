# Annotator selectors: v0.2.8 → v0.2.9

For scripts that drive the annotator (`ui/canvas.js`) by CSS selector: QA checks, demo recorders. v0.2.9 redesigned
the layout (tool rail on the left, tabbed inspector on the right, transport and status ribbon at the bottom). Almost
every `data-*` hook kept its name; what changed is **where** elements live and **when** they are visible.

## The selectors most scripts use

| Selector | v0.2.9 | What to change |
|---|---|---|
| `button[data-tool="box"]` | Same selector, now in the left rail (`.pl-rail`; was `.pl-side .pl-toolsec`). | Nothing for clicking it. The Box tool now has modes (Drag, Two clicks, Four edge points): every page load starts in Drag, clicking stays in the current mode, and pressing **B again** steps to the next one. To be sure of drag-to-draw after other steps, also click `button[data-boxmode="drag"]`. Its label is "Box" (was "Draw box"): text selectors break, `data-tool` does not. |
| `[data-a="next"]` | Same, in the bottom transport (`.pl-dock .pl-transport`; was the top bar `.pl-bar-tools`). | Nothing. |
| `[data-a="nextTodo"]` | Same, bottom transport. | Nothing for the `data-a` selector. Its "Next to check" text is hidden when the annotator is under 1100 px wide (the `aria-label` stays). |
| `[data-a="review"]` | Same, bottom right of the transport (`button.primary.pl-confirm`). | Its text sits in an inner `<span>` next to an icon; `textContent` still reads "Confirm frame" / "Unconfirm frame" (trim it). Hidden in image-classification projects, as before. |
| `[data-b="suggest"]`, `[data-b="accept"]` | Same, but inside the inspector's **Find** tab (`[data-panel="find"]`). | That panel is `hidden` unless the Find tab is open: click `button[data-tab="find"]` first (or press S / Y, unchanged). |
| `button.pl-row[data-cls]` | Same, inside the **Classes** tab (`[data-panel="classes"]`). | Click `button[data-tab="classes"]` first if another tab may be open (or press 1–9, 0, unchanged). Each row is now wrapped in `div.pl-rowwrap` next to a delete button `button.pl-x[data-delcls]`: never click the wrapper or a `[data-delcls]`. The row's text now ends with this frame's count (e.g. "1 chip 2"): match the name with `.pl-name` instead of the row's whole text. |

The inspector remembers its last tab per browser (`localStorage` key `pl-tab`, default `classes`), so a script that
does not click a tab sees whatever the last run left open. Click the tab you need every time.

## Moved or renamed

| v0.2.8 | v0.2.9 |
|---|---|
| `.pl-bar-tools` (top action bar) | `.pl-dock .pl-transport` (bottom) |
| `.pl-main` | `.pl-body` (grid: `.pl-rail`, `.pl-work`, `.pl-side`) |
| `.pl-side .pl-toolsec` (Tool section) | `.pl-rail` (`button.pl-tool[data-tool=…]`) |
| `.pl-bottom` (timeline) | `.pl-dock` (`canvas.pl-strip` + `.pl-foot`) |
| `.pl-boxes` (parts list, always shown) | same class, in the **Parts** tab (`[data-panel="parts"]`) |
| `[data-a="exportJump"]` | removed: open `button[data-tab="export"]` |
| `.pl-counts` in the action bar | in the footer, `.pl-foot .pl-counts` |
| `[data-a="stop"]`: always shown, disabled when idle | hidden when idle (`hidden` attribute), shown while a job runs, now with the text "Stop" |

## Labels that collapse on narrower screens

The `data-*` hooks never change with width; only visible text does (the `aria-label`s stay):

- Under 1360 px: the rail's key caps hide; the Back / Ahead / After this frame texts hide (icons stay).
- Under 1100 px: the rail shows icons only; the Next to check and Undo texts and the "Track" label hide.
- Under 980 px: the inspector moves under the frame.
- Under 900 px (web only): the annotator shows a "needs a bigger screen" notice instead of the canvas, as before.

A recorder that wants every label on screen should run at 1440 × 900 or wider.

## New hooks

| Hook | What it is |
|---|---|
| `button[data-tab="classes"\|"parts"\|"find"\|"export"\|"settings"]` | Inspector tabs; panels are `[data-panel="…"]`. |
| `button[data-tool=…]` | Tools: `click`, `select`, `box`, `polygon`, `ellipse`, `brush`, `erase`, `smartBrush`, `smartErase`, `pan`. `ellipse`, `erase` and the smart tools are segmentation projects only (class `pl-seg`, hidden otherwise). |
| `button[data-boxmode="drag"\|"two"\|"four"]` | Box modes in the options bar (`.pl-opts`). |
| `button[data-polymode="new"\|"add"\|"cut"]` | Polygon / ellipse modes, segmentation projects only. |
| `[data-a="deleteFrame"]` | Delete frame (opens the confirm dialog). |
| `button.pl-x[data-delcls]` | Delete a class (opens the confirm dialog). |
| `[data-b="resetTask"]`, `[data-b="resetProject"]` | Settings › Start over (open the confirm dialog). |
| `dialog.pl-del-dlg` | The one confirm dialog for all of the above: `.pl-del-keep` (autofocused, cancels), `.pl-del-ok` (does it). |
| `[data-a="zoomFit"]` | Now two buttons (rail and zoom widget): use `.pl-zoom [data-a="zoomFit"]` for one. |

## Keyboard

Every v0.2.8 shortcut still works. New: V select, O polygon, L ellipse, B again cycles the box mode, P brush also in
box projects, Enter / double-click closes a polygon, Backspace takes back a corner, Shift+Delete deletes the frame
(asks first).
