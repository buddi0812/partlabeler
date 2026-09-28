# PartLabeler — notes for Claude

- Design and roadmap: `docs/PLAN.md` (approved 2026-09-25). Follow its build order: Step 0 tools → Step 1 spikes (S1–S7, go/no-go) → M1 → M2 → M3 → M4 (Teach & Transfer) → Phase B (Colab first, then small GPU/CPU).
- Dev machine: Windows 11, RTX 3060 12 GB, 16 GB RAM. Phase A targets only this machine; keep all device/dtype logic in `engine/hw.py` so Phase B is a local change.
- Scope: a general-purpose industrial annotation tool (any product, any part list). The car-lamp example data is only an example application: no car-specific defaults in engine, UI or docs (parent-object crop is an optional per-project setting; left/right handling only when class names form left_/right_ pairs).
- Core principle: the output is the annotated dataset. Interactive annotation must work with no model training; the in-tool detector ("Boost") is opt-in, off by default, never auto-started. Ask before any training run outside Teach & Transfer.
- Key constraints from research:
  - Use SAM 3 via Hugging Face `transformers` (`Sam3TrackerVideoModel`, `Sam3Model`), not Meta's repo (Windows/T4 problems). SAM 3 exemplar boxes only work on the frame they're drawn on — cross-frame discovery is DINOv3-based.
  - SAM 3.1 Object Multiplex (`Tracker31`) is an opt-in tracker ("tracker": "sam3.1" in app.json) through
    `engine/vendor/muggled_sam` (Apache-2.0, pinned, unmodified; weights `sam3.1_multiplex.pt` from a hash-pinned
    mirror, `torch.load(weights_only=True)`): 1.7x faster at 4 parts, but an outline ballooned in S19-S22 where SAM 3
    held, so SAM 3 HF `Tracker` stays the default (`make_tracker`).
  - Parts that change look (project.json `states`, `engine/states.py`, S16): tracking picks each frame's state class
    from person-vouched examples in the task; a class change between states touches one frame only.
  - No Gradio: Colab free tier forbids web-UI-first use. UI = one `ui/canvas.js` served by FastAPI locally and by anywidget in notebooks.
  - Laya / Laya Vision / openjev were measured as box checkers and are not used (S4b, S4c in `spikes/REPORT.md`); review flags come from `Project.flags` (area vs the person's box, jumps, lost parts). PyPI `laya` and the laya-vision fork share a package name: never install both.
  - Chunk video sessions (~180 frames), offload state to CPU, fully tear down sessions (VRAM leak, sam3 issue #305).
- Refresh the knowledge graph with `/graphify` after each milestone.
- Rivet, the in-app helper (`engine/assistant.py`, UI in `ui/canvas.js`): Gemini Flash-Lite, answers from `docs/GUIDE.md`
  (keep the guide accurate when features change). The API key comes only from `GEMINI_API_KEY` or `~/.partlabeler/assistant.json`:
  never write a key into the repo, and scan for `AIza` before any public push. Tips and the pros are fixed text, not model calls.
- Label types (project.json `task`): detect (boxes), segment (outlines: masks as COCO RLE, box = mask box,
  polygons only at export, `engine/masks.py`) and classify (image classes on a grid). Smart sorting
  (`engine/sort.py`: grouping, suggestions, wrong-label check; S10) also drives "Sort parts" in box/outline projects.
- Updates (`engine/update.py`, start-screen button, `update_windows.bat`, `partlabeler update`): only the app's own
  files change; projects/, data/, .venv and models never; labels are backed up first. Keep it that way.
- Accounts (`engine/accounts.py`): local only, `~/.partlabeler/accounts.json` (scrypt hashes; PARTLABELER_CONFIG moves it);
  every page and API call of the local host needs a sign-in (/login); per-account prefs are applied server-side
  (`themed()`: data-theme/-accent on <html>, `window.PL`) and each account may pick its own projects folder
  (`home()` per request; background jobs keep it). Notebooks have no sign-in. Theme tokens: `ui/theme.css`
  (`--c-*`, modules fall back to their light values). Never commit an accounts file.
- API for programs (`ui/mcp.py`): MCP at `/mcp` (streamable HTTP, plain JSON replies, stateless) and the same tools at
  `/api/tools/<name>`; Bearer API tokens (accounts.py, hashed, revocable) are checked in the `signed_in` middleware;
  tools call `Session.call` (engine/api.py) so edits are live in open tabs and undoable. Default projects folder of a
  computer: `~/.partlabeler/app.json` {"home": ...} (`accounts.default_home`).
- Layout: `engine/` (no UI code; `api.py` is the one message protocol), `ui/` (`canvas.js` annotator, `home.js` start screen, two hosts), `engine/cli.py` (`partlabeler` CLI). User data lives in `data/` and `projects/`, both gitignored: never commit them.
