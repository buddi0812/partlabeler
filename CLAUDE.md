# PartLabeler — notes for Claude

- Design and roadmap: `docs/PLAN.md` (approved 2026-09-25). Follow its build order: Step 0 tools → Step 1 spikes (S1–S7, go/no-go) → M1 → M2 → M3 → M4 (Teach & Transfer) → Phase B (Colab first, then small GPU/CPU).
- Dev machine: Windows 11, RTX 3060 12 GB, 16 GB RAM. Phase A targets only this machine; keep all device/dtype logic in `engine/hw.py` so Phase B is a local change.
- Scope: a general-purpose industrial annotation tool (any product, any part list). The car-lamp example data is only an example application: no car-specific defaults in engine, UI or docs (parent-object crop is an optional per-project setting; left/right handling only when class names form left_/right_ pairs).
- Core principle: the output is the annotated dataset. Interactive annotation must work with no model training; the in-tool detector ("Boost") is opt-in, off by default, never auto-started. Ask before any training run outside Teach & Transfer.
- Key constraints from research:
  - Use SAM 3 via Hugging Face `transformers` (`Sam3TrackerVideoModel`, `Sam3Model`), not Meta's repo (Windows/T4 problems). SAM 3 exemplar boxes only work on the frame they're drawn on — cross-frame discovery is DINOv3-based.
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
- Layout: `engine/` (no UI code; `api.py` is the one message protocol), `ui/` (`canvas.js` annotator, `home.js` start screen, two hosts), `engine/cli.py` (`partlabeler` CLI). User data lives in `data/` and `projects/`, both gitignored: never commit them.
