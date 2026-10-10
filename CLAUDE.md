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
  app.json `"assistant_llm"` (a local OpenAI-compatible URL) switches Rivet to `ui/rivet_tools.py` instead: it drives
  PartLabeler through a small, fixed subset of `ui/mcp.py`'s tools, pauses on destructive ones (`{"confirm": ...}`)
  until a person clicks Confirm, and logs every tool call to `<projects>/.partlabeler/assistant-actions.jsonl`. The
  model id is read from `GET <url>/v1/models` at each chat, never hardcoded. app.json `"assistant_llm_key"`
  (optional, e.g. for a Hawk-AI box's llama-server running with `--api-key`) is sent as `Authorization: Bearer
  <key>` on every call to that server; absent, no header is sent (older boxes, or a key-less server) - never
  logged. app.json `"assistant_cloud": false`
  makes Rivet say "No assistant is installed on this computer" instead of ever offering Gemini. `engine/assistant.py`'s
  `backend()` ("local" | "none" | "gemini", reading `assistant_llm`/`assistant_cloud` fresh every call) is the one
  decision every entry point shares — both REST (`ui/host_fastapi.py`) and the websocket's `on_assistant*`
  (`engine/api.py`, used by the main app too, not just notebooks) — so a box set up for a local model or with
  the cloud switched off can't be reached by Gemini through either path. Keep new entry points on `backend()`.
- Label types (project.json `task`): detect (boxes), segment (outlines: masks as COCO RLE, box = mask box,
  polygons only at export, `engine/masks.py`) and classify (image classes on a grid). Smart sorting
  (`engine/sort.py`: grouping, suggestions, wrong-label check; S10) also drives "Sort parts" in box/outline projects.
- Releases: tags `vX.Y.Z` (= pyproject version) on the published commit; a downstream USB/box installer pins a tag
  and relies on: install.ps1 `-Cuda cu130|cu132|cu126|cpu`, `-Cpu`, `-NoModels`, `-Python` default 3.12, offline
  with `UV_OFFLINE`/`HF_HUB_OFFLINE` + the uv on PATH; `engine.models` (`SAM3`, `DINOV3`, `SAM31`, `fetch`); the
  service command `engine.cli app --no-browser --host --port --home`; the ONNX export format. Announce changes to
  those first. `.gitattributes` export-ignore keeps notes, spikes, videos and samples out of release zips.
  install.ps1 (2026-10-08): a re-run never switches an already-installed PyTorch build on its own (CPU stays
  CPU, a CUDA build stays that CUDA build) — only `-Cpu`/`-Cuda <build>` do that; no unasked multi-GB download.
  "GPU always" is a downstream installer's own job (it knows a box has a capable GPU and asks for `-Cuda <build>`
  explicitly) — a CUDA build runs fine with no GPU anyway (`engine/hw.py` picks the device at runtime; the start
  screen's device chip shows which). `-Plan` prints the decision as JSON (`installed_build`, `chosen_build`,
  `ideal_build`, `action`, `reason`, `approx_bytes`, `wheels`: real name/url/sha256/bytes for `chosen_build`'s
  torch+torchvision from the package index) and changes nothing on disk. `-Prefetch <dir>` downloads those
  wheels (plain `pip download` — uv has no `pip download` subcommand — into `<dir>\prefetch.json`, resumable at
  the whole-file level); `-Wheels <dir>` installs from one offline, refusing unless it's complete and matches.
  `-NoSelfCheck` skips step 7 (for a build machine that only collects packages, e.g. a USB stick, and never
  runs them); otherwise an import failure naming Windows' app control (WinError 4551) gets a specific message,
  never advice to turn that off (it cannot be turned back on without a reset). Tests in `tests/test_install_ps1.py`.
- Updates (`engine/update.py`, start-screen button, `update_windows.bat`, `partlabeler update`): only the app's own
  files change; projects/, data/, .venv and models never; labels are backed up first. Keep it that way.
- Teach's GPU memory (`engine/detector.py` `_vram_needed_gb`/`batch_plan`/`fits_gpu`): `_VRAM_CAL` holds real
  measured peaks (`docs/VRAM_CALIBRATION.md`, `tests/test_vram_calibration.py`) per size at 320px - nano and
  small only so far; other sizes fall back to the older `_GB_BASE`/`_GB_PER_IMAGE` guess until measured too.
  `batch_plan()` picks the largest batch whose own (not the previous batch's) predicted peak fits free GPU
  memory minus `VRAM_MARGIN_GB` (1 GB headroom for whatever else is using the card). Add new sizes/resolutions
  to the doc and the test fixture together, from a real measurement, never a guess.
- Embedded in another product (`engine.accounts.box_managed()`, app.json `"box_managed": true`, set only by that
  product's own installer, e.g. Hawk-AI): developer-only UI is hidden from its operator - no raw storage path on
  the start screen, `/api/update` refuses (that product updates PartLabeler, not this), and Rivet's local-model
  name is not reported (`/api/assistant`, the websocket's `assistant_info`). Separately, and for every Rivet
  caller regardless of box_managed: the system prompt (`ui/rivet_tools.py`) never lets the model name a tool or
  show a technical id to the person - plain English and the names they gave things, always.
- Accounts (`engine/accounts.py`): local only, `~/.partlabeler/accounts.json` (scrypt hashes; PARTLABELER_CONFIG moves it);
  every page and API call of the local host needs a sign-in (/login); per-account prefs are applied server-side
  (`themed()`: data-theme/-accent on <html>, `window.PL`) and each account may pick its own projects folder
  (`home()` per request; background jobs keep it). Notebooks have no sign-in. Theme tokens: `ui/theme.css`
  (`--c-*`, modules fall back to their light values). Never commit an accounts file.
- API for programs (`ui/mcp.py`): MCP at `/mcp` (streamable HTTP, plain JSON replies, stateless) and the same tools at
  `/api/tools/<name>`; Bearer API tokens (accounts.py, hashed, revocable) are checked in the `signed_in` middleware;
  tools call `Session.call` (engine/api.py) so edits are live in open tabs and undoable. Default projects folder of a
  computer: `~/.partlabeler/app.json` {"home": ...} (`accounts.default_home`).
- Integration API (`ui/v1.py`; contract and production design guide `docs/INTEGRATION.md`; stdlib client
  `client/partlabeler_client.py`): `/api/v1`, changes only by addition (breaking = `/api/v2`). Network mode (`--host`):
  other machines and proxied requests reach only `/api/v1`; path tools (`LOCAL_ONLY_TOOLS`) are refused there.
  `GET /api/v1/models` (hawk-ai-internal §12.3) is the one exception: a service key (`X-Hawk-Token` = app.json
  `"models_token"`, not a Bearer account token) lists every export (`ui/host_fastapi.py` `models_for_deploy`) for
  a same-box production system's deploy picker, local-only (127.0.0.1) even with the right key since `onnx_path`
  is an absolute path. Rivet's `deploy_link`/`list_stations` tools build the dashboard deep link
  (`app.json "dashboard_url"`, default `http://127.0.0.1:8080`) a person opens and approves on the box's own
  screen - PartLabeler never deploys anything itself.
- Team server (`"team": true` + `"host"` in ~/.partlabeler/app.json, or `app --team`): other machines get the whole
  app signed in; `ui/v1.py` TEAM_LOCAL_ONLY stays local (signup, open-folder, update, assistant key, projects folder);
  login attempt limit; engine/api.py `ACTOR` (set by the host) makes undo per person and lets a teammate's job wait.
- Annotate by text (`engine/textlabel.py`, `ConceptFinder.detect_texts`: one image encoding for all phrases): scopes
  frame / rest / task / project, finds stored as `imported` parts to check, never doubled, one undo step.
- Cycle recordings (`engine/recordings.py`, "Recordings" tab): browse another local system's cameras/videos and add
  them as tasks. `app.json "recordings_api"` (must be 127.0.0.1/localhost) + `"recordings_token"` (sent as header
  `X-Hawk-Token`, read fresh per call, never logged or put in a URL - the other system regenerates it on its own
  install/repair/update); else a plain `"recordings_dir"` folder. `resolve()` pages past the API's default
  listing limit (a camera can keep 100+ segments/day).
  "Crop to camera ROI" cuts every kept frame to that camera's ROI at ingest (`Project.add_source`'s `camera_id`/
  `crop`, `engine.recordings.crop_rect`/`valid_crop` - the same floor/ceil rounding and 0..1-share validity rule
  hawk-vision uses, so a crop drawn here lines up with what the box cuts at inference). A task's `camera_id`/
  `crop` flow into a Teach run's settings.json `task_cameras`; `detector.crop_card` writes model.json's "crop"
  (one rectangle per camera, only when every task of that camera used the same one) or refuses export by name
  when one camera's tasks were cut differently (hawk-ai-internal §13.3).
- Layout: `engine/` (no UI code; `api.py` is the one message protocol), `ui/` (`canvas.js` annotator, `home.js` start screen, two hosts), `engine/cli.py` (`partlabeler` CLI). User data lives in `data/` and `projects/`, both gitignored: never commit them.
