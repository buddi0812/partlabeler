"""install.ps1's PyTorch build decision and -Plan (Windows only; skipped elsewhere). Runs the real script against
a throwaway copy of the repo with a fake .venv (a real empty venv plus a hand-written torch dist-info, so no
network, no GPU and no real multi-hundred-MB torch install are needed): the bug this guards against is an update
silently replacing an already-installed PyTorch build (fixture3, 2026-10-08, docs/team/README.md "Release gate")."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="install.ps1 is Windows only")


def make_box(tmp_path: Path, torch_version: str = "", poison_torch_import: str = "") -> Path:
    """A throwaway copy of the repo root (install.ps1 needs pyproject.toml next to it) with a real, empty venv;
    torch_version ("2.14.1+cpu", "2.14.0+cu130", ...) writes a dist-info for it without installing torch.
    poison_torch_import: also makes "torch" a real (fake) package that raises this on import - for step 7's
    self-check, which "import torch, engine.hw, ..., transformers, ..." stops at the first failing import."""
    box = tmp_path / "box"
    box.mkdir()
    shutil.copy(ROOT / "install.ps1", box / "install.ps1")
    shutil.copy(ROOT / "pyproject.toml", box / "pyproject.toml")
    subprocess.run(["uv", "venv", "--python", "3.12", str(box / ".venv")], check=True, capture_output=True)
    if torch_version:
        site = next((box / ".venv").glob("Lib/site-packages"))
        info = site / f"torch-{torch_version.replace('+', '+')}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: torch\nVersion: {torch_version}\n", encoding="utf-8")
        if poison_torch_import:
            pkg = site / "torch"
            pkg.mkdir()
            (pkg / "__init__.py").write_text(f"raise OSError({poison_torch_import!r})", encoding="utf-8")
    return box


def run_plan(box: Path) -> dict:
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1", "-Plan"],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


def snapshot(box: Path) -> set:
    # __pycache__: Python's own bytecode cache, written the first time a .py runs in the venv; not state -Plan
    # itself could be said to "change", and not deterministic enough to assert on.
    return {(p.relative_to(box), p.stat().st_mtime_ns) for p in box.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def test_existing_cpu_build_is_kept_on_a_rerun_even_with_a_gpu_present(tmp_path):
    """A re-run never switches on its own, even a CPU box with a capable GPU: "GPU always" is a downstream
    installer's own job, asking explicitly (-Cuda <build>), not install.ps1's own default (2026-10-08: keep
    the conservative default - a CUDA build runs fine with no GPU anyway, engine/hw.py picks the device)."""
    box = make_box(tmp_path, "2.14.1+cpu")
    d = run_plan(box)
    assert d["torch"]["installed_build"] == "cpu"
    assert d["torch"]["chosen_build"] == "cpu"
    assert d["torch"]["action"] == "keep"
    assert d["torch"]["approx_bytes"] == 0


def test_existing_gpu_build_is_kept_even_when_a_newer_one_is_ideal(tmp_path):
    """Once on a GPU build, a re-run doesn't chase a different/newer one by itself (that would reintroduce the
    original unasked-download bug, just for CUDA point releases instead of CPU->CUDA)."""
    box = make_box(tmp_path, "2.14.0+cu130")
    d = run_plan(box)
    assert d["torch"]["installed_build"] == "cu130"
    assert d["torch"]["chosen_build"] == "cu130"
    assert d["torch"]["action"] == "keep"
    assert d["torch"]["approx_bytes"] == 0


def test_explicit_cuda_switches_even_from_a_working_gpu_build(tmp_path):
    box = make_box(tmp_path, "2.14.0+cu130")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1", "-Plan", "-Cuda", "cu126"],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(r.stdout)
    assert d["torch"]["action"] == "reinstall" and d["torch"]["chosen_build"] == "cu126"


def test_explicit_cuda_matching_installed_never_reinstalls(tmp_path):
    box = make_box(tmp_path, "2.14.0+cu130")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1", "-Plan", "-Cuda", "cu130"],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(r.stdout)
    assert d["torch"]["action"] == "keep" and d["torch"]["chosen_build"] == "cu130"


def test_plan_changes_nothing_on_disk(tmp_path):
    box = make_box(tmp_path, "2.14.1+cpu")
    before = snapshot(box)
    run_plan(box)
    assert snapshot(box) == before


# These three use an explicit -Cuda (always different from the installed +cpu, so $torchAction is always
# "reinstall" and -Wheels is actually consulted) rather than relying on a real GPU being present on the runner.


def test_wheels_refuses_without_a_prefetch(tmp_path):
    box = make_box(tmp_path, "2.14.1+cpu")
    before = snapshot(box)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1",
                       "-Cuda", "cu126", "-Wheels", str(tmp_path / "nowhere")],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "prefetch.json" in (r.stdout + r.stderr)
    assert snapshot(box) == before          # refused before touching anything


def test_wheels_refuses_an_incomplete_prefetch(tmp_path):
    box = make_box(tmp_path, "2.14.1+cpu")
    pf = tmp_path / "pf"
    pf.mkdir()
    (pf / "prefetch.json").write_text(json.dumps({"variant": "cu126", "torch": "", "files": [], "complete": False}))
    before = snapshot(box)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1",
                       "-Cuda", "cu126", "-Wheels", str(pf)],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "not complete" in (r.stdout + r.stderr)
    assert snapshot(box) == before


def test_wheels_refuses_a_variant_mismatch(tmp_path):
    box = make_box(tmp_path, "2.14.1+cpu")
    pf = tmp_path / "pf"
    pf.mkdir()
    (pf / "prefetch.json").write_text(json.dumps({"variant": "cu999", "torch": "0", "files": [], "complete": True}))
    before = snapshot(box)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1",
                       "-Cuda", "cu126", "-Wheels", str(pf)],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "is for cu999" in (r.stdout + r.stderr)
    assert snapshot(box) == before


def test_plan_resolves_the_ideal_builds_real_wheels(tmp_path):
    """torch.wheels: the real current torch+torchvision wheel for ideal_build (name/url/sha256/bytes), resolved
    from the package index - needs the internet, unlike the rest of -Plan (there's no other way to know what a
    real install would actually fetch right now; skip if it's not reachable here)."""
    box = make_box(tmp_path, "2.14.1+cpu")
    d = run_plan(box)
    assert d["torch"]["chosen_build"] == "cpu"          # a plain re-run keeps it, so wheels describes cpu here
    wheels = d["torch"]["wheels"]
    if not wheels:
        pytest.skip("no internet reachable from this runner")
    names = {w["name"].split("-")[0] for w in wheels}
    assert names == {"torch", "torchvision"}
    for w in wheels:
        assert w["bytes"] > 1_000_000 and len(w["sha256"]) == 64 and w["url"].startswith("https://")
        assert "-cp" in w["name"] and "+cu" not in w["name"]          # the plain (cpu) wheel, not a cuXXX one


def test_plan_wheels_follows_an_explicit_chosen_build(tmp_path):
    box = make_box(tmp_path, "2.14.0+cu130")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1", "-Plan", "-Cuda", "cu126"],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(r.stdout)
    wheels = d["torch"]["wheels"]
    if not wheels:
        pytest.skip("no internet reachable from this runner")
    assert all("+cu126" in w["name"] for w in wheels)    # the asked-for build, not what's installed (cu130)


def test_plan_reports_prefetch_status(tmp_path):
    box = make_box(tmp_path, "2.14.1+cpu")
    d = run_plan(box)
    chosen = d["torch"]["chosen_build"]          # a plain re-run keeps the installed build (cpu here)
    pf = tmp_path / "pf"
    pf.mkdir()
    (pf / "prefetch.json").write_text(json.dumps({"variant": chosen, "torch": "2.14.1", "files": [], "complete": True}))
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1", "-Plan", "-Prefetch", str(pf)],
                       cwd=box, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    d2 = json.loads(r.stdout)
    assert d2["prefetch"] == {"dir": str(pf), "exists": True, "complete": True, "variant_matches": True}


def test_no_self_check_skips_step_7_entirely(tmp_path):
    """-NoSelfCheck: for a build machine (e.g. make-bundle collecting packages for a USB stick) where torch
    never actually runs - importing it there is meaningless and must not be able to fail the build."""
    box = make_box(tmp_path, "2.14.1+cpu")    # no "engine" package here either: a real self-check would fail
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1",
                       "-Cpu", "-NoModels", "-NoTeach", "-NoSelfCheck"],
                       cwd=box, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Self-check (skipped" in r.stdout
    assert "CUDA available" not in r.stdout     # the self-check's own import line never ran


def test_self_check_names_an_application_control_block_plainly(tmp_path):
    box = make_box(tmp_path, "2.14.1+cpu", poison_torch_import=(
        "[WinError 4551] An Application Control policy has blocked this file. "
        "Error loading ...\\.venv\\Lib\\site-packages\\torch\\lib\\caffe2_nvrtc.dll"))
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "install.ps1",
                       "-Cpu", "-NoModels", "-NoTeach"],
                       cwd=box, capture_output=True, text=True, timeout=180)
    assert r.returncode != 0
    assert "Smart App Control" in r.stdout and "PartLabeler cannot run on this PC" in r.stdout
    assert "turning" not in r.stdout.lower() or "will not help" in r.stdout   # never advises turning it off
