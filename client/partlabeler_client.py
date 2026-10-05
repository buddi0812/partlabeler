"""PartLabeler integration client: one file, Python standard library only, for test suites and production code.
Copy it into your framework. It speaks the integration API (/api/v1, docs/INTEGRATION.md).

    from partlabeler_client import PartLabeler
    pl = PartLabeler("http://127.0.0.1:8765", token="plt_...")        # token: Settings → API access
    pl.health()                                                       # {"status": "ok", ...}
    pl.annotate("frame.jpg", ["car", "wheel"])["parts"]               # stateless: nothing stored
    pl.create_project("line", type="detect", labels=["car"])
    job = pl.upload("line", "cam0.mp4", every=5)                      # a video (or several images) -> a task
    pl.wait(job["job"])
    pl.annotate_text("line", ["car", "truck cabin"], scope="task", task=1, wait_s=0)
    pl.wait_idle("line")                                              # long scopes run in the background
    pl.annotations("line", task=1)["frames"]
    pl.export("line", "coco", path="line_coco.zip")                   # waits, then downloads the zip

Errors raise PartLabelerError(status, detail): 400 bad input, 401 token, 403 not allowed from this machine,
404 unknown project/job, 409 the project is busy with another job, 413 too big. Reads and /annotate are retried
on connection errors and 502/503/504 with backoff; calls that change something are not (they may have happened).
"""
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class PartLabelerError(Exception):
    def __init__(self, status: int, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class PartLabeler:
    def __init__(self, url: str = "http://127.0.0.1:8765", token: str | None = None, timeout: float = 600,
                 retries: int = 3):
        self.base = url.rstrip("/") + "/api/v1"
        self.token, self.timeout, self.retries = token, timeout, retries

    # ---- transport ---------------------------------------------------------------------------------------
    def _call(self, method: str, path: str, body=None, data: bytes | None = None, ctype: str | None = None,
              raw: bool = False, retry: bool | None = None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            data, ctype = json.dumps(body).encode(), "application/json"
        if ctype:
            headers["Content-Type"] = ctype
        retry = method == "GET" if retry is None else retry
        for attempt in range(self.retries + 1 if retry else 1):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = r.read()
                return payload if raw else json.loads(payload or b"null")
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                try:
                    detail = json.loads(text).get("detail", text)
                except (ValueError, AttributeError):
                    detail = text
                if retry and e.code in (502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise PartLabelerError(e.code, detail) from None
            except urllib.error.URLError as e:              # not reachable (yet)
                if retry and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise PartLabelerError(0, str(e.reason)) from None

    @staticmethod
    def _multipart(fields: dict, files: list) -> tuple[bytes, str]:
        """multipart/form-data from {name: value} and [(field, filename, bytes)]."""
        b = uuid.uuid4().hex
        out = []
        for k, v in fields.items():
            if v is None:
                continue
            for one in v if isinstance(v, list) else [v]:
                out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{one}\r\n'.encode())
        for field, name, content in files:
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            out.append(f'--{b}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                       f"Content-Type: {kind}\r\n\r\n".encode() + content + b"\r\n")
        out.append(f"--{b}--\r\n".encode())
        return b"".join(out), f"multipart/form-data; boundary={b}"

    # ---- calls -------------------------------------------------------------------------------------------
    def health(self) -> dict:
        return self._call("GET", "/health")

    def annotate(self, image, objects, threshold: float = 0.5, outlines: bool = False) -> dict:
        """Parts of the named objects on one image (a path or bytes): {width, height, parts: [{label, box, score,
        outline?}], ms}. Nothing is stored on the server."""
        content = Path(image).read_bytes() if isinstance(image, (str, Path)) else bytes(image)
        body = {"image": base64.b64encode(content).decode(), "threshold": threshold, "outlines": outlines,
                "objects": objects if isinstance(objects, list) else [o.strip() for o in str(objects).split(",")]}
        return self._call("POST", "/annotate", body, retry=True)

    def create_project(self, name: str, type: str = "detect", labels=()) -> dict:
        return self._call("POST", "/projects", {"name": name, "type": type, "labels": list(labels)})

    def project(self, name: str) -> dict:
        return self._call("GET", f"/projects/{urllib.parse.quote(name)}")

    def upload(self, project: str, *paths, name: str | None = None, every: int | None = None,
               subset: str | None = None) -> dict:
        """One video, or any number of images, as one new task. Returns the add_tasks job ({"job": id, ...})."""
        if not paths:
            raise ValueError("give a video or images")
        field = "file" if len(paths) == 1 and Path(paths[0]).suffix.lower() not in (
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp") else "files"
        data, ctype = self._multipart({"name": name, "every": every, "subset": subset},
                                      [(field, Path(p).name, Path(p).read_bytes()) for p in paths])
        return self._call("POST", f"/projects/{urllib.parse.quote(project)}/uploads", data=data, ctype=ctype)

    def annotate_text(self, project: str, objects, scope: str = "frame", item: int | None = None,
                      task: int | None = None, frame: int | None = None, labels=None, threshold: float = 0.5,
                      include_confirmed: bool = False, wait_s: float = 120) -> dict:
        """SAM 3 finds the objects on a frame / rest (to the end of its task) / task / project; parts to check."""
        body = {"objects": objects if isinstance(objects, list) else [o.strip() for o in str(objects).split(",")],
                "labels": labels, "scope": scope, "item": item, "task": task, "frame": frame, "threshold": threshold,
                "include_confirmed": include_confirmed, "wait_s": wait_s}
        return self._call("POST", f"/projects/{urllib.parse.quote(project)}/annotate-text", body)

    def status(self, project: str, task: int | None = None) -> dict:
        q = f"?task={task}" if task is not None else ""
        return self._call("GET", f"/projects/{urllib.parse.quote(project)}/status{q}")

    def annotations(self, project: str, task: int | None = None, item: int | None = None, items=None,
                    outlines: bool = False, only_labeled: bool = True) -> dict:
        q = {"task": task, "item": item, "items": ",".join(map(str, items)) if items else None,
             "outlines": str(outlines).lower(), "only_labeled": str(only_labeled).lower()}
        query = urllib.parse.urlencode({k: v for k, v in q.items() if v is not None})
        return self._call("GET", f"/projects/{urllib.parse.quote(project)}/annotations?{query}")

    def export(self, project: str, format: str, path=None, tasks=None, confirmed_only: bool = False,
               save_images: bool = True, timeout: float = 3600) -> dict:
        """Export the dataset; with `path`, wait for it and download it there as a zip."""
        job = self._call("POST", f"/projects/{urllib.parse.quote(project)}/export",
                         {"format": format, "tasks": tasks, "confirmed_only": confirmed_only,
                          "save_images": save_images, "wait_s": 0})
        if path is None:
            return job
        done = self.wait(job["job"], timeout)
        zipped = self._call("GET", f"/projects/{urllib.parse.quote(project)}/exports/{job['job']}.zip", raw=True)
        Path(path).write_bytes(zipped)
        return {**done, "path": str(path)}

    def job(self, job: str) -> dict:
        return self._call("GET", f"/jobs/{job}")

    def stop(self, job: str) -> dict:
        return self._call("POST", f"/jobs/{job}/stop", {})

    def wait(self, job: str, timeout: float = 3600, every: float = 1.0) -> dict:
        """Poll a job until it finishes; raises PartLabelerError if it failed, TimeoutError if it takes too long."""
        t0 = time.time()
        while True:
            j = self.job(job)
            if j["finished"]:
                if j.get("error"):
                    raise PartLabelerError(500, j["error"])
                return j
            if time.time() - t0 > timeout:
                raise TimeoutError(f"job {job} still running after {timeout:.0f} s")
            time.sleep(every)

    def wait_idle(self, project: str, timeout: float = 3600, every: float = 2.0) -> dict:
        """Poll a project until its running job (tracking, annotate by text…) is over; returns its status."""
        t0 = time.time()
        while True:
            s = self.status(project)
            if not s.get("busy"):
                return s
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{project} still busy after {timeout:.0f} s")
            time.sleep(every)

    def tool(self, name: str, **args) -> dict:
        """Any other tool (GET /api/v1/tools lists them, with their inputs)."""
        return self._call("POST", f"/tools/{name}", args)
