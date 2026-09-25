"""Notebook host for the annotator (Jupyter, JupyterLab, VS Code, Colab).

    from ui.host_widget import Annotator
    Annotator("data/projects/my_run")        # last expression in a cell displays it

Colab delivers a widget's messages only while the kernel is handling one from the page, so messages sent
from job threads (tracking progress, suggestions, Rivet's answers) would be lost. Those wait in an outbox
instead, and the page sends {"type": "poll"} about twice a second; each poll sends what is waiting.
"""
import queue
import threading
from pathlib import Path

import anywidget

from engine.api import Session
from engine.project import Project


class Outbox:
    """Send from the main thread right away; from other threads, hold until flush() (called on each poll)."""

    def __init__(self, send):
        self._send, self._q = send, queue.SimpleQueue()

    def __call__(self, msg: dict) -> None:
        if threading.current_thread() is threading.main_thread():
            self.flush()                                     # keep the order
            self._send(msg)
        else:
            self._q.put(msg)

    def flush(self) -> None:
        while True:
            try:
                self._send(self._q.get_nowait())
            except queue.Empty:
                return


class Annotator(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "canvas.js"

    def __init__(self, project, **kwargs):
        super().__init__(**kwargs)
        self.outbox = Outbox(self.send)
        self.session = Session(project if isinstance(project, Project) else Project(project), self.outbox)
        self.on_msg(self._on_msg)

    def _on_msg(self, _widget, msg, _buffers):
        if msg.get("type") != "poll":
            self.session.handle(msg)
        self.outbox.flush()
