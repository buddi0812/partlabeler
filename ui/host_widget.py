"""Notebook host for the annotator (Jupyter, JupyterLab, VS Code, Colab).

    from ui.host_widget import Annotator
    Annotator("data/projects/my_run").run()    # or leave `Annotator(...)` as the cell's last expression

Colab delivers a widget's messages only while the kernel is handling one from the page, so messages sent
from job threads (tracking progress, suggestions, Rivet's answers) would be lost. Those wait in an outbox
instead, and the page sends {"type": "poll"} about twice a second; each poll sends what is waiting and
{"type": "pong"}, so the page notices when the kernel stops answering (busy, or the session disconnected).

Colab disconnects a runtime 30 minutes after the last cell finished (its idle timer is re-armed when a cell
completes and paused while one runs; clicks in the page or in a widget do not count). run() therefore keeps
the cell running while the annotator is in use, handling its messages meanwhile (jupyter-ui-poll), and
returns after `idle_minutes` without any action in it, so an unattended session still times out as usual.
"""
import os
import queue
import threading
import time
from pathlib import Path

import anywidget

from engine.api import Session
from engine.project import Project

IDLE_MINUTES = 25


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
        self.last_action = time.time()
        self.on_msg(self._on_msg)

    def _on_msg(self, _widget, msg, _buffers):
        if msg.get("type") != "poll":                        # polls are automatic; anything else is the user
            self.last_action = time.time()
            self.session.handle(msg)
        self.outbox.flush()
        if msg.get("type") == "poll":
            self.send({"type": "pong"})

    def run(self, idle_minutes: float = IDLE_MINUTES) -> None:
        """Show the annotator. On Colab, keep this cell running while it is used (so the runtime is not
        disconnected as idle) until `idle_minutes` pass without any action in it, or the cell is stopped."""
        from IPython.display import display
        display(self)
        if not os.environ.get("COLAB_RELEASE_TAG"):
            return
        from jupyter_ui_poll import ui_events
        self.last_action = time.time()
        try:
            with ui_events() as poll:
                while time.time() - self.last_action < idle_minutes * 60:
                    poll(50)                                 # handle the annotator's messages
                    time.sleep(0.05)
            print(f"No action in the annotator for {idle_minutes:g} minutes, so this cell stopped and Colab's usual "
                  "idle timeout applies again. The annotator still works and your work is saved; run this cell "
                  "again to keep the runtime awake while you annotate.")
        except KeyboardInterrupt:
            print("Stopped. The annotator still works and your work is saved. Colab's usual idle timeout applies "
                  "again (30 minutes after the last cell) until you run this cell again.")
