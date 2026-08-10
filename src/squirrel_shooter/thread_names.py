"""Best-effort names for Python and native worker threads."""

from __future__ import annotations

import threading
from pathlib import Path


def set_current_thread_name(name: str) -> None:
    """Name the current Python thread and its Linux task when available."""

    threading.current_thread().name = name
    comm = Path("/proc/self/task") / str(threading.get_native_id()) / "comm"
    try:
        comm.write_text(name[:15] + "\n", encoding="utf-8")
    except OSError:
        # Windows development and restricted Linux environments do not expose comm.
        pass
