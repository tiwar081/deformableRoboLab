"""Shared helpers valid for every example — a catch-all for small cross-example utilities.

Add any function that should be identical across all demos here (the first one is the terminal
tee below). Two rules:

1. Each example imports this module BEFORE ``import warp`` (so ``install_terminal_log`` can install
   the tee before Warp's module-load messages are emitted). Therefore keep module-scope imports
   light — **no ``warp``/``newton`` at the top of this file**. If a shared helper needs Warp,
   import it lazily inside the function, or put it in ``examples/franka_common.py`` (the
   Warp/Newton-dependent Franka building blocks + the ``GraspExample`` base).
2. Prefer parameters from ``assets.params`` so behavior stays centralized.
"""
from __future__ import annotations

import sys
from pathlib import Path


class _TerminalTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def install_terminal_log():
    """Redirect stdout/stderr to both the console and ``outputs/terminal``; returns the log file."""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = (output_dir / "terminal").open("w", buffering=1)
    sys.stdout = _TerminalTee(sys.__stdout__, log)
    sys.stderr = _TerminalTee(sys.__stderr__, log)
    return log
