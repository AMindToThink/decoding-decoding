"""Tee stdout/stderr to a logfile so every experiment run is recoverable later.

Use:
    from decoding_decoding.logging_utils import start_logging
    log_path = start_logging(out_dir / "logs", "extend_to_2048")
    print("...")  # goes to console AND log_path

Logs land at  {log_dir}/{name}_{YYYYMMDD-HHMMSS}.log
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import IO, TextIO


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        # Some libs (vLLM/tqdm) probe isatty(). Inherit from the first stream.
        return getattr(self.streams[0], "isatty", lambda: False)()

    def fileno(self) -> int:
        # vLLM's suppress_stdout context calls sys.stdout.fileno() and
        # os.dup2()s /dev/null on top. Returning the original stdout fd is
        # the right thing: that fd-level redirection happens out-of-band of
        # Python's print(), so Python writes still flow through this Tee
        # to the logfile while C-level vLLM internals temporarily silence
        # the terminal copy.
        return self.streams[0].fileno()


def start_logging(log_dir: Path, name: str) -> Path:
    """Begin teeing stdout/stderr to a logfile under log_dir.

    Returns the path to the opened logfile. The file handle stays open for
    the life of the process; callers don't need to close it.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"

    fp: IO = open(log_path, "w", buffering=1, encoding="utf-8")
    fp.write(f"# Log started {dt.datetime.now().isoformat()}\n")
    fp.write(f"# argv: {sys.argv}\n")
    fp.flush()

    sys.stdout = _Tee(sys.__stdout__, fp)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, fp)  # type: ignore[assignment]
    print(f"[logging] tee'ing stdout/stderr to {log_path}")
    return log_path
