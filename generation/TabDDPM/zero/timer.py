"""Minimal timer utilities compatible with the expected libzero API."""
import time
from typing import Optional


class Timer:
    """A simple wall-clock timer with start/stop semantics."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def run(self) -> "Timer":
        if self._start is None:
            self._start = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        if self._start is not None:
            self._elapsed += time.perf_counter() - self._start
            self._start = None
        return self

    def reset(self) -> "Timer":
        self._start = None
        self._elapsed = 0.0
        return self

    @property
    def seconds(self) -> float:
        if self._start is not None:
            return self._elapsed + time.perf_counter() - self._start
        return self._elapsed

    def __enter__(self) -> "Timer":
        return self.run()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __str__(self) -> str:
        total = self.seconds
        hours, remainder = divmod(int(total), 3600)
        minutes, seconds = divmod(remainder, 60)
        frac = total - int(total)
        seconds_with_frac = seconds + frac
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds_with_frac:06.3f}"
        if minutes:
            return f"{minutes:02d}:{seconds_with_frac:06.3f}"
        return f"{seconds_with_frac:.3f}s"
