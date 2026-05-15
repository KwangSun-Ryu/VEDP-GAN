"""TADGAN 진행바 유틸."""

import os
import shutil
import sys
import threading

from tqdm.auto import tqdm


class SilentBar:
    def __init__(self):
        self.total = 0
        self.n = 0

    def update(self, n=1):
        return

    def set_postfix_str(self, text="", refresh=True):
        return

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        return

    def set_description_str(self, desc=None, refresh=True):
        return

    def refresh(self):
        return

    def close(self):
        return


class NullProgressReporter:
    def __init__(self, verbose=False, emit_logs=True):
        self.verbose = bool(verbose)
        self.emit_logs = bool(emit_logs)

    def add_total(self, n):
        return

    def step(self, phase, experiment, variant, data, metric=None, stage=None):
        return

    def ok(self, message):
        if not self.emit_logs:
            return
        print(message, flush=True)

    def fail(self, message):
        if not self.emit_logs:
            return
        print(message, flush=True)

    def info(self, message, verbose_only=True):
        if not self.emit_logs:
            return
        if verbose_only and not self.verbose:
            return
        print(message, flush=True)

    def create_epoch_bar(self, total, desc, enabled=True, colour="#36cfc9"):
        return SilentBar()

    def create_detail_bar(self, total, desc, enabled=True, colour="#ff9f1c", verbose_only=True):
        return SilentBar()

    def close(self):
        return


class ProgressReporter:
    def __init__(self, verbose=False, colour="#1ab6ff"):
        self.verbose = bool(verbose)
        self._lock = threading.Lock()
        self._is_windows = os.name == "nt"
        self._use_tqdm = self._resolve_tqdm_output()
        self._bar = tqdm(total=0, **self._build_tqdm_kwargs(colour=colour)) if self._use_tqdm else SilentBar()

    def _resolve_tqdm_output(self):
        force = os.getenv("AB_STUDY_FORCE_TQDM", "").strip().lower()
        disable = os.getenv("AB_STUDY_DISABLE_TQDM", "").strip().lower()

        if force in {"1", "true", "yes", "on"}:
            return True
        if disable in {"1", "true", "yes", "on"}:
            return False
        return sys.stderr.isatty()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _build_tqdm_kwargs(self, colour=None, position=0, leave=True, desc=None):
        kwargs = {
            "position": position,
            "leave": leave,
            "dynamic_ncols": True,
            "ascii": self._is_windows,
        }
        if desc is not None:
            kwargs["desc"] = desc
        if not self._is_windows and colour is not None:
            kwargs["colour"] = colour
        return kwargs

    def _terminal_width(self):
        return shutil.get_terminal_size(fallback=(120, 24)).columns

    def _ellipsize_middle(self, text, max_len):
        text = str(text or "")
        if max_len <= 0 or len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        head = max(1, (max_len - 3) // 2)
        tail = max(1, max_len - 3 - head)
        return f"{text[:head]}...{text[-tail:]}"

    def _compact_phase(self, phase, metric=None):
        phase_map = {
            "prepare": "prep",
            "train": "train",
            "sample": "sample",
            "aggregate": "agg",
            "eval-ML": "ml",
            "eval-SDMetrics": "sdm",
            "eval-Utils": "util",
            "eval-DCR": "dcr",
        }
        if phase in phase_map:
            return phase_map[phase]
        if metric is not None:
            return f"{phase}:{metric}"
        return str(phase or "")

    def _compact_desc(self, variant, data):
        base = f"{variant}/{data}"
        max_len = 28 if self._is_windows else 40
        return self._ellipsize_middle(base, max_len)

    def _compact_postfix(self, phase, stage=None, metric=None):
        parts = [self._compact_phase(phase, metric=metric)]
        if stage:
            parts.append(str(stage))
        text = " ".join(part for part in parts if part)
        reserved = 52 if self._is_windows else 72
        max_len = max(24, self._terminal_width() - reserved)
        return self._ellipsize_middle(text, max_len)

    def add_total(self, n):
        n = int(n or 0)
        if n <= 0:
            return
        with self._lock:
            current_total = int(self._bar.total or 0)
            self._bar.total = current_total + n
            self._bar.refresh()

    def step(self, phase, experiment, variant, data, metric=None, stage=None):
        desc = self._compact_desc(variant, data)
        postfix = self._compact_postfix(phase, stage=stage, metric=metric)
        with self._lock:
            self._bar.set_description_str(desc, refresh=False)
            self._bar.set_postfix_str(postfix, refresh=False)
            self._bar.update(1)

    def ok(self, message):
        with self._lock:
            if self._use_tqdm:
                self._bar.write(message)
            else:
                print(message, flush=True)

    def fail(self, message):
        with self._lock:
            if self._use_tqdm:
                self._bar.write(message)
            else:
                print(message, flush=True)

    def info(self, message, verbose_only=True):
        if verbose_only and not self.verbose:
            return
        with self._lock:
            if self._use_tqdm:
                self._bar.write(message)
            else:
                print(message, flush=True)

    def _create_bar(self, total, desc, position, enabled=True, colour="#36cfc9"):
        if not enabled or not self._use_tqdm:
            return SilentBar()
        with self._lock:
            return tqdm(
                total=total,
                **self._build_tqdm_kwargs(
                    desc=self._ellipsize_middle(desc, 24 if self._is_windows else 32),
                    position=position,
                    leave=False,
                    colour=colour,
                ),
            )

    def create_epoch_bar(self, total, desc, enabled=True, colour="#36cfc9"):
        return self._create_bar(total=total, desc=desc, position=1, enabled=enabled, colour=colour)

    def create_detail_bar(self, total, desc, enabled=True, colour="#ff9f1c", verbose_only=True):
        enabled = enabled and (self.verbose or not verbose_only)
        return self._create_bar(total=total, desc=desc, position=2, enabled=enabled, colour=colour)

    def close(self):
        with self._lock:
            if self._use_tqdm:
                self._bar.close()
