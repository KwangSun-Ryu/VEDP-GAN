"""prediction 평가 실행 전역 진행/로그 유틸."""

import threading

from tqdm.auto import tqdm


class NullProgressReporter:
    """진행바를 사용하지 않는 fallback 리포터."""

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)

    def add_total(self, n):
        return

    def step(self, metric, model, data, multiples=None, stage=None):
        return

    def ok(self, message):
        print(message)

    def fail(self, message):
        print(message)

    def info(self, message, verbose_only=True):
        if verbose_only and not self.verbose:
            return
        print(message)

    def close(self):
        return


class ProgressReporter:
    """전역 단일 tqdm와 스레드 안전 로그 출력을 담당."""

    def __init__(self, verbose=False, colour="#0075f2"):
        self.verbose = bool(verbose)
        self._lock = threading.Lock()
        self._bar = tqdm(total=0, colour=colour, dynamic_ncols=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def add_total(self, n):
        n = int(n or 0)
        if n <= 0:
            return
        with self._lock:
            current_total = int(self._bar.total or 0)
            self._bar.total = current_total + n
            self._bar.refresh()

    def _format_multiples(self, multiples):
        if multiples is None:
            return "all"
        try:
            return f"{int(multiples):02d}x"
        except Exception:
            return str(multiples)

    def step(self, metric, model, data, multiples=None, stage=None):
        postfix = (
            f"metric={metric} model={model} data={data} "
            f"mul={self._format_multiples(multiples)} stage={stage or '-'}"
        )
        with self._lock:
            self._bar.set_postfix_str(postfix)
            self._bar.update(1)

    def ok(self, message):
        with self._lock:
            self._bar.write(message)

    def fail(self, message):
        with self._lock:
            self._bar.write(message)

    def info(self, message, verbose_only=True):
        if verbose_only and not self.verbose:
            return
        with self._lock:
            self._bar.write(message)

    def close(self):
        with self._lock:
            self._bar.close()
