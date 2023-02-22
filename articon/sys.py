from __future__ import annotations
from progress import counter, bar, Infinite


class ProgressMixin(Infinite):
    VERBOSE = True

    def reset(self, message: str) -> None:
        self.index = 0
        self.message = f'{message} '

    def set_verbose(self, on: bool):
        self.VERBOSE = on

    def next(self, *args, **kwargs):
        if not self.VERBOSE:
            return
        return super().next(*args, **kwargs)

    def finish(self):
        if not self.VERBOSE:
            return
        return super().finish()


class Bar(bar.IncrementalBar, ProgressMixin):
    def reset(self, message: str, max: int | float, item: str = "") -> None:
        self.max = max
        self.suffix = '%(index)d/%(max)d ' + item
        super().reset(message)


class Counter(counter.Counter, ProgressMixin):
    ...
