from __future__ import annotations
from progress import counter, bar


class ProgressMixin(object):
    def reset(self, message: str) -> None:
        self.index = 0
        self.message = f'{message} '


class Bar(bar.IncrementalBar, ProgressMixin):
    def reset(self, message: str, max: int | float, item: str = "") -> None:
        self.max = max
        self.suffix = '%(index)d/%(max)d ' + item
        super().reset(message)


class Counter(counter.Counter, ProgressMixin):
    ...
