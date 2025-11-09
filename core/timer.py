from __future__ import annotations


class Timer:
    def __init__(self, consumed: float = 0) -> None:
        self.consumed = consumed

    def copy(self) -> Timer:
        return Timer(self.consumed)
