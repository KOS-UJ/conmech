"""
timing helpers
"""
import time
from contextlib import ContextDecorator
from ctypes import ArgumentError
from typing import Any

import pandas as pd

# Inspired by https://github.com/realpython/codetiming


class SingleTimer(ContextDecorator, list):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.start_time = None

    def start(self) -> None:
        if self.start_time is not None:
            raise ArgumentError

        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            raise ArgumentError

        timer = time.perf_counter() - self.start_time
        self.start_time = None

        self.append(timer)
        return timer

    def __enter__(self) -> "SingleTimer":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


class Timer(dict):
    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = SingleTimer(key)

        return super().__getitem__(key)

    @property
    def dt(self):
        return self.to_dataframe()

    def to_dataframe(self, subset=None):
        if subset is None:
            values = self
        else:
            values = {key: value for (key, value) in self.items() if key in subset}

        return pd.DataFrame.from_dict(values)
