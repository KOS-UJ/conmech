import numpy as np
from typing import Callable, Dict


class BoundariesDescription:
    def __init__(self, **kwargs):
        self.boundaries: Dict[str, Callable[[np.ndarray], bool]] = {
            "contact": lambda _: False,
            "dirichlet": lambda _: False,
        }
        for key, value in kwargs.items():
            assert isinstance(key, str)
            self.boundaries[key] = value

    def __getitem__(self, item):
        return self.boundaries[item]
