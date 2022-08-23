from typing import Callable, Dict, Optional

import numpy as np


class BoundariesDescription:
    def __init__(self, **kwargs):
        self.indicators: Dict[str, Callable[[np.ndarray], bool]] = {
            "contact": lambda _: False,
            "dirichlet": lambda _: False,
        }
        self.conditions: Dict[str, Callable[[np.ndarray], Optional[float]]] = {
            "dirichlet": lambda x: np.zeros(x.shape[0]),
        }
        for key, value in kwargs.items():
            assert isinstance(key, str)
            if isinstance(value, tuple):
                indicator, condition = value
                self.conditions[key] = condition
            else:
                indicator = value
            self.indicators[key] = indicator

    def __getitem__(self, item):
        return self.indicators[item]
