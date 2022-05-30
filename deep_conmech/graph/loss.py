from ctypes import ArgumentError
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Loss:
    main: float = 0.0
    inner_energy: float = 0.0
    energy: float = 0.0
    boundary_integral: float = 0.0
    mean: float = 0.0
    rmse: Optional[float] = None
    relative_energy: Optional[float] = None

    count: int = 0

    def add(self, loss: "Loss", normalize=True):
        self_vars = vars(self)
        loss_vars = vars(loss)

        total_count = self.count + loss.count
        for (key, value) in self_vars.items():
            if value is not None:
                if normalize:
                    p1 = value * float(self.count) / total_count
                    p2 = loss_vars[key] * float(loss.count) / total_count
                    self_vars[key] = p1 + p2
                else:
                    self_vars[key] += loss_vars[key]
            elif loss_vars[key] is not None:
                raise ArgumentError

        self.count = total_count

    def normalize(self):
        self_vars = vars(self)
        for (key, value) in self_vars.items():
            if value is not None:
                self_vars[key] /= self.count
