from ctypes import ArgumentError
from dataclasses import dataclass
from typing import Optional


@dataclass
class LossRaport:
    main: float = 0.0
    inner_energy: float = 0.0
    energy: float = 0.0
    boundary_integral: float = 0.0
    mean: float = 0.0
    rmse: Optional[float] = None
    relative_energy: Optional[float] = None

    _count: int = 0

    def add(self, loss: "LossRaport", normalize=True):
        self_vars = vars(self)
        loss_vars = vars(loss)

        total_count = self._count + loss._count
        for (key, value) in self_vars.items():
            if value is not None:
                if normalize:
                    p1 = value * float(self._count) / total_count
                    p2 = loss_vars[key] * float(loss._count) / total_count
                    self_vars[key] = p1 + p2
                else:
                    self_vars[key] += loss_vars[key]
            elif loss_vars[key] is not None:
                raise ArgumentError

        self._count = total_count

    def normalize(self):
        self_vars = vars(self)
        for (key, value) in self_vars.items():
            if key is not "_count" and value is not None:
                self_vars[key] /= self._count

    def get_iterator(self):
        data = vars(self).copy()
        del data["_count"]
        for (key, value) in vars(self).items():
            if value is None:
                del data[key]
        return data.items()
