from ctypes import ArgumentError
from dataclasses import dataclass


@dataclass
class LossRaport:
    main: float = 0.0
    displacement_loss: float = 0.0
    acceleration_loss: float = 0.0
    reduced_lifted_displacement_loss: float = 0.0
    reduced_lifted_acceleration_loss: float = 0.0

    _count: int = 0

    @property
    def count(self):
        return self._count

    def add(self, loss: "LossRaport", normalize=True):
        self_vars = vars(self)
        loss_vars = vars(loss)

        total_count = self._count + loss.count
        for (key, value) in self_vars.items():
            if value is not None:
                if normalize:
                    p1 = value * float(self._count) / total_count
                    p2 = loss_vars[key] * float(loss.count) / total_count
                    self_vars[key] = p1 + p2
                else:
                    self_vars[key] += loss_vars[key]
            elif loss_vars[key] is not None:
                raise ArgumentError

        self._count = total_count

    def normalize(self):
        self_vars = vars(self)
        for (key, value) in self_vars.items():
            if key != "_count" and value is not None:
                self_vars[key] /= self._count

    def get_iterator(self):
        data = vars(self).copy()
        del data["_count"]
        for (key, value) in vars(self).items():
            if value is None:
                del data[key]
        return data.items()
