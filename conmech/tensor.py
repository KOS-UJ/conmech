import numpy as np


class Tensor:
    def __init__(self, data_dim: tuple, physical_dim: int):
        self.data_dim = data_dim
        self.physical_dim = physical_dim
        self.data: np.ndarray = np.empty((physical_dim, *data_dim))

    @classmethod
    def empty(cls, data_dim: tuple, physical_dim: int):
        return cls(data_dim, physical_dim)
