from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MeshProperties:
    mesh_type: str
    mesh_density: List[float]
    scale: List[float]
    dimension: int
    is_adaptive: bool = False
    initial_base: Optional[np.ndarray] = None
    initial_position: Optional[np.ndarray] = None

    @staticmethod
    def _get_modulo(array, index):
        return array[index % len(array)]

    @property
    def scale_x(self) -> float:
        return self._get_modulo(self.scale, 0)

    @property
    def scale_y(self) -> float:
        return self._get_modulo(self.scale, 1)

    @property
    def scale_z(self) -> float:
        return self._get_modulo(self.scale, 2)

    @property
    def mesh_density_x(self) -> float:
        return self._get_modulo(self.mesh_density, 0)

    @property
    def mesh_density_y(self) -> float:
        return self._get_modulo(self.mesh_density, 1)

    @property
    def mesh_density_z(self) -> float:
        return self._get_modulo(self.mesh_density, 2)
