from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class MeshProperties:
    mesh_type: str

    mesh_density: Optional[List[float]] = None
    grid_height: Optional[float] = None
    scale: Optional[List[float]] = field(init=False)
    dimension: Optional[int] = None

    path: Optional[str] = None

    initial_base: Optional[np.ndarray] = None
    initial_position: Optional[np.ndarray] = None
    mean_at_origin: bool = False
    corners_vector: Optional[np.ndarray] = None
    corner_mesh_data: Optional[np.ndarray] = None

    def __post_init__(self):
        self.do_sanity_checks()
        if self.mesh_type != "msh_file":
            self.scale = [
                (self.grid_height / self.mesh_density[1]) * elems_num
                for elems_num in self.mesh_density
            ]
            self.dimension = self.dimension or 2

    def do_sanity_checks(self):
        if not (
            self.mesh_type == "msh_file" and self.path is not None and self.mesh_density is None
        ) and not (
            self.mesh_type != "msh_file" and self.path is None and self.mesh_density is not None
        ):
            raise ValueError("Improper combination of mesh parameters")

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
