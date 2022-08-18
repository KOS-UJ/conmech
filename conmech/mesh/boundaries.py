from typing import Dict

import numpy as np

from conmech.mesh.boundary import Boundary


class Boundaries:
    def __init__(self, boundary_internal_indices: np.ndarray, **kwargs):
        self.boundaries: Dict[str, Boundary] = {}
        for key, value in kwargs.items():
            assert isinstance(key, str)
            assert isinstance(value, Boundary)
            self.boundaries[key] = value

        self.boundary_internal_indices = boundary_internal_indices

    @property
    def boundary_surfaces(self):
        return np.unique(
            np.vstack((self.contact_boundary, self.neumann_boundary, self.dirichlet_boundary)),
            axis=1,
        )

    @property
    def boundary_nodes_count(self):
        return self.contact_nodes_count + self.neumann_nodes_count + self.dirichlet_nodes_count

    @property
    def boundary_indices(self):
        return slice(self.boundary_nodes_count)

    @property
    def contact_boundary(self):
        return self.boundaries["contact"].surfaces

    @property
    def neumann_boundary(self):
        return self.boundaries["neumann"].surfaces

    @property
    def dirichlet_boundary(self):
        return self.boundaries["dirichlet"].surfaces

    @property
    def contact_nodes_count(self) -> int:
        return self.boundaries["contact"].node_count

    @property
    def neumann_nodes_count(self) -> int:
        return self.boundaries["neumann"].node_count

    @property
    def dirichlet_nodes_count(self) -> int:
        return self.boundaries["dirichlet"].node_count

    def get_all_boundary_indices(self, boundary, total_node_count, dimension):
        i = self.boundaries[boundary].node_indices
        for d in range(dimension):
            if isinstance(i, slice):
                yield slice(i.start + d * total_node_count, i.stop + d * total_node_count)
            else:
                yield i + d * total_node_count
