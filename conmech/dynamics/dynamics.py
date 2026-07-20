# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2022-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
# Copyright (C) 2022  Michał Jureczka <michal.jureczka@uj.edu.pl>
# Copyright (C) 2023 Wiktor Prządka
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
from typing import Optional

import numpy as np
import scipy.sparse

from conmech.struct.types import FeatureMatrix
from conmech.dynamics.factory.dynamics_factory_method import (
    get_dynamics,
    get_basic_matrices,
    get_factory,
)
from conmech.properties.body_properties import (
    ElasticRelaxationProperties,
)
from conmech.scene.body_forces import BodyForces


class Dynamics:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        body: "Body",
    ):
        self.body = body
        self.body.dynamics = self

        self.force = BodyForces(body)
        self.temperature = BodyForces(body)

        self.factory = get_factory(body.mesh.dimension)
        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self._w_matrix: Optional[np.ndarray] = None
        self._local_stifness_matrices: Optional[np.ndarray] = None
        self.__relaxation: Optional[np.ndarray] = None
        self.__relaxation_tensor: Optional[float] = None
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray
        self.piezoelectricity: np.ndarray
        self.permittivity: np.ndarray
        self.poisson_operator: np.ndarray

        self.reinitialize_matrices()

    def reinitialize_matrices(self, elements_density: Optional[np.ndarray] = None):
        (
            self.element_initial_volume,
            self.volume_at_nodes,
            U,
            V,
            self._w_matrix,
            self._local_stifness_matrices,
        ) = get_basic_matrices(
            elements=self.body.mesh.elements,
            nodes=self.body.mesh.nodes,
        )  # + self.displacement_old)

        if elements_density is not None:
            self._w_matrix = self.asembly_w_matrix_with_density(elements_density)

        (
            self.acceleration_operator,
            self.elasticity,
            self.viscosity,
            self.thermal_expansion,
            self.thermal_conductivity,
            self.piezoelectricity,
            self.permittivity,
            self.poisson_operator,
        ) = get_dynamics(
            elements=self.body.mesh.elements,
            body_prop=self.body.properties,
            U=U,
            V=V,
            W=self._w_matrix,
        )

    def asembly_w_matrix_with_density(self, elements_density: np.ndarray):
        # COO accumulation: one (row, col) entry per (element, i, j); scipy sums
        # duplicates on CSR construction.
        elements = self.body.mesh.elements
        nodes_count = self.body.mesh.nodes_count
        dim, _, elements_count, element_size, _ = self._local_stifness_matrices.shape

        nnz = elements_count * element_size * element_size
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        entry = 0
        for element in elements:
            for global_i in element:
                for global_j in element:
                    rows[entry] = global_i
                    cols[entry] = global_j
                    entry += 1

        blocks = []
        for k in range(dim):
            row_blocks = []
            for m in range(dim):
                data = np.empty(nnz, dtype=np.double)
                entry = 0
                for element_index in range(elements_count):
                    scale = elements_density[element_index]
                    lsm = self._local_stifness_matrices[k, m, element_index]
                    for i in range(element_size):
                        for j in range(element_size):
                            data[entry] = scale * lsm[i, j]
                            entry += 1
                row_blocks.append(
                    scipy.sparse.csr_matrix((data, (rows, cols)), shape=(nodes_count, nodes_count))
                )
            blocks.append(row_blocks)
        return FeatureMatrix(blocks)

    def relaxation(self, time: float = 0):
        # TODO handle others
        if isinstance(self.body.properties, ElasticRelaxationProperties):
            relaxation_tensor = self.body.properties.relaxation(time)
            if (relaxation_tensor != self.__relaxation_tensor).any():
                self.__relaxation_tensor = relaxation_tensor
                self.__relaxation = self.factory.get_relaxation_tensor(
                    self._w_matrix, relaxation_tensor
                )
        else:
            raise TypeError("There is no relaxation operator!")

        return self.__relaxation
