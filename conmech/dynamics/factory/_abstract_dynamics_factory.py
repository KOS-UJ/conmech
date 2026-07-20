# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2022-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from typing import Tuple

import numpy as np
import scipy.sparse

from conmech.struct.stiffness_matrix import StiffnessMatrix, SM1
from conmech.struct.types import FeatureVector, FeatureMatrix


class AbstractDynamicsFactory:
    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def get_edges_features_matrix(self, elements, nodes) -> Tuple:
        raise NotImplementedError()

    def calculate_constitutive_matrices(
        self, W: FeatureMatrix, mu: float, lambda_: float
    ) -> StiffnessMatrix:
        raise NotImplementedError()

    def get_relaxation_tensor(self, W: FeatureMatrix, coeff: np.ndarray) -> StiffnessMatrix:
        raise NotImplementedError()

    def calculate_acceleration(self, U: scipy.sparse.spmatrix, density: float) -> StiffnessMatrix:
        raise NotImplementedError()

    def calculate_thermal_expansion(self, V: FeatureVector, coeff: np.ndarray) -> StiffnessMatrix:
        raise NotImplementedError()

    def calculate_thermal_conductivity(
        self, W: FeatureMatrix, coeff: np.ndarray
    ) -> StiffnessMatrix:
        raise NotImplementedError()

    def get_piezoelectric_tensor(self, W: FeatureMatrix, coeff: np.ndarray) -> StiffnessMatrix:
        raise NotImplementedError()

    def get_permittivity_tensor(self, W: FeatureMatrix, coeff: np.ndarray) -> StiffnessMatrix:
        raise NotImplementedError()

    @staticmethod
    def calculate_poisson_matrix(W: FeatureMatrix, propagation: float) -> SM1:
        return SM1(propagation**2 * W.diagonal_sum())

    @staticmethod
    def calculate_wave_matrix(W: FeatureMatrix) -> scipy.sparse.spmatrix:
        return W.diagonal_sum()
