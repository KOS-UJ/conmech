# CONMECH @ Jagiellonian University in Kraków
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

import random

import numpy as np

from conmech.helpers import nph
from conmech.mesh.utils import interpolate_nodes


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(options):
    return random.choice(options)


def get_mean(dimension, scale):
    return np.random.uniform(
        low=-scale,
        high=scale,
        size=(1, dimension),
    )


def get_corner_vectors_rotate(dimension, scale):
    if dimension != 2:
        raise NotImplementedError
    # 1 2
    # 0 3
    corner_vector = nph.generate_normal(rows=1, columns=dimension, scale=scale)
    corner_vectors = corner_vector * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    return corner_vectors


def get_corner_vectors_all(dimension, scale):
    corner_vectors = nph.generate_normal(rows=dimension * 2, columns=dimension, scale=scale)
    return corner_vectors


def scale_nodes_to_square(nodes):
    nodes_min = np.min(nodes, axis=0)
    nodes_max = np.max(nodes, axis=0)
    scaled_nodes = (nodes - nodes_min) / (nodes_max - nodes_min)
    return scaled_nodes


def get_nodes_interpolation(nodes: np.ndarray, base: np.ndarray, corner_vectors: np.ndarray):
    # orthonormal matrix; inverse equals transposition
    denormalized_nodes = nph.get_in_base(nodes, base.T)
    denormalized_scaled_nodes = denormalized_nodes  # scale_nodes_to_square(denormalized_nodes)

    denormalized_nodes_interpolation = interpolate_nodes(
        scaled_nodes=denormalized_scaled_nodes,
        corner_vectors=corner_vectors,
    )
    nodes_interpolation = (
        denormalized_nodes_interpolation  # nph.get_in_base(denormalized_nodes_interpolation, base)
    )
    return nodes_interpolation
