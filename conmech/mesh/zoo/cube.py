# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
import numpy as np

from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.properties.mesh_properties import CubeMeshDescription


class Cube(RawMesh):
    def __init__(self, _mesh_descr: CubeMeshDescription):
        nodes = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [0.0, 1.0, 0.5],
                [0.5, 1.0, 0.5],
                [1.0, 1.0, 0.5],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.5, 1.0],
                [0.5, 0.5, 1.0],
                [1.0, 0.5, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        elements = np.asarray(
            [
                [0, 1, 3, 9],
                [12, 13, 15, 21],
                [10, 11, 13, 19],
                [4, 5, 7, 13],
                [1, 4, 3, 13],
                [13, 16, 15, 25],
                [11, 14, 13, 23],
                [5, 8, 7, 17],
                [1, 3, 9, 13],
                [13, 15, 21, 25],
                [11, 13, 19, 23],
                [5, 7, 13, 17],
                [1, 9, 10, 13],
                [13, 21, 22, 25],
                [11, 19, 20, 23],
                [5, 13, 14, 17],
                [3, 9, 13, 12],
                [15, 21, 25, 24],
                [13, 19, 23, 22],
                [7, 13, 17, 16],
                [18, 19, 9, 21],
                [12, 13, 3, 15],
                [10, 11, 1, 13],
                [22, 23, 13, 25],
                [19, 22, 13, 21],
                [13, 16, 7, 15],
                [11, 14, 5, 13],
                [23, 26, 17, 25],
                [19, 21, 13, 9],
                [13, 15, 7, 3],
                [11, 13, 5, 1],
                [23, 25, 17, 13],
                [19, 9, 13, 10],
                [13, 3, 7, 4],
                [11, 1, 5, 2],
                [23, 13, 17, 14],
                [21, 9, 12, 13],
                [15, 3, 6, 7],
                [13, 1, 4, 5],
                [25, 13, 16, 17],
            ]
        )
        super().__init__(nodes, elements)
