# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from conmech.properties.mesh_description import NestedRectangleMeshDescription


class NestedRectangle(RawMesh):
    """
    Structured triangulation of a rectangle whose refinements are nested.

    * the mesh size is exactly `1 / cells_per_unit` in both directions, so
      doubling `cells_per_unit` halves `h` and the coarse vertices are a subset
      of the fine ones;
    * diagonals are parallel to `(1, 1)`, so coarse diagonal is a union of fine
      diagonals.

    They give `V_h(n) subset V_h(2 n)` for the P1 space: a P1 function of
    the coarse mesh is represented exactly on the fine mesh, so carrying it over
    introduces no error `H^1`.
    """

    def __init__(self, mesh_descr: NestedRectangleMeshDescription):
        scale_x, scale_y = mesh_descr.scale
        n_x = int(round(scale_x * mesh_descr.cells_per_unit))
        n_y = int(round(scale_y * mesh_descr.cells_per_unit))
        if n_x < 1 or n_y < 1:
            raise ValueError(
                f"cells_per_unit={mesh_descr.cells_per_unit} is too small for "
                f"scale={mesh_descr.scale}: it gives {n_x} x {n_y} cells"
            )

        grid_x, grid_y = np.meshgrid(
            np.linspace(0.0, scale_x, n_x + 1), np.linspace(0.0, scale_y, n_y + 1)
        )
        nodes = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

        i_cell, j_cell = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
        i_cell, j_cell = i_cell.ravel(), j_cell.ravel()

        def index(i, j):
            return i + (n_x + 1) * j

        # lower-right and upper-left triangle of each cell, split along (1, 1)
        lower = np.stack(
            (index(i_cell, j_cell), index(i_cell + 1, j_cell), index(i_cell + 1, j_cell + 1)),
            axis=1,
        )
        upper = np.stack(
            (index(i_cell, j_cell), index(i_cell + 1, j_cell + 1), index(i_cell, j_cell + 1)),
            axis=1,
        )
        super().__init__(nodes, np.concatenate((lower, upper), axis=0))
