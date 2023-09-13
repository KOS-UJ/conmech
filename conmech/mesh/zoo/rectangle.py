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
import meshzoo
import numpy as np

from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.properties.mesh_description import RectangleMeshDescription


class Rectangle(RawMesh):
    def __init__(self, mesh_descr: RectangleMeshDescription):
        scale_x, scale_y = mesh_descr.scale
        mesh_density = [
            int(np.ceil(scale / mesh_descr.max_element_perimeter)) for scale in mesh_descr.scale
        ]

        # pylint: disable=no-member
        nodes, elements = meshzoo.rectangle_tri(
            np.linspace(0.0, scale_x, int(mesh_density[0]) + 1),
            np.linspace(0.0, scale_y, int(mesh_density[1]) + 1),
            variant="zigzag",
        )
        super().__init__(nodes, elements)
