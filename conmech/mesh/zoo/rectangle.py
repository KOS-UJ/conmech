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
from conmech.mesh.zoo import MeshZOO
from conmech.properties.mesh_properties import MeshProperties


@MeshZOO.register("rectangle", "rectangle_2d", "meshzoo_rectangle_2d")
class Rectangle(RawMesh):
    def __init__(self, mesh_prop: MeshProperties):
        # pylint: disable=no-member
        nodes, elements = meshzoo.rectangle_tri(
            np.linspace(0.0, mesh_prop.scale_x, int(mesh_prop.mesh_density_x) + 1),
            np.linspace(0.0, mesh_prop.scale_y, int(mesh_prop.mesh_density_y) + 1),
            variant="zigzag",
        )
        super().__init__(nodes, elements)
