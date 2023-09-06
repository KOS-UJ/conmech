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
import pygmsh

from conmech.mesh.zoo import MeshZOO
from conmech.mesh.zoo.pygmsh import _utils
from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.properties.mesh_properties import GeneratedMeshProperties


@MeshZOO.register("pygmsh_rectangle", "pygmsh_rectangle_2d")
class Rectangle(RawMesh):
    def __init__(self, mesh_prop: GeneratedMeshProperties):
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(
                [
                    [0.0, 0.0],
                    [0.0, mesh_prop.scale_y],
                    [mesh_prop.scale_x, mesh_prop.scale_y],
                    [mesh_prop.scale_x, 0.0],
                ]
            )
            _utils.set_mesh_size(geom, mesh_prop)
            nodes, elements = _utils.get_nodes_and_elements(geom, 2)
        super().__init__(nodes, elements)
