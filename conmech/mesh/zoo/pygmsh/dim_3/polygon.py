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

from conmech.mesh.zoo.pygmsh import _utils
from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.mesh.zoo import MeshZOO
from conmech.properties.mesh_properties import MeshProperties


@MeshZOO.register("polygon_3d", "pygmsh_polygon_3d")
class Polygon(RawMesh):
    def __init__(self, mesh_prop: MeshProperties):
        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(
                [
                    [0.0, 0.0],
                    [1.0, -0.2],
                    [1.1, 1.2],
                    [0.1, 0.7],
                ]
            )
            geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)

            _utils.set_mesh_size(geom, mesh_prop)
            nodes, elements = _utils.get_nodes_and_elements(geom, 3)
            nodes = _utils.normalize_nodes(nodes)
        super().__init__(nodes, elements)