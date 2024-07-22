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

from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.properties.mesh_description import Barboteu2008MeshDescription


class Barboteu2008(RawMesh):
    def __init__(self, mesh_descr: Barboteu2008MeshDescription):
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [3.0, 1.0],
                    [1.5, 1.0],
                    [1.0, 1.5],
                    [1.0, 4.0],
                    [0.0, 4.0],
                ],
                mesh_size=0.1,
            )
            Barboteu2008._set_mesh_size(geom, mesh_descr)
            nodes, elements = Barboteu2008._get_nodes_and_elements(geom, 2)
        super().__init__(nodes, elements)

    @staticmethod
    def _set_mesh_size(geom, mesh_descr: Barboteu2008MeshDescription):
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, *_: mesh_descr.max_element_perimeter
        )

    @staticmethod
    def _get_nodes_and_elements(geom, dim):
        geom_mesh = geom.generate_mesh()
        nodes = geom_mesh.points.copy()
        elements = geom_mesh.cells[-2].data.astype("long").copy()
        return nodes[:, :dim], elements
