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
import dmsh

from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.mesh.zoo import MeshZOO
from conmech.properties.mesh_properties import MeshProperties


@MeshZOO.register("sofochbar", "sofochbar_2023", "tunnel")
class JOB2023(RawMesh):
    def __init__(self, mesh_prop: MeshProperties):
        # pylint: disable=no-member  # for dmsh
        diameter = 3.0
        thickness = 1.0
        pillar_height = 2.0
        eps = 0.01
        width = diameter + 2 * thickness
        geo = dmsh.Circle([width / 2, pillar_height], width / 2)
        geo = geo - dmsh.Circle([width / 2, pillar_height], diameter / 2)
        geo = geo - dmsh.Polygon(
            [
                [0.0, pillar_height - width / 2],
                [width, pillar_height - width / 2],
                [width, pillar_height],
                [0.0, pillar_height],
            ]
        )
        geo = geo + dmsh.Polygon(
            [
                [0.0, 0.0],
                [thickness, 0.0],
                [thickness, pillar_height + eps],
                [0.0, pillar_height + eps],
            ]
        )
        geo = geo + dmsh.Polygon(
            [
                [diameter + thickness, 0.0],
                [width, 0.0],
                [width, pillar_height + eps],
                [diameter + thickness, pillar_height + eps],
            ]
        )

        foundation = dmsh.Path([[0., 0.], [diameter + 2 * thickness, 0.]])

        dense_y = mesh_prop.mesh_density_y
        dense_x = mesh_prop.mesh_density_x

        def target_edge_length(x):
            return dense_y**-1 + dense_x**-1 * foundation.dist(x)

        nodes, elements = dmsh.generate(geo, target_edge_length)
        super().__init__(nodes, elements)
