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


@MeshZOO.register("JOB2023", "jurochbar_2023", "bow")
class JOB2023(RawMesh):
    def __init__(self, mesh_prop: MeshProperties):
        # pylint: disable=no-member  # for dmsh
        geo = dmsh.Polygon(
            [
                [0.0, 0.0],
                [1.2, 0.0],
                [1.2, 0.6],
                [0.0, 0.6],
            ]
        )
        x1 = 0.15
        x2 = 1.05
        y1 = 0.15
        y2 = 0.45
        r = 0.05
        geo = geo - dmsh.Circle([0.6, 0.0], 0.3)
        geo = geo - dmsh.Circle([x1, y1], r)
        geo = geo - dmsh.Circle([x2, y1], r)
        geo = geo - dmsh.Circle([x1, y2], r)
        geo = geo - dmsh.Circle([x2, y2], r)
        nodes, elements = dmsh.generate(geo, 1 / mesh_prop.mesh_density[0])
        super().__init__(nodes, elements)
