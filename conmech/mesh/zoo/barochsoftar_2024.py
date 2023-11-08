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
from conmech.properties.mesh_description import JOB2023MeshDescription


class BOST2023(RawMesh):
    def __init__(self, mesh_descr: JOB2023MeshDescription):
        # pylint: disable=no-member  # for dmsh
        geo = dmsh.Polygon(
            [
                [0.0, 0.2],
                [0.2, 0.4],
                [0.6, 0.4],
                [0.8, 0.2],
                [0.8, 0.0],
                [1.2, 0.0],
                [1.2, 1.2],
                [0.8, 1.2],
                [0.8, 1.0],
                [0.6, 0.8],
                [0.2, 0.8],
                [0.0, 1.0],
            ]
        )
        nodes, elements = dmsh.generate(geo, mesh_descr.max_element_perimeter)
        super().__init__(nodes, elements)
