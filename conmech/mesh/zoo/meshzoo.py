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


class ClassRegistrationError(ImportError):
    pass


class MeshZOO:
    ZOO = {}

    @staticmethod
    def register(*names: str):
        def add_to_zoo(mesh):
            for name in names:
                lower_name = name.lower()
                if lower_name in MeshZOO.ZOO:
                    raise ClassRegistrationError(f"Name {name} is already taken")
                MeshZOO.ZOO[lower_name] = mesh
            return mesh

        return add_to_zoo

    @staticmethod
    def get_by_name(mesh_name: str) -> type:
        return MeshZOO.ZOO[mesh_name.lower()]
