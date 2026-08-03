# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2019-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

from matplotlib import pyplot as plt

from conmech.helpers.config import Config
from conmech.plotting.drawer import Drawer


def draw(state, config: Config, dimension: int = 2):
    if dimension == 2:
        Drawer(state=state, config=config).draw(show=config.show, save=config.save)
        return

    fig = plt.figure()
    axs = fig.add_subplot(111, projection="3d")
    # Draw nodes
    nodes = state.body.mesh.nodes
    axs.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="b", marker="o")

    # Draw elements
    faces = state.displaced_nodes[state.body.mesh.boundary_surfaces]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    axs.add_collection3d(
        Poly3DCollection(faces, facecolors="cyan", linewidths=1, edgecolors="r", alpha=0.25)
    )
    plt.show()
