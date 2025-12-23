# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2025-2026  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.tri as tri

from conmech.simulations.problem_solver import PoissonSolver
from conmech.properties.mesh_description import CrossMeshDescription
from examples.example_poisson import StaticPoissonSetup


def run_simulation(element_size, label):
    print(f"{label}: (element size ~{element_size})...")
    mesh_descr = CrossMeshDescription(
        initial_position=None,
        max_element_perimeter=element_size,
        scale=[1, 1]
    )
    setup = StaticPoissonSetup(mesh_descr)
    runner = PoissonSolver(setup, "direct")

    state = runner.solve(verbose=False)
    return state


def main():
    coarse_size = 0.25
    state_coarse = run_simulation(coarse_size, "Coarse Mesh")

    fine_size = 0.04
    state_fine = run_simulation(fine_size, "Reference Mesh")

    nodes_c = state_coarse.body.mesh.nodes
    vals_c = state_coarse.temperature.flatten()

    nodes_f = state_fine.body.mesh.nodes
    vals_f = state_fine.temperature.flatten()

    vals_c_interp = griddata(nodes_c, vals_c, nodes_f, method='linear')

    mask = ~np.isnan(vals_c_interp)

    # error(x) = | u_exact(x) - u_approx(x) |
    local_error = np.zeros_like(vals_f)
    local_error[mask] = np.abs(vals_f[mask] - vals_c_interp[mask])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    triang_c = tri.Triangulation(nodes_c[:, 0], nodes_c[:, 1], state_coarse.body.mesh.elements)
    triang_f = tri.Triangulation(nodes_f[:, 0], nodes_f[:, 1], state_fine.body.mesh.elements)

    tc1 = axes[0].tripcolor(triang_c, vals_c, cmap='viridis', shading='gouraud')
    axes[0].triplot(triang_c, 'k-', alpha=0.3, linewidth=0.5)
    axes[0].set_title(f"Coarse mesh, h={coarse_size}")
    plt.colorbar(tc1, ax=axes[0])
    axes[0].set_aspect('equal')

    tc2 = axes[1].tripcolor(triang_f, vals_f, cmap='viridis', shading='gouraud')
    axes[1].set_title(f"Fine mesh, h={fine_size}")
    plt.colorbar(tc2, ax=axes[1])
    axes[1].set_aspect('equal')

    tc3 = axes[2].tripcolor(triang_f, local_error, cmap='inferno', shading='gouraud')
    axes[2].set_title("\n$|u_{ref} - u_{h}|$")
    cb3 = plt.colorbar(tc3, ax=axes[2])
    cb3.set_label('Error magnitude')
    axes[2].set_aspect('equal')

    threshold = 0.3 * np.max(local_error)
    axes[2].tricontour(triang_f, local_error, levels=[threshold], colors='white',
                       linestyles='dashed', linewidths=2)
    axes[2].text(0.5, -0.15,
                 "In white dashed line h-adaptive refinement needed",
                 ha='center', transform=axes[2].transAxes, fontsize=10, style='italic')

    plt.suptitle("Mesh refinement potential zones", fontsize=16)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
