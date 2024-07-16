# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import ContactLaw, StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver as StaticProblemSolver
from conmech.properties.mesh_description import BOST2023MeshDescription
from examples.error_estimates import error_estimates

GAP = 0.0
E = 12000
kappa = 0.42


class BOST23(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def potential_tangential_direction(u_tau: np.ndarray) -> float:
        return np.sum(u_tau * u_tau) ** 0.5

    @staticmethod
    def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
        return 0

    @staticmethod
    def regularized_subderivative_tangential_direction(
        u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7
    ) -> float:
        """
        Coulomb regularization
        """
        return 0


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = E / (1 + kappa)
    la_coef: ... = E * kappa / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = BOST23

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([-0.2, -0.8]) * 1e3

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        if u_nu < 0:
            return 0
        return u_nu

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def prepare_setup(ig, setup):
    if ig == "inf":

        def potential_normal_direction(u_nu: float) -> float:
            return 0

        kwargs = {"method": "constrained"}
    else:

        def potential_normal_direction(u_nu: float) -> float:
            u_nu -= GAP
            # EXAMPLE 10
            a = 0.1
            b = 0.1
            if u_nu <= 0:
                result = 0.0
            elif u_nu < b:
                result = (a + np.exp(-b)) / (2 * b) * u_nu**2
            else:
                result = a * u_nu - np.exp(-u_nu) + ((b + 2) * np.exp(-b) - a * b) / 2
            return ig * result

        kwargs = {"method": "POWELL"}

    setup.contact_law.potential_normal_direction = potential_normal_direction
    return kwargs


def main(
        config: Config,
        igs: Optional[List[int]] = None,
        highlighted: Optional[Iterable] = None
):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    PREFIX = "BOST"
    to_simulate = []
    igs = igs or [1024, "inf"]
    highlighted = highlighted or igs
    for ig in igs:
        try:
            with open(f"{config.outputs_path}/{PREFIX}_ig_{ig}", "rb") as output:
                _ = pickle.load(output)
        except IOError as ioe:
            print(ioe)
            to_simulate.append(ig)
    if config.force:
        to_simulate = igs

    mesh_descr = BOST2023MeshDescription(
        initial_position=None,
        max_element_perimeter=1 / 32 if not config.test else 1 / 6
    )

    if to_simulate:
        print("Simulating...")
        print(to_simulate)
        setup = StaticSetup(mesh_descr)

        if to_simulate[0] == igs[0]:
            initial_displacement = setup.initial_displacement
        else:
            ig_prev = igs[igs.index(to_simulate[0]) - 1]
            with open(f"{config.outputs_path}/{PREFIX}_ig_{ig_prev}", "rb") as output:
                state = pickle.load(output)
            initial_displacement = lambda _: state.displacement.copy()

        for i, ig in enumerate(to_simulate):
            kwargs = prepare_setup(ig, setup)
            if config.test:
                setup.elements_number = (2, 4)
            runner = StaticProblemSolver(setup, "schur")

            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=initial_displacement,
                **kwargs,
            )

            if i + 1 < len(to_simulate):
                ig_prev = igs[igs.index(to_simulate[i + 1]) - 1]
                if ig == ig_prev:
                    initial_displacement = setup.initial_displacement
                else:
                    with open(f"{config.outputs_path}/{PREFIX}_ig_{ig_prev}", "rb") as output:
                        state = pickle.load(output)
                    initial_displacement = lambda _: state.displacement.copy()

            state.displaced_nodes[: state.body.mesh.nodes_count, :] = (
                state.body.mesh.nodes[: state.body.mesh.nodes_count, :] + state.displacement[:, :]
            )
            with open(f"{config.outputs_path}/{PREFIX}_ig_{ig}", "wb+") as output:
                state.body.dynamics.force.outer.source = None
                state.body.dynamics.force.inner.source = None
                state.body.properties.relaxation = None
                state.setup = None
                state.constitutive_law = None
                pickle.dump(state, output)

    print(f"Plotting {igs=}")
    for i, ig in enumerate(igs):
        with open(f"{config.outputs_path}/{PREFIX}_ig_{ig}", "rb") as output:
            state = pickle.load(output)

        state.displaced_nodes[:, 1] += GAP
        state.body.mesh.nodes[:, 1] += GAP

        if ig in highlighted:
            drawer = Drawer(state=state, config=config)
            drawer.colorful = False
            drawer.outer_forces_scale = 1
            if ig == 0:
                title = r"$\lambda^{-1} = 0.000$"
            else:
                value = f"{float(np.log10(ig)):.3f}" if isinstance(ig, int) else str(ig)
                title = r"$\log_{10}\lambda^{-1} = $" + value
            if config.save:
                save = str(ig)
            else:
                save = False
            drawer.draw(show=config.show, save=save, title=title)

    print("Error estimate")
    errors = error_estimates(
        f"{config.outputs_path}/{PREFIX}_ig_inf",
        *[f"{config.outputs_path}/{PREFIX}_ig_{ig}" for ig in igs],
    )
    X = [ig for ig in igs if not isinstance(ig, str)]
    Y = list(errors.values())[:-1]
    Y = [v[0] for v in Y]
    Y = -np.asarray(Y)
    plot_errors(X, Y, highlighted_id=None, save=f"{config.outputs_path}/convergence.pdf")


def plot_errors(X, Y, highlighted_id, save: Optional[str] = None):
    plt.plot(X[:], Y[:], marker="o", color="gray")
    if highlighted_id is not None:
        plt.plot(X[highlighted_id], Y[highlighted_id], "ro", color="black")
    plt.loglog()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.xlabel(r"$\lambda^{-1}_{\,\, n}$")
    plt.ylabel(r"$||\mathbf{u}^h_n - \mathbf{u}||$")
    if save is None:
        plt.show()
    else:
        plt.savefig(save, format="pdf")


if __name__ == "__main__":
    highlighted = (1, 1024, 1216, 2048, 5931641, "inf")

    show = True

    igs = {int(2 ** (i / 2)) for i in range(0, 48)}
    eigs = {
        1024 - 128,
        1024 + 64,
        1024 + 96,
        1024 + 128,
        1024 + 128 + 64,
    }
    igs.update(eigs)
    igs = [0] + sorted(list(igs)) + ["inf"]  #
    igs = igs[:]
    main(Config(save=not show, show=show, force=False).init(), igs, highlighted)
