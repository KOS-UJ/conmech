# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023-2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from conmech.dynamics.contact.contact_law import PotentialOfContactLaw, DirectContactLaw
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.solvers.optimization.optimization import Optimization

mesh_density = 4
kN = 1000
mm = 0.001
E = 1.378e8 * kN
kappa = 0.3
surface = 5 * mm * 80 * mm
k0 = 30e6 * kN * surface
k10 = 10e6 * kN * surface
k11 = 10e3 * kN * surface
k20 = 5e6 * kN * surface
k21 = 5e3 * kN * surface


class MMLV99(PotentialOfContactLaw, DirectContactLaw):
    @staticmethod
    def potential_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        u_nu = -var_nu
        if u_nu <= 0.0:
            return 0.0
        if u_nu < 0.5 * mm:
            return k0 * u_nu**2
        if u_nu < 1.0 * mm:
            return k10 * u_nu**2 + k11 * u_nu
        if u_nu < 2.0 * mm:
            return k20 * u_nu**2 + k21 * u_nu + 4
        return 16.0

    @staticmethod
    def subderivative_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        u_nu = -var_nu
        if u_nu <= 0.0:
            return 0.0
        if u_nu < 0.5 * mm:
            return k0 * u_nu * 2
        if u_nu < 1.0 * mm:
            return k10 * (u_nu * 2) + k11
        if u_nu < 2.0 * mm:
            return k20 * (u_nu * 2) + k21
        return 0.0

    @staticmethod
    def sub2derivative_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        u_nu = -var_nu

        acc = 0.0
        p1 = 0.5 * mm
        if u_nu <= p1:
            return acc

        acc += (k0 * p1 * 2) - (k10 * (p1 * 2) + k11)
        p2 = 1.0 * mm
        if u_nu < p2:
            return acc

        acc += (k10 * (p2 * 2) + k11) - (k20 * (p2 * 2) + k21)
        p3 = 2.0 * mm
        if u_nu < p3:
            return acc

        acc += (k20 * (p3 * 2) + k21) - 0.0
        return acc

    @staticmethod
    def potential_tangential_direction(
        var_tau: float, static_displacement_tau: float, dt: float
    ) -> float:
        return np.log(np.sum(var_tau**2) ** 0.5 + 1)

    @staticmethod
    def subderivative_tangential_direction(
        var_tau: float, static_displacement_tau: float, dt: float
    ) -> float:
        quadsum = np.sum(var_tau**2)
        norm = quadsum**0.5
        denom = norm + quadsum
        coef = 1 / denom if denom != 0.0 else 0.0
        return var_tau * coef


@dataclass()
class StaticSetup(StaticDisplacementProblem):
    grid_height: ... = 10 * mm
    elements_number: ... = (mesh_density, 8 * mesh_density)
    mu_coef: ... = (E * surface) / (2 * (1 + kappa))
    la_coef: ... = ((E * surface) * kappa) / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = MMLV99()

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 5])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0.0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config, methods, forces):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    PREFIX = "BBO"
    if config.force:
        to_simulate = [(m, f) for m in methods for f in forces]
    else:
        to_simulate = []
        for m in methods:
            for f in forces:
                try:
                    path = f"{config.outputs_path}/{PREFIX}_mtd_{m}_frc_{f:.2e}"
                    with open(path, "rb") as output:
                        _ = pickle.load(output)
                except IOError as ioe:
                    print(ioe)
                    to_simulate.append((m, f))

    mesh_descr = RectangleMeshDescription(
        initial_position=None,
        max_element_perimeter=0.25 * 10 * mm,
        scale=[8 * 10 * mm, 10 * mm],
    )

    if to_simulate:
        print("Simulating...")
        setup = StaticSetup(mesh_descr=mesh_descr)

        losses = {}
        for method, force in to_simulate:
            print(method, force)
            m_loss = losses.get(method, {})
            losses[method] = m_loss

            def outer_forces(x, t=None):
                if x[1] >= 0.0099:
                    return np.array([0, force * surface])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticSolver(setup, "global")
            validator = StaticSolver(setup, "global")

            start = time.time()
            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                method=method,
            )
            m_loss[force] = loss_value(state, validator), time.time() - start
            path = f"{config.outputs_path}/{PREFIX}_mtd_{method}_frc_{force:.2e}"
            with open(path, "wb+") as output:
                state.body.dynamics.force.outer.source = None
                state.body.dynamics.force.inner.source = None
                state.body.properties.relaxation = None
                state.setup = None
                state.constitutive_law = None
                pickle.dump(state, output)
        path = f"{config.outputs_path}/{PREFIX}_losses"
        with open(path, "wb+") as output:
            pickle.dump(losses, output)

    print("Plotting...")
    path = f"{config.outputs_path}/{PREFIX}_losses"
    # if config.show:
    plot_losses(path)

    for m in methods:
        for f in forces:
            path = f"{config.outputs_path}/{PREFIX}_mtd_{m}_frc_{f:.2e}"
            with open(path, "rb") as output:
                state = pickle.load(output)

            drawer = Drawer(state=state, config=config)
            drawer.colorful = True
            drawer.draw(
                show=config.show,
                save=config.save,
                # title=f"{m}: {f}, "
                # f"time: {runner.step_solver.last_timing}"
            )

            x = state.body.mesh.nodes[: state.body.mesh.contact_nodes_count - 1, 0]
            u = state.displacement[: state.body.mesh.contact_nodes_count - 1, 1]
            y1 = [MMLV99().subderivative_normal_direction(-u_, 0.0, 0.0) for u_ in u]
            plt.plot(x, y1, label=f"{f:.2e}")
        plt.ylabel("Interlaminar binding force [kN/m$^2$]")
        plt.xlabel(r"Contact interface [mm]")
        plt.grid()
        plt.title(m)
        plt.legend()
        plt.show()


def plot_losses(path, slopes=None):
    print("Plotting...")
    with open(path, "rb") as output:
        losses = pickle.load(output)
    data = {
        "BFGS": (losses["BFGS"], "0.7"),
        "BFGS (subgrad)": (losses["gradiented BFGS"], "0.6"),
        "CG": (losses["CG"], "0.4"),
        "CG (subgrad)": (losses["gradiented CG"], "0.3"),
        "Powell": (losses["Powell"], "blue"),
        "subgradient": (losses["qsm"], "pink"),
        "global subgradient": (losses["globqsm"], "red"),
    }

    # Grid of plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    styles = [
        "-",
        "--",
        "-.",
        ":",
        "-.",
        "--",
        "-",
    ]

    # loss
    for i, (method, (results, color)) in enumerate(data.items()):
        forces = np.array(list(results.keys())) / 1e6  # N/m2 to MPa
        losses = -np.array([val[0] for val in results.values()])
        axes[0].plot(
            forces,
            losses,
            linestyle=styles[i % len(styles)],
            marker="o",
            label=method,
            linewidth=2,
            color=color,
        )

    axes[0].set_ylabel("$-\mathcal{L}(u)$", fontsize=14)
    axes[0].set_xlabel("Load [MPa]", fontsize=14)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10, loc="lower right")
    # axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    if slopes is not None:
        fig.suptitle(f"Num. of layers: {slopes + 2}")

    for i, (method, (results, color)) in enumerate(data.items()):
        forces = np.array(list(results.keys())) / 1e6
        times = np.array([val[1] for val in results.values()])

        best_losses = {
            force: min(data[m][0][force][0] for m in data if force in data[m][0])
            for force in results.keys()
        }

        markers = []
        for force, (loss, _) in results.items():
            best_loss = best_losses[force]
            if loss == best_loss:  # best solution
                marker = "s"
            elif loss <= best_loss * 0.9 + 0.01:  # near to solution
                marker = "o"
            else:  # fail
                marker = "x"
            markers.append(marker)

        (line,) = axes[1].plot(
            forces,
            times,
            linestyle=styles[i % len(styles)],
            linewidth=2,
            label=method,
            alpha=0.7,
            color=color,
        )
        color = line.get_color()  # line color
        for j, force in enumerate(forces):
            axes[1].scatter(force, times[j], marker=markers[j], color=color, s=70)

    axes[1].set_ylabel("Computation Time [s]", fontsize=14)
    axes[1].set_xlabel("Load [MPa]", fontsize=14)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_yscale("log")
    # axes[1].legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(path + ".pdf")
    plt.show()

    return
    for i, (mtd, values) in enumerate(losses.items()):
        forces_ = np.asarray(list(values.keys()))
        values_ = np.asarray(list(values.values()))[:, 0]
        times_ = np.asarray(list(values.values()))[:, 1]
        # plt.plot(forces_ / 1e3, -1 * values_, "-o", label=mtd, linewidth=3*(len(losses.items()) - i))
        total_width = 300
        width = total_width / len(losses.items())
        shift = total_width / 2 + i * width + width / 2
        plt.bar(forces_ / 1e3 + shift, -1 * values_, label=mtd, width=width)
    plt.legend()
    plt.ylabel("$-\mathcal{L}(u)$")
    plt.xlabel(r"Load [kN/m$^2$]")
    plt.grid()
    plt.show()
    for mtd, values in losses.items():
        forces_ = np.asarray(list(values.keys()))
        values_ = np.asarray(list(values.values()))[:, 0]
        times_ = np.asarray(list(values.values()))[:, 1]
        plt.plot(forces_ / 1e3, times_, "-o", label=mtd)
    plt.legend()
    plt.ylabel("$time$")
    plt.xlabel(r"Load [kN/m$^2$]")
    plt.grid()
    plt.show()


def loss_value(state, runner) -> float:
    initial_guess = state["displacement"].T.ravel().reshape(state.body.mesh.dimension, -1)
    solution = np.squeeze(initial_guess.copy().reshape(1, -1))
    self: Optimization = runner.step_solver
    args = (
        np.zeros_like(solution),  # variable_old
        self.body.mesh.nodes,
        self.body.mesh.contact_boundary,
        self.body.mesh.boundaries.contact_normals,
        self.lhs,
        self.rhs,
        np.zeros_like(solution),  # displacement
        np.ascontiguousarray(self.body.dynamics.acceleration_operator.SM1.data),
        self.time_step,
    )
    result = self.loss(solution, *args)[0]
    print(result)
    return result


if __name__ == "__main__":
    X = np.linspace(0, -3 * mm, 1000)
    Y = np.empty(1000)
    for i in range(1000):
        Y[i] = MMLV99.potential_normal_direction(X[i], 0, 0)
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = MMLV99.subderivative_normal_direction(X[i], 0, 0)
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = MMLV99.sub2derivative_normal_direction(X[i], 0, 0)
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = MMLV99.subderivative_normal_direction(
            X[i], 0, 0
        ) + MMLV99.sub2derivative_normal_direction(X[i], 0, 0)
    plt.plot(X, Y)
    plt.show()

    forces = np.asarray(
        (
            20e3 * kN,
            21e3 * kN,
            22e3 * kN,
            23e3 * kN,
            25e3 * kN,
            26e3 * kN,
            26.2e3 * kN,
            27e3 * kN,
            # 30e3 * kN,
        )
    )[:]

    methods = (
        "gradiented BFGS",
        "gradiented CG",
        "BFGS",
        "CG",
        "Powell",
        "globqsm",
        "qsm",
        "dc globqsm",
        "dc qsm",
    )[-4:]
    main(Config(save=False, show=False, force=True).init(), methods, forces)
