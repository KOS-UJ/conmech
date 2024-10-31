"""
Created at 21.08.2019
"""

import pickle
import time
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import ContactLaw, StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from examples.Makela_et_al_1998 import loss_value, plot_losses

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
k30 = 2.5e12 * kN * surface


class MMLV99(ContactLaw):
    @staticmethod
    def potential_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        if var_nu <= 0:
            return 0.0
        if var_nu < 0.5 * mm:
            return k0 * var_nu**2
        if var_nu < 1 * mm:
            return k10 * var_nu**2 + k11 * var_nu
        if var_nu < 2 * mm:
            return k20 * var_nu**2 + k21 * var_nu + 4
        return var_nu**4 * k30

    @staticmethod
    def potential_tangential_direction(
        var_tau: float, static_displacement_tau: float, dt: float
    ) -> float:
        return np.log(np.sum(var_tau**2) ** 0.5 + 1)

    @staticmethod
    def subderivative_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        if var_nu <= 0:
            return 0.0
        if var_nu < 0.5 * mm:
            return k0 * var_nu * 2
        if var_nu < 1 * mm:
            return k10 * (var_nu * 2) + k11
        if var_nu < 2 * mm:
            return k20 * (var_nu * 2) + k21
        return var_nu**3 * 4 * k30

    @staticmethod
    def sub2derivative_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        acc = 0.0
        p1 = 0.5 * mm
        if var_nu <= p1:
            return acc

        acc += (k0 * p1 * 2) - (k10 * (p1 * 2) + k11)
        p2 = 1 * mm
        if var_nu < p2:
            return acc

        acc += (k10 * (p2 * 2) + k11) - (k20 * (p2 * 2) + k21)
        p3 = 2 * mm
        if var_nu < p3:
            return acc

        return acc


@dataclass()
class StaticSetup(StaticDisplacementProblem):
    grid_height: ... = 10 * mm
    elements_number: ... = (mesh_density, 8 * mesh_density)
    mu_coef: ... = (E * surface) / (2 * (1 + kappa))
    la_coef: ... = ((E * surface) * kappa) / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = MMLV99

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
    PREFIX = "DBBO"
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
                    return np.array([0, -1 * force * surface])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticSolver(setup, "schur")
            validator = StaticSolver(setup, "global")

            start = time.time()
            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                method=method,
                maxiter=100,
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
    if config.show:
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
                title=f"{m}: {f}, ",
                # f"time: {runner.step_solver.last_timing}"
            )
            x = state.body.mesh.nodes[: state.body.mesh.contact_nodes_count - 1, 0]
            u = state.displacement[: state.body.mesh.contact_nodes_count - 1, 1]
            y1 = [MMLV99().subderivative_normal_direction(-u_, 0.0, 0.0) for u_ in u]
            print(f)
            plt.plot(x, y1, label=f"{f:.2e}")
        plt.title(m)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    X = np.linspace((2 - 2) * mm, (2 + 2) * mm, 1000)
    Y = np.empty(1000)
    for i in range(1000):
        Y[i] = MMLV99.potential_normal_direction(X[i], 0.0, 0.0)
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = MMLV99().subderivative_normal_direction(X[i], 0.0, 0.0)
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = MMLV99().sub2derivative_normal_direction(X[i], 0.0, 0.0)
    plt.plot(X, Y)
    plt.show()

    methods = ("gradiented BFGS", "gradiented CG", "BFGS", "CG", "qsm", "Powell", "globqsm")[:]
    forces = (
        23e3 * kN,
        25e3 * kN,
        25.6e3 * kN,
        25.9e3 * kN,
        26e3 * kN,
        26.1e3 * kN,
        26.2e3 * kN,
        27e3 * kN,
        30e3 * kN,
    )[:]
    main(Config(save=False, show=False, force=False).init(), methods, forces)
