"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import ContactLaw, Static
from conmech.simulations.problem_solver import Static as StaticProblemSolver


class MMLV99(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        # if u_nu >= 0:
        #     return 0.0
        # if u_nu > -0.5e-3:
        #     return (30 / 200) * u_nu ** 2
        # if u_nu > -1e-3:
        #     return (10 / 200) * (u_nu ** 2 + u_nu)
        # if u_nu > -2e-3:
        #     return (5 / 200) * (u_nu ** 2 + u_nu + 2)
        # return (40 / 200)
        u_nu = -u_nu
        if u_nu <= 0:
            return 0.0
        if u_nu < 0.5 * mm:
            return (30e6 * kN * surface) * u_nu ** 2
        if u_nu < 1 * mm:
            return (10e6 * kN * surface) * (u_nu ** 2 + u_nu) - 4995 * surface * 1000
        if u_nu < 2 * mm:
            return (5e6 * kN * surface) * (u_nu ** 2 + u_nu) + 10 * surface * 1000
        return 10030 * surface * 1000

    @staticmethod
    def potential_tangential_direction(u_tau: np.ndarray) -> float:
        return np.log(np.sum(u_tau * u_tau) ** 0.5 + 1)

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


mesh_density = 4
kN = 1000
mm = 0.001
E = 1.378e8 * kN
kappa = 0.3
surface = 5 * mm * 80 * mm * 8


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 10 * mm
    elements_number: ... = (mesh_density, 8 * mesh_density)
    mu_coef: ... = (E) / (2 * (1 + kappa))
    la_coef: ... = ((E) * kappa) / ((1 + kappa) * (1 - 2 * kappa))
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


def main(save: bool = False):
    setup = StaticSetup(mesh_type="cross")

    for method in ("Powell", "BFGS", "qsm")[2:]:
        for force in np.arange(20e3 * kN, 30e3 * kN + 1, 1e3 * kN):
            def outer_forces(x, t=None):
                if x[1] >= 0.0099:
                    return np.array([0, force])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticProblemSolver(setup, "schur")

            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                method=method
            )
            config = Config()
            drawer = Drawer(state=state, config=config)
            drawer.colorful = True
            drawer.draw(show=not save, save=save, title=f"{method}: {force}")


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    X = np.linspace(0, -3 * mm, 1000)
    Y = np.empty(1000)
    for i in range(1000):
        Y[i] = MMLV99.potential_normal_direction(X[i])
    plt.plot(X, Y)
    plt.show()
    main(save=True)
