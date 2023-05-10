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
        if u_nu >= 0:
            return 0.0
        if u_nu > -0.5e-6:
            return (30e6) * u_nu ** 2
        if u_nu > -1e-3:
            return (10e6) * (u_nu ** 2 + u_nu) - 4995
        if u_nu > -2e-3:
            return (5e6) * (u_nu ** 2 + u_nu) + 10
        return 10030

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


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 0.01
    elements_number: ... = (mesh_density, 8 * mesh_density)
    mu_coef: ... = (1.378e8) / (2 * (1 + 0.3))
    la_coef: ... = ((1.378e8) * 0.3) / ((1 + 0.3) * (1 - 2 * 0.3))
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
        for force in np.arange(80000, 200000 + 1, 10000):
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
    main(save=True)
