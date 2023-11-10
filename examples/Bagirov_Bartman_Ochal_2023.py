"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import ContactLaw, StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver

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


def normal_direction(u_nu: float) -> float:
    u_nu = -u_nu
    if u_nu <= 0:
        return 0.0
    if u_nu < 0.5 * mm:
        return k0 * u_nu * 2
    if u_nu < 1 * mm:
        return k10 * (u_nu * 2) + k11
    if u_nu < 2 * mm:
        return k20 * (u_nu * 2) + k21
    return 0


class MMLV99(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        u_nu = -u_nu
        if u_nu <= 0:
            return 0.0
        if u_nu < 0.5 * mm:
            return k0 * u_nu**2
        if u_nu < 1 * mm:
            return k10 * u_nu**2 + k11 * u_nu
        if u_nu < 2 * mm:
            return k20 * u_nu**2 + k21 * u_nu + 4
        return 16

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


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = RectangleMeshDescription(
        initial_position=None,
        max_element_perimeter=0.25 * 10 * mm,
        scale=[8 * 10 * mm, 10 * mm],
    )

    setup = StaticSetup(mesh_descr=mesh_descr)

    for method in ("Powell", "BFGS", "CG", "qsm")[3:]:
        for force in (
            np.asarray([23e3 * kN, 26.2e3 * kN, 27e3 * kN, 30e3 * kN]) * surface
        ):

            def outer_forces(x, t=None):
                if x[1] >= 0.0099:
                    return np.array([0, force])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticSolver(setup, "schur")

            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                method=method,
            )
            drawer = Drawer(state=state, config=config)
            drawer.colorful = True
            drawer.draw(show=config.show, save=config.save, title=f"{method}: {force}")


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    X = np.linspace(0, -3 * mm, 1000)
    Y = np.empty(1000)
    for i in range(1000):
        Y[i] = MMLV99.potential_normal_direction(X[i])
    plt.plot(X, Y)
    plt.show()
    main(Config().init())
