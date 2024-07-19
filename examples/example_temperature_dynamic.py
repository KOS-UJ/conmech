"""
Created at 21.08.2019
"""

from dataclasses import dataclass, field

import numpy as np

from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import TemperatureDynamicProblem
from conmech.simulations.problem_solver import TemperatureTimeDependentSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law


class TPSlopeContactLaw(PotentialOfContactLaw):
    @staticmethod
    def normal_bound(
            var_nu: float,
            static_displacement_nu: float,
            dt: float
    ) -> float:
        """
        Direction of heat flux
        """
        return - 1.0

    @staticmethod
    def potential_normal_direction(
            var_nu: float,
            static_displacement_nu: float,
            dt: float
    ) -> float:
        """Temperature exchange"""
        return 0.0

    @staticmethod
    def potential_tangential_direction(
            var_tau: float,
            static_displacement_tau: float,
            dt: float
    ) -> float:
        """Friction generated temperature"""
        return 0 * np.linalg.norm(var_tau)


@dataclass()
class TDynamicSetup(TemperatureDynamicProblem):
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.02
    contact_law: ... = make_slope_contact_law(slope=1e1)
    contact_law_2: ... = TPSlopeContactLaw
    thermal_expansion: ... = field(
        default_factory=lambda: np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    )
    thermal_conductivity: ... = field(
        default_factory=lambda: np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    )

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.25])

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, -1.0])

    @staticmethod
    def outer_forces(x, t=None):
        if x[0] == 0:
            return np.array([48.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        if x[0] == 2.5:
            return np.array([-48.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(contact=lambda x: x[1] == 0, dirichlet=lambda x: False)


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.25, scale=[2.5, 1]
    )
    setup = TDynamicSetup(mesh_descr)
    runner = TemperatureTimeDependentSolver(setup, solving_method="schur")
    n_steps = 32 if not config.test else 8

    states = runner.solve(
        n_steps=n_steps,
        output_step=range(0, n_steps, 4),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_temperature=setup.initial_temperature,
    )
    states = list(states)
    T_max = -np.inf
    T_min = np.inf
    for state in states:
        T_max = max(T_max, np.max(state.temperature))
        T_min = min(T_min, np.min(state.temperature))
    for state in states:
        drawer = Drawer(state=state, config=config)
        drawer.cmap = "plasma"
        drawer.field_name = "temperature"
        drawer.draw(field_max=T_max, field_min=T_min, show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
