"""
Created at 21.08.2019
"""

from dataclasses import dataclass, field

import numpy as np

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import PiezoelectricDynamicProblem
from conmech.simulations.problem_solver import PiezoelectricTimeDependentSolver
from conmech.properties.mesh_description import Barboteu2008MeshDescription
from examples.p_slope_contact_law import make_slope_contact_law


# TODO # 48
class PPSlopeContactLaw(make_slope_contact_law(slope=1e1)):
    @staticmethod
    def h_nu(uN, t):
        raise NotImplementedError()

    @staticmethod
    def h_tau(uN, t):
        raise NotImplementedError()

    @staticmethod
    def electric_charge_tangetial(u_tau):  # potential
        return 0 * 0.1 * 0.5 * np.linalg.norm(u_tau) ** 2

    @staticmethod
    def electric_charge_flux(charge):
        return 0 * charge


@dataclass()
class PDynamicSetup(PiezoelectricDynamicProblem):
    mu_coef: ... = 45
    la_coef: ... = 105
    th_coef: ... = 4.5
    ze_coef: ... = 10.5
    time_step: ... = 0.01
    contact_law: ... = PPSlopeContactLaw
    piezoelectricity: ... = field(
        default_factory=lambda: np.array(
            [
                [[0.0, -0.59, 0.0], [-0.61, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-0.59, 0.0, 0.0], [0.0, 1.14, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
    )
    permittivity: ... = field(
        default_factory=lambda: np.array([[8.3, 0.0, 0.0], [0.0, 8.8, 0.0], [0.0, 0.0, -8]])
    )

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.0])

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: 0.0 <= x[0] <= 1.0 and 0.0 <= x[1] <= 1.0,
        dirichlet=lambda x: 1.0 <= x[0] <= 1.5 and 1.0 <= x[1] <= 1.5,
        piezo_dirichlet_0=(
            lambda x: (x[0] == 1.0 and 1.0 <= x[1] <= 4.0),
            lambda x: np.full(x.shape[0], 0),
        ),
        piezo_dirichlet_1=(
            lambda x: (1.5 <= x[0] <= 3.0 and x[1] == 1.0),
            lambda x: np.full(x.shape[0], 20),
        ),
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = Barboteu2008MeshDescription(initial_position=None, max_element_perimeter=0.25)
    setup = PDynamicSetup(mesh_descr)
    runner = PiezoelectricTimeDependentSolver(setup, solving_method="global")

    steps = 100 if not config.test else 5
    output = steps // 5
    states = runner.solve(
        n_steps=steps,
        output_step=range(0, steps + 1, output),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_electric_potential=setup.initial_electric_potential,
    )
    T_max = -np.inf
    T_min = np.inf
    for state in states:
        T_max = max(T_max, np.max(state.electric_potential))
        T_min = min(T_min, np.min(state.electric_potential))
    for state in states:
        Drawer(state=state, config=config).draw(
            field_max=T_max, field_min=T_min, show=config.show, save=config.save
        )


if __name__ == "__main__":
    main(Config().init())
