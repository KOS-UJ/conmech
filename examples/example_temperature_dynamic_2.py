"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
import pickle
import matplotlib.pyplot as plt


from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import TemperatureDynamicProblem
from conmech.simulations.problem_solver import (
    TemperatureTimeDependentSolver as TDynamicProblemSolver,
)
from conmech.state.state import TemperatureState
from conmech.properties.mesh_description import CrossMeshDescription
from examples.p_slope_contact_law import make_slope_contact_law
import matplotlib.tri as tri

# TODO #99


def compute_error(ref: TemperatureState, sol: TemperatureState):
    x = sol.body.mesh.initial_nodes[:, 0]
    y = sol.body.mesh.initial_nodes[:, 1]

    soltri = tri.Triangulation(x, y, triangles=sol.body.mesh.elements)
    u1hi = tri.LinearTriInterpolator(soltri, sol.velocity[:, 0])
    u2hi = tri.LinearTriInterpolator(soltri, sol.velocity[:, 1])
    thi = tri.LinearTriInterpolator(soltri, sol.temperature)

    total_abs_error = np.full_like(ref.temperature, fill_value=np.nan)

    for element in ref.body.mesh.elements:
        x0 = ref.body.mesh.initial_nodes[element[0]]
        x1 = ref.body.mesh.initial_nodes[element[1]]
        x2 = ref.body.mesh.initial_nodes[element[2]]
        if total_abs_error[element[0]] != np.nan:
            total_abs_error[element[0]] = (  # abs(ref.velocity[element[0], 0] - u1hi(*x0))
                # + abs(ref.velocity[element[0], 1] - u2hi(*x0))
                +abs(ref.temperature[element[0]] - thi(*x0))
                / ref.temperature[element[0]]
            )
        if total_abs_error[element[1]] != np.nan:
            total_abs_error[element[1]] = (  # abs(ref.velocity[element[1], 0] - u1hi(*x1))
                # + abs(ref.velocity[element[1], 1] - u2hi(*x1))
                +abs(ref.temperature[element[1]] - thi(*x1))
                / ref.temperature[element[1]]
            )
        if total_abs_error[element[2]] != np.nan:
            total_abs_error[element[2]] = (  # abs(ref.velocity[element[2], 0] - u1hi(*x2))
                # + abs(ref.velocity[element[2], 1] - u2hi(*x2))
                +abs(ref.temperature[element[2]] - thi(*x2))
                / ref.temperature[element[2]]
            )

    return total_abs_error


class TPSlopeContactLaw(make_slope_contact_law(slope=1e1)):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        if u_nu <= 0:
            return 0.0
        return (0.5 * 1e3 * u_nu) * u_nu

    @staticmethod
    def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
        if u_nu <= 0:
            return 0 * v_nu
        return (1e3 * u_nu) * v_nu

    @staticmethod
    def potential_tangential_direction(u_tau: np.ndarray) -> float:
        return np.log(np.sum(u_tau * u_tau) ** 0.5 + 1)

    @staticmethod
    def h_nu(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 100.0 * (uN - g_t)
        return 0

    @staticmethod
    def h_tau(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 10.0 * (uN - g_t)
        return 0

    @staticmethod
    def temp_exchange(temp):  # potential  # TODO # 48
        return 0.1 * 0.5 * (temp - 0.27) ** 2

    @staticmethod
    def h_temp(u_tau):  # potential  # TODO # 48
        return 0.1 * 0.5 * u_tau**2


@dataclass()
class TDynamicSetup(TemperatureDynamicProblem):
    mu_coef: ... = 45
    la_coef: ... = 105
    th_coef: ... = 4.5
    ze_coef: ... = 10.5
    time_step: ... = 0.01
    contact_law: ... = TPSlopeContactLaw
    thermal_expansion: ... = np.eye(3) * 0.5

    thermal_conductivity: ... = np.eye(3) * 0.033

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.25])

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, -1])

    @staticmethod
    def outer_forces(x, t=None):
        if x[0] == 0:
            return np.array([48.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        if x[0] == 1.5:
            return np.array([-44.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 1

    boundaries: ... = BoundariesDescription(contact=lambda x: x[1] == 0)


# TODO: #99
def main(steps, setup, config: Config):
    simulate = True
    output_step = (2**i for i in range(int(np.log2(steps))))

    if setup is None:
        mesh_descr = CrossMeshDescription(
            initial_position=None, max_element_perimeter=0.25, scale=[1, 1]
        )
        setup = TDynamicSetup(mesh_descr)
    runner = TDynamicProblemSolver(setup, solving_method="schur")

    # for step, state in zip(output_step, runner.solve(
    #     n_steps=257,
    #     output_step=output_step,
    #     verbose=True,
    #     initial_displacement=setup.initial_displacement,
    #     initial_velocity=setup.initial_velocity,
    #     initial_temperature=setup.initial_temperature,
    # )):
    #     with open(f'output/animation/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}_t_{step}',
    #               'wb') as output:
    #         pickle.dump(state, output)

    # fig, axes = plt.subplots(3, 2)
    # for si, step in enumerate([0, 64, 256]):
    #     with open(f'output/animation/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}_t_{step}',
    #               'rb') as output:
    #         state6 = pickle.load(output)
    #         print(f"k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}")
    #         # Drawer(state=state6, config=config).draw(
    #         #     temp_max=np.max(state6.temperature), temp_min=np.min(state6.temperature), draw_mesh=False, show=show, save=save
    #         # )
    #     with open(f'output/animation/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))-1}_t_{step}',
    #               'rb') as output:
    #         state5 = pickle.load(output)
    #         print(f"k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}")
    #         drawer = Drawer(state=state5, config=config)
    #         drawer.node_size = 0
    #         drawer.draw(#fig_axes=(fig, axes[si, 0]),
    #             temp_max=0.275, temp_min=0.25, draw_mesh=False, show=show, save=False, title=f"t={step / 512}"
    #         )
    #     state_err = state6.copy()
    #     state_err.temperature = compute_error(state6, state5)
    #     drawer = Drawer(state=state_err, config=config, colormap="PuRd")
    #     drawer.node_size = 0
    #     drawer.draw(#fig_axes=(fig, axes[si, 1]),
    #         temp_max=0.0005, temp_min=0, draw_mesh=False, show=show,
    #         save=save, title=f"t={step / 512}"
    #     )
    # plt.show()

    states = runner.solve(
        n_steps=steps // 2 + 1,
        output_step=(0, 4, 8, 16, 32, 64, 128, 256),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_temperature=setup.initial_temperature,
    )
    for state in states:
        # with open(f'output/animation/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}',
        #           'wb') as output:
        #     pickle.dump(state, output)

        # with open(f'output/animation/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}',
        #           'rb') as output:
        #     state = pickle.load(output)
        #     print(f"k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}")
        Drawer(state=state, config=config).draw(
            field_max=np.max(state.temperature),
            field_min=np.min(state.temperature),
            show=config.show,
            save=config.save,
        )


if __name__ == "__main__":
    T = 1
    ks = [2**i for i in [2]]
    hs = [2**i for i in [2]]
    for h in hs:
        for k in ks:
            mesh_descr = CrossMeshDescription(
                initial_position=None, max_element_perimeter=1 / h, scale=[1.5, 1]
            )
            setup = TDynamicSetup(mesh_descr)
            setup.time_step = T / k
            main(setup=setup, steps=k, config=Config().init())
