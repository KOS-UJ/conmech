"""
Created at 21.10.2022
"""
import pickle
from dataclasses import dataclass

import numba
import numpy as np
from matplotlib.colors import Normalize
import scipy.interpolate
import matplotlib.pyplot as plt

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import ContactLaw, Quasistatic
from conmech.simulations.problem_solver import TimeDependent as QuasistaticProblemSolver


def make_contact_law(limit_value, limit):
    class JureczkaOchalBartman2023(ContactLaw):
        @staticmethod
        def potential_normal_direction(u_nu: float) -> float:
            if u_nu <= 0:
                return 0.0
            if u_nu < limit:
                return limit_value * u_nu
            return limit_value * limit

        @staticmethod
        def potential_tangential_direction(u_tau: np.ndarray) -> float:
            return -0.3 * np.exp(-np.sum(u_tau * u_tau) ** 0.5) + 0.7 * np.sum(u_tau * u_tau) ** 0.5
            # return np.log(np.sum(u_tau * u_tau) ** 0.5 + 1)\

        @staticmethod
        def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
            return 0

        @staticmethod
        def regularized_subderivative_tangential_direction(
            u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7
        ) -> float:
            return 0

    return JureczkaOchalBartman2023


x1 = 0.15
x2 = 1.05
y1 = 0.15
y2 = 0.45
r = 0.05
eps = 0.01


def make_setup(mesh_type_, boundaries_, contact_law_, elements_number_, friction_bound_):
    @dataclass()
    class QuasistaticSetup(Quasistatic):
        grid_height: ... = 1.0
        elements_number: ... = elements_number_
        mu_coef: ... = 40
        la_coef: ... = 100
        th_coef: ... = 40
        ze_coef: ... = 100
        time_step: ... = 1 / 128
        contact_law: ... = contact_law_

        @staticmethod
        def inner_forces(x):
            return np.array([0.0, -1.0])

        @staticmethod
        def outer_forces(x):
            if x[1] == 0.6:
                return np.array([-4.0, -4.0])
            if x[0] == 1.2:
                return np.array([-4.0, -4.0])
            return np.array([0.0, 0.0])

        @staticmethod
        def friction_bound(u_nu: float) -> float:
            n = friction_bound_
            b = 0.1
            if u_nu <= 0:
                return 0.0
            if u_nu < b:
                return n * u_nu
            return n * b

        boundaries: ... = boundaries_

    return QuasistaticSetup(mesh_type=mesh_type_)


def get_interpolated(u, vertices):
    x = vertices[:, 0].copy()
    y = vertices[:, 1].copy()
    u_x = u[:, 0].copy()
    u_y = u[:, 1].copy()
    u_fun_x = scipy.interpolate.LinearNDInterpolator(list(zip(x, y)), u_x)
    u_fun_y = scipy.interpolate.LinearNDInterpolator(list(zip(x, y)), u_y)
    return u_fun_x, u_fun_y


@numba.njit()
def calculate_dx_dy(x0, u0, x1, u1, x2, u2):
    a1 = x1[0] - x0[0]
    b1 = x1[1] - x0[1]
    c1 = u1 - u0
    a2 = x2[0] - x0[0]
    b2 = x2[1] - x0[1]
    c2 = u2 - u0
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    dx = a / c
    dy = b / c
    return dx, dy


def gradient(elements, initial_nodes, f):
    result = np.zeros((len(f), 2))
    norm = np.zeros(len(f))
    for element in elements:
        x0 = initial_nodes[element[0]]
        x1 = initial_nodes[element[1]]
        x2 = initial_nodes[element[2]]
        f0 = f[element[0]]
        f1 = f[element[1]]
        f2 = f[element[2]]
        dx, dy = calculate_dx_dy(x0, f0, x1, f1, x2, f2)
        result[element[0], 0] += dx
        result[element[0], 1] += dy
        norm[element[0]] += 1
        result[element[1], 0] += dx
        result[element[1], 1] += dy
        norm[element[1]] += 1
        result[element[2], 0] += dx
        result[element[2], 1] += dy
        norm[element[2]] += 1

    result[:, 0] /= norm
    result[:, 1] /= norm

    return result


def constitutive_law(u, v, setup, elements, nodes):
    grad_x = gradient(elements, nodes, u[:, 0])
    grad_y = gradient(elements, nodes, u[:, 1])
    grad_u = np.concatenate((grad_x, grad_y), axis=1).reshape(-1, 2, 2)

    stress_u = np.zeros_like(grad_u)
    stress_u[:, 0, 0] = 2 * setup.mu_coef * grad_u[:, 0, 0] + setup.la_coef * (
        grad_u[:, 0, 0] + grad_u[:, 1, 1]
    )
    stress_u[:, 1, 1] = 2 * setup.mu_coef * grad_u[:, 1, 1] + setup.la_coef * (
        grad_u[:, 0, 0] + grad_u[:, 1, 1]
    )
    stress_u[:, 0, 1] = setup.mu_coef * (grad_u[:, 0, 1] + grad_u[:, 1, 0])
    stress_u[:, 1, 0] = stress_u[:, 0, 1]

    grad_x = gradient(elements, nodes, v[:, 0])
    grad_y = gradient(elements, nodes, v[:, 1])
    grad_v = np.concatenate((grad_x, grad_y), axis=1).reshape(-1, 2, 2)

    stress_v = np.zeros_like(grad_v)
    stress_v[:, 0, 0] = 2 * setup.th_coef * grad_v[:, 0, 0] + setup.ze_coef * (
        grad_v[:, 0, 0] + grad_v[:, 1, 1]
    )
    stress_v[:, 1, 1] = 2 * setup.th_coef * grad_v[:, 1, 1] + setup.ze_coef * (
        grad_v[:, 0, 0] + grad_v[:, 1, 1]
    )
    stress_v[:, 0, 1] = setup.th_coef * (grad_v[:, 0, 1] + grad_v[:, 1, 0])
    stress_v[:, 1, 0] = stress_v[:, 0, 1]
    return stress_u + stress_v


def main(show: bool = True, save: bool = False):
    simulate = False
    config = Config()
    names = ("four_screws",)  # "one_screw", "friction", "hard")
    h = 64
    n_steps = 32
    output_steps = range(0, n_steps)

    four_screws = BoundariesDescription(
        contact=lambda x: x[1] == 0,
        dirichlet=lambda x: (
            (x[0] - x1) ** 2 + (x[1] - y1) ** 2 <= (r + eps) ** 2
            or (x[0] - x1) ** 2 + (x[1] - y2) ** 2 <= (r + eps) ** 2
            or (x[0] - x2) ** 2 + (x[1] - y1) ** 2 <= (r + eps) ** 2
            or (x[0] - x2) ** 2 + (x[1] - y2) ** 2 <= (r + eps) ** 2
        ),
    )
    one_screw = BoundariesDescription(
        contact=lambda x: x[1] == 0,
        dirichlet=lambda x: (x[0] - x1) ** 2 + (x[1] - y1) ** 2 <= (r + eps) ** 2,
    )
    soft_foundation = make_contact_law(300, 0.1)
    hard_foundation = make_contact_law(3000, 0.1)
    friction_bound = 3

    if simulate:
        for name in names:
            boundaries = four_screws
            contact_law = soft_foundation()
            if name != "four_screws":
                boundaries = one_screw
            if name == "hard":
                contact_law = hard_foundation()
            if name == "friction":
                friction_bound = 300
            setup = make_setup(
                mesh_type_="bow",
                boundaries_=boundaries,
                contact_law_=contact_law,
                elements_number_=(h, h),
                friction_bound_=friction_bound,
            )
            runner = QuasistaticProblemSolver(setup, "schur")
            states = runner.solve(
                n_steps=n_steps,
                output_step=output_steps,
                verbose=True,
                # fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                initial_velocity=setup.initial_velocity,
            )
            for state in states:
                with open(
                    f"./output/2023/{name}_t_{int(state.time//setup.time_step)}_h_{h}",
                    "wb+",
                ) as output:
                    pickle.dump(state, output)

    for name in names:
        boundaries = four_screws
        contact_law = soft_foundation()
        setup = make_setup(
            mesh_type_="bow",
            boundaries_=boundaries,
            contact_law_=contact_law,
            elements_number_=(h, h),
            friction_bound_=friction_bound,
        )
        if name == names[0]:
            steps = (0, *output_steps[1:])
        else:
            steps = output_steps[1:]
        for time_step in steps:
            with open(f"./output/2023/{name}_t_{time_step}_h_{h}", "rb") as output:
                state = pickle.load(output)
            if time_step == 0:
                drawer = Drawer(state=state, config=config)
                drawer.node_size = 1
                drawer.original_mesh_color = "k"
                drawer.deformed_mesh_color = None
                drawer.draw(show=show, temp_min=0, temp_max=40, save=save)
            stress = constitutive_law(
                state.displacement,
                state.velocity,
                setup,
                state.body.mesh.elements,
                state.body.mesh.initial_nodes,
            )
            c = np.linalg.norm(stress, axis=(1, 2))
            state.temperature = c  # stress[:, 0, 1]
            drawer = Drawer(state=state, config=config)
            drawer.node_size = 0
            drawer.original_mesh_color = None
            drawer.deformed_mesh_color = None
            drawer.cmap = plt.cm.rainbow
            drawer.draw(show=show, temp_min=0, temp_max=30, save=True)


if __name__ == "__main__":
    main(show=False)
