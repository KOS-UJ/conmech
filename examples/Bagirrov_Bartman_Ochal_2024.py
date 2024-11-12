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

cc = 1.5
mesh_density = 8
kN = 1000
mm = 0.001
E = cc * 1.378e8 * kN
kappa = 0.4
surface = 10 * mm * 100 * mm


def make_composite_contact_law(layers, alpha, beta):
    alphas = np.asarray([alpha(lay) for lay in layers])
    betas = np.asarray([beta(lay) for lay in layers])
    betas[0] = 0
    integ = np.asarray([0.5 * (layers[n+1] - layers[n]) * (betas[n+1] + alphas[n])
                        for n in range(len(layers) - 1)])
    cumm = np.cumsum(integ)
    slope = np.asarray([(betas[n + 1] - alphas[n]) / (layers[n + 1] - layers[n]) for n in range(len(layers) - 1)])
    const = np.asarray([alphas[n] - layers[n] * slope[n] for n in range(len(layers) - 1)])

    class Composite(ContactLaw):

        @staticmethod
        def potential_tangential_direction(
                var_tau: float, static_displacement_tau: float, dt: float
        ) -> float:
            return np.log(np.sum(var_tau ** 2) ** 0.5 + 1)

        @staticmethod
        def potential_normal_direction(
                var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            val = 0.0
            if var_nu <= layers[0]:
                return 0.0
            for n in range(len(layers)):
                if var_nu > layers[n]:
                    continue
                if n > 1:
                    val += cumm[n-2]
                val += 0.5 * (var_nu - layers[n-1]) * (var_nu * slope[n-1] + const[n-1] + alphas[n-1])
                break
            else:
                val = cumm[-1] + alphas[-1] * (var_nu - layers[-1])
            return val

        @staticmethod
        def subderivative_normal_direction(
                var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            if var_nu <= layers[0]:
                return 0.0
            if layers[-1] < var_nu:
                return alphas[-1]
            for n in range(len(layers) - 1):
                if var_nu <= layers[n+1]:
                    return var_nu * slope[n] + const[n]

        @staticmethod
        def sub2derivative_normal_direction(
                var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            acc = 0.0
            if var_nu <= layers[0]:
                return 0.0
            for n in range(len(layers)):
                if var_nu >= layers[n]:
                    acc += max(0, betas[n] - alphas[n])
                    if n > 0:
                        acc += max(0, alphas[n-1] - betas[n])
                else:
                    acc += max(0, - slope[n-1]) * (var_nu - layers[n-1])
                    break
            return acc

    return Composite

@dataclass()
class StaticSetup(StaticDisplacementProblem):
    grid_height: ... = 10 * mm
    elements_number: ... = (mesh_density, 6 * mesh_density)
    mu_coef: ... = (E * surface) / (2 * (1 + kappa))
    la_coef: ... = ((E * surface) * kappa) / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = None

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


def main(config: Config, methods, forces, contact, prefix=""):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    PREFIX = f"BBO{prefix}"
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
        scale=[4 * 10 * mm, 10 * mm],
    )

    if to_simulate:
        print("Simulating...")
        setup = StaticSetup(mesh_descr=mesh_descr, contact_law=contact)

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
            m_loss[force] = loss_value(state, validator), runner.step_solver.computation_time
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

    for m in methods[:]:
        for f in forces[3:4]:
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
            # x = state.body.mesh.nodes[: state.body.mesh.contact_nodes_count - 1, 0]
            # u = state.displacement[: state.body.mesh.contact_nodes_count - 1, 1]
            # y1 = [MMLV99().subderivative_normal_direction(-u_, 0.0, 0.0) for u_ in u]
            # print(f)
            # plt.plot(x, u, label=f"{f:.2e}")
        # plt.title(m)
        # plt.legend()
        # plt.show()


def composite_problem(config, layers_limits, thickness, methods, forces):
    def bottom(x):
        if x >= thickness:
            return 0.0
        return 0.0 #(x / mm) ** 2 / mm

    def top(x):
        # if x >= thickness:
        #     return 0.0
        return (15.0 / mm + (x / mm) / mm) * surface / 400 / mm**2 * cc

    contact = make_composite_contact_law(layers_limits, alpha=bottom, beta=top)

    # X = np.linspace(-thickness / 4, 5 / 4 * thickness, 1000)
    # Y = np.empty(1000)
    # for i in range(1000):
    #     Y[i] = contact.potential_normal_direction(X[i], 0.0, 0.0)
    # plt.plot(X, Y)
    # plt.show()
    # for i in range(1000):
    #     Y[i] = contact.subderivative_normal_direction(X[i], 0.0, 0.0)
    # plt.plot(X, Y)
    # # plt.show()
    # for i in range(1000):
    #     Y[i] = contact.sub2derivative_normal_direction(X[i], 0.0, 0.0)
    # plt.plot(X, Y)
    # plt.show()
    main(config, methods, forces, contact, prefix=str(len(layers_limits)))


def survey(config):
    methods = (
        "gradiented BFGS",
        # "gradiented CG",
        "BFGS",
        # "CG",
        "Powell",
        "qsm",
        # "globqsm",
        # "dc qsm",
        "dc globqsm"
    )[:]
    forces = np.asarray((
        # 23e3 * kN,
        24e3 * kN,
        # 25e3 * kN,
        # 25.5e3 * kN,
        26e3 * kN,
        # 26.5e3 * kN,
        # 27e3 * kN,
        28e3 * kN,
        # 29e3 * kN,
        30e3 * kN,
    )) * cc
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8][4::2]: #range(1, 10 + 1, 2):
        thickness = 3 * mm
        layers_num = i
        # layers_limits = np.logspace(0, thickness, layers_num + 1)
        layers_limits = partition(0, thickness, layers_num + 1, p=1.25)
        composite_problem(config, layers_limits, thickness, methods, forces)


def partition(start, stop, num, p=1.0):
    length = stop - start
    if p == 1:
        grad = num
    else:
        grad = (1 - p**num) / (1 - p)

    first_part = length / grad
    points = [start]
    curr = start
    for i in range(num):
        curr += first_part * (p ** i)
        points.append(curr)

    return np.asarray(points)


if __name__ == "__main__":
    config_ = Config(save=False, show=True, force=True).init()
    survey(config_)
