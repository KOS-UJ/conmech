"""
Created at 21.08.2019
"""

import pickle
import time
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import ContactLaw, StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.state.state import State
from examples.Makela_et_al_1998 import loss_value, plot_losses

cc = 0.7
mesh_density = 20
kN = 1000
mm = 0.001
E = cc * 1.378e8 * kN
kappa = 0.4
surface = 10 * mm * 100 * mm
LENGTH = 210 * mm


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
        LAYERS = layers

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
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0 or x[0] >= LENGTH - 1 * mm
    )


def plot_setup(config: Config, mesh_descr, contact):
    setup = StaticSetup(mesh_descr=mesh_descr, contact_law=contact)
    runner = StaticSolver(setup, "global")
    state = State(runner.body)
    drawer = Drawer(state=state, config=config)
    drawer.node_size = 0
    drawer.colorful = False
    drawer.original_mesh_color = None
    drawer.deformed_mesh_color = "white"
    # drawer.normal_stress_scale = 10
    drawer.field_name = None
    drawer.xlabel = "x"
    drawer.ylabel = "y"

    fig, axes = plt.subplots(1, 1)
    axes = (axes,)
    drawer.outer_forces_scale = 0.35
    drawer.outer_forces_size = 5
    # plt.title("Reference configuration")
    # to have nonzero force interface on Neumann boundary.
    # state.time = 4
    drawer.x_min = 0
    drawer.x_max = 0.2
    drawer.y_min = -0.025
    drawer.y_max = 0.05
    # f_limits = [-1, 2]
    drawer.colorful = False
    x_min = min(state.displaced_nodes[:, 0])
    x_max = max(state.displaced_nodes[:, 0])
    y_min = min(state.displaced_nodes[:, 1])
    y_max = max(state.displaced_nodes[:, 1])
    dirichlet_arrows = np.asarray([
        (x_min, y_min, 1.0, 0.0),
        (x_max, y_min, -1.0, 0.0),
        (x_max, y_max, -1.0, 0.0),
        (x_min, y_max, 1.0, 0.0)])
    drawer.dirichlet_scale = 200
    drawer.dirichlet_size = 4
    drawer.dirichlet_arrows = dirichlet_arrows
    f_x = np.linspace(x_min, x_max, num=40)
    coef = lambda x: 1 - ((x - LENGTH / 2) ** 2) / ((LENGTH / 2) ** 2)
    d_y = np.asarray([-0.01 * coef(x) for x in f_x])
    # f_y_ort = np.asarray([y_max, 0.0, -0.01])
    f_y = np.full_like(f_x, y_max)
    d_x = np.zeros_like(f_x)
    drawer.outer_forces_arrows = np.column_stack((f_x, f_y, d_x, d_y))
    drawer.draw(
        fig_axes=(fig, axes[0]),
        show=False,
        # field_min=f_limits[0],
        # field_max=f_limits[1],
        save="reference.pdf",
    )


def main(config: Config, methods, forces, contact, prefix="", layers_num=None):
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
                        print(path)
                        _ = pickle.load(output)
                except IOError as ioe:
                    print(ioe)
                    to_simulate.append((m, f))

    mesh_descr = RectangleMeshDescription(
        initial_position=None,
        max_element_perimeter=0.25 * 10 * mm,
        scale=[LENGTH, 10 * mm],
    )

    plot_setup(config, mesh_descr, contact)
    # return

    if to_simulate:
        print("Simulating...")
        setup = StaticSetup(mesh_descr=mesh_descr, contact_law=contact)

        losses = {}
        initial_displacement = None
        mtd = None
        for method, force in to_simulate:
            print(method, force)
            if "qsm" not in method or mtd != method:
                initial_displacement = None
                mtd = method
            m_loss = losses.get(method, {})
            losses[method] = m_loss

            def outer_forces(x, t=None):
                if x[1] >= 0.0099:
                    coef = 1 - ((x[0] - LENGTH / 2) ** 2) / ((LENGTH / 2) ** 2)
                    return coef * np.array([0, -1 * force * surface])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticSolver(setup, "schur")
            validator = StaticSolver(setup, "global")

            start = time.time()
            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement if initial_displacement is None else initial_displacement,
                method=method,
                maxiter=100,
            )
            initial_guess = state["displacement"] #.T.ravel().reshape(state.body.mesh.dimension, -1)
            initial_displacement = initial_guess.copy() #np.squeeze(initial_guess.copy().reshape(1, -1))

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
    plot_losses(path, slopes=layers_num)

    colors = ("0.7", "0.6", "0.4", "0.3", "blue", "pink", "red")
    for i, m in enumerate(methods[:]):
        n_ = 5
        for f in forces[n_:n_+1]:
            path = f"{config.outputs_path}/{PREFIX}_mtd_{m}_frc_{f:.2e}"
            with open(path, "rb") as output:
                state = pickle.load(output)

            drawer = Drawer(state=state, config=config)
            # drawer.deformed_mesh_color = "white"
            # drawer.colorful = True
            # drawer.draw(
            #     show=config.show,
            #     save=config.save,
            #     title=f"{m}: {f}, ",
            #     # f"time: {runner.step_solver.last_timing}"
            # )
            # print(state.displaced_nodes[drawer.mesh.contact_indices][:, 1])
            x = state.body.mesh.nodes[: state.body.mesh.contact_nodes_count - 1, 0]
            if m == methods[0]:
                for layer in contact.LAYERS:
                    plt.plot(x * 1000, np.full_like(x, -layer) * 1000, color="skyblue")
            u = state.displacement[: state.body.mesh.contact_nodes_count - 1, 1]
            y1 = [contact.subderivative_normal_direction(-u_, 0.0, 0.0) for u_ in u]
            # print(f)
            if m == "qsm":
                m = "subgradient"
            if m == "globqsm":
                m = "global subgradient"
            plt.plot(x * 1000, u * 1000, label=f"{m}", color=colors[i])


def composite_problem(config, layers_limits, thickness, methods, forces, layers_num, axes=None):
    def bottom(x):
        if x >= thickness:
            return 0.0
        return 0.0

    def top(x):
        return (15.0 / mm + (x / mm) / mm) * surface / 400 / mm**2 * cc

    contact = make_composite_contact_law(layers_limits, alpha=bottom, beta=top)
    if axes is not None:
        X = np.linspace(-thickness / 4, 5 / 4 * thickness, 1000) * 1000  # m to mm
        Y = np.empty(1000)
        T = np.empty(1000)
        for i in range(1000):
            Y[i] = contact.potential_normal_direction(X[i] / 1e3, 0.0, 0.0)
        axes[1].plot(X, Y, color=f"{layers_num * 2 / 20}")
        # plt.show()
        for i in range(1000):
            Y[i] = contact.subderivative_normal_direction(X[i] / 1e3, 0.0, 0.0)
            Y[i] /= 1000  # kPa to
            T[i] = top(X[i]) / 1e6 + 26.2
        axes[0].plot(X, Y, color=f"{layers_num * 2 / 20}")
        axes[0].plot(X, T, linestyle="--", color="skyblue")
        # plt.show()
        # for i in range(1000):
        #     Y[i] = contact.sub2derivative_normal_direction(X[i], 0.0, 0.0)
        # plt.plot(X, Y)
        # plt.show()

    main(config, methods, forces, contact, prefix=str(len(layers_limits)), layers_num=layers_num)


def survey(config):
    methods = (
        "gradiented BFGS",
        "gradiented CG",
        "BFGS",
        "CG",
        "Powell",
        "qsm",
        "globqsm",
    )[:]
    forces = np.asarray((
        # 15e3 * kN,
        16e3 * kN,
        17e3 * kN,
        18e3 * kN,
        19e3 * kN,
        20e3 * kN,
        # 21e3 * kN,
        22.5e3 * kN,
        # 23e3 * kN,
        25e3 * kN,
        30e3 * kN,
        35e3 * kN,
        40e3 * kN,
    ))

    # fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    axes = None
    draw_boundary = True
    for i in reversed([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12][:-2]):
        if axes is not None and i not in (0, 5):
            continue
        if draw_boundary and i not in (0, 5):
            continue
        thickness = 3 * mm
        layers_num = i
        # layers_limits = np.logspace(0, thickness, layers_num + 1)
        layers_limits = partition(0, thickness, layers_num + 1, p=1.25)
        composite_problem(config, layers_limits, thickness, methods, forces, layers_num, axes)
        if draw_boundary:
            plt.title(f"Num. of layers: {layers_num + 2}; Load: {forces[5] / 1e6} MPa")
            plt.ylim(- 4, 1)
            plt.xlabel(r"Contact Boundary [mm]")
            plt.ylabel(r"Penetration [mm]")
            if i == 5:
                plt.legend()
            plt.savefig(config.outputs_path + f"/boundary{layers_num + 2}.pdf")
    if axes is not None:
        axes[0].set_ylabel("Force ($\partial j(u)$) [MPa]", fontsize=14)
        axes[0].set_xlabel("Penetration ($u$) [mm]", fontsize=14)
        axes[0].grid()
        axes[1].set_ylabel("$j(u)$", fontsize=14)
        axes[1].set_xlabel("Penetration ($u$) [mm]", fontsize=14)
        axes[1].grid()
        plt.savefig(config.outputs_path + "/functional.pdf")


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
    config_ = Config(save=False, show=True, force=False).init()
    survey(config_)
