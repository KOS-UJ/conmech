"""
Created at 21.08.2019
"""
import pickle
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import ContactLaw, StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.solvers.optimization.optimization import Optimization

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


class MMLV99(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        u_nu = -u_nu
        coef = 1.
        if u_nu <= 0:
            return 0.0
        if u_nu < 0.5 * mm:
            return k0 * u_nu ** 2 * coef
        if u_nu < 1 * mm:
            return (k10 * u_nu ** 2 + k11 * u_nu) * coef
        if u_nu < 2 * mm:
            return (k20 * u_nu ** 2 + k21 * u_nu + 4) * coef
        return 16 * coef

    @staticmethod
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


def main(config: Config, methods, forces):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    PREFIX = "BBO"
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

        for method, force in to_simulate:
            print(method, force)

            def outer_forces(x, t=None):
                if x[1] >= 0.0099:
                    return np.array([0, force * surface])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticSolver(setup, "schur")

            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                method=method,
            )
            path = f"{config.outputs_path}/{PREFIX}_mtd_{method}_frc_{force:.2e}"
            with open(path, "wb+") as output:
                state.body.dynamics.force.outer.source = None
                state.body.dynamics.force.inner.source = None
                state.body.properties.relaxation = None
                state.setup = None
                state.constitutive_law = None
                pickle.dump(state, output)
        print(Optimization.RESULTS)

    for m in methods:
        for f in forces:
            path = f"{config.outputs_path}/{PREFIX}_mtd_{m}_frc_{f:.2e}"
            with open(path, "rb") as output:
                state = pickle.load(output)

            print("drawing")
            drawer = Drawer(state=state, config=config)
            drawer.colorful = True
            drawer.draw(
                show=config.show,
                save=config.save,
                # title=f"{m}: {f}, "
                # f"time: {runner.step_solver.last_timing}"
            )
            x = state.body.mesh.nodes[:state.body.mesh.contact_nodes_count - 1,
                0]
            u = state.displacement[:state.body.mesh.contact_nodes_count - 1, 1]
            y1 = [MMLV99().normal_direction(-u_) for u_ in u]
            plt.plot(x, y1, label=f"{f:.2e}")
        plt.ylabel("Interlaminar binding force [kN/m$^2$]")
        plt.xlabel(r"Contact interface [mm]")
        plt.grid()
        plt.title(m)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    X = np.linspace(0, -3 * mm, 1000)
    Y = np.empty(1000)
    for i in range(1000):
        Y[i] = MMLV99.potential_normal_direction(X[i])
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = MMLV99.normal_direction(X[i])
    plt.plot(X, Y)
    plt.show()
    results = {
        "BFGS": [-0.061546678021737036,
                 -0.06782602334922566,
                 -0.07441012406759984,
                 -0.08129875924227234,
                 -0.0959892642846613,
                 -0.10379118250601398,
                 -0.10538811134540409,
                 -0.8584224789292736,
                 -0.14133884664811114, ],
        "CG": [-0.07225702623584927,
               -0.07966800277816762,
               -0.08744039267159345,
               -0.09557428287965247,
               -0.12191044159984168,
               -0.1358025353476277,
               -0.13865495481609302,
               -0.15028696247286885,
               -1.265832916470563, ],
        "Powell": [-0.0723012449592487,
                   -0.07971212256709342,
                   -0.0874845064006726,
                   -0.0978621160055679,
                   -0.12214289905071576,
                   -0.13588717513833654,
                   -0.7582249892835198,
                   -0.8589012526317955,
                   -1.2688709207679356, ],
        "subgradient": [-0.05079652409797247,
                        -0.046161334145372934,
                        -0.04120648554585715,
                        -0.3859157295854724,
                        -0.6104716467978587,
                        -0.7302821710666211,
                        -0.7554950402698594,
                        -0.8555741662642888,
                        -1.2663638426265278, ],
    }
    forces = np.asarray((20e3 * kN, 21e3 * kN, 21e3 * kN, 23e3 * kN,
              25e3 * kN, 26e3 * kN, 26.2e3 * kN, 27e3 * kN, 30e3 * kN))[::2]
    # for m, losses in results.items():
    #     plt.plot(forces/1e3, -1 * np.asarray(losses[:]), "-o", label=m)
    plt.legend()
    plt.ylabel("$-\mathcal{L}(u)$")
    plt.xlabel(r"Load [kN/m$^2$]")
    plt.grid()
    plt.show()
    methods = ("BFGS", "CG", "Powell", "subgradient")[-1:]
    main(Config(save=True, show=False, force=False).init(), methods, forces)
