"""
Created at 21.08.2019
"""
import pickle
from dataclasses import dataclass
from matplotlib import pyplot as plt

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
k30 = 2.5e12 * kN * surface


def normal_direction(u_nu: float) -> float:
    if u_nu <= 0:
        return 0.0
    if u_nu < 0.5 * mm:
        return k0 * u_nu * 2
    if u_nu < 1 * mm:
        return k10 * (u_nu * 2) + k11
    if u_nu < 2 * mm:
        return k20 * (u_nu * 2) + k21
    return u_nu ** 3 * 4 * k30


class MMLV99(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        if u_nu <= 0:
            return 0.0
        if u_nu < 0.5 * mm:
            return k0 * u_nu ** 2
        if u_nu < 1 * mm:
            return k10 * u_nu ** 2 + k11 * u_nu
        if u_nu < 2 * mm:
            return k20 * u_nu ** 2 + k21 * u_nu + 4
        return u_nu ** 4 * k30

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
    PREFIX = "DBBO"
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
                    return np.array([0, -1 * force * surface])
                return np.array([0, 0])

            setup.outer_forces = outer_forces

            runner = StaticSolver(setup, "schur")

            state = runner.solve(
                verbose=True,
                fixed_point_abs_tol=0.001,
                initial_displacement=setup.initial_displacement,
                method=method,
                maxiter=100,
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

            # drawer = Drawer(state=state, config=config)
            # drawer.colorful = True
            # drawer.draw(
            #     show=config.show,
            #     save=config.save,
            #     title=f"{m}: {f}, "
            #     # f"time: {runner.step_solver.last_timing}"
            # )
            x = state.body.mesh.nodes[:state.body.mesh.contact_nodes_count - 1,
                0]
            u = state.displacement[:state.body.mesh.contact_nodes_count - 1, 1]
            y1 = [normal_direction(-u_) for u_ in u]
            print(f)
            plt.plot(x, y1, label=f"{f:.2e}")
        plt.title(m)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    X = np.linspace((2-2) * mm, (2+2) * mm, 1000)
    Y = np.empty(1000)
    for i in range(1000):
        Y[i] = MMLV99.potential_normal_direction(X[i])
    plt.plot(X, Y)
    plt.show()
    for i in range(1000):
        Y[i] = normal_direction(X[i])
    plt.plot(X, Y)
    plt.show()
    # results = {
    #     "Powell": [-0.09786211600599237,
    #                -0.12214289905239312,
    #                -0.13027212877878766,
    #                -0.13447218948364842,
    #                -0.13588717514960513,
    #                -0.1373096435316275,
    #                -0.7582249893948801,
    #                -0.8589012530191608,
    #                -1.2688709210981735, ],
    #     "BFGS": [-0.09487765162385353,
    #              -0.12207326519092926,
    #              -0.11772218280878324,
    #              -0.1198269497911567,
    #              -0.12061095335641955,
    #              -0.1219350781729528,
    #              -0.12279409585312624,
    #              -0.8584230093357013,
    #              -1.2687124265093166, ],
    #     "CG": [-0.0955742828809952,
    #            -0.12191044159984168,
    #            -0.13009806547436803,
    #            -0.1341887930175023,
    #            -0.1358025353476277,
    #            -0.136904521914724,
    #            -0.13865495481609302,
    #            -0.8584104766729636,
    #            -1.2658836730355307, ],
    #     "subgradient2": [-0.09786204500205781,
    #                      -0.12214281874382482,
    #                      -0.13027204041320914,
    #                      -0.15450061948841598,
    #                      -0.1571765749815719,
    #                      -0.15986547858657962,
    #                      -0.7582249071621823,
    #                      -0.8589012119331098,
    #                      -1.2688708874747643, ],
    # }
    # for m, losses in results.items():
    #     plt.plot(-1 * np.asarray(losses), label=m)
    # plt.legend()
    # # plt.loglog()
    # plt.show()
    methods = ("BFGS", "CG", "qsm", "Powell", "subgradient", 'qsm')[-1:]
    forces = (23e3 * kN, 25e3 * kN, 25.6e3 * kN, 25.9e3 * kN, 26e3 * kN,
              26.1e3 * kN, 26.2e3 * kN, 27e3 * kN, 30e3 * kN)[-1::4]
    main(Config(save=False, show=True, force=True).init(), methods, forces)
