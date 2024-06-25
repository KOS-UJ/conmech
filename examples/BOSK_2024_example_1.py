from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.membrane import plot_in_columns, plot_limit_points
from conmech.scenarios.problems import ContactLaw, ContactWaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.state.products.intersection import Intersection
from conmech.state.products.intersection_contact_limit_points import \
    IntersectionContactLimitPoints
from conmech.state.state import State

TESTING = False
FORCE_SIMULATION = True
FULL = False
PRECISION = 32 if not TESTING else 3
OBSTACLE_LEVEL = 1.0
T = 10.0


def make_DNC(obstacle_level: float, kappa: float, beta: float):
    class DampedNormalCompliance(ContactLaw):
        @staticmethod
        def general_contact_condition(u, v):
            if u < obstacle_level:
                return 0
            return kappa * (u - obstacle_level) + beta * v

    return DampedNormalCompliance


@dataclass()
class MembraneSetup(ContactWaveProblem):
    time_step: ... = 1 / 512
    propagation: ... = 4.0
    contact_law: ... = make_DNC(OBSTACLE_LEVEL, kappa=10.0, beta=0.5)()

    @staticmethod
    def inner_forces(
            x: np.ndarray,
            t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([100])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(
        # dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
        dirichlet=lambda x: x[0] in (0,) or x[1] in (0, 1),
        contact=lambda x: x[0] == 1,
    )


def main(config: Config, setup, name, steps):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    print(name)
    output_step = (steps,)
    to_simulate = False
    if config.force:
        to_simulate = True
    else:
        for step in output_step:
            try:
                State.load(f"{config.path}/{name}_step_{step}")
            except IOError:
                to_simulate = True

    if to_simulate:
        runner = WaveSolver(setup, "schur")

        states = runner.solve(
            n_steps=steps,
            output_step=output_step,
            products=[IntersectionContactLimitPoints(
                obstacle_level=OBSTACLE_LEVEL, x=1.0), Intersection(x=1.0)],
            initial_displacement=setup.initial_displacement,
            initial_velocity=setup.initial_velocity,
            verbose=True,
            method="Powell",
        )
        for step, state in enumerate(states):
            state.save(f"{config.path}/{name}_step_{step}")
    else:
        states = []
        for step in output_step:
            states.append(
                State.load(f"{config.path}/{name}_step_{step}"))

    plot_limit_points(states[-1].products['limit points at 1.00'])
    # intersect = states[-1].products['intersection at 1.00']
    # for t, v in intersect.data.items():
    #     if t in (s / 2 for s in range(int(2 * T) + 1)):
    #         plt.title(f'{t:.2f}')
    #         plt.plot(*v)
    #         plt.show()
    if not FULL:
        return

    states_ids = list(range(len(states)))
    to_plot = [10, 30, 50, 210, 320, 430, 540, 600] #states_ids[1:4] #+ states_ids[::4][1:]
    vmin = np.inf
    vmax = -np.inf
    field = "velocity"
    zmin = np.inf
    zmax = -np.inf
    zfield = "displacement"
    for i, state in enumerate(states):
        if i not in to_plot:
            continue
        state["velocity"][:] = np.abs(state["velocity"])[:]
        vmax = max(max(getattr(state, field)[:, 0]), vmax)
        vmin = min(min(getattr(state, field)[:, 0]), vmin)
        zmax = max(max(getattr(state, zfield)[:, 0]), zmax)
        zmin = min(min(getattr(state, zfield)[:, 0]), zmin)
    prec = 1
    zmax = round(zmax + 0.05 * prec, prec)
    zmin = round(zmin - 0.05 * prec, prec)
    states_ = []
    for i, state in enumerate(states):
        if i not in to_plot:
            continue
        states_.append(state)
    plot_in_columns(
        states_, field=field, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax,
        title=f"velocity"  #: {i * setup.time_step:.2f}s"
    )
    plot_in_columns(
        states_, field=field, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax,
        in3d=True,
        title=f"velocity"  #: {i * setup.time_step:.2f}s"
    )


if __name__ == "__main__":
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / PRECISION, scale=[1, 1]
    )
    setups = dict()

    kappas = (0.0, 0.5, 1.0, 5.0, 10.0, 100.0)[-2:]
    betas = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)[1:-1]
    to_simulate = []
    for kappa in kappas:
        for beta in betas:
            label = f"kappa={kappa:.2f};beta={beta:.2f}"
            to_simulate.append(label)
            setup = MembraneSetup(mesh_descr)
            setup.contact_law = make_DNC(OBSTACLE_LEVEL, kappa=kappa, beta=beta)()
            setups[label] = setup
    # to_simulate = [
    #     "no contact",
    #     # "plain",
    #     # "force"
    #     "beta=0.00",
    #     "beta=0.25",
    #     "beta=0.50",
    #     "beta=0.75",
    #     "beta=1.00",
    # ]
    #
    # setup = MembraneSetup(mesh_descr)
    # setups["plain"] = setup
    #
    # setup = MembraneSetup(mesh_descr)
    # setup.contact_law = make_DNC(OBSTACLE_LEVEL, kappa=0.0, beta=0.0)()
    # setups["no contact"] = setup
    #
    # def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
    #     return np.array([max(100 * (1 - t), 0)])
    # setup = MembraneSetup(mesh_descr)
    # setup.inner_forces = inner_forces
    # setups["force"] = setup
    #
    # setup = MembraneSetup(mesh_descr)
    # setup.contact_law = make_DNC(OBSTACLE_LEVEL, kappa=10.0, beta=0.0)()
    # setups["beta=0"] = setup

    for name in to_simulate:
        main(
            Config(output_dir="BOSK.1", force=FORCE_SIMULATION).init(),
            setups[name],
            name=name,
            steps=int(T / setups[name].time_step) + 1
        )
