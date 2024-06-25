from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.membrane import plot as membrane_plot, plot_in_columns, \
    plot_limit_points
from conmech.scenarios.problems import InteriorContactWaveProblem, \
    InteriorContactLaw
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.state.products.intersection import Intersection
from conmech.state.products.intersection_contact_limit_points import \
    IntersectionContactLimitPoints
from conmech.state.state import State

TESTING = False
FORCE_SIMULATION = True
PRECISION = 8 if not TESTING else 3
OBSTACLE_LEVEL = 1.0
KAPPA = 10.0
BETA = 0.0
FORCE = 250.0
PROPAGATION = 2.5
TIMESTEP = 1/256

FULL = False
END = (0.75, 1)


def make_DNC(obstacle_level: float, kappa: float, beta: float):
    class DampedNormalCompliance(InteriorContactLaw):
        KAPPA = kappa
        BETA = beta

        @staticmethod
        def general_contact_condition(u, v):
            if u < obstacle_level:
                return 0
            return kappa * (u - obstacle_level) + beta * v

    return DampedNormalCompliance


@dataclass()
class MembraneSetup(InteriorContactWaveProblem):
    time_step: ... = TIMESTEP
    propagation: ... = PROPAGATION
    contact_law: ... = make_DNC(OBSTACLE_LEVEL, kappa=KAPPA, beta=BETA)()

    @staticmethod
    def inner_forces(
            x: np.ndarray,
            t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([FORCE])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.])

    boundaries: ... = BoundariesDescription(
        dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
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
        runner = WaveSolver(setup, "global")

        states = runner.solve(
            n_steps=steps,
            output_step=output_step,
            products=[IntersectionContactLimitPoints(
                obstacle_level=OBSTACLE_LEVEL, x=0.50), Intersection(x=0.50)],
            initial_displacement=setup.initial_displacement,
            initial_velocity=setup.initial_velocity,
            verbose=True,
            method="Powell",
        )
        for step, state in enumerate(states):
            state.save(f"{config.path}/{name}_step_{int(state.time / setup.time_step)}")
    else:
        states = []
        for step in output_step:
            states.append(
                State.load(f"{config.path}/{name}_step_{step}"))

    plot_limit_points(
        states[-1].products['limit points at 0.50'],
        title=fr'$\kappa={setup.contact_law.KAPPA}$ '
              fr'$\beta={setup.contact_law.BETA}$')
    intersect = states[-1].products['intersection at 0.50']
    results = tuple(intersect.data.items())
    T = results[-1][0]
    num = len(results)
    i = 0
    for t, v in intersect.data.items():
        i += 1
        # if t in (s / 2 for s in range(int(2 * T) + 1)):
        plt.plot(*v, color=f'{1 - t / T:.3f}')
        if num * END[0] < i < num * END[1]:
            break
    plt.title(f'{t:.2f}')
    plt.show()
    if not FULL:
        return

    states_ids = list(range(len(states)))
    to_plot = states_ids[:64:4] # [2, 4, 8, 11, 13, 15, 17, 20] #states_ids[1:4] + states_ids[-1:]
    vmin = np.inf
    vmax = -np.inf
    field = "velocity"
    zmin = np.inf
    zmax = -np.inf
    zfield = "displacement"
    for i, state in enumerate(states):
        if i not in to_plot:
            continue
        vmax = max(max(getattr(state, field)[:, 0]), vmax)
        vmin = min(min(getattr(state, field)[:, 0]), vmin)
        zmax = max(max(getattr(state, zfield)[:, 0]), zmax)
        zmin = min(min(getattr(state, zfield)[:, 0]), zmin)
    # vmax = min(vmax, 1)
    # vmin = max(vmin, -1)
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
        x=0.5,
        title=f"velocity"  #: {i * setup.time_step:.2f}s"
    )
    # plot_in_columns(
    #     states_, field=field, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax,
    #     in3d=True,
    #     title=f"velocity"  #: {i * setup.time_step:.2f}s"
    # )


if __name__ == "__main__":
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / PRECISION, scale=[1, 1]
    )
    T = 2.0 if not TESTING else MembraneSetup(None).time_step * 2
    setups = dict()
    to_simulate = [
        "plain",
        # "nonzero",
        # "velocity",
        # "force",
        # "beta=0",
        # "beta=0.5",
        # "beta=1.0",

    ]

    setup = MembraneSetup(mesh_descr)
    setups["plain"] = setup

    def initial_displacement(x: np.ndarray) -> np.ndarray:
        a = (x[:, 1] - 1) * 4 * (x[:, 0] * (x[:, 0] - 1))
        b = np.zeros(x.shape[0])
        c = np.stack((a, b), axis=1)
        return c
    setup = MembraneSetup(mesh_descr)
    setup.initial_displacement = initial_displacement
    setups["nonzero"] = setup

    setup = MembraneSetup(mesh_descr)
    setup.contact_law = make_DNC(OBSTACLE_LEVEL, kappa=10.0, beta=0.5)()
    setups["beta=0.5"] = setup

    setup = MembraneSetup(mesh_descr)
    setup.contact_law = make_DNC(OBSTACLE_LEVEL, kappa=1.0, beta=10.0)()
    setups["beta=1.0"] = setup

    def initial_velocity(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x) * 10
    setup = MembraneSetup(mesh_descr)
    setup.initial_velocity = initial_velocity
    setups["velocity"] = setup

    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([max(15 * (1 - t), 0)])
    setup = MembraneSetup(mesh_descr)
    setup.inner_forces = inner_forces
    setups["force"] = setup

    setup = MembraneSetup(mesh_descr)
    setup.contact_law = make_DNC(OBSTACLE_LEVEL, kappa=10.0, beta=0.0)()
    setups["beta=0"] = setup

    for name in to_simulate:
        main(
            Config(output_dir="BOSK.2", force=FORCE_SIMULATION).init(),
            setups[name],
            name=name,
            steps=int(T / setups[name].time_step) + 1
        )
