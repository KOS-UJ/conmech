from dataclasses import dataclass
from typing import Optional

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.membrane import plot as membrane_plot
from conmech.scenarios.problems import ContactWaveProblem, InteriorContactLaw
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.state.state import State

PRECISION = 8


class DampedNormalCompliance(InteriorContactLaw):
    @staticmethod
    def general_contact_condition(u, v):
        k = 50
        obstacle_level = 1
        beta = 150
        if u < obstacle_level:
            return 0
        return k * (u - obstacle_level) + beta * v


@dataclass()
class MembraneSetup(ContactWaveProblem):
    time_step: ... = 2 / PRECISION
    contact_law: ... = DampedNormalCompliance()

    @staticmethod
    def inner_forces(
            x: np.ndarray,
            t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([100.])

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
    to_simulate = False
    if config.force:
        to_simulate = True
    else:
        for step in range(steps):
            try:
                State.load(f"{config.path}/{name}_step_{step}")
            except IOError:
                to_simulate = True

    if to_simulate:
        runner = WaveSolver(setup, "global")

        states = runner.solve(
            n_steps=steps,
            output_step=range(0, steps),
            initial_displacement=setup.initial_displacement,
            initial_velocity=setup.initial_velocity,
            verbose=True,
            method="Powell",
        )
        for step, state in enumerate(states):
            state.save(f"{config.path}/{name}_step_{step}")
    else:
        states = []
        for step in range(steps):
            states.append(
                State.load(f"{config.path}/{name}_step_{step}"))

    states_ids = list(range(len(states)))
    to_plot = states_ids[1:4] + states_ids[::16][1:]
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
    prec = 1
    zmax = round(zmax + 0.05 * prec, prec)
    zmin = round(zmin - 0.05 * prec, prec)
    for i, state in enumerate(states):
        if i not in to_plot:
            continue
        membrane_plot(
            state, field=field, vmin=vmin, vmax=vmax, zmin=zmin, zmax=zmax,
            title=f"{name}: {i * setup.time_step:.2f}s"
        )


if __name__ == "__main__":
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / PRECISION, scale=[1, 1]
    )
    T = 3.2
    setups = dict()

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

    def initial_velocity(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x) * 10
    setup = MembraneSetup(mesh_descr)
    setup.initial_velocity = initial_velocity
    setups["velocity"] = setup

    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([max(100 * (1 - t), 0)])
    setup = MembraneSetup(mesh_descr)
    setup.inner_forces = inner_forces
    setups["force"] = setup

    def general_contact_condition(u, v):
        k = 300
        obstacle_level = 1
        beta = 0
        if u < obstacle_level:
            return 0
        return k * (u - obstacle_level) + beta * max(v, 0)
    setup = MembraneSetup(mesh_descr)
    setup.contact_law.general_contact_condition = general_contact_condition
    setups["beta=0"] = setup

    for name, setup in setups.items():
        main(
            Config(output_dir="BOSK.ORG", force=True).init(),
            setup, name=name, steps=int(T/list(setups.values())[0].time_step)+1
        )
