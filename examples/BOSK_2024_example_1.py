from dataclasses import dataclass
from typing import Optional

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.membrane import plot_in_columns
from conmech.scenarios.problems import ContactLaw, ContactWaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.state.state import State

TESTING = True
PRECISION = 12 if not TESTING else 3


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
    time_step: ... = 1 / 20
    contact_law: ... = make_DNC(1.0, kappa=100.0, beta=150.0)()

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
        runner = WaveSolver(setup, "schur")

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
    to_plot = [1, 2, 5, 8, 11, 14, 17, 20] #states_ids[1:4] #+ states_ids[::4][1:]
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
        title=f"velocity" #: {i * setup.time_step:.2f}s"
    )


if __name__ == "__main__":
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / PRECISION, scale=[1, 1]
    )
    T = 2.0
    setups = dict()
    to_simulate = [
        "plain",
        "beta=0",
    ]

    setup = MembraneSetup(mesh_descr)
    setups["plain"] = setup

    setup = MembraneSetup(mesh_descr)
    setup.contact_law = make_DNC(1.0, kappa=100.0, beta=0.0)()
    setups["beta=0"] = setup

    for name in to_simulate:
        main(
            Config(output_dir="BOSK.1", force=True).init(),
            setups[name],
            name=name,
            steps=int(T / setups[name].time_step) + 1
        )
