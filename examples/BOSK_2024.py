from dataclasses import dataclass
from typing import Optional

import numpy as np

from conmech.dynamics.contact.damped_normal_compliance import make_damped_norm_compl
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.membrane import plot_in_columns, plot_limit_points
from conmech.scenarios.problems import ContactWaveProblem, InteriorContactWaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.state.products.verticalintersection import VerticalIntersection
from conmech.state.products.intersection_contact_limit_points import (
    VerticalIntersectionContactLimitPoints,
)
from conmech.state.state import State


@dataclass()
class BoundaryContactMembrane(ContactWaveProblem):
    time_step: ... = 1 / 64
    propagation: ... = np.sqrt(4.0)
    contact_law: ... = make_damped_norm_compl(obstacle_level=1.0, kappa=10.0, beta=0.1)()

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([100])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(
        dirichlet=lambda x: x[0] in (0,) or x[1] in (0, 1),
        contact=lambda x: x[0] == 1,
    )


@dataclass()
class InteriorContactMembrane(InteriorContactWaveProblem):
    time_step: ... = 1 / 512
    propagation: ... = np.sqrt(100.00)
    contact_law: ... = make_damped_norm_compl(
        obstacle_level=1.0, kappa=10.0, beta=100.0, interior=True
    )()

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([2500])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1))


def runner(config: Config, setup, name, steps, intersect, continuing, solving_method):
    print(name)
    output_step = (
        int(steps * 0.25),
        int(steps * 0.5),
        int(steps * 0.75),
        steps,
    )

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
        runner = WaveSolver(setup, solving_method)

        if continuing:
            path = f"{config.path}/{continuing}"
            print("Loading:", path)
            state = State.load(path)
            state.products = {}
        else:
            state = None

        states = runner.solve(
            n_steps=steps,
            output_step=output_step,
            products=[
                VerticalIntersectionContactLimitPoints(
                    obstacle_level=setup.contact_law.obstacle_level, x=intersect
                ),
                VerticalIntersection(x=intersect),
            ],
            initial_displacement=setup.initial_displacement,
            initial_velocity=setup.initial_velocity,
            state=state,
            verbose=True,
        )
        for step, state in enumerate(states):
            state.save(f"{config.path}/{name}_step_{step}")
    else:
        states = []
        for step in output_step:
            states.append(State.load(f"{config.path}/{name}_step_{step}"))

    if continuing:
        path = f"{config.path}/{continuing}"
        print("Loading:", path)
        state = State.load(path)
        plot_limit_points(
            state.products[f"limit points at {intersect:.2f}"],
            title=rf"$\kappa={setup.contact_law.kappa}$ " rf"$\beta={setup.contact_law.beta}$",
            finish=False,
        )
    plot_limit_points(
        states[-1].products[f"limit points at {intersect:.2f}"],
        title=rf"$\kappa={setup.contact_law.kappa}$, $\beta={setup.contact_law.beta}$",
        finish=config.show,
    )

    states_ids = list(range(len(states)))
    to_plot = states_ids
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
    states_ = []
    for i, state in enumerate(states):
        if i not in to_plot:
            continue
        states_.append(state)
    plot_in_columns(
        states_,
        field=field,
        vmin=vmin,
        vmax=vmax,
        zmin=zmin,
        zmax=zmax,
        x=intersect,
        title=f"intersection at {intersect:.2f}",
        finish=config.show,
    )
    plot_in_columns(
        states_,
        field=field,
        vmin=vmin,
        vmax=vmax,
        zmin=zmin,
        zmax=zmax,
        in3d=True,
        title=f"velocity",
        finish=config.show,
    )


def boundary_contact(config: Config):
    precision = 16 if not config.test else 3
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / precision, scale=[1, 1]
    )
    default = BoundaryContactMembrane(mesh_descr)
    T = 4.0 if not config.test else default.time_step * 2

    setups = dict()
    to_simulate = [
        "no contact",
        "plain",
        "beta=0.00",
    ]

    setup = default
    setups["plain"] = setup

    setup = BoundaryContactMembrane(mesh_descr)
    setup.contact_law = make_damped_norm_compl(
        obstacle_level=default.contact_law.obstacle_level, kappa=0.00, beta=0.0, interior=False
    )()
    setups["no contact"] = setup

    setup = BoundaryContactMembrane(mesh_descr)
    setup.contact_law = make_damped_norm_compl(
        obstacle_level=default.contact_law.obstacle_level,
        kappa=default.contact_law.kappa,
        beta=0.0,
        interior=False,
    )()
    setups["beta=0.00"] = setup

    for name in to_simulate:
        runner(
            config,
            setups[name],
            name="case_2_bdry_" + name,
            steps=int(T / setups[name].time_step) + 1,
            intersect=1.00,
            continuing=None,
            solving_method="schur",
        )


def interior_contact(config: Config):
    precision = 4 if not config.test else 3
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / precision, scale=[1, 1]
    )
    default = InteriorContactMembrane(mesh_descr)
    T = 0.5 if not config.test else default.time_step * 2

    setups = dict()
    to_simulate = [
        "plain",
        "beta=0.00",
        "beta=1000.0",
    ]

    setup = default
    setups["plain"] = setup

    setup = InteriorContactMembrane(mesh_descr)
    setup.contact_law = make_damped_norm_compl(
        obstacle_level=default.contact_law.obstacle_level,
        kappa=default.contact_law.kappa,
        beta=0.0,
        interior=True,
    )()
    setups["beta=0.00"] = setup

    setup = InteriorContactMembrane(mesh_descr)
    setup.contact_law = make_damped_norm_compl(
        obstacle_level=default.contact_law.obstacle_level,
        kappa=default.contact_law.kappa,
        beta=1000.0,
        interior=True,
    )()
    setups["beta=1000.0"] = setup

    for name in to_simulate:
        runner(
            config,
            setups[name],
            name="case_1_int_" + name,
            steps=int(T / setups[name].time_step) + 1,
            intersect=0.50,
            continuing=None,
            solving_method="global",
        )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    boundary_contact(config)
    interior_contact(config)


if __name__ == "__main__":
    main(Config().init())
