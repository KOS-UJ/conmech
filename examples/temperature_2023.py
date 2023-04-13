import argparse
from argparse import ArgumentParser

import numpy as np

from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.body_properties import TimeDependentTemperatureBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import TemperatureObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import M_BUNNY_3D_LIFTED, TemperatureScenario
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from conmech.state.obstacle import Obstacle


def main(mesh_density=32, final_time=2.5, plot_animation=True, shell=False):
    config = Config(shell=shell)
    config.print_skip = 0.05
    mesh_prop = MeshProperties(
        dimension=3,
        mesh_type=M_BUNNY_3D_LIFTED,
        scale=[1],
        mesh_density=[mesh_density],
    )
    simulation_config = SimulationConfig(
        use_normalization=False,
        use_linear_solver=False,
        use_green_strain=False,
        use_nonconvex_friction_law=True,
        use_constant_contact_integral=False,
        use_lhs_preconditioner=False,
        with_self_collisions=False,
        use_pca=False,
    )
    schedule = Schedule(final_time=final_time, time_step=0.01)

    def forces_function(*_):
        return np.array([-0.7, 0, -2])

    obstacle_geometry = np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.76]]])

    thermal_expansion_coefficients = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    thermal_conductivity_coefficients = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    )
    temp_body_prop = TimeDependentTemperatureBodyProperties(
        mass_density=1.0,
        mu=10.0,
        lambda_=10.0,
        theta=20.0,
        zeta=20.0,
        thermal_expansion=thermal_expansion_coefficients,
        thermal_conductivity=thermal_conductivity_coefficients,
    )

    all_scenarios = [
        TemperatureScenario(
            name="temperature_3d_bunny_push_base",
            mesh_prop=mesh_prop,
            body_prop=temp_body_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacle=Obstacle(
                obstacle_geometry,
                TemperatureObstacleProperties(hardness=200.0, friction=0.1, heat=0.2),
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
        TemperatureScenario(
            name="temperature_3d_bunny_push_friction",
            mesh_prop=mesh_prop,
            body_prop=temp_body_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacle=Obstacle(
                obstacle_geometry,
                TemperatureObstacleProperties(hardness=200.0, friction=1.0, heat=0.2),
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
        TemperatureScenario(
            name="temperature_3d_bunny_push_heat",
            mesh_prop=mesh_prop,
            body_prop=temp_body_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacle=Obstacle(
                obstacle_geometry,
                TemperatureObstacleProperties(hardness=200.0, friction=0.1, heat=0.6),
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
        TemperatureScenario(
            name="temperature_3d_bunny_push_soft",
            mesh_prop=mesh_prop,
            body_prop=temp_body_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacle=Obstacle(
                obstacle_geometry,
                TemperatureObstacleProperties(hardness=50.0, friction=0.1, heat=0.2),
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
    ]

    return simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=config,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--shell", action=argparse.BooleanOptionalAction, default=False
    )  # Python 3.9+
    args = parser.parse_args()
    main(shell=args.shell)
