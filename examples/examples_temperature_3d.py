import argparse
from argparse import ArgumentParser

import numpy as np

from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.body_properties import TimeDependentTemperatureBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import TemperatureObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (
    M_BUNNY_3D,
    M_CUBE_3D,
    TemperatureScenario,
    default_temp_body_prop,
    default_temp_obstacle_prop,
    default_thermal_conductivity_coefficients,
    default_thermal_expansion_coefficients,
    f_rotate_3d,
    get_temp_body_prop,
)
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle

simulation_config = SimulationConfig(
    use_normalization=False,
    use_linear_solver=False,
    use_green_strain=False,
    use_nonconvex_friction_law=False,
    use_constant_contact_integral=False,
    use_lhs_preconditioner=False,
    use_pca=False,
)


def get_constitutive_temp_scenarios(mesh_density, final_time):
    constitutive_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            thermal_expansion_coeff=np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.5]]),
            thermal_conductivity_coeff=default_thermal_conductivity_coefficients,
        ),
        get_temp_body_prop(
            thermal_expansion_coeff=np.array(
                [[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]]
            ),
            thermal_conductivity_coeff=default_thermal_conductivity_coefficients,
        ),
        # not allowed in physical law
        get_temp_body_prop(
            thermal_expansion_coeff=np.array(
                [[1.0, -0.5, -0.5], [0.5, 1.0, -0.5], [0.5, 0.5, 1.0]]
            ),
            thermal_conductivity_coeff=default_thermal_conductivity_coefficients,
        ),
    ]

    # obstacle = Obstacle(None, default_temp_obstacle_prop)
    obstacle = Obstacle(
        np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]]), default_temp_obstacle_prop
    )
    return [
        TemperatureScenario(
            name=f"C_{i}",
            mesh_prop=MeshProperties(
                dimension=3, mesh_type=M_CUBE_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0, 0]),
            obstacle=obstacle,
            heat_function=np.array([2]),
            simulation_config=simulation_config,
        )
        for i, temp_body_prop in enumerate(constitutive_temp_body_prop)
    ]


def get_expansion_temp_scenarios(mesh_density, final_time):
    expansion_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
        ),
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array(
                [[0.1, 0.0, 0.1], [0.0, 0.1, 0.0], [0.1, 0.0, 0.1]]
            ),
        ),
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array(
                [[0.1, -0.1, 0.0], [-0.1, 0.1, 0.0], [0.0, 0.0, 0.1]]
            ),
        ),
        # not allowed by physical law
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array(
                [[0.1, 0.1, -0.1], [-0.1, 0.1, 0.1], [0.1, -0.1, 0.1]]
            ),
        ),
    ]

    def h_corner(
        initial_node: np.ndarray,
        moved_node: np.ndarray,
        mesh_prop: MeshProperties,
        t: float,
    ):
        _ = moved_node, t
        x_scaled = initial_node[0] / mesh_prop.scale_x
        y_scaled = initial_node[1] / mesh_prop.scale_y
        z_scaled = initial_node[2] / mesh_prop.scale_z
        if x_scaled > 0.9 and y_scaled < 0.1 and z_scaled < 0.1:
            return -1000
        return 0.0

    # obstacle = Obstacle(None, default_temp_obstacle_prop)
    obstacle = Obstacle(
        np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]]), default_temp_obstacle_prop
    )
    return [
        TemperatureScenario(
            name=f"K_{i}",
            mesh_prop=MeshProperties(
                dimension=3, mesh_type=M_CUBE_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0, 0]),
            obstacle=obstacle,
            heat_function=h_corner,
            simulation_config=simulation_config,
        )
        for i, temp_body_prop in enumerate(expansion_temp_body_prop)
    ]


def main(mesh_density=32, final_time=2.5, plot_animation=True, shell=False):
    config = Config(shell=shell)
    config.print_skip = 0.05
    mesh_prop = MeshProperties(
        dimension=3,
        mesh_type=M_BUNNY_3D,
        scale=[1],
        mesh_density=[mesh_density],
    )

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

    all_scenarios = []
    advanced_scenarios = [
        TemperatureScenario(
            name="temperature_3d_bunny_throw",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=default_temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacle=Obstacle(
                np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_temp_obstacle_prop
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
        TemperatureScenario(
            name="temperature_3d_cube_throw",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_CUBE_3D,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=default_temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacle=Obstacle(
                np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_temp_obstacle_prop
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
        TemperatureScenario(
            name="temperature_3d_bunny_push_base",
            mesh_prop=mesh_prop,
            body_prop=temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=lambda *_: np.array([0, 0, -2]),
            obstacle=Obstacle(
                np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]),
                TemperatureObstacleProperties(hardness=800.0, friction=0.0, heat=0.0),
            ),
            heat_function=np.array([0]),
            simulation_config=simulation_config,
        ),
    ]

    constitutive_temp_scenarios = get_constitutive_temp_scenarios(mesh_density, final_time)
    expansion_temp_scenarios = get_expansion_temp_scenarios(mesh_density, final_time)

    all_scenarios = [
        *advanced_scenarios,
        *constitutive_temp_scenarios,
        *expansion_temp_scenarios,
    ]
    simulation_runner.run_examples(
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
