import numpy as np

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (
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


def get_C_temp_scenarios(mesh_density, final_time):
    C_temp_body_prop = [
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
        np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_temp_obstacle_prop
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
        )
        for i, temp_body_prop in enumerate(C_temp_body_prop)
    ]


def get_K_temp_scenarios(mesh_density, final_time):
    K_temp_body_prop = [
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
        # not allowed in physical law
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
        x_scaled = initial_node[0] / mesh_prop.scale_x
        y_scaled = initial_node[1] / mesh_prop.scale_y
        z_scaled = initial_node[2] / mesh_prop.scale_z
        if x_scaled > 0.9 and y_scaled < 0.1 and z_scaled < 0.1:
            return -1000
        return 0.0

    obstacle = Obstacle(None, default_temp_obstacle_prop)
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
        )
        for i, temp_body_prop in enumerate(K_temp_body_prop)
    ]


def main(mesh_density=5, final_time=3, plot_animation=True):
    all_scenarios = []
    all_scenarios.extend(get_C_temp_scenarios(mesh_density, final_time))
    all_scenarios.extend(get_K_temp_scenarios(mesh_density, final_time))
    obstacle = Obstacle(
        np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_temp_obstacle_prop
    )
    all_scenarios.extend(
        [
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
                obstacle=obstacle,
                heat_function=np.array([0]),
            ),
        ]
    )

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=Config(shell=False),
    )


if __name__ == "__main__":
    main()
