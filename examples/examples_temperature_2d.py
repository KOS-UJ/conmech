import numpy as np

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (
    TemperatureScenario,
    default_thermal_expansion_coefficients,
    default_thermal_conductivity_coefficients,
    default_temp_body_prop,
    default_temp_obstacle_prop,
    f_rotate_fast,
    get_temp_body_prop,
    M_CIRCLE,
    M_POLYGON,
    M_RECTANGLE,
)
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle


def get_C_temp_scenarios(mesh_density, final_time):
    C_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            thermal_expansion_coeff=np.array([[1.5, 0], [0, 0.5]]),
            thermal_conductivity_coeff=default_thermal_conductivity_coefficients,
        ),
        get_temp_body_prop(
            thermal_expansion_coeff=np.array([[1.0, -0.5], [-0.5, 1.0]]),
            thermal_conductivity_coeff=default_thermal_conductivity_coefficients,
        ),
        # not allowed in physical law
        get_temp_body_prop(
            thermal_expansion_coeff=np.array([[1.0, 0.5], [-0.5, 1.0]]),
            thermal_conductivity_coeff=default_thermal_conductivity_coefficients,
        ),
    ]
    obstacle = Obstacle.get_linear_obstacle("side", default_temp_obstacle_prop)
    return [
        TemperatureScenario(
            name=f"C_{i}",
            mesh_prop=MeshProperties(
                dimension=2, mesh_type=M_RECTANGLE, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0]),
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
            thermal_conductivity_coeff=np.array([[0.5, 0], [0, 0.5]]),
        ),
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array([[0.1, 0.1], [0.1, 0.1]]),
        ),
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array([[0.1, -0.1], [-0.1, 0.1]]),
        ),
        # not allowed in physical law
        get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array([[0.1, -0.1], [0.1, 0.1]]),
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
        if x_scaled < 0.1 and y_scaled < 0.1:
            return -50.0  # -100
        return 0.0

    obstacle = Obstacle.get_linear_obstacle("side", default_temp_obstacle_prop)

    return [
        TemperatureScenario(
            name=f"K_{i}",
            mesh_prop=MeshProperties(
                dimension=2, mesh_type=M_RECTANGLE, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=temp_body_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0]),
            obstacle=obstacle,
            heat_function=h_corner,
        )
        for i, temp_body_prop in enumerate(K_temp_body_prop)
    ]


def get_polygon_scenarios(mesh_density, final_time):
    polygon_scenario = lambda i, forces_function, obstacle: TemperatureScenario(
        name=f"polygon_{i}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_POLYGON, scale=[1], mesh_density=[mesh_density]
        ),
        body_prop=default_temp_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=forces_function,
        obstacle=Obstacle.get_linear_obstacle(obstacle, default_temp_obstacle_prop),
        heat_function=np.array([0]),
    )

    return [
        polygon_scenario(i=0, forces_function=f_rotate_fast, obstacle="side"),
        polygon_scenario(i=1, forces_function=f_rotate_fast, obstacle="front"),
        polygon_scenario(i=2, forces_function=np.array([1, 0]), obstacle="front"),
    ]


def get_friction_scenarios(mesh_density, final_time):
    obstacle = Obstacle(np.array([[[0.0, 1.0]], [[0.0, 0.0]]]), default_temp_obstacle_prop)
    friction_scenario = lambda i: TemperatureScenario(
        name="circle_flat_A_roll",
        mesh_prop=MeshProperties(
            dimension=2,
            mesh_type=M_CIRCLE,
            scale=[1],
            mesh_density=[mesh_density],
        ),
        body_prop=get_temp_body_prop(
            thermal_expansion_coeff=default_thermal_expansion_coefficients,
            thermal_conductivity_coeff=np.array([[0.01, 0], [0, 0.01]]),
        ),
        schedule=Schedule(final_time=final_time),
        forces_function=np.array([1.0, -0.5]),
        obstacle=obstacle,
        heat_function=np.array([0]),
    )

    return [
        friction_scenario(i=0),
    ]


def main(mesh_density=5, final_time=3, plot_animation=True):
    all_scenarios = []
    all_scenarios.extend(get_friction_scenarios(mesh_density, final_time))
    all_scenarios.extend(get_polygon_scenarios(mesh_density, final_time))
    all_scenarios.extend(get_K_temp_scenarios(mesh_density, final_time))
    all_scenarios.extend(get_C_temp_scenarios(mesh_density, final_time))

    return simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=Config(shell=False),
    )


if __name__ == "__main__":
    main()
