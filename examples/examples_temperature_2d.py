import numpy as np

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (TemperatureScenario, default_C_coeff,
                                         default_K_coeff, default_temp_body_prop,
                                         default_temp_obstacle_prop, f_rotate_fast,
                                         get_temp_body_prop, m_circle, m_polygon,
                                         m_rectangle, o_front, o_side)
from conmech.simulations import simulation_runner


def get_C_temp_scenarios(mesh_density, final_time):
    C_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            C_coeff=np.array([[1.5, 0], [0, 0.5]]), K_coeff=default_K_coeff,
        ),
        get_temp_body_prop(
            C_coeff=np.array([[1.0, -0.5], [-0.5, 1.0]]), K_coeff=default_K_coeff,
        ),
        # not allowed in physical law
        get_temp_body_prop(
            C_coeff=np.array([[1.0, 0.5], [-0.5, 1.0]]), K_coeff=default_K_coeff,
        ),
    ]
    return [
        TemperatureScenario(
            id=f"C_{i}",
            mesh_data=MeshProperties(
                dimension=2,
                mesh_type=m_rectangle,
                scale=[1],
                mesh_density=[mesh_density],
                is_adaptive=False,
            ),
            body_prop=temp_body_prop,
            obstacle_prop=default_temp_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0]),
            obstacles=o_side,
            heat_function=np.array([2]),
        )
        for i, temp_body_prop in enumerate(C_temp_body_prop)
    ]


def get_K_temp_scenarios(mesh_density, final_time):
    K_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            C_coeff=default_C_coeff, K_coeff=np.array([[0.5, 0], [0, 0.5]]),
        ),
        get_temp_body_prop(
            C_coeff=default_C_coeff, K_coeff=np.array([[0.1, 0.1], [0.1, 0.1]]),
        ),
        get_temp_body_prop(
            C_coeff=default_C_coeff, K_coeff=np.array([[0.1, -0.1], [-0.1, 0.1]]),
        ),
        # not allowed in physical law
        get_temp_body_prop(
            C_coeff=default_C_coeff, K_coeff=np.array([[0.1, -0.1], [0.1, 0.1]]),
        ),
    ]

    def h_corner(ip, mp, md, t):
        x_scaled = ip[0] / md.scale_x
        y_scaled = ip[1] / md.scale_y
        if x_scaled < 0.1 and y_scaled < 0.1:
            return -50.0  # -100
        return 0.0

    return [
        TemperatureScenario(
            id=f"K_{i}",
            mesh_data=MeshProperties(
                dimension=2,
                mesh_type=m_rectangle,
                scale=[1],
                mesh_density=[mesh_density],
                is_adaptive=False,
            ),
            body_prop=temp_body_prop,
            obstacle_prop=default_temp_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0]),
            obstacles=o_side,
            heat_function=h_corner,
        )
        for i, temp_body_prop in enumerate(K_temp_body_prop)
    ]


def get_polygon_scenarios(mesh_density, final_time):
    polygon_scenario = lambda i, forces_function, obstacle: TemperatureScenario(
        id=f"polygon_{i}",
        mesh_data=MeshProperties(
            dimension=2,
            mesh_type=m_polygon,
            scale=[1],
            mesh_density=[mesh_density],
            is_adaptive=False,
        ),
        body_prop=default_temp_body_prop,
        obstacle_prop=default_temp_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=forces_function,
        obstacles=obstacle,
        heat_function=np.array([0]),
    )

    return [
        polygon_scenario(i=0, forces_function=f_rotate_fast, obstacle=o_side),
        polygon_scenario(i=1, forces_function=f_rotate_fast, obstacle=o_front),
        polygon_scenario(i=2, forces_function=np.array([1, 0]), obstacle=o_front),
    ]


def get_friction_scenarios(mesh_density, final_time):
    friction_scenario = lambda i: TemperatureScenario(
        id="circle_flat_A_roll",
        mesh_data=MeshProperties(
            dimension=2,
            mesh_type=m_circle,
            scale=[1],
            mesh_density=[mesh_density],
        ),
        body_prop=
        get_temp_body_prop(
            C_coeff=default_C_coeff, K_coeff=np.array([[0.01, 0], [0, 0.01]]),
        ),
        obstacle_prop=default_temp_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=np.array([1.0, -0.5]),
        obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
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

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=Config(SHELL=False),
    )


if __name__ == "__main__":
    main()
