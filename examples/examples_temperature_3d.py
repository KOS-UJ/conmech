from deep_conmech.common import simulation_runner
from deep_conmech.scenarios import *


def get_C_temp_scenarios(mesh_density, final_time):
    C_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            C_coeff=np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.5]]),
            K_coeff=default_K_coeff,
        ),
        get_temp_body_prop(
            C_coeff=np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]]),
            K_coeff=default_K_coeff,
        ),
        # not allowed in physical law
        get_temp_body_prop(
            C_coeff=np.array([[1.0, -0.5, -0.5], [0.5, 1.0, -0.5], [0.5, 0.5, 1.0]]),
            K_coeff=default_K_coeff,
        ),
    ]

    return [
        TemperatureScenario(
            id=f"C_{i}",
            mesh_data=MeshData(
                dimension=3,
                mesh_type=m_cube_3d,
                scale=[1],
                mesh_density=[mesh_density],
                is_adaptive=False,
            ),
            body_prop=temp_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0, 0]),
            obstacles=None,  # np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]]),
            heat_function=np.array([2]),
        )
        for i, temp_body_prop in enumerate(C_temp_body_prop)
    ]


def get_K_temp_scenarios(mesh_density, final_time):
    K_temp_body_prop = [
        default_temp_body_prop,
        get_temp_body_prop(
            C_coeff=default_C_coeff,
            K_coeff=np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
        ),
        get_temp_body_prop(
            C_coeff=default_C_coeff,
            K_coeff=np.array([[0.1, 0., 0.1], [0., 0.1, 0.], [0.1, 0., 0.1]]),
        ),
        get_temp_body_prop(
            C_coeff=default_C_coeff,
            K_coeff=np.array([[0.1, -0.1, 0.], [-0.1, 0.1, 0.], [0., 0., 0.1]]),
        ),
        # not allowed in physical law
        get_temp_body_prop(
            C_coeff=default_C_coeff,
            K_coeff=np.array([[0.1, 0.1, -0.1], [-0.1, 0.1, 0.1], [0.1, -0.1, 0.1]]),
        ),
    ]

    def h_corner(ip, mp, md, t):
        x_scaled = ip[0] / md.scale_x
        y_scaled = ip[1] / md.scale_y
        z_scaled = ip[2] / md.scale_z
        if x_scaled > 0.9 and y_scaled < 0.1 and z_scaled < 0.1:
            return -1000
        return 0.0

    return [
        TemperatureScenario(
            id=f"K_{i}",
            mesh_data=MeshData(
                dimension=3,
                mesh_type=m_cube_3d,
                scale=[1],
                mesh_density=[mesh_density],
                is_adaptive=False,
            ),
            body_prop=temp_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0, 0, 0]),
            obstacles=None,
            heat_function=h_corner,
        )
        for i, temp_body_prop in enumerate(K_temp_body_prop)
    ]


def main(mesh_density=5, final_time=3, plot_animation=True):
    all_scenarios = []
    # all_scenarios.extend(get_C_temp_scenarios(mesh_density, final_time))
    all_scenarios.extend(get_K_temp_scenarios(mesh_density, final_time))
    all_scenarios.extend(
        [
            TemperatureScenario(
                id="temperature_3d_cube_throw",
                mesh_data=MeshData(
                    dimension=3,
                    mesh_type=m_cube_3d,
                    scale=[1],
                    mesh_density=[mesh_density],
                ),
                body_prop=default_temp_body_prop,
                obstacle_prop=default_obstacle_prop,
                schedule=Schedule(final_time=final_time),
                forces_function=f_rotate_3d,
                obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
                heat_function=np.array([0]),
            ),
        ]
    )

    simulation_runner.run_examples(
        all_scenarios=all_scenarios, file=__file__, plot_animation=plot_animation,
    )


if __name__ == "__main__":
    main()
