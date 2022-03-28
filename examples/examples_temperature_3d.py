from deep_conmech.common import simulation_runner
from deep_conmech.scenarios import *


def main(mesh_density=3, final_time=1, plot_animation=True):
    all_temp_body_prop = [
        get_temp_body_prop(
            C_coeff=np.array([[1, 0], [0, 0.5]]),
            K_coeff=np.array([[0.1, 0], [0, 0.1]]),
        ),
    ]

    all_scenarios = []
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
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
    )


if __name__ == "__main__":
    main()
