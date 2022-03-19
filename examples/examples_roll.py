import deep_conmech.common.config as config
import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main():
    path = f"EXAMPLES ROLL - {cmh.CURRENT_TIME}"

    all_scenarios = [
        Scenario(
            id="circle_slide_roll",
            mesh_data=MeshData(
                dimension=2, mesh_type=scenarios.m_circle, scale=[1], mesh_density=[8]
            ),
            body_coeff=scenarios.body_coeff,
            obstacle_coeff=scenarios.obstacle_coeff,
            time_data=TimeData(final_time=8.0),
            forces_function=np.array([0.0, -0.5]),
            obstacles=np.array([[[0.7, 1.0], [-0.3, 1.0]], [[0.0, -0.01], [4.0, -0.01]]]),
        ),
        Scenario(
            id="circle_flat_roll",
            mesh_data=MeshData(
                dimension=2, mesh_type=scenarios.m_circle, scale=[1], mesh_density=[5]
            ),
            body_coeff=scenarios.body_coeff,
            obstacle_coeff=scenarios.obstacle_coeff,
            time_data=TimeData(final_time=4.0),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]])
        ),
        Scenario(
            id="rectangle_flat_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_rectangle,
                scale=[1],
                mesh_density=[5],
            ),
            body_coeff=scenarios.body_coeff,
            obstacle_coeff=scenarios.obstacle_coeff,
            time_data=TimeData(final_time=4.0),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]])
        )
    ]
    # change name boundary to contact
    # ball falling from staircase
    # if step does not cause penetration, use fsolve, otherwise return and use
    # remove dim from Scenario
    # check different time steps (and mesh sizes)
    # standardize boundary indices and initial_vector
    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            path,
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
            description="Printing",
        )


if __name__ == "__main__":
    main()
