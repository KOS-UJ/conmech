import deep_conmech.common.config as config
import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main():
    path = f"SIMULATOR 2D - {cmh.CURRENT_TIME}"

    all_scenarios = [
        Scenario(
            id="circle_slide_roll",
            mesh_data=MeshData(
                dimension=2, mesh_type=scenarios.m_circle, scale=[1], mesh_density=[5]
            ),
            coefficients=scenarios.coefficients,
            obstacle_coefficients=scenarios.obstacle_coefficients,
            forces_function=np.array([0.0, -0.5]),
            obstacles=np.array([[[0.7, 1.0]], [[0.0, -0.01]]]),
            # obstacles=np.array([[[0.7, 1.0], [-0.5, 1.0]], [[0.0, -0.01], [4.0, -0.01]]]),
            episode_steps=400,
        ),
        Scenario(
            id="circle_flat_roll",
            mesh_data=MeshData(
                dimension=2, mesh_type=scenarios.m_circle, scale=[1], mesh_density=[5]
            ),
            coefficients=scenarios.coefficients,
            obstacle_coefficients=scenarios.obstacle_coefficients,
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
            episode_steps=200,
        ),
        Scenario(
            id="rectangle_flat_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_rectangle,
                scale=[1],
                mesh_density=[8],
            ),
            coefficients=scenarios.coefficients,
            obstacle_coefficients=scenarios.obstacle_coefficients,
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
            episode_steps=200,
        ),
    ]
    # change name boundary to contact
    # ball falling from staircase
    # if step does not cause penetration, use fsolve, otherwise return and use
    # remove dim from Scenario
    # standardize episode steps
    # check different time steps (and mesh sizes)
    # use boundary normals instead of obstacle (?)
    # standardize boundary indices and initial_vector
    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            path,
            simulate_dirty_data=config.SIMULATE_DIRTY_DATA,
            draw_base=False,
            draw_detailed=True,
            description=f"Printing simulator 2D",
        )


if __name__ == "__main__":
    main()

