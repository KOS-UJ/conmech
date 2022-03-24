from conmech.helpers import cmh
from deep_conmech.common import config
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *


def main(mesh_density=4, final_time=3.0):
    all_scenarios = [
        Scenario(
            id="ball_roll",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_ball_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacles=np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]),
        ),
        Scenario(
            id="ball_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_ball_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
        Scenario(
            id="cube_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_cube_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
    ]

    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES 3D",
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
        )


if __name__ == "__main__":
    main()
