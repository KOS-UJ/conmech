from conmech.helpers import cmh
from deep_conmech.common import config
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *


def main():
    path = f"EXAMPLES 3D - {cmh.CURRENT_TIME}"

    all_scenarios = [
        Scenario(
            id="ball_roll",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_ball_3d, scale=[1], mesh_density=[2]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=0.5),
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacles=np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]),
        ),
        Scenario(
            id="ball_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_ball_3d, scale=[1], mesh_density=[4]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=4.0),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
        Scenario(
            id="cube_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_cube_3d, scale=[1], mesh_density=[3]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=4.0),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
    ]

    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            SettingRandomized.get_setting,
            path,
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
            description="Printing simulator 3D",
        )


if __name__ == "__main__":
    main()
