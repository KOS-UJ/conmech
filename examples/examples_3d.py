from conmech.helpers import cmh

from deep_conmech.common import config
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *

def main():
    path = f"EXAMPLES 3D - {cmh.CURRENT_TIME}"
        
    scenario = Scenario(
        id="scenario_3d",
        mesh_type=m_cube_3d,
        mesh_density=3,
        scale=1,
        forces_function=f_rotate_3d,
        obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        is_adaptive=False,
        dim=3,
    )

    plotter_mapper.print_one_dynamic(
        Calculator.solve,
        scenario,
        path,
        simulate_dirty_data=config.SIMULATE_DIRTY_DATA,
        draw_base=False,
        draw_detailed=True,
        description="Printing simulator 3D",
    )


if __name__ == "__main__":
    main()
