from matplotlib.pyplot import draw
from conmech.helpers import cmh
from deep_conmech.graph.helpers import thh
import deep_conmech.common.config as config
import deep_conmech.scenarios as scenarios
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.scenarios import *


def main():
    path = f"SIMULATOR 2D - {cmh.CURRENT_TIME}"

    scenario = Scenario(
        id="scenario",
        dim=2,
        mesh_type=scenarios.m_rectangle,
        mesh_density=16,
        scale=1,
        forces_function=np.array([0.5, 0.0]),
        obstacles=np.array([[[0., 1.0]], [[0.0, 0.01]]]),
        is_adaptive=False,
        episode_steps=200
    )
    #remove dim ftom Scenario
    #standardize episode steps
    # turn off normalization
    # check different time steps (and mesh sizes)
    # use boundary normals instead of obstacle (?)
    # standardize boundary indices and initial_vector
    plotter_mapper.print_one_dynamic(
        Calculator.solve,
        scenario,
        path,
        simulate_dirty_data=config.SIMULATE_DIRTY_DATA,
        draw_base=False,
        draw_detailed=True,
        description="Printing simulator 2D",
    )


if __name__ == "__main__":
    main()


'''
    scenario = Scenario(
        id="scenario",
        dim=2,
        mesh_type=scenarios.m_rectangle,
        mesh_density=2,
        scale=1,
        forces_function=np.array([0.0, -0.5]),
        obstacles=np.array([[[0.7, 1.0]], [[0.0, -0.01]]]),
        is_adaptive=False,
        episode_steps=400
    )
'''