from matplotlib.pyplot import draw
from conmech.helpers import helpers
from deep_conmech.graph.helpers import thh
import deep_conmech.common.config as config
import deep_conmech.scenarios as scenarios
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.scenarios import *


def main():
    path = f"SIMULATOR 2D - {helpers.CURRENT_TIME}"

    scenario = Scenario(
        id="scenario_test",
        mesh_type=scenarios.m_circle,
        mesh_density=4,
        scale=1,
        forces_function=np.array([0.0, -0.01]),
        obstacles=np.array([[[0.5, 1.0]], [[0.0, -0.0]]]),
        is_adaptive=False,
        episode_steps=400,
        dim=2
    )

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
