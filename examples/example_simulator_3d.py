import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import nph
from matplotlib.gridspec import GridSpec

from deep_conmech.common import config
from deep_conmech.common.plotter import plotter_3d
from deep_conmech.graph.helpers import thh
from deep_conmech import scenarios
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles
from deep_conmech.common.plotter import plotter_mapper

catalog = f"output/3D - {thh.CURRENT_TIME}"



def main():
    path = f"SIMULATOR 3D - {thh.CURRENT_TIME}"

    plotter_mapper.print_one_dynamic(
        Calculator.solve,
        scenarios.scenario_3d,
        path,
        simulate_dirty_data=config.SIMULATE_DIRTY_DATA,
        print_base=False,
        description="Printing simulator 3D",
    )



if __name__ == "__main__":
    main()

