"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np
from simulation.simulation_runner import SimulationRunner


class Setup:
    timeStep = 1
    gridHeight = 1

    gridSizeH = 2
    gridSizeL = 5
    F0 = np.array([-0.2, -0.2])
    FN = np.array([0, 0])
    mi = 4
    la = 4


if __name__ == '__main__':
    setup = Setup()
    SimulationRunner.run(setup)
