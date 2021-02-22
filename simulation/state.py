"""
Created at 18.02.2021
"""

import numpy as np


class State:

    def __init__(self, grid):
        self.grid = grid
        self.displacement: np.ndarray = np.zeros((self.grid.independent_num, 2))
        self.displaced_points: np.ndarray = np.zeros([len(self.grid.Points), 3])
        for i in range(0, len(self.grid.Points)):
            self.displaced_points[i] = self.grid.Points[i]

    def set_u_and_displaced_points(self, displacement_vector: np.ndarray):
        self.displacement = displacement_vector.reshape((2, -1)).T
        self.displaced_points[:self.grid.independent_num, :2] = \
            self.grid.Points[:self.grid.independent_num, :2] + self.displacement[:, :2]
