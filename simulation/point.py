"""
Created at 22.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np

from simulation.grid import Grid


class Point:
    LEFT_BOTTOM_CORNER = 0
    LEFT_SIDE = 1
    LEFT_TOP_CORNER = 2
    TOP = 3
    RIGHT_TOP_CORNER = 4
    RIGHT_SIDE = 5
    RIGHT_BOTTOM_CORNER = 6
    BOTTOM = 7
    NORMAL_MIDDLE = 8
    CROSS = 9

    @staticmethod
    def ax_ay(point):
        n_dx = np.array([1., 1., -1., -1., -1., -1., 1., 1.]) * 0.5  # normal dx
        n_dy = np.array([-1., -1., -1., -1., 1., 1., 1., 1.]) * 0.5

        c_dx = np.array([1., 0., -1., 0., 0., 0., 0., 0.])  # cross dx
        c_dy = np.array([0., -1., 0., 1., 0., 0., 0., 0.])

        dx = n_dx
        dy = n_dy
        if point[2] == Grid.TOP:
            f = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        elif point[2] == Grid.RIGHT_TOP_CORNER:
            f = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        elif point[2] == Grid.RIGHT_SIDE:
            f = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        elif point[2] == Grid.RIGHT_BOTTOM_CORNER:
            f = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        elif point[2] == Grid.BOTTOM:
            f = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        elif point[2] == Grid.NORMAL_MIDDLE:
            f = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        elif point[2] == Grid.CROSS:
            f = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # only 4 used
            dx = c_dx
            dy = c_dy
        else:
            raise ValueError

        result = (f * dx, f * dy)
        return result

