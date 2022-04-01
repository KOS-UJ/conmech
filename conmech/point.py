"""
Created at 22.08.2019
"""

from typing import Tuple

import numpy as np

from conmech.grid import Grid


class Point:
    GRID_POINT_DX = np.array([1., 1., -1., -1., -1., -1., 1., 1.]) * 0.5
    GRID_POINT_DY = np.array([-1., -1., -1., -1., 1., 1., 1., 1.]) * 0.5
    CROSS_POINT_DX = np.array([1., 0., -1., 0., 0., 0., 0., 0.])
    CROSS_POINT_DY = np.array([0., -1., 0., 1., 0., 0., 0., 0.])

    TOP_ID = 3
    assert Grid.TOP == TOP_ID
    assert Grid.RIGHT_TOP_CORNER == TOP_ID + 1
    assert Grid.RIGHT_SIDE == TOP_ID + 2
    assert Grid.RIGHT_BOTTOM_CORNER == TOP_ID + 3
    assert Grid.BOTTOM == TOP_ID + 4
    assert Grid.NORMAL_MIDDLE == TOP_ID + 5
    assert Grid.CROSS == TOP_ID + 6
    F_SLOPES = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 1, 1],
                           [1, 1, 0, 0, 0, 0, 1, 1],
                           [1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 0, 0, 0, 0]])

    X = 0
    Y = 1
    TYPE = 2

    @staticmethod
    def ax_ay(point):
        slopes = Point.get_slopes(point_type=int(point[Point.TYPE]))
        dx, dy = Point.get_slope_direction(point_type=int(point[Point.TYPE]))
        result = (slopes * dx, slopes * dy)
        return result

    @staticmethod
    def get_slopes(point_type: int) -> np.ndarray:
        return Point.F_SLOPES[point_type - Point.TOP_ID]

    @staticmethod
    def get_slope_direction(point_type: int) -> Tuple[int, int]:
        if point_type == Grid.CROSS:
            dx = Point.CROSS_POINT_DX
            dy = Point.CROSS_POINT_DY
        else:
            dx = Point.GRID_POINT_DX
            dy = Point.GRID_POINT_DY
        return dx, dy
