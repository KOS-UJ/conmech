"""
Created at 12.04.2021
"""

import numpy as np
import pytest

from conmech.edge import Edge
from conmech.grid import Grid
from conmech.grid_factory import GridFactory
from conmech.point import Point


class TestGridFactory:

    @staticmethod
    @pytest.mark.parametrize("length, repeated", ((1, 0), (6, 0), (9, 3)))
    def test_add_multiple_points_with_repeats(length, repeated):
        # Arrange
        grid = Grid()
        grid.Edges = np.zeros((5, 3))
        x = [0.0, 1, 3, 0, -1, -3.14, 0, -1, -3.14]
        y = [0.0, 1, -2.3, 5, 16, -10, 5, 16, -10]
        t = [Grid.CROSS, Grid.TOP, Grid.BOTTOM,
             Grid.RIGHT_BOTTOM_CORNER, Grid.LEFT_BOTTOM_CORNER, Grid.NORMAL_MIDDLE,
             Grid.CROSS, Grid.TOP, Grid.BOTTOM]

        # Act
        for i in range(length):
            GridFactory.add_point(grid, x[i], y[i], t[i])

        # Assert
        points_num = length - repeated
        assert points_num == len(grid.Points)
        assert np.all(grid.Edges[:, Edge.TYPE] == 0)
        assert np.all(grid.Edges[:, Edge.START] == points_num)
        assert np.all(grid.Edges[:, Edge.STOP] == points_num)
        for i in range(points_num):
            point = grid.Points[i]
            assert x[points_num - i - 1] == point[Point.X]
            assert y[points_num - i - 1] == point[Point.Y]
            assert t[points_num - i - 1] == point[Point.TYPE]
