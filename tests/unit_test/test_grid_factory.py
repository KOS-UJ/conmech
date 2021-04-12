"""
Created at 12.04.2021
"""

import numpy as np
from simulation.grid import Grid
from simulation.grid_factory import GridFactory
from simulation.point import Point
from simulation.edge import Edge


class TestGridFactory:

    @staticmethod
    def test_add_single_point():
        # Arrange
        grid = Grid()
        grid.Edges = np.zeros((1, 3))
        x = 0.0
        y = 0.0
        t = Grid.CROSS

        # Act
        GridFactory.add_point(grid, x, y, t)

        # Assert
        assert 1 == len(grid.Points)
        assert np.all(grid.Edges[:, Edge.TYPE] == 0)
        assert np.all(grid.Edges[:, Edge.START] == 1)
        assert np.all(grid.Edges[:, Edge.STOP] == 1)
        point = grid.Points[0]
        assert x == point[Point.X]
        assert y == point[Point.Y]
        assert t == point[Point.TYPE]

    @staticmethod
    def test_add_multiple_points():
        # Arrange
        grid = Grid()
        grid.Edges = np.zeros((5, 3))
        length = 6
        x = [0.0, 1, 3, 0, -1, -3.14]
        y = [0.0, 1, -2.3, 5,16, -10, 1]
        t = [Grid.CROSS, Grid.TOP, Grid.BOTTOM, Grid.RIGHT_BOTTOM_CORNER, Grid.LEFT_BOTTOM_CORNER, Grid.NORMAL_MIDDLE]

        # Act
        for i in range(length):
            GridFactory.add_point(grid, x[i], y[i], t[i])

        # Assert
        assert length == len(grid.Points)
        print(grid.Edges)
        assert np.all(grid.Edges[:, Edge.TYPE] == 0)
        assert np.all(grid.Edges[:, Edge.START] == length)
        assert np.all(grid.Edges[:, Edge.STOP] == length)
        for i in range(length):
            point = grid.Points[i]
            assert x[length - i - 1] == point[Point.X]
            assert y[length - i - 1] == point[Point.Y]
            assert t[length - i - 1] == point[Point.TYPE]

    @staticmethod
    def test_add_multiple_points_with_repeats():
        # Arrange
        grid = Grid()
        grid.Edges = np.zeros((5, 3))
        length = 9
        repeated = 3
        x = [0.0, 1,    3, 0, -1, -3.14, 0, -1, -3.14]
        y = [0.0, 1, -2.3, 5, 16,   -10, 5, 16,   -10]
        t = [Grid.CROSS, Grid.TOP, Grid.BOTTOM,
             Grid.RIGHT_BOTTOM_CORNER, Grid.LEFT_BOTTOM_CORNER, Grid.NORMAL_MIDDLE,
             Grid.CROSS, Grid.TOP, Grid.BOTTOM]

        # Act
        for i in range(length):
            GridFactory.add_point(grid, x[i], y[i], t[i])

        # Assert
        assert length - repeated == len(grid.Points)
        assert np.all(grid.Edges[:, Edge.TYPE] == 0)
        assert np.all(grid.Edges[:, Edge.START] == length - repeated)
        assert np.all(grid.Edges[:, Edge.STOP] == length - repeated)
        for i in range(length - repeated):
            point = grid.Points[i]
            assert x[length - repeated - i - 1] == point[Point.X]
            assert y[length - repeated - i - 1] == point[Point.Y]
            assert t[length - repeated - i - 1] == point[Point.TYPE]
