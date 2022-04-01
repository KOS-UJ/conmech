"""
Created at 22.08.2019
"""

import numpy as np

from conmech.edge import Edge
from conmech.grid import Grid
from conmech.point import Point


class GridFactory:
    @staticmethod
    def add_point(grid, x, y, t):
        # avoid adding two points in the same position, BUT MAYBE IT SHOULD OVERRIDE TYPE
        for point in grid.Points:
            if point[Point.X] == x and point[Point.Y] == y:
                return
        grid.Points = np.append([[x, y, t]], grid.Points, axis=0)  # add new point at the beginning
        # increment all edges ends
        grid.Edges[:, Edge.START] += 1
        grid.Edges[:, Edge.STOP] += 1

    @staticmethod
    def add_edge(grid, start_point: int, stop_point: int, t: int):
        stop_is_smaller_in_x = grid.Points[stop_point][Point.X] < grid.Points[start_point][Point.X]
        stop_is_equals_in_x = grid.Points[stop_point][Point.X] == grid.Points[start_point][Point.X]
        stop_is_smaller_in_y = grid.Points[stop_point][Point.Y] < grid.Points[start_point][Point.Y]
        stop_is_smaller = stop_is_smaller_in_x or (stop_is_equals_in_x and stop_is_smaller_in_y)

        if stop_is_smaller:
            start_point, stop_point = stop_point, start_point
        grid.Edges = np.append([[start_point, stop_point, t]], grid.Edges, axis=0)

    @staticmethod
    def startBorder(grid, x, y):
        GridFactory.add_point(grid, x, y, 0)

    @staticmethod
    def addBorderD(grid, x, y):
        GridFactory.add_point(grid, x, y, 1)
        GridFactory.add_edge(grid, 1, 0, Edge.VERTICAL)
        grid.BorderEdgesD += 1

    @staticmethod
    def addBorderDLast(grid, x, y):
        GridFactory.add_point(grid, x, y, 2)
        GridFactory.add_edge(grid, 1, 0, Edge.VERTICAL)
        grid.BorderEdgesD += 1

    @staticmethod
    def addBorderNTop(grid, x, y):
        GridFactory.add_point(grid, x, y, 3)
        GridFactory.add_edge(grid, 1, 0, Edge.HORIZONTAL)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderNTopLast(grid, x, y):
        GridFactory.add_point(grid, x, y, 4)
        GridFactory.add_edge(grid, 1, 0, Edge.HORIZONTAL)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderNSide(grid, x, y):
        GridFactory.add_point(grid, x, y, 5)
        GridFactory.add_edge(grid, 0, 1, Edge.VERTICAL)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderNSideLast(grid, x, y):
        GridFactory.add_point(grid, x, y, 6)
        GridFactory.add_edge(grid, 0, 1, Edge.VERTICAL)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderC(grid, x, y):
        GridFactory.add_point(grid, x, y, 7)
        GridFactory.add_edge(grid, 0, 1, Edge.HORIZONTAL)
        grid.BorderEdgesC += 1

    @staticmethod
    def stopBorder(grid):
        GridFactory.add_edge(grid, len(grid.Points) - 1, 0, Edge.HORIZONTAL)
        grid.BorderEdgesC += 1

    @staticmethod
    def construct(sizeH, sizeL, height):
        grid = Grid()
        grid.SizeH = sizeH
        grid.SizeL = sizeL
        grid.Height = height
        grid.longTriangleSide = height / sizeH
        grid.Length = grid.longTriangleSide * sizeL

        grid.halfLongTriangleSide = grid.longTriangleSide * 0.5
        grid.shortTriangleSide = grid.longTriangleSide * np.sqrt(2) * 0.5
        grid.halfShortTriangleSide = grid.shortTriangleSide * 0.5
        grid.TriangleArea = (grid.longTriangleSide * grid.longTriangleSide) / 4.

        GridFactory.startBorder(grid, 0, 0)
        GridFactory.add_dirichlet_points(grid)
        GridFactory.add_neumann_points(grid)
        GridFactory.add_contact_points(grid)
        GridFactory.stopBorder(grid)
        GridFactory.add_interior_points(grid)
        GridFactory.sort_points(grid)
        GridFactory.add_edges_idx(grid)
        GridFactory.add_edges_start(grid)
        return grid

    @staticmethod
    def add_dirichlet_points(grid):
        for i in range(1, grid.SizeH):
            GridFactory.addBorderD(grid, 0, i * grid.longTriangleSide)
        GridFactory.addBorderDLast(grid, 0, grid.SizeH * grid.longTriangleSide)

    @staticmethod
    def add_neumann_points(grid):
        for i in range(1, grid.SizeL):
            GridFactory.addBorderNTop(grid, i * grid.longTriangleSide, grid.Height)
        GridFactory.addBorderNTopLast(grid, grid.SizeL * grid.longTriangleSide, grid.Height)

        for i in range(grid.SizeH - 1, 0, -1):
            GridFactory.addBorderNSide(grid, grid.Length, i * grid.longTriangleSide)
        GridFactory.addBorderNSideLast(grid, grid.Length, 0.)

    @staticmethod
    def add_contact_points(grid):
        for i in range(grid.SizeL - 1, 0, -1):
            GridFactory.addBorderC(grid, i * grid.longTriangleSide, 0)

    @staticmethod
    def add_interior_points(grid):
        for i in range(0, grid.SizeL):
            for j in range(1, grid.SizeH):
                x1 = i * grid.longTriangleSide
                x2 = (i + 1) * grid.longTriangleSide
                y = j * grid.longTriangleSide
                GridFactory.add_point(grid, x1, y, 8)
                GridFactory.add_point(grid, x2, y, 8)
                a = grid.getPoint(x1, y)
                b = grid.getPoint(x2, y)
                GridFactory.add_edge(grid, a, b, Edge.HORIZONTAL)

        for i in range(1, grid.SizeL):
            for j in range(0, grid.SizeH):
                x = i * grid.longTriangleSide
                y1 = j * grid.longTriangleSide
                y2 = (j + 1) * grid.longTriangleSide
                GridFactory.add_point(grid, x, y1, 8)
                GridFactory.add_point(grid, x, y2, 8)
                a = grid.getPoint(x, y1)
                b = grid.getPoint(x, y2)
                GridFactory.add_edge(grid, a, b, Edge.VERTICAL)

        for i in range(0, grid.SizeL):
            for j in range(0, grid.SizeH):
                x = (i + 0.5) * grid.longTriangleSide
                y = (j + 0.5) * grid.longTriangleSide
                GridFactory.add_point(grid, x, y, 9)
                a = grid.getPoint(x, y)

                b = grid.getPoint(i * grid.longTriangleSide, (j + 1) * grid.longTriangleSide)
                GridFactory.add_edge(grid, a, b, Edge.DIAGONAL_DOWN_TO_CROSS_POINT)

                b = grid.getPoint((i + 1) * grid.longTriangleSide, (j + 1) * grid.longTriangleSide)
                GridFactory.add_edge(grid, a, b, Edge.DIAGONAL_UP_TO_GRID_POINT)

                b = grid.getPoint((i + 1) * grid.longTriangleSide, j * grid.longTriangleSide)
                GridFactory.add_edge(grid, a, b, Edge.DIAGONAL_DOWN_TO_GRID_POINT)

                b = grid.getPoint(i * grid.longTriangleSide, j * grid.longTriangleSide)
                GridFactory.add_edge(grid, a, b, Edge.DIAGONAL_UP_TO_CROSS_POINT)

    @staticmethod
    def sort_points(grid):
        start_interior = 0
        for i in range(len(grid.Points)):
            point = grid.Points[i]
            if point[Point.Y] == 0 and point[Point.X] != 0:  # Assume it is contact
                GridFactory.swap_rows(grid.Points, i, start_interior)
                GridFactory.replace_point_id(grid.Edges, i, start_interior)
                start_interior += 1
        start_interior_edge = 0
        for i in range(len(grid.Edges)):
            edge = grid.Edges[i]
            if edge[Edge.START] < start_interior and edge[Edge.STOP] < start_interior \
                    or edge[Edge.START] == len(grid.Points) - 1 and edge[Edge.STOP] == 0:
                GridFactory.swap_rows(grid.Edges, i, start_interior_edge)
                start_interior_edge += 1

    @staticmethod
    def swap_rows(rows, i, j):
        row = np.empty(3)
        row[:] = rows[j, :]
        rows[j, :] = rows[i, :]
        rows[i, :] = row[:]

    @staticmethod
    def replace_point_id(edges, i, j):
        for edge in edges:
            if edge[Edge.START] == i:
                edge[Edge.START] = j
            elif edge[Edge.START] == j:
                edge[Edge.START] = i

            if edge[Edge.STOP] == i:
                edge[Edge.STOP] = j
            elif edge[Edge.STOP] == j:
                edge[Edge.STOP] = i

    @staticmethod
    def add_edges_idx(grid):
        grid.edges_idx = np.argsort(grid.Edges[:, 0])

    @staticmethod
    def add_edges_start(grid):
        grid.edges_start = np.zeros(grid.Points.shape[0] + 1, dtype=int)
        point = 0
        for i in range(len(grid.Edges)):
            while grid.Edges[grid.edges_idx[i], 0] >= point:
                grid.edges_start[point] = i
                point += 1
        grid.edges_start[-1] = grid.Points.shape[0]
