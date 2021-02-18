"""
Created at 22.08.2019
"""

import numpy as np
from simulation.grid import Grid
from simulation.point import Point
from simulation.edge import Edge


class GridFactory:
    @staticmethod
    def addPoint(grid, x, y, t):
        i = 0
        while i < len(grid.Points):
            if grid.Points[i][0] == x and grid.Points[i][1] == y:
                return
            else:
                i += 1
        grid.Points = np.append([[x, y, t]], grid.Points, axis=0)
        for i in range(0, len(grid.Edges)):
            grid.Edges[i][0] += 1
            grid.Edges[i][1] += 1

    @staticmethod
    def addEdge(grid, i, j, t):  # always (i,j) has i<j on x OR xi equals xj and i<j on y
        a = i
        b = j
        if (grid.Points[j][0] < grid.Points[i][0] or
                (grid.Points[j][0] == grid.Points[i][0] and grid.Points[j][1] < grid.Points[i][1])):
            a = j
            b = i
        grid.Edges = np.append([[a, b, t]], grid.Edges, axis=0)

    @staticmethod
    def startBorder(grid, x, y):
        GridFactory.addPoint(grid, x, y, 0)

    @staticmethod
    def addBorderD(grid, x, y):
        GridFactory.addPoint(grid, x, y, 1)
        GridFactory.addEdge(grid, 1, 0, 2)
        grid.BorderEdgesD += 1

    @staticmethod
    def addBorderDLast(grid, x, y):
        GridFactory.addPoint(grid, x, y, 2)
        GridFactory.addEdge(grid, 1, 0, 2)
        grid.BorderEdgesD += 1

    @staticmethod
    def addBorderNTop(grid, x, y):
        GridFactory.addPoint(grid, x, y, 3)
        GridFactory.addEdge(grid, 1, 0, 1)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderNTopLast(grid, x, y):
        GridFactory.addPoint(grid, x, y, 4)
        GridFactory.addEdge(grid, 1, 0, 1)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderNSide(grid, x, y):
        GridFactory.addPoint(grid, x, y, 5)
        GridFactory.addEdge(grid, 0, 1, 2)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderNSideLast(grid, x, y):
        GridFactory.addPoint(grid, x, y, 6)
        GridFactory.addEdge(grid, 0, 1, 2)
        grid.BorderEdgesN += 1

    @staticmethod
    def addBorderC(grid, x, y):
        GridFactory.addPoint(grid, x, y, 7)
        GridFactory.addEdge(grid, 0, 1, 1)
        grid.BorderEdgesC += 1

    @staticmethod
    def stopBorder(grid):
        GridFactory.addEdge(grid, len(grid.Points) - 1, 0, 1)
        grid.BorderEdgesC += 1

    @staticmethod
    def construct(sizeH, sizeL, height):
        grid = Grid()
        grid.SizeH = sizeH
        grid.SizeL = sizeL
        grid.Height = height
        grid.longTriangleSide = float(height) / sizeH
        grid.Length = grid.longTriangleSide * sizeL

        grid.halfLongTriangleSide = float(grid.longTriangleSide) * 0.5
        grid.shortTriangleSide = float(grid.longTriangleSide) * np.sqrt(2) * 0.5
        grid.halfShortTriangleSide = float(grid.shortTriangleSide) * 0.5
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
            GridFactory.addBorderD(grid, 0, float(i) * grid.longTriangleSide)
        GridFactory.addBorderDLast(grid, 0, float(grid.SizeH) * grid.longTriangleSide)

    @staticmethod
    def add_neumann_points(grid):
        for i in range(1, grid.SizeL):
            GridFactory.addBorderNTop(grid, float(i) * grid.longTriangleSide, grid.Height)
        GridFactory.addBorderNTopLast(grid, float(grid.SizeL) * grid.longTriangleSide, grid.Height)

        for i in range(grid.SizeH - 1, 0, -1):
            GridFactory.addBorderNSide(grid, grid.Length, float(i) * grid.longTriangleSide)
        GridFactory.addBorderNSideLast(grid, grid.Length, float(0))

    @staticmethod
    def add_contact_points(grid):
        for i in range(grid.SizeL - 1, 0, -1):
            GridFactory.addBorderC(grid, float(i) * grid.longTriangleSide, 0)

    @staticmethod
    def add_interior_points(grid):
        for i in range(0, grid.SizeL):
            for j in range(1, grid.SizeH):
                x1 = float(i) * grid.longTriangleSide
                x2 = float(i + 1) * float(grid.longTriangleSide)
                y = float(j) * grid.longTriangleSide
                GridFactory.addPoint(grid, x1, y, 8)
                GridFactory.addPoint(grid, x2, y, 8)
                a = grid.getPoint(x1, y)
                b = grid.getPoint(x2, y)
                GridFactory.addEdge(grid, a, b, 1)

        for i in range(1, grid.SizeL):
            for j in range(0, grid.SizeH):
                x = float(i) * grid.longTriangleSide
                y1 = float(j) * grid.longTriangleSide
                y2 = float(j + 1) * grid.longTriangleSide
                GridFactory.addPoint(grid, x, y1, 8)
                GridFactory.addPoint(grid, x, y2, 8)
                a = grid.getPoint(x, y1)
                b = grid.getPoint(x, y2)
                GridFactory.addEdge(grid, a, b, 2)

        for i in range(0, grid.SizeL):
            for j in range(0, grid.SizeH):
                x = (float(i) + 0.5) * grid.longTriangleSide
                y = (float(j) + 0.5) * grid.longTriangleSide
                GridFactory.addPoint(grid, x, y, 9)
                a = grid.getPoint(x, y)
                b = grid.getPoint((float(i)) * grid.longTriangleSide, (float(j) + 1.0) * grid.longTriangleSide)
                GridFactory.addEdge(grid, a, b, 5)
                b = grid.getPoint((float(i) + 1.0) * grid.longTriangleSide, (float(j) + 1.0) * grid.longTriangleSide)
                GridFactory.addEdge(grid, a, b, 4)
                b = grid.getPoint((float(i) + 1.0) * grid.longTriangleSide, (float(j)) * grid.longTriangleSide)
                GridFactory.addEdge(grid, a, b, 6)
                b = grid.getPoint((float(i)) * grid.longTriangleSide, (float(j)) * grid.longTriangleSide)
                GridFactory.addEdge(grid, a, b, 3)

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
            if edge[Edge.START] < start_interior and edge[Edge.STOP] < start_interior\
                    or edge[Edge.START] == len(grid.Points) -1 and edge[Edge.STOP] == 0:
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

