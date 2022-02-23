"""
Created at 21.08.2019
"""

import numpy as np

from conmech.point import Point
from conmech.edge import Edge


class Matrices:
    @staticmethod
    def construct_B(grid, mi, la):
        AX = np.zeros([grid.independent_nodes_count, 8])  # area with dx
        AY = np.zeros([grid.independent_nodes_count, 8])  # area with dy
        for i in range(grid.independent_nodes_count):
            p = grid.Points[i]
            AX[i], AY[i] = Point.ax_ay(p)

        W11 = Matrices.multiply(grid, AX, AX)
        W12 = Matrices.multiply(grid, AX, AY)
        W21 = Matrices.multiply(grid, AY, AX)  # np.transpose(W12)
        W22 = Matrices.multiply(grid, AY, AY)

        B = [
            [(2 * mi + la) * W11 + mi * W22, mi * W21 + la * W12],
            [la * W21 + mi * W12, mi * W11 + (2 * mi + la) * W22],
        ]

        B = np.asarray(B)

        return B

    @staticmethod
    def construct_K(grid):
        AX = np.zeros([grid.independent_nodes_count, 8])  # area with dx
        AY = np.zeros([grid.independent_nodes_count, 8])  # area with dy
        for i in range(grid.independent_nodes_count):
            p = grid.Points[i]
            AX[i], AY[i] = Point.ax_ay(p)

        W11 = Matrices.multiply(grid, AX, AX)
        W12 = Matrices.multiply(grid, AX, AY)
        W21 = Matrices.multiply(grid, AY, AX)  # np.transpose(W12)
        W22 = Matrices.multiply(grid, AY, AY)

        # TODO
        k11 = k22 = 0.1
        k12 = k21 = 0
        return k11 * W11 + k12 * W12 + k21 * W21 + k22 * W22

    @staticmethod
    def construct_C2(grid):
        A = np.zeros([grid.independent_nodes_count, 8])
        for i in range(grid.independent_nodes_count):
            p = grid.Points[i]
            A[i] = Point.get_slopes(point_type=int(p[Point.TYPE]))

        AX = np.zeros([grid.independent_nodes_count, 8])  # area with dx
        AY = np.zeros([grid.independent_nodes_count, 8])  # area with dy
        for i in range(grid.independent_nodes_count):
            p = grid.Points[i]
            AX[i], AY[i] = Point.ax_ay(p)

        A2 = A * (grid.shortTriangleSide / 6)

        U1 = Matrices.multiply(grid, A2, AX)
        U2 = Matrices.multiply(grid, A2, AY)

        # TODO
        c11 = c22 = 0.5
        c12 = c21 = 0
        C2X = c11 * U1 + c21 * U2
        C2Y = c12 * U1 + c22 * U2

        return C2X, C2Y

    @staticmethod
    def construct_U(grid):
        A = np.zeros([grid.independent_nodes_count, 8])
        for i in range(grid.independent_nodes_count):
            p = grid.Points[i]
            A[i] = Point.get_slopes(point_type=int(p[Point.TYPE]))

        _U = Matrices.multiply(grid, A, A)
        _U *= grid.shortTriangleSide ** 2 / 24
        for i in range(grid.independent_nodes_count):
            _U[i, i] *= 2

        _U = [[_U, np.zeros_like(_U)], [np.zeros_like(_U), _U]]

        U = np.asarray(_U)

        return U

    @staticmethod
    def multiply(grid, AK, AL):
        W = np.zeros([grid.independent_nodes_count, grid.independent_nodes_count])

        for i in range(grid.independent_nodes_count):
            W[i][i] = np.sum(AK[i] * AL[i])

        for edge in grid.Edges:
            i = edge[0]
            j = edge[1]
            if i < grid.independent_nodes_count and j < grid.independent_nodes_count:
                c1i, c1j, c2i, c2j = Edge.c(edge)
                W[i][j] = AK[i][c1i] * AL[j][c1j] + AK[i][c2i] * AL[j][c2j]
                W[j][i] = AL[i][c1i] * AK[j][c1j] + AL[i][c2i] * AK[j][c2j]

        return W
