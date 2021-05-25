"""
Created at 21.08.2019
"""

import numpy as np

from conmech.point import Point
from conmech.edge import Edge


class Matrices:
    @staticmethod
    def construct_A(grid, th, ze):

        # Added matrix A taking into account problems with velocity.
        #  Now C from Schur Method will depend on A instead of B

        AX = np.zeros([grid.independent_num, 8])  # area with dx
        AY = np.zeros([grid.independent_num, 8])  # area with dy
        for i in range(grid.independent_num):
            p = grid.Points[i]
            AX[i], AY[i] = Point.ax_ay(p)

        W11 = Matrices.multiply(grid, AX, AX)
        W12 = Matrices.multiply(grid, AX, AY)
        W21 = Matrices.multiply(grid, AY, AX)  # np.transpose(W12)
        W22 = Matrices.multiply(grid, AY, AY)

        A = [[(2 * th + ze) * W11 + th * W22,
              th * W21 + ze * W12],
             [ze * W21 + th * W12,
              th * W11 + (2 * th + ze) * W22]
             ]


        A = np.asarray(A)

        #self.A12 = scipy.sparse.lil_matrix((th * self.W21 + ze * self.W12)).astype(np.single)
        #self.A21 = scipy.sparse.lil_matrix((ze * self.W21 + th * self.W12)).astype(np.single)
        #self.A11 = scipy.sparse.lil_matrix(((2 * th + ze) * self.W11 + th * self.W22)).astype(np.single)
        #self.A22 = scipy.sparse.lil_matrix((th * self.W11 + (2 * th + ze) * self.W22)).astype(np.single)

        return A

    @staticmethod
    def construct_B(grid, mi, la):
        AX = np.zeros([grid.independent_num, 8])  # area with dx
        AY = np.zeros([grid.independent_num, 8])  # area with dy
        for i in range(grid.independent_num):
            p = grid.Points[i]
            AX[i], AY[i] = Point.ax_ay(p)

        W11 = Matrices.multiply(grid, AX, AX)
        W12 = Matrices.multiply(grid, AX, AY)
        W21 = Matrices.multiply(grid, AY, AX)  # np.transpose(W12)
        W22 = Matrices.multiply(grid, AY, AY)

        B = [[(2 * mi + la) * W11 + mi * W22,
              mi * W21 + la * W12],
             [la * W21 + mi * W12,
              mi * W11 + (2 * mi + la) * W22]
             ]

        B = np.asarray(B)

        return B

    @staticmethod
    def construct_U(grid):
        A = np.zeros([grid.independent_num, 8])
        for i in range(grid.independent_num):
            p = grid.Points[i]
            A[i] = Point.get_slopes(point_type=int(p[Point.TYPE]))

        _U = Matrices.multiply(grid, A, A)
        _U *= grid.shortTriangleSide ** 2 / 24
        for i in range(grid.independent_num):
            _U[i, i] *= 2

        _U = [[_U, np.zeros_like(_U)],
              [np.zeros_like(_U), _U]]

        U = np.asarray(_U)

        return U

    @staticmethod
    def multiply(grid, AK, AL):
        W = np.zeros([grid.independent_num, grid.independent_num])

        for i in range(grid.independent_num):
            W[i][i] = np.sum(AK[i] * AL[i])

        for edge in grid.Edges:
            i = edge[0]
            j = edge[1]
            if i < grid.independent_num and j < grid.independent_num:
                c1i, c1j, c2i, c2j = Edge.c(edge)
                W[i][j] = AK[i][c1i] * AL[j][c1j] + AK[i][c2i] * AL[j][c2j]
                W[j][i] = AL[i][c1i] * AK[j][c1j] + AL[i][c2i] * AK[j][c2j]

        return W
