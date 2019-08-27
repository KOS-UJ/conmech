"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np

from simulation.point import Point
from simulation.edge import Edge


class Matrices:

    @staticmethod
    def construct_B(grid, mi, la):
        AX = np.zeros([grid.indNumber(), 8])  # area with dx
        AY = np.zeros([grid.indNumber(), 8])  # area with dy
        for i in range(grid.indNumber()):
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

        return B

    @staticmethod
    def multiply(grid, AK, AL):
        W = np.zeros([grid.indNumber(), grid.indNumber()])

        for i in range(grid.indNumber()):
            W[i][i] = np.sum(AK[i] * AL[i])

            # c - contacting triangles numbers
            for j in range(grid.indNumber()):
                edge = grid.get_edge(i, j)
                c1i, c1j, c2i, c2j = Edge.c(edge)

                if c1i >= 0:  # edge was found
                    W[i][j] = AK[i][c1i] * AL[j][c1j] + AK[i][c2i] * AL[j][c2j]
                    W[j][i] = AL[i][c1i] * AK[j][c1j] + AL[i][c2i] * AK[j][c2j]

        return W
