"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np

from simulation.point import Point
from simulation.edge import Edge


class Matrices:

    def __init__(self, grid, mi, la):
        self.grid = grid
        self.AX = np.zeros([self.grid.indNumber(), 8])  # area with dx
        self.AY = np.zeros([self.grid.indNumber(), 8])  # area with dy
        self.W11 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])
        self.W12 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])
        self.W21 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])
        self.W22 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])

        for i in range(self.grid.indNumber()):
            p = self.grid.Points[i]
            self.AX[i], self.AY[i] = Point.ax_ay(p)

        self.multiply(self.W11, self.AX, self.AX)
        self.multiply(self.W12, self.AX, self.AY)
        self.multiply(self.W21, self.AY, self.AX)  # np.transpose(self.W12)
        self.multiply(self.W22, self.AY, self.AY)

        self.B11 = (2 * mi + la) * self.W11 + mi * self.W22
        self.B12 = mi * self.W21 + la * self.W12
        self.B21 = la * self.W21 + mi * self.W12
        self.B22 = mi * self.W11 + (2 * mi + la) * self.W22

    def multiply(self, W, AK, AL):
        for i in range(self.grid.indNumber()):
            W[i][i] = np.sum(AK[i] * AL[i])

            # c - contacting triangles numbers
            for j in range(self.grid.indNumber()):
                edge = self.grid.get_edge(i, j)
                c1i, c1j, c2i, c2j = Edge.c(edge)

                if c1i >= 0:  # edge was found
                    W[i][j] = AK[i][c1i] * AL[j][c1j] + AK[i][c2i] * AL[j][c2j]
                    W[j][i] = AL[i][c1i] * AK[j][c1j] + AL[i][c2i] * AK[j][c2j]
