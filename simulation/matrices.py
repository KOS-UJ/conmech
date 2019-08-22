"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np


class Matrices:

    def __init__(self, grid, mi, la):
        self.grid = grid
        self.AX = np.zeros([self.grid.indNumber(), 8])  # area with dx
        self.AY = np.zeros([self.grid.indNumber(), 8])  # area with dy
        self.W11 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])
        self.W12 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])
        self.W21 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])
        self.W22 = np.zeros([self.grid.indNumber(), self.grid.indNumber()])

        n_dx = np.array([1., 1., -1., -1., -1., -1., 1., 1.]) * 0.5  # normal dx
        n_dy = np.array([-1., -1., -1., -1., 1., 1., 1., 1.]) * 0.5

        c_dx = np.array([1., 0., -1., 0., 0., 0., 0., 0.])  # cross dx
        c_dy = np.array([0., -1., 0., 1., 0., 0., 0., 0.])

        for i in range(0, self.grid.indNumber()):
            p = self.grid.Points[i]
            dx = n_dx
            dy = n_dy
            if p[2] == self.grid.TOP:
                f = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            elif p[2] == self.grid.RIGHT_TOP_CORNER:
                f = np.array([0, 0, 0, 0, 0, 0, 1, 1])
            elif p[2] == self.grid.RIGHT_SIDE:
                f = np.array([1, 1, 0, 0, 0, 0, 1, 1])
            elif p[2] == self.grid.RIGHT_BOTTOM_CORNER:
                f = np.array([1, 1, 0, 0, 0, 0, 0, 0])
            elif p[2] == self.grid.BOTTOM:
                f = np.array([1, 1, 1, 1, 0, 0, 0, 0])
            elif p[2] == self.grid.NORMAL_MIDDLE:
                f = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            elif p[2] == self.grid.CROSS:
                f = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # only 4 used
                dx = c_dx
                dy = c_dy
            else:
                raise ValueError
            self.AX[i] = f * dx
            self.AY[i] = f * dy

        self.multiply(self.W11, self.AX, self.AX)
        self.multiply(self.W12, self.AX, self.AY)
        self.multiply(self.W21, self.AY, self.AX)  # np.transpose(self.W12)
        self.multiply(self.W22, self.AY, self.AY)

        self.B11 = (2 * mi + la) * self.W11 + mi * self.W22
        self.B12 = mi * self.W21 + la * self.W12
        self.B21 = la * self.W21 + mi * self.W12
        self.B22 = mi * self.W11 + (2 * mi + la) * self.W22

    def multiply(self, W, AK, AL):
        for i in range(0, self.grid.indNumber()):
            W[i][i] = np.sum(AK[i] * AL[i])

            # c - contacting triangles numbers
            for j in range(0, self.grid.indNumber()):
                c1i = -1
                c1j = -1
                c2i = -1
                c2j = -1

                if (self.grid.getEdgeType(i, j) == 1):  # 1 - from normal go right to normal
                    c1i = 3
                    c1j = 0
                    c2i = 4
                    c2j = 7
                elif (self.grid.getEdgeType(i, j) == 2):  # 2 - from normal go up to normal
                    c1i = 1
                    c1j = 6
                    c2i = 2
                    c2j = 5
                elif (self.grid.getEdgeType(i, j) == 3):  # 3 - from normal go right and up to cross
                    c1i = 2
                    c1j = 0
                    c2i = 3
                    c2j = 3
                elif (self.grid.getEdgeType(i, j) == 4):  # 4 - from cross go right and up to normal
                    c1i = 1
                    c1j = 7
                    c2i = 2
                    c2j = 6
                elif (self.grid.getEdgeType(i, j) == 5):  # 5 - from normal go right and down to cross
                    c1i = 4
                    c1j = 1
                    c2i = 5
                    c2j = 0
                elif (self.grid.getEdgeType(i, j) == 6):  # 6 - from cross go right and down to normal
                    c1i = 2
                    c1j = 1
                    c2i = 3
                    c2j = 0

                if (c1i >= 0):  # edge was found
                    W[i][j] = AK[i][c1i] * AL[j][c1j] + AK[i][c2i] * AL[j][c2j]
                    W[j][i] = AL[i][c1i] * AK[j][c1j] + AL[i][c2i] * AK[j][c2j]
