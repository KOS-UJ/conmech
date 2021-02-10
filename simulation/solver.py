"""
Created at 21.08.2019
"""

import numpy as np
import numba
from simulation.matrices import Matrices
from simulation.f import F


class Solver:

    def __init__(self, grid, F0, FN, mi, la):
        self.mi = mi
        self.la = la

        self.grid = grid
        # self.time_step = time_step
        # self.currentTime = 0

        self.B = Matrices.construct_B(grid, mi, la)
        self.F = F(grid, F0, FN)

        self.u = np.zeros((self.grid.indNumber(), 2))

        self.DisplacedPoints = np.zeros([len(self.grid.Points), 3])

        for i in range(0, len(self.grid.Points)):
            self.DisplacedPoints[i] = self.grid.Points[i]

    @staticmethod
    def length(p1, p2):
        return float(np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))

    def n_down(self, e):
        # [0,-1]
        e1 = int(self.grid.Edges[e][0])
        e2 = int(self.grid.Edges[e][1])
        dx = self.grid.Points[e2][0] - self.grid.Points[e1][0]
        dy = self.grid.Points[e2][1] - self.grid.Points[e1][1]
        norm = np.sqrt(dx * dx + dy * dy)
        n = np.array([float(dy) / norm, float(-dx) / norm])
        if n[1] > 0:
            n = -n
        return n

    def JZu(self):
        # JZu = np.zeros([self.grid.indNumber(), 2])
        #
        # for i in range(0, self.grid.indNumber()):
        #     for e in range(-self.grid.BorderEdgesD - self.grid.BorderEdgesN - self.grid.BorderEdgesC,
        #                    -self.grid.BorderEdgesD - self.grid.BorderEdgesN):
        #         e1 = int(self.grid.Edges[e][0])
        #         e2 = int(self.grid.Edges[e][1])
        #         if i == e1 or i == e2:
        #             umL = 0  # u at mL
        #             if e1 < self.grid.indNumber():
        #                 umL += self.u[e1] * 0.5
        #             if e2 < self.grid.indNumber():
        #                 umL += self.u[e2] * 0.5
        #
        #             p1 = self.grid.Points[int(e1)][0:2]
        #             p2 = self.grid.Points[int(e2)][0:2]
        #             mL = (p1 + p2) * 0.5
        #             L = self.length(p1, p2)
        #             nmL = self.n_down(e)  # n at mL
        #
        #             uNmL = umL[0] * nmL[0] + umL[1] * nmL[1]
        #             uTmL = umL - uNmL * nmL
        #
        #             vNZero = nmL[0]
        #             vNOne = nmL[1]
        #             vThauZero = [1. - float(nmL[0] * nmL[0]), - float(nmL[0] * nmL[1])]
        #             vThauOne = [- float(nmL[0] * nmL[1]), 1. - float(nmL[1] * nmL[1])]
        #
        #             JZu[i][0] += L * 0.5 * (self.jnZ(uNmL, vNZero) + self.h(uNmL) * self.jtZ(uTmL, vThauZero))
        #             JZu[i][1] += L * 0.5 * (self.jnZ(uNmL, vNOne) + self.h(uNmL) * self.jtZ(uTmL, vThauOne))

        return JZu(self.grid.indNumber(), self.grid.BorderEdgesD, self.grid.BorderEdgesN, self.grid.BorderEdgesC, self.grid.Edges, self.u, self.grid.Points, self.knu)

    def set_u_and_displaced_points(self, u_vector):
        self.u = u_vector.reshape((2, -1)).T

        self.DisplacedPoints[:self.grid.indNumber(), :2] = self.grid.Points[:self.grid.indNumber(), :2] + self.u[:, :2]

    def Bu1(self):
        result = np.dot(self.B[0][0], self.u[:, 0]) + np.dot(self.B[0][1], self.u[:, 1])
        return result

    def Bu2(self):
        result = np.dot(self.B[1][0], self.u[:, 0]) + np.dot(self.B[1][1], self.u[:, 1])
        return result

    def f(self, u_vector):
        self.u = np.zeros([self.grid.indNumber(), 2])
        self.u[:, 0] = u_vector[0:self.grid.indNumber()]
        self.u[:, 1] = u_vector[self.grid.indNumber():2 * self.grid.indNumber()]

        X = Bu1() \
            + self.JZu()[:, 0] \
            - self.F.Zero

        Y = Bu2() \
            + self.JZu()[:, 1] \
            - self.F.One

        return 100000000 * np.append(X, Y)  # 10000000000

    ########################################################

    knu = 1.
    delta = 0.1

    def jnZ(self, uN, vN):  # un, vN - scalars
        if uN <= 0:
            return 0 * vN
        return (self.knu * uN) * vN

    @staticmethod
    def h(uN):
        return 0

    @staticmethod
    def jtZ(uT, vT, rho=0.0000001):  # uT, vT - vectors; REGULARYZACJA Coulomba
        M = 1 / np.sqrt(float(uT[0] * uT[0] + uT[1] * uT[1]) + float(rho ** 2))
        result = M * float(uT[0]) * float(vT[0]) + M * float(uT[1]) * float(vT[1])
        return result

@numba.njit()
def JZu(indNumber, BorderEdgesD, BorderEdgesN, BorderEdgesC, Edges, u, Points, knu):
    JZu = np.zeros((indNumber, 2))

    for i in range(0, indNumber):
        for e in range(-BorderEdgesD - BorderEdgesN - BorderEdgesC,
                       -BorderEdgesD - BorderEdgesN):
            e1 = int(Edges[e][0])
            e2 = int(Edges[e][1])
            if i == e1 or i == e2:
                umL = np.zeros(2)  # u at mL
                if e1 < indNumber:
                    umL += u[e1] * 0.5
                if e2 < indNumber:
                    umL += u[e2] * 0.5

                p1 = Points[int(e1)][0:2]
                p2 = Points[int(e2)][0:2]
                mL = (p1 + p2) * 0.5
                L = length(p1, p2)
                nmL = n_down(Points, Edges, e)  # n at mL

                uNmL = umL[0] * nmL[0] + umL[1] * nmL[1]
                uTmL = umL - uNmL * nmL

                vNZero = nmL[0]
                vNOne = nmL[1]
                vThauZero = [1. - float(nmL[0] * nmL[0]), - float(nmL[0] * nmL[1])]
                vThauOne = [- float(nmL[0] * nmL[1]), 1. - float(nmL[1] * nmL[1])]

                JZu[i][0] += L * 0.5 * (jnZ(uNmL, vNZero, knu) + h(uNmL) * jtZ(uTmL, vThauZero))
                JZu[i][1] += L * 0.5 * (jnZ(uNmL, vNOne , knu) + h(uNmL) * jtZ(uTmL, vThauOne))

    return JZu


@numba.njit()
def length(p1, p2):
    return float(np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))


@numba.njit()
def n_down(Points, Edges, e):
    # [0,-1]
    e1 = int(Edges[e][0])
    e2 = int(Edges[e][1])
    dx = Points[e2][0] - Points[e1][0]
    dy = Points[e2][1] - Points[e1][1]
    norm = np.sqrt(dx * dx + dy * dy)
    n = np.array([float(dy) / norm, float(-dx) / norm])
    if n[1] > 0:
        n = -n
    return n


@numba.njit()
def jnZ(uN, vN, knu):  # un, vN - scalars
    if uN <= 0:
        return 0 * vN
    return (knu * uN) * vN


@numba.njit()
def h(uN):
    return 0


@numba.njit()
def jtZ(uT, vT, rho=0.0000001):  # uT, vT - vectors; REGULARYZACJA Coulomba
    M = 1 / np.sqrt(float(uT[0] * uT[0] + uT[1] * uT[1]) + float(rho ** 2))
    result = M * float(uT[0]) * float(vT[0]) + M * float(uT[1]) * float(vT[1])
    return result


@numba.njit()
def Bu1(B, u):
    result = np.dot(B[0][0], u[:, 0]) + np.dot(B[0][1], u[:, 1])
    return result


@numba.njit()
def Bu2(B, u):
    result = np.dot(B[1][0], u[:, 0]) + np.dot(B[1][1], u[:, 1])
    return result


@numba.njit()
def f(u_vector, indNumber, BorderEdgesD, BorderEdgesN, BorderEdgesC, Edges, Points, knu, B, F_Zero, F_One):
    u = np.zeros((indNumber, 2))
    u[:, 0] = u_vector[0:indNumber]
    u[:, 1] = u_vector[indNumber:2 * indNumber]

    X = Bu1(B, u) \
        + JZu(indNumber, BorderEdgesD, BorderEdgesN, BorderEdgesC, Edges, u, Points, knu)[:, 0] \
        - F_Zero

    Y = Bu2(B, u) \
        + JZu(indNumber, BorderEdgesD, BorderEdgesN, BorderEdgesC, Edges, u, Points, knu)[:, 1] \
        - F_One

    return 100000000 * np.append(X, Y)  # 10000000000
