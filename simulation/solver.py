"""
Created at 21.08.2019
"""

import numpy as np
import numba
from simulation.matrices import Matrices
from simulation.f import F


class Solver:

    def __init__(self,
                 grid,
                 F0,
                 FN,
                 mi,
                 la,
                 contact_law_normal_direction,
                 contact_law_tangential_direction,
                 friction_bound
    ):
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

        self.f = make_f(jnZ=contact_law_normal_direction, jtZ=contact_law_tangential_direction, h=friction_bound)

    def set_u_and_displaced_points(self, u_vector):
        self.u = u_vector.reshape((2, -1)).T

        self.DisplacedPoints[:self.grid.indNumber(), :2] = self.grid.Points[:self.grid.indNumber(), :2] + self.u[:, :2]

    ########################################################

    knu = 1.
    delta = 0.1


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
def Bu1(B, u):
    result = np.dot(B[0][0], u[:, 0]) + np.dot(B[0][1], u[:, 1])
    return result


@numba.njit()
def Bu2(B, u):
    result = np.dot(B[1][0], u[:, 0]) + np.dot(B[1][1], u[:, 1])
    return result

def make_f(jnZ, jtZ, h):
    jnZ = numba.njit(jnZ)
    jtZ = numba.njit(jtZ)
    h = numba.njit(h)

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
    def f(u_vector, indNumber, BorderEdgesD, BorderEdgesN, BorderEdgesC, Edges, Points, knu, B, F_Zero, F_One):
        u = np.zeros((indNumber, 2))
        u[:, 0] = u_vector[0:indNumber]
        u[:, 1] = u_vector[indNumber:2 * indNumber]

        jZu = JZu(indNumber, BorderEdgesD, BorderEdgesN, BorderEdgesC, Edges, u, Points, knu)

        X = Bu1(B, u) \
            + jZu[:, 0] \
            - F_Zero

        Y = Bu2(B, u) \
            + jZu[:, 1] \
            - F_One

        return 100000000 * np.append(X, Y)  # 10000000000

    return f
