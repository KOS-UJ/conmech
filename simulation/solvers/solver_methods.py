"""
Created at 21.08.2019
"""

import numpy as np
import numba


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
    def JZu(indNumber, BorderEdgesC, Edges, u, Points):
        JZu = np.zeros((indNumber, 2))

        for i in range(0, indNumber):
            for e in range(0, BorderEdgesC):
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
                    vThauZero = np.asarray([1. - float(nmL[0] * nmL[0]), - float(nmL[0] * nmL[1])])
                    vThauOne = np.asarray([- float(nmL[0] * nmL[1]), 1. - float(nmL[1] * nmL[1])])

                    JZu[i][0] += L * 0.5 * (jnZ(uNmL, vNZero)) + h(uNmL) * jtZ(uTmL, vThauZero)
                    JZu[i][1] += L * 0.5 * (jnZ(uNmL, vNOne)) + h(uNmL) * jtZ(uTmL, vThauOne)
        return JZu

    @numba.njit()
    def f(u_vector, indNumber, BorderEdgesC, Edges, Points, B, F_Zero, F_One):
        u = np.zeros((indNumber, 2))
        u[:, 0] = u_vector[0:indNumber]
        u[:, 1] = u_vector[indNumber:2 * indNumber]

        jZu = JZu(indNumber, BorderEdgesC, Edges, u, Points)

        X = Bu1(B, u) + jZu[:, 0] - F_Zero

        Y = Bu2(B, u) + jZu[:, 1] - F_One

        return np.append(X, Y)  # 10000000000

    return f


def make_L2(jn):
    jn = numba.njit(jn)
    DIMENSION = 2

    @numba.njit()
    def Ju(indNumber, BorderEdgesC, Edges, ut_vector, Points):
        J = 0
        for e in range(0, BorderEdgesC):
            nmL = n_down(Points, Edges, e)  # n at mL

            firstPointIndex = Edges[e][0]
            secondPointIndex = Edges[e][1]

            umLx = 0.
            umLy = 0.
            offset = len(ut_vector) // DIMENSION
            if firstPointIndex < indNumber: # exclude points from Gamma_D
                umLx += 0.5 * ut_vector[firstPointIndex]
                umLy += 0.5 * ut_vector[offset + firstPointIndex]
            if secondPointIndex < indNumber: # exclude points from Gamma_D
                umLx += 0.5 * ut_vector[secondPointIndex]
                umLy += 0.5 * ut_vector[offset + secondPointIndex]

            uNmL = umLx * nmL[0] + umLy * nmL[1]
            uTmLx = umLx - uNmL * nmL[0]
            uTmLy = umLy - uNmL * nmL[1]

            firstPointCoordinates = Points[int(firstPointIndex)][0:2]
            secondPointCoordinates = Points[int(secondPointIndex)][0:2]
            edgeLength = length(firstPointCoordinates, secondPointCoordinates)

            J += edgeLength * (jn(uNmL))

        return J

    @numba.njit()
    def L2(ut_vector, indNumber, BorderEdgesC, Edges, Points, C, E):
        ju = Ju(indNumber, BorderEdgesC, Edges, ut_vector, Points)
        return 0.5*np.dot(np.dot(C, ut_vector), ut_vector) - np.dot(E, ut_vector) + ju

    return L2
