"""
Created at 21.08.2019
"""
from typing import Callable, Optional, Any

import numpy as np
import numba

from conmech.vertex_utils import length


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

    # @numba.njit()
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

                    # TODO: Move  to seperate method, take into account also problems with velocity
                    # u will be calculated by suming ptevious velocities times time step

                    # TODO: To make function h dependent on u_nu, we need Uzawa approach
                    #       For now, for validating we can ignore it.
                    JZu[i][0] += L * 0.5 * (jnZ(uNmL, vNZero)) + h(uNmL) * jtZ(uTmL, vThauZero)
                    JZu[i][1] += L * 0.5 * (jnZ(uNmL, vNOne)) + h(uNmL) * jtZ(uTmL, vThauOne)
        return JZu

    # @numba.njit()
    def f(u_vector, indNumber, BorderEdgesC, Edges, Points, B, F_Zero, F_One):
        u = np.zeros((indNumber, 2))
        u[:, 0] = u_vector[0:indNumber]
        u[:, 1] = u_vector[indNumber:2 * indNumber]

        jZu = JZu(indNumber, BorderEdgesC, Edges, u, Points)

        X = Bu1(B, u) + jZu[:, 0] - F_Zero

        Y = Bu2(B, u) + jZu[:, 1] - F_One

        return np.append(X, Y)  # 10000000000

    return f


def njit(func: Optional[Callable], value: Optional[Any] = 0) -> Callable:
    if func is None:
        @numba.njit()
        def const(_):
            return value
        return const
    return numba.njit(func)


def make_L2(jn: Callable, jt: Optional[Callable] = None, h: Optional[Callable] = None):
    jn = njit(jn)
    jt = njit(jt)
    h = njit(h)
    DIMENSION = 2

    @numba.njit(inline='always')
    def interpolate_point_between(first_point_id, second_point_id, vector):
        result = np.zeros(DIMENSION)
        offset = len(vector) // DIMENSION
        for i in range(DIMENSION):
            if first_point_id < offset:  # exclude points from Gamma_D, TODO ind_num
                result[i] += 0.5 * vector[i * offset + first_point_id]
        for i in range(DIMENSION):
            if second_point_id < offset:  # exclude points from Gamma_D, TODO ind_num
                result[i] += 0.5 * vector[i * offset + second_point_id]
        return result

    @numba.njit()
    def cost_functional(ut_vector, ut_vector_old, vertices, contact_boundaries):
        cost = 0
        nmL = np.asarray([0, -1])
        for contact_boundary in contact_boundaries:
            for i in range(1, len(contact_boundary)):
                v1_id = contact_boundary[i - 1]
                v2_id = contact_boundary[i]

                # nmL = n_down(Points, Edges, e)  # TODO n at mL

                um = interpolate_point_between(v1_id, v2_id, ut_vector)
                um_old = interpolate_point_between(v1_id, v2_id, ut_vector_old)

                uNmL = (um * nmL).sum()
                uNmL_old = (um_old * nmL).sum()
                uTmL = np.empty(DIMENSION)
                for i in range(DIMENSION):
                    uTmL[i] = um[i] - uNmL * nmL[i]

                firstPointCoordinates = vertices[v1_id]
                secondPointCoordinates = vertices[v2_id]
                edgeLength = length(firstPointCoordinates, secondPointCoordinates)

                cost += edgeLength * (jn(uNmL) + h(uNmL_old) * jt(uTmL))

        return cost

    @numba.njit()
    def L2(ut_vector, ut_vector_old, vertices, contact_boundaries, C, E, t_vector):
        ju = cost_functional(ut_vector, ut_vector_old, vertices, contact_boundaries)
        result = (0.5 * np.dot(np.dot(C, ut_vector), ut_vector) - np.dot(E, ut_vector)
                  + ju)
        result = np.asarray(result).ravel()
        return result

    return L2


def make_L2_t_new(jn: Callable, jt: Optional[Callable] = None, h: Optional[Callable] = None):
    jn = njit(jn)
    jt = njit(jt)
    h = njit(h)
    DIMENSION = 2

    @numba.njit(inline='always')
    def interpolate_point_between(first_point_id, second_point_id, vector, ind_number):
        result = np.zeros(DIMENSION)
        offset = len(vector) // DIMENSION
        for i in range(DIMENSION):
            if first_point_id < ind_number:  # exclude points from Gamma_D
                result[i] += 0.5 * vector[i * offset + first_point_id]
        for i in range(DIMENSION):
            if second_point_id < ind_number:  # exclude points from Gamma_D
                result[i] += 0.5 * vector[i * offset + second_point_id]
        return result

    @numba.njit()
    def cost_functional(indNumber, BorderEdgesC, Edges, ut_vector, ut_vector_old, Points):
        cost = 0
        for e in range(0, BorderEdgesC):
            nmL = n_down(Points, Edges, e)  # n at mL

            firstPointIndex = Edges[e][0]
            secondPointIndex = Edges[e][1]

            um = interpolate_point_between(
                firstPointIndex, secondPointIndex, ut_vector, indNumber)
            um_old = interpolate_point_between(
                firstPointIndex, secondPointIndex, ut_vector_old, indNumber)

            uNmL = (um * nmL).sum()
            uNmL_old = (um_old * nmL).sum()
            uTmL = np.empty(DIMENSION)
            for i in range(DIMENSION):
                uTmL[i] = um[i] - uNmL * nmL[i]

            firstPointCoordinates = Points[int(firstPointIndex)][0:2]
            secondPointCoordinates = Points[int(secondPointIndex)][0:2]
            edgeLength = length(firstPointCoordinates, secondPointCoordinates)

            cost += edgeLength * (jn(uNmL) + h(uNmL_old) * jt(uTmL))

        return cost

    @numba.njit()
    def L2(ut_vector, wt_vector, indNumber, BorderEdgesC, Edges, Points, C, E, t_vector):
        ju = cost_functional(indNumber, BorderEdgesC, Edges, ut_vector, wt_vector, Points)
        result = (0.5 * np.dot(np.dot(C, ut_vector), ut_vector) - np.dot(E, ut_vector)
                  + ju)
        result = np.asarray(result).ravel()
        return result

    return L2


def make_L2_t(hn: Callable, ht: Optional[Callable] = None, h: Optional[Callable] = None):
    hn = njit(hn)
    ht = njit(ht)
    h = numba.njit(h)
    DIMENSION = 2

    @numba.njit()
    def cost_functional(indNumber, BorderEdgesC, Edges, tt_vector, ut_vector, Points):
        cost = 0.

        for e in range(0, BorderEdgesC):
            nmL = n_down(Points, Edges, e)  # n at mL

            firstPointIndex = Edges[e][0]
            secondPointIndex = Edges[e][1]

            umLx = 0.
            umLy = 0.
            tmL = 0.
            offset = len(ut_vector) // DIMENSION
            if firstPointIndex < indNumber:  # exclude points from Gamma_D
                umLx += 0.5 * ut_vector[firstPointIndex]
                umLy += 0.5 * ut_vector[offset + firstPointIndex]
                tmL += 0.5 * tt_vector[firstPointIndex]
            if secondPointIndex < indNumber:  # exclude points from Gamma_D
                umLx += 0.5 * ut_vector[secondPointIndex]
                umLy += 0.5 * ut_vector[offset + secondPointIndex]
                tmL += 0.5 * tt_vector[secondPointIndex]

            uNmL = umLx * nmL[0] + umLy * nmL[1]
            uTmLx = umLx - uNmL * nmL[0]
            uTmLy = umLy - uNmL * nmL[1]

            firstPointCoordinates = Points[int(firstPointIndex)][0:2]
            secondPointCoordinates = Points[int(secondPointIndex)][0:2]
            edgeLength = length(firstPointCoordinates, secondPointCoordinates)

            # cost += edgeLength * (hn(uNmL, tmL) + h(np.linalg.norm(np.asarray((uTmLx, uTmLy)))) * ht(uNmL, tmL))
            cost += edgeLength * (h(np.linalg.norm(np.asarray((uTmLx, uTmLy)))) * tmL)

        return cost

    @numba.njit()
    def L2(tt_vector, indNumber, BorderEdgesC, Edges, Points, T, Q, ut_vector):
        # TODO #31
        return 0.5 * np.dot(np.dot(T, tt_vector), tt_vector) - np.dot(Q, tt_vector) \
               - cost_functional(indNumber, BorderEdgesC, Edges, tt_vector, ut_vector, Points)

    return L2
