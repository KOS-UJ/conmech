"""
Created at 21.08.2019
"""
from typing import Callable, Optional, Any

import numba
import numpy as np

from conmech.helpers import nph

DIMENSION = 2


@numba.njit()
def n_down(n0, n1):
    # [0,-1]
    x = 0
    y = 1
    dx = n0[x] - n1[x]
    dy = n0[y] - n1[y]
    norm = np.sqrt(dx**2 + dy**2)
    n = np.array([float(dy) / norm, float(-dx) / norm])
    if n[1] > 0:
        n = -n
    return n


@numba.njit(inline="always")
def interpolate_node_between(node_id_0, node_id_1, vector, dimension=DIMENSION):
    result = np.zeros(dimension)
    offset = len(vector) // dimension
    for i in range(dimension):
        if node_id_0 < offset:  # exclude dirichlet nodes (and inner nodes in schur)
            result[i] += 0.5 * vector[i * offset + node_id_0]
    for i in range(dimension):
        if node_id_1 < offset:  # exclude dirichlet nodes (and inner nodes in schur)
            result[i] += 0.5 * vector[i * offset + node_id_1]
    return result


def make_equation(jn, jt, h_functional):
    jn = numba.njit(jn)
    jt = numba.njit(jt)
    h_functional = numba.njit(h_functional)

    @numba.njit()
    def contact_part(u_vector, nodes, contact_boundary):
        contact_vector = np.zeros_like(u_vector)
        offset = len(u_vector) // DIMENSION

        for edge in contact_boundary:
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]

            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between(n_id_0, n_id_1, u_vector)

            normal_vector = n_down(n_0, n_1)

            um_normal = (um * normal_vector).sum()
            um_tangential = um - um_normal * normal_vector

            v_tau_0 = np.asarray(
                [
                    1 - normal_vector[0] * normal_vector[0],
                    0 - normal_vector[0] * normal_vector[1],
                ]
            )
            v_tau_1 = np.asarray(
                [
                    0 - normal_vector[0] * normal_vector[1],
                    1 - normal_vector[1] * normal_vector[1],
                ]
            )

            edge_len = nph.length(n_0, n_1)
            j_x = edge_len * 0.5 * (jn(um_normal, normal_vector[0])) + h_functional(um_normal) * jt(
                um_tangential, v_tau_0
            )
            j_y = edge_len * 0.5 * (jn(um_normal, normal_vector[1])) + h_functional(um_normal) * jt(
                um_tangential, v_tau_1
            )

            if n_id_0 < offset:
                contact_vector[n_id_0] += j_x
                contact_vector[n_id_0 + offset] += j_y

            if n_id_1 < offset:
                contact_vector[n_id_1] += j_x
                contact_vector[n_id_1 + offset] += j_y

        return contact_vector

    @numba.njit()
    def equation(u_vector, vertices, contact_boundary, lhs, rhs):
        c_part = 0 #contact_part(u_vector, vertices, contact_boundary)
        result = np.dot(lhs, u_vector) + c_part - rhs
        return result

    return equation


def njit(func: Optional[Callable], value: Optional[Any] = 0) -> Callable:
    if func is None:

        @numba.njit()
        def const(_):
            return value

        return const
    return numba.njit(func)


def make_cost_functional(
    jn: Callable, jt: Optional[Callable] = None, h_functional: Optional[Callable] = None
):
    jn = njit(jn)
    jt = njit(jt)
    h_functional = njit(h_functional)

    @numba.njit()
    def contact_cost_functional(u_vector, u_vector_old, nodes, contact_boundary):
        cost = 0
        offset = len(u_vector) // DIMENSION

        for edge in contact_boundary:
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]

            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between(n_id_0, n_id_1, u_vector)
            um_old = interpolate_node_between(n_id_0, n_id_1, u_vector_old)

            normal_vector = n_down(n_0, n_1)

            um_normal = (um * normal_vector).sum()
            um_old_normal = (um_old * normal_vector).sum()
            um_tangential = um - um_normal * normal_vector

            if n_id_0 < offset and n_id_1 < offset:
                cost += nph.length(n_0, n_1) * (
                    jn(um_normal) + h_functional(um_old_normal) * jt(um_tangential)
                )
        return cost

    @numba.njit()
    def cost_functional(u_vector, nodes, contact_boundary, lhs, rhs, u_vector_old):
        ju = contact_cost_functional(u_vector, u_vector_old, nodes, contact_boundary)
        result = 0.5 * np.dot(np.dot(lhs, u_vector), u_vector) - np.dot(rhs, u_vector) + ju
        result = np.asarray(result).ravel()
        return result

    return cost_functional


def make_cost_functional_temperature(
    hn: Callable, ht: Optional[Callable] = None, h_functional: Optional[Callable] = None
):
    _hn = njit(hn)  # TODO #48
    _ht = njit(ht)
    h_functional = numba.njit(h_functional)

    @numba.njit()
    def contact_cost_functional(u_vector, nodes, contact_boundary):
        cost = 0
        offset = len(u_vector) // DIMENSION

        for edge in contact_boundary:
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]

            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between(n_id_0, n_id_1, u_vector)

            normal_vector = n_down(n_0, n_1)

            um_normal = (um * normal_vector).sum()
            um_tangential = um - um_normal * normal_vector

            if n_id_0 < offset and n_id_1 < offset:
                # cost += edgeLength * (hn(uNmL, tmL)
                #      + h(np.linalg.norm(np.asarray((uTmLx, uTmLy)))) * ht(uNmL, tmL))
                cost += nph.length(n_0, n_1) * h_functional(np.linalg.norm(um_tangential))
        return cost

    @numba.njit()
    def cost_functional(temp_vector, nodes, contact_boundary, lhs, rhs, u_vector):
        result = (
            0.5 * np.dot(np.dot(lhs, temp_vector), temp_vector)
            - np.dot(rhs, temp_vector)
            - contact_cost_functional(u_vector, nodes, contact_boundary)
        )
        result = np.asarray(result).ravel()
        return result

    return cost_functional
