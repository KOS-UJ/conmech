"""
Created at 21.08.2019
"""
from typing import Callable, Optional, Any

import numba
import numpy as np

from conmech.helpers import nph

DIMENSION = 2


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


@numba.njit(inline="always")
def interpolate_node_between_prb(edge, vector, dimension=DIMENSION):
    result = np.zeros(dimension)
    offset = len(vector) // dimension
    for k in range(len(edge)):
        for i in range(dimension):
            if edge[k] < offset:  # exclude dirichlet nodes (and inner nodes in schur)
                result[i] += 0.5 * vector[i * offset + edge[k]]
    return result


# TODO #97
@numba.njit(inline="always")
def interpolate_node_between_2023(node_id_0, _node_id_1, vector, dimension=DIMENSION):
    result = np.zeros(dimension)
    offset = len(vector) // dimension
    for i in range(dimension):
        if node_id_0 < offset:  # exclude dirichlet nodes (and inner nodes in schur)
            result[i] += vector[i * offset + node_id_0]
    return result


def make_equation(
    jn: Optional[callable], jt: Optional[callable], h_functional: Optional[callable]
) -> callable:
    # TODO Make it prettier
    if jn is None:

        @numba.njit
        def equation(
            u_vector: np.ndarray, _, __, lhs: np.ndarray, rhs: np.ndarray
        ) -> np.ndarray:
            result = np.dot(lhs, u_vector) - rhs
            return result

    else:
        jn = numba.njit(jn)
        jt = numba.njit(jt)
        h_functional = numba.njit(h_functional)

        @numba.njit()
        def contact_part(u_vector, nodes, contact_boundary, contact_normals):
            contact_vector = np.zeros_like(u_vector)
            offset = len(u_vector) // DIMENSION

            for ei, edge in enumerate(contact_boundary):
                n_id_0 = edge[0]
                n_id_1 = edge[1]
                n_0 = nodes[n_id_0]
                n_1 = nodes[n_id_1]

                # ASSUMING `u_vector` and `nodes` have the same order!
                um = interpolate_node_between(n_id_0, n_id_1, u_vector)

                normal_vector = contact_normals[ei]

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
                j_x = edge_len * 0.5 * (jn(um_normal, normal_vector[0])) + h_functional(
                    um_normal
                ) * jt(um_tangential, v_tau_0)
                j_y = edge_len * 0.5 * (jn(um_normal, normal_vector[1])) + h_functional(
                    um_normal
                ) * jt(um_tangential, v_tau_1)

                if n_id_0 < offset:
                    contact_vector[n_id_0] += j_x
                    contact_vector[n_id_0 + offset] += j_y

                if n_id_1 < offset:
                    contact_vector[n_id_1] += j_x
                    contact_vector[n_id_1 + offset] += j_y

            return contact_vector

        @numba.njit
        def equation(
            u_vector: np.ndarray,
            vertices: np.ndarray,
            contact_boundary: np.ndarray,
            contact_normals: np.ndarray,
            lhs: np.ndarray,
            rhs: np.ndarray,
        ) -> np.ndarray:
            c_part = contact_part(u_vector, vertices, contact_boundary, contact_normals)
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
    jn: Callable,
    jt: Optional[Callable] = None,
    h_functional: Optional[Callable] = None,
    dim=2,
):
    jn = njit(jn)
    jt = njit(jt)
    h_functional = njit(h_functional)

    @numba.njit()
    def contact_cost(length, normal, tangential, tangential_second):
        return length * (normal + tangential_second * tangential)

    @numba.njit()
    def contact_cost_functional(
        u_vector, u_vector_old, nodes, contact_boundary, contact_normals
    ):
        offset = len(u_vector) // dim

        cost = 0.0
        for ei in numba.prange(len(contact_boundary)):
            edge = contact_boundary[ei]
            normal_vector = contact_normals[ei]
            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between_prb(edge, u_vector, dimension=dim)
            um_normal = (um * normal_vector).sum()
            um_tangential = um - um_normal * normal_vector

            um_old = interpolate_node_between_prb(edge, u_vector_old, dimension=dim)
            um_old_normal = (um_old * normal_vector).sum()
            um_old_tangential = um_old - um_old_normal * normal_vector

            # temp_m = interpolate_node_between_prb(edge, temp_vector, dimension=1)

            for node_id in edge:
                if node_id >= offset:
                    continue

            cost += contact_cost(
                nph.length_prb(edge, nodes),
                jn(um_normal),
                jt(um_tangential),
                h_functional(um_old_normal),
            )
        return cost

    # pylint: disable=unused-argument # 'dt'
    @numba.njit()
    def cost_functional(
        u_vector, nodes, contact_boundary, contact_normals, lhs, rhs, u_vector_old, dt
    ):
        ju = contact_cost_functional(
            u_vector, u_vector_old, nodes, contact_boundary, contact_normals
        )
        result = (
            0.5 * np.dot(np.dot(lhs, u_vector), u_vector) - np.dot(rhs, u_vector) + ju
        )
        result = np.asarray(result).ravel()
        return result

    return cost_functional


def make_cost_functional_poisson(jn: Callable):
    jn = njit(jn)

    @numba.njit()
    def contact_cost_functional(u_vector, nodes, contact_boundary, contact_normals):
        cost = 0
        offset = len(u_vector) // DIMENSION

        for ei, edge in enumerate(contact_boundary):
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]

            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between(n_id_0, n_id_1, u_vector)

            normal_vector = contact_normals[ei]

            um_normal = (um * normal_vector).sum()

            if n_id_0 < offset and n_id_1 < offset:
                cost += nph.length(n_0, n_1) * (jn(um_normal))
        return cost

    # pylint: disable=unused-argument # u_vector_old, dt
    # @numba.njit()
    def cost_functional(
        u_vector, nodes, contact_boundary, contact_normals, lhs, rhs, u_vector_old, dt
    ):
        ju = contact_cost_functional(u_vector, nodes, contact_boundary, contact_normals)
        result = (
            0.5 * np.dot(np.dot(lhs, u_vector), u_vector) - np.dot(rhs, u_vector) + ju
        )
        result = np.asarray(result).ravel()
        return result

    return cost_functional


def make_cost_functional_2023(  # TODO #97
    jn: Callable, jt: Optional[Callable] = None, h_functional: Optional[Callable] = None
):
    jn = njit(jn)
    jt = njit(jt)
    h_functional = njit(h_functional)

    @numba.njit()
    def contact_cost_functional(
        v_vector, u_vector_old, nodes, contact_boundary, contact_normals
    ):
        cost = 0
        offset = len(v_vector) // DIMENSION

        for ei, edge in enumerate(contact_boundary):
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]

            # ASSUMING `u_vector` and `nodes` have the same order!
            vm = interpolate_node_between_2023(n_id_0, n_id_1, v_vector)
            vm_2 = interpolate_node_between_2023(n_id_1, n_id_0, v_vector)
            um_old = interpolate_node_between_2023(n_id_0, n_id_1, u_vector_old)
            um_old_2 = interpolate_node_between_2023(n_id_1, n_id_0, u_vector_old)

            normal_vector = contact_normals[ei]

            vm_normal = (vm * normal_vector).sum()
            um_old_normal = (um_old * normal_vector).sum()
            vm_tangential = vm - vm_normal * normal_vector
            vm_normal_2 = (vm_2 * normal_vector).sum()
            um_old_normal_2 = (um_old_2 * normal_vector).sum()
            vm_tangential_2 = vm_2 - vm_normal * normal_vector

            if n_id_0 < offset and n_id_1 < offset:
                cost += (
                    0.5
                    * nph.length(n_0, n_1)
                    * (
                        jn(um_old_normal) * vm_normal
                        + h_functional(um_old_normal) * jt(vm_tangential)
                        + jn(um_old_normal_2) * vm_normal_2
                        + h_functional(um_old_normal_2) * jt(vm_tangential_2)
                    )
                )
        return cost

    @numba.njit()
    def cost_functional(
        v_vector, nodes, contact_boundary, contact_normals, lhs, rhs, u_vector_old, dt
    ):
        u_vector = u_vector_old + dt * v_vector
        ju = contact_cost_functional(
            v_vector, u_vector, nodes, contact_boundary, contact_normals
        )
        result = (
            0.5 * np.dot(np.dot(lhs, v_vector), v_vector) - np.dot(rhs, v_vector) + ju
        )
        result = np.asarray(result).ravel()
        return result

    return cost_functional


def make_cost_functional_temperature(
    normal_condition: Callable,
    tangential_condition: Optional[Callable] = None,
    h_functional: Optional[Callable] = None,
    heat_exchange: Optional[Callable] = None,
):
    jn = njit(normal_condition)
    jt = njit(tangential_condition)
    h_functional = njit(h_functional)
    heat_exchange = njit(heat_exchange)
    #
    # cost += contact_cost(
    #     nph.length_prb(edge, nodes),
    #     jn(um_normal),
    #     jt(um_tangential),
    #     h_functional(um_old_normal),
    # )
    #
    # @numba.njit()
    # def contact_cost(length, normal, tangential, tangential_second):
    #     return length * (normal + tangential_second)
    #
    # (h_functional(np.linalg.norm(um_tangential)) - heat_exchange(temp_m[0]))

    @numba.njit()
    def contact_cost_functional(
        u_vector, temp_vector, nodes, contact_boundary, contact_normals
    ):
        offset = len(u_vector) // DIMENSION

        cost = 0.0
        for ei in numba.prange(len(contact_boundary)):
            edge = contact_boundary[ei]
            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between_prb(edge, u_vector)
            temp_m = interpolate_node_between_prb(edge, temp_vector, dimension=1)

            normal_vector = contact_normals[ei]

            um_normal = (um * normal_vector).sum()
            um_tangential = um - um_normal * normal_vector

            for node_id in edge:
                if node_id >= offset:
                    continue

            cost += nph.length_prb(edge, nodes) * (
                h_functional(np.linalg.norm(um_tangential)) - heat_exchange(temp_m[0])
            )
        return cost

    # pylint: disable=unused-argument # 'dt'
    @numba.njit()
    def cost_functional(
        temp_vector, nodes, contact_boundary, contact_normals, lhs, rhs, u_vector, dt
    ):
        ju = contact_cost_functional(
            u_vector,
            temp_vector,
            nodes,
            contact_boundary,
            contact_normals,
        )
        result = (
            0.5 * np.dot(np.dot(lhs, temp_vector), temp_vector)
            - np.dot(rhs, temp_vector)
            - ju
        )
        result = np.asarray(result).ravel()
        return result

    return cost_functional


def make_cost_functional_piezoelectricity(
    hn: Callable,
    ht: Optional[Callable] = None,
    h_functional: Optional[Callable] = None,
    electric_charge_exchange: Optional[Callable] = None,
):
    _hn = njit(hn)  # TODO #48
    _ht = njit(ht)
    h_functional = numba.njit(h_functional)
    electric_charge_exchange = njit(electric_charge_exchange)

    @numba.njit()
    def contact_cost_functional(
        u_vector, nodes, contact_boundary, contact_normals, temp_vector
    ):
        cost = 0
        offset = len(u_vector) // DIMENSION

        for ei, edge in enumerate(contact_boundary):
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]

            # ASSUMING `u_vector` and `nodes` have the same order!
            um = interpolate_node_between(n_id_0, n_id_1, u_vector)
            temp_m = interpolate_node_between(n_id_0, n_id_1, temp_vector, dimension=1)

            normal_vector = contact_normals[ei]

            um_normal = (um * normal_vector).sum()
            um_tangential = um - um_normal * normal_vector

            if n_id_0 < offset and n_id_1 < offset:
                # cost += edgeLength * (hn(uNmL, tmL)
                #      + h(np.linalg.norm(np.asarray((uTmLx, uTmLy)))) * ht(uNmL, tmL))
                cost += nph.length(n_0, n_1) * (
                    h_functional(np.linalg.norm(um_tangential))
                    - electric_charge_exchange(temp_m[0])
                )
        return cost

    # pylint: disable=unused-argument # 'dt'
    @numba.njit()
    def cost_functional(
        temp_vector, nodes, contact_boundary, contact_normals, lhs, rhs, u_vector, dt
    ):
        result = (
            0.5 * np.dot(np.dot(lhs, temp_vector), temp_vector)
            - np.dot(rhs, temp_vector)
            - contact_cost_functional(
                u_vector, nodes, contact_boundary, contact_normals, temp_vector
            )
        )
        result = np.asarray(result).ravel()
        return result

    return cost_functional
