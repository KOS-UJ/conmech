"""
Created at 21.08.2019
"""

from typing import Callable, Optional, Any

import numba
import numpy as np

from conmech.helpers import nph


@numba.njit(inline="always")
def interpolate_node_between(edge, vector, full_vector, dimension):
    result = np.zeros(dimension)
    offset = len(vector) // dimension
    offset_full = len(full_vector) // dimension
    for node in edge:
        for i in range(dimension):
            if node < offset:  # exclude dirichlet nodes (and inner nodes in schur)
                result[i] += 0.5 * vector[i * offset + node]
            else:
                # old values
                result[i] += 0.5 * full_vector[i * offset_full + node]
    return result


# pylint: disable=too-many-arguments
def make_equation(
    jn: Optional[callable],
    contact: Optional[callable] = None,
    problem_dimension=2,
) -> callable:
    if jn is None or contact is not None:
        contact = njit(contact, value=0.0)

        # pylint: disable=unused-argument
        @numba.njit
        def equation(
            var: np.ndarray,
            var_old: np.ndarray,
            _,
            __,
            ___,
            lhs: np.ndarray,
            rhs: np.ndarray,
            displacement: np.ndarray,
            volume_multiplier: np.ndarray,
            time_step,
        ) -> np.ndarray:
            ind = lhs.shape[0]
            response = np.zeros(ind)
            for i in range(ind):
                response[i] = contact(var[i], displacement[i], time_step)
            res = (
                0.5 * np.dot(np.dot(lhs, var[:ind]), var[:ind])
                - np.dot(rhs, var[:ind])
                + 0.5 * np.dot(np.dot(volume_multiplier, response), np.ones_like(var[:ind]))
                + np.dot(var[ind:], var[ind:].T)
            )

            result = np.asarray(res).ravel()
            return result

    else:
        jn = numba.njit(jn)

        @numba.njit()
        def contact_part(u_vector, nodes, contact_boundary, contact_normals):
            contact_vector = np.zeros_like(u_vector)
            offset = len(u_vector) // problem_dimension

            for ei, edge in enumerate(contact_boundary):
                n_id_0 = edge[0]
                n_id_1 = edge[1]
                normal_vector = contact_normals[ei]

                # ASSUMING `u_vector` and `nodes` have the same order!
                um = interpolate_node_between(edge, u_vector, u_vector, dimension=problem_dimension)
                um_normal = (um * normal_vector).sum()

                edge_len = nph.length(edge, nodes)
                j_x = edge_len * 0.5 * (jn(um_normal, normal_vector[0], 0.0))
                j_y = edge_len * 0.5 * (jn(um_normal, normal_vector[1], 0.0))

                if n_id_0 < offset:
                    contact_vector[n_id_0] += j_x
                    contact_vector[n_id_0 + offset] += j_y

                if n_id_1 < offset:
                    contact_vector[n_id_1] += j_x
                    contact_vector[n_id_1 + offset] += j_y

            return contact_vector

        # pylint: disable=unused-argument,too-many-arguments
        @numba.njit
        def equation(
            var: np.ndarray,
            var_old: np.ndarray,
            nodes: np.ndarray,
            contact_boundary: np.ndarray,
            contact_normals: np.ndarray,
            lhs: np.ndarray,
            rhs: np.ndarray,
            displacement: np.ndarray,
            base_integrals,
            time_step,
        ) -> np.ndarray:
            c_part = contact_part(var, nodes, contact_boundary, contact_normals)
            result = np.dot(lhs, var) + c_part - rhs
            return result

    return equation


def njit(func: Optional[Callable], value: Optional[Any] = 0) -> Callable:
    if func is None or isinstance(func, (int, float)):
        ret_val = func if func is not None else value

        @numba.njit()
        def const(_, __, ___):
            return ret_val

        return const
    return numba.njit(func)


def make_cost_functional(
    normal_condition: Callable,
    normal_condition_bound: Optional[Callable] = None,
    tangential_condition: Optional[Callable] = None,
    tangential_condition_bound: Optional[Callable] = None,
    problem_dimension=2,
    variable_dimension=2,
):
    normal_condition = njit(normal_condition)
    normal_condition_bound = njit(normal_condition_bound, value=1)
    tangential_condition = njit(tangential_condition)
    tangential_condition_bound = njit(tangential_condition_bound, value=1)

    @numba.njit()
    def contact_cost(length, normal, normal_bound, tangential, tangential_bound):
        return length * (normal_bound * normal + tangential_bound * tangential)

    @numba.njit()
    def contact_cost_functional(
        var, var_old, static_displacement, nodes, contact_boundary, contact_normals, dt
    ):
        offset = len(var) // problem_dimension

        cost = 0.0
        # pylint: disable=not-an-iterable
        for ei in numba.prange(len(contact_boundary)):
            edge = contact_boundary[ei]
            normal_vector = contact_normals[ei]
            # ASSUMING `u_vector` and `nodes` have the same order!
            vm = interpolate_node_between(edge, var, var_old, dimension=variable_dimension)
            if variable_dimension == 1:
                vm_normal = vm[0]
                vm_tangential = np.empty(0)
            else:
                vm_normal = (vm * normal_vector).sum()
                vm_tangential = vm - vm_normal * normal_vector

            static_displacement_mean = interpolate_node_between(
                edge,
                static_displacement,
                static_displacement,
                dimension=problem_dimension,
            )
            static_displacement_normal = (static_displacement_mean * normal_vector).sum()
            static_displacement_tangential = (
                static_displacement_mean - static_displacement_normal * normal_vector
            )

            for node_id in edge:
                if node_id >= offset:
                    continue

            cost += contact_cost(
                nph.length(edge, nodes),
                normal_condition(vm_normal, static_displacement_normal, dt),
                normal_condition_bound(vm_normal, static_displacement_normal, dt),
                tangential_condition(vm_tangential, static_displacement_tangential, dt),
                tangential_condition_bound(vm_normal, static_displacement_normal, dt),
            )
        return cost

    # pylint: disable=too-many-arguments,unused-argument # 'base_integrals'
    @numba.njit()
    def cost_functional(
        var,
        var_old,
        nodes,
        contact_boundary,
        contact_normals,
        lhs,
        rhs,
        u_vector,
        base_integrals,
        dt,
    ):
        ju = contact_cost_functional(
            var, var_old, u_vector, nodes, contact_boundary, contact_normals, dt
        )
        ind = lhs.shape[0]
        result = 0.5 * np.dot(np.dot(lhs, var[:ind]), var[:ind]) - np.dot(rhs, var[:ind]) + ju
        result = np.asarray(result).ravel()
        return result

    return cost_functional

def make_cost_functional_subgradient(
    djn: Callable, djt: Optional[Callable] = None, dh_functional: Optional[Callable] = None
):
    djn = njit(djn)
    djt = njit(djt)
    dh_functional = njit(dh_functional)

    @numba.njit()
    def contact_subgradient(u_vector, u_vector_old, nodes, contact_boundary,
                            contact_normals):
        cost = np.zeros_like(u_vector)
        offset = len(u_vector) // DIMENSION

        for edge in contact_boundary:
            n_id_0 = edge[0]
            n_id_1 = edge[1]
            n_0 = nodes[n_id_0]
            n_1 = nodes[n_id_1]
            if n_id_0 < offset:
                um_normal_0 = -n_0[0]  # TODO
                cost[n_id_0] = djn(um_normal_0)
                cost[n_id_0 + offset] = cost[n_id_0]
            if n_id_1 < offset:
                um_normal_1 = -n_1[0]  # TODO
                cost[n_id_1] = djn(um_normal_1)
                cost[n_id_1 + offset] = cost[n_id_1]
        return cost

    # pylint: disable=unused-argument # 'dt'
    @numba.njit()
    def subgradient(
            u_vector, nodes, contact_boundary, contact_normals, lhs, rhs,
            u_vector_old, dt
    ):
        dj = contact_subgradient(
            u_vector, u_vector_old, nodes, contact_boundary, contact_normals
        )
        result = np.dot(lhs, u_vector) - rhs + dj
        result = result.ravel()
        return result

    return subgradient
