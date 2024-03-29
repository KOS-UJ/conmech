"""
Created at 21.08.2019
"""

from typing import Callable, Optional, Any

import numba
import numpy as np

from conmech.helpers import nph


@numba.njit(inline="always")
def interpolate_node_between(edge, vector, dimension):
    result = np.zeros(dimension)
    offset = len(vector) // dimension
    for node in edge:
        for i in range(dimension):
            if node < offset:  # exclude dirichlet nodes (and inner nodes in schur)
                result[i] += 0.5 * vector[i * offset + node]
    return result


def make_equation(
    jn: Optional[callable],
    jt: Optional[callable],
    h_functional: Optional[callable],
    problem_dimension=2,
) -> callable:
    # TODO Make it prettier
    if jn is None:

        @numba.njit
        def equation(u_vector: np.ndarray, _, __, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            result = np.dot(lhs, u_vector) - rhs
            return result

    else:
        jn = numba.njit(jn)
        jt = numba.njit(jt)
        h_functional = numba.njit(h_functional)

        @numba.njit()
        def contact_part(u_vector, nodes, contact_boundary, contact_normals):
            contact_vector = np.zeros_like(u_vector)
            offset = len(u_vector) // problem_dimension

            for ei, edge in enumerate(contact_boundary):
                n_id_0 = edge[0]
                n_id_1 = edge[1]
                normal_vector = contact_normals[ei]

                # ASSUMING `u_vector` and `nodes` have the same order!
                um = interpolate_node_between(edge, u_vector, dimension=problem_dimension)
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
                        0 - normal_vector[1] * normal_vector[0],
                        1 - normal_vector[1] * normal_vector[1],
                    ]
                )

                edge_len = nph.length(edge, nodes)
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
    if func is None or isinstance(func, (int, float)):
        ret_val = func if func is not None else value

        @numba.njit()
        def const(_):
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
    def contact_cost_functional(var, displacement, nodes, contact_boundary, contact_normals):
        offset = len(var) // problem_dimension

        cost = 0.0
        # pylint: disable=not-an-iterable
        for ei in numba.prange(len(contact_boundary)):
            edge = contact_boundary[ei]
            normal_vector = contact_normals[ei]
            # ASSUMING `u_vector` and `nodes` have the same order!
            vm = interpolate_node_between(edge, var, dimension=variable_dimension)
            if variable_dimension == 1:
                vm_normal = vm[0]
                vm_tangential = np.empty(0)
            else:
                vm_normal = (vm * normal_vector).sum()
                vm_tangential = vm - vm_normal * normal_vector

            um = interpolate_node_between(edge, displacement, dimension=problem_dimension)
            um_normal = (um * normal_vector).sum()

            for node_id in edge:
                if node_id >= offset:
                    continue

            cost += contact_cost(
                nph.length(edge, nodes),
                normal_condition(vm_normal),
                normal_condition_bound(um_normal),
                tangential_condition(vm_tangential),
                tangential_condition_bound(um_normal),
            )
        return cost

    # pylint: disable=unused-argument # 'dt'
    @numba.njit()
    def cost_functional(var, nodes, contact_boundary, contact_normals, lhs, rhs, u_vector, dt):
        ju = contact_cost_functional(var, u_vector, nodes, contact_boundary, contact_normals)
        result = 0.5 * np.dot(np.dot(lhs, var), var) - np.dot(rhs, var) + ju
        result = np.asarray(result).ravel()
        return result

    return cost_functional
