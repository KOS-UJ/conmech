# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2019-2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
from typing import Callable, Optional, Any

import numba
import numpy as np

from conmech.struct.types import f64, cf64, ci64
from conmech.helpers import nph


@numba.njit(inline="always")
def interpolate_node_between(edge, vector, full_vector, dimension):
    result = np.zeros(dimension)
    offset = len(vector) // dimension
    offset_full = len(full_vector) // dimension
    for node in edge:
        for i in range(dimension):
            if node < offset:  # exclude dirichlet nodes (and inner nodes in schur)
                result[i] += vector[i * offset + node] / len(edge)
            else:
                # old values
                result[i] += full_vector[i * offset_full + node] / len(edge)
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
                j_x = edge_len * 0.5 * (jn(um_normal, 0.0, 0.0) * normal_vector[0])
                j_y = edge_len * 0.5 * (jn(um_normal, 0.0, 0.0) * normal_vector[1])

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

    @numba.njit(f64(f64, f64, f64, f64, f64))
    def contact_cost(length, normal, normal_bound, tangential, tangential_bound):
        return length * (normal_bound * normal + tangential_bound * tangential)

    @numba.njit(f64(f64[:], cf64[:], cf64[:], cf64[:, :], ci64[:, :], cf64[:, :], f64))
    def contact_cost_functional(
        var, var_old, static_displacement, nodes, contact_boundary, contact_normals, dt
    ):
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

            cost += contact_cost(
                nph.length(edge, nodes),
                normal_condition(vm_normal, static_displacement_normal, dt),
                normal_condition_bound(vm_normal, static_displacement_normal, dt),
                tangential_condition(vm_tangential, static_displacement_tangential, dt),
                tangential_condition_bound(vm_normal, static_displacement_normal, dt),
            )
        return cost

    # pylint: disable=too-many-arguments,unused-argument # 'base_integrals'
    @numba.njit(
        f64[:](
            f64[:],
            cf64[:],
            cf64[:, :],
            ci64[:, :],
            cf64[:, :],
            cf64[:, :],
            cf64[:],
            cf64[:],
            cf64[:, :],
            f64,
        )
    )
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


def make_subgradient(
    normal_condition: Callable,
    normal_condition_bound: Optional[Callable] = None,
    tangential_condition: Optional[Callable] = None,
    tangential_condition_bound: Optional[Callable] = None,
    problem_dimension=2,
    variable_dimension=2,
    only_boundary=False,
):
    normal_condition = njit(normal_condition)
    normal_condition_bound = njit(normal_condition_bound, value=1)
    tangential_condition = njit(tangential_condition)
    tangential_condition_bound = njit(tangential_condition_bound, value=1)

    @numba.njit()
    def contact_cost(length, normal, normal_bound, tangential, tangential_bound):
        return length * (normal_bound * normal + tangential_bound * tangential)

    @numba.njit()
    def contact_subgradient(
        var, var_old, static_displacement, nodes, contact_boundary, contact_normals, dt
    ):
        cost = np.zeros_like(var)
        offset = len(var) // variable_dimension

        # pylint: disable=not-an-iterable
        for ei in numba.prange(len(contact_boundary)):
            edge = contact_boundary[ei]
            normal_vector = contact_normals[ei]
            # ASSUMING `u_vector` and `nodes` have the same order!
            vm = interpolate_node_between(edge, var, var_old, dimension=variable_dimension)
            if variable_dimension == 1:
                raise NotImplementedError()  # TODO
                # vm_normal = vm[0]
                # vm_tangential = np.empty(0)
            # else:
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

            subgrad = contact_cost(
                nph.length(edge, nodes),
                normal_condition(vm_normal, static_displacement_normal, dt),
                normal_condition_bound(vm_normal, static_displacement_normal, dt),
                tangential_condition(vm_tangential, static_displacement_tangential, dt),
                tangential_condition_bound(vm_normal, static_displacement_normal, dt),
            )

            for node in edge:
                for i in range(variable_dimension):
                    if node < offset:
                        cost[i * offset + node] += normal_vector[i] / len(edge) * subgrad

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
        result = np.zeros_like(var)
        dj = contact_subgradient(
            var, var_old, u_vector, nodes, contact_boundary, contact_normals, dt
        )
        ind = lhs.shape[0]
        if only_boundary:
            result_ = dj
        else:
            result_ = np.dot(lhs, var[:ind]) - rhs + dj
        result[:ind] = result_.ravel()
        return result

    return cost_functional


def make_subgradient_dc(
    normal_condition: Callable,
    normal_condition_sub2: Callable,
    normal_condition_bound: Optional[Callable] = None,
    tangential_condition: Optional[Callable] = None,
    tangential_condition_bound: Optional[Callable] = None,
    problem_dimension=2,
    variable_dimension=2,
    only_boundary=False,
):
    if only_boundary:
        pass  # TODO
    normal_condition = njit(normal_condition)
    normal_condition_sub2 = njit(normal_condition_sub2)
    normal_condition_bound = njit(normal_condition_bound, value=1)
    tangential_condition = njit(tangential_condition)
    tangential_condition_bound = njit(tangential_condition_bound, value=1)

    @numba.njit()
    def contact_cost(length, normal, normal_bound, tangential, tangential_bound):
        return length * (normal_bound * normal + tangential_bound * tangential)

    @numba.njit()
    def contact_subgradient(
        var, var1, var_old, static_displacement, nodes, contact_boundary, contact_normals, dt
    ):
        cost = np.zeros_like(var)
        offset = len(var) // variable_dimension

        # pylint: disable=not-an-iterable
        for ei in numba.prange(len(contact_boundary)):
            edge = contact_boundary[ei]
            normal_vector = contact_normals[ei]
            # ASSUMING `u_vector` and `nodes` have the same order!
            vm = interpolate_node_between(edge, var, var_old, dimension=variable_dimension)
            vm1 = interpolate_node_between(edge, var1, var_old, dimension=variable_dimension)
            if variable_dimension == 1:
                raise NotImplementedError()  # TODO
                # vm_normal = vm[0]
                # vm_tangential = np.empty(0)
            # else:
            vm_normal = (vm * normal_vector).sum()
            vm_tangential = vm - vm_normal * normal_vector
            vm_normal1 = (vm1 * normal_vector).sum()

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

            subgrad = (
                contact_cost(
                    nph.length(edge, nodes),
                    normal_condition(vm_normal1, static_displacement_normal, dt),
                    normal_condition_bound(vm_normal1, static_displacement_normal, dt),
                    tangential_condition(vm_tangential, static_displacement_tangential, dt),
                    tangential_condition_bound(vm_normal1, static_displacement_normal, dt),
                )
                + contact_cost(
                    nph.length(edge, nodes),
                    normal_condition_sub2(vm_normal1, static_displacement_normal, dt),
                    normal_condition_bound(vm_normal1, static_displacement_normal, dt),
                    tangential_condition(vm_tangential, static_displacement_tangential, dt),
                    tangential_condition_bound(vm_normal1, static_displacement_normal, dt),
                )
                - contact_cost(
                    nph.length(edge, nodes),
                    normal_condition_sub2(vm_normal, static_displacement_normal, dt),
                    normal_condition_bound(vm_normal, static_displacement_normal, dt),
                    tangential_condition(vm_tangential, static_displacement_tangential, dt),
                    tangential_condition_bound(vm_normal, static_displacement_normal, dt),
                )
            )

            for node in edge:
                for i in range(variable_dimension):
                    if node < offset:
                        cost[i * offset + node] += normal_vector[i] / len(edge) * subgrad

        return cost

    # pylint: disable=too-many-arguments,unused-argument # 'base_integrals'
    @numba.njit()
    def cost_functional(
        var,
        var1,
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
        result = np.zeros_like(var)
        dj = contact_subgradient(
            var, var1, var_old, u_vector, nodes, contact_boundary, contact_normals, dt
        )
        ind = lhs.shape[0]
        result_ = np.dot(lhs, var1[:ind]) - rhs + dj
        result[:ind] = result_.ravel()
        return result

    return cost_functional
