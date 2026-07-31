# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
"""
Error norms for scalar P1 finite element solutions on triangular meshes.

Norms: `L2`, `H1_semi` (the seminorm of the gradient), `V` (the full `H^1` norm,
`sqrt(L2**2 + H1_semi**2)`) and `Linf_nodal`.
"""

import warnings
from typing import Callable, Dict, Sequence

import numpy as np
import matplotlib.tri as mtri

# Radon 7-point rule of degree 5 on a triangle.
# Columns: (l1, l2, l3, weight). Weights sum to 1.
_TRI_RULE_DEG5 = np.array(
    [
        [1 / 3, 1 / 3, 1 / 3, 0.225],
        [0.059715871789770, 0.470142064105115, 0.470142064105115, 0.132394152788506],
        [0.470142064105115, 0.059715871789770, 0.470142064105115, 0.132394152788506],
        [0.470142064105115, 0.470142064105115, 0.059715871789770, 0.132394152788506],
        [0.797426985353087, 0.101286507323456, 0.101286507323456, 0.125939180544827],
        [0.101286507323456, 0.797426985353087, 0.101286507323456, 0.125939180544827],
        [0.101286507323456, 0.101286507323456, 0.797426985353087, 0.125939180544827],
    ]
)

_TRI_BARY = _TRI_RULE_DEG5[:, :3]
_TRI_W = _TRI_RULE_DEG5[:, 3]

MASKED_FRACTION_LIMIT = 1e-3


def _corners(nodes: np.ndarray, elements: np.ndarray):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)
    return nodes[elements[:, 0]], nodes[elements[:, 1]], nodes[elements[:, 2]]


def signed_double_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """`2A` with sign, one entry per element."""
    p_0, p_1, p_2 = _corners(nodes, elements)
    return (p_1[:, 0] - p_0[:, 0]) * (p_2[:, 1] - p_0[:, 1]) - (p_2[:, 0] - p_0[:, 0]) * (
        p_1[:, 1] - p_0[:, 1]
    )


def element_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(signed_double_areas(nodes, elements))


def p1_gradients(nodes: np.ndarray, elements: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Element-wise constant gradient of the P1 function given by nodal `values`.
    """
    values = np.asarray(values, dtype=float).ravel()
    p_0, p_1, p_2 = _corners(nodes, elements)
    corner_values = values[np.asarray(elements, dtype=np.int64)]
    two_area = signed_double_areas(nodes, elements)

    grad_x = (
        (p_1[:, 1] - p_2[:, 1]) * corner_values[:, 0]
        + (p_2[:, 1] - p_0[:, 1]) * corner_values[:, 1]
        + (p_0[:, 1] - p_1[:, 1]) * corner_values[:, 2]
    ) / two_area
    grad_y = (
        (p_2[:, 0] - p_1[:, 0]) * corner_values[:, 0]
        + (p_0[:, 0] - p_2[:, 0]) * corner_values[:, 1]
        + (p_1[:, 0] - p_0[:, 0]) * corner_values[:, 2]
    ) / two_area
    return np.stack((grad_x, grad_y), axis=1)


def quadrature_points(nodes: np.ndarray, elements: np.ndarray):
    """
    Physical quadrature points and integration weights.

    Shapes are `(n_elements * 7, 2)` and `(n_elements * 7,)`, ordered
    element-major so a reshape to `(n_elements, 7, ...)` recovers the element
    structure.
    """
    p_0, p_1, p_2 = _corners(nodes, elements)
    points = (
        _TRI_BARY[None, :, 0, None] * p_0[:, None, :]
        + _TRI_BARY[None, :, 1, None] * p_1[:, None, :]
        + _TRI_BARY[None, :, 2, None] * p_2[:, None, :]
    )
    weights = element_areas(nodes, elements)[:, None] * _TRI_W[None, :]
    return points.reshape(-1, 2), weights.reshape(-1)


def p1_at_quadrature(elements: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Exact values of the P1 function at the quadrature points of every element."""
    values = np.asarray(values, dtype=float).ravel()
    corner_values = values[np.asarray(elements, dtype=np.int64)]
    return (corner_values[:, None, :] * _TRI_BARY[None, :, :]).sum(axis=2).reshape(-1)


def integrate(nodes: np.ndarray, elements: np.ndarray, fun: Callable) -> float:
    """Integrate `fun(points) -> (n,)` over the mesh with the degree-5 rule."""
    points, weights = quadrature_points(nodes, elements)
    return float(np.dot(weights, np.asarray(fun(points), dtype=float).ravel()))


def norms_of_nodal_field(nodes: np.ndarray, elements: np.ndarray, values: np.ndarray) -> Dict:
    """
    Norms of the P1 function with the given nodal values.

    Used for the difference of two discrete solutions on one mesh: the
    difference is itself P1, so all the norms are exact.
    """
    values = np.asarray(values, dtype=float).ravel()
    _, weights = quadrature_points(nodes, elements)
    l2_sq = float(np.dot(weights, p1_at_quadrature(elements, values) ** 2))

    gradients = p1_gradients(nodes, elements, values)
    h1_sq = float(np.dot(element_areas(nodes, elements), (gradients**2).sum(axis=1)))

    return {
        "L2": np.sqrt(max(l2_sq, 0.0)),
        "H1_semi": np.sqrt(max(h1_sq, 0.0)),
        "V": np.sqrt(max(l2_sq + h1_sq, 0.0)),
        "Linf_nodal": float(np.max(np.abs(values))) if values.size else 0.0,
    }


def errors_vs_exact(
    nodes: np.ndarray,
    elements: np.ndarray,
    u_h_nodal: np.ndarray,
    u_exact_fn: Callable[[np.ndarray], np.ndarray],
    grad_u_exact_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict:
    """
    Errors of a P1 solution against a closed-form solution.

    :param u_exact_fn: `(n, 2) -> (n,)`, called once on all quadrature points.
    :param grad_u_exact_fn: `(n, 2) -> (n, 2)`, called once as well.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)
    u_h_nodal = np.asarray(u_h_nodal, dtype=float).ravel()

    points, weights = quadrature_points(nodes, elements)
    u_h = p1_at_quadrature(elements, u_h_nodal)
    u_exact = np.asarray(u_exact_fn(points), dtype=float).ravel()
    if u_exact.shape != u_h.shape:
        raise ValueError(f"u_exact_fn returned shape {u_exact.shape}, expected {u_h.shape}")
    l2_sq = float(np.dot(weights, (u_h - u_exact) ** 2))

    gradients = p1_gradients(nodes, elements, u_h_nodal)
    grad_exact = np.asarray(grad_u_exact_fn(points), dtype=float).reshape(-1, len(_TRI_W), 2)
    difference = grad_exact - gradients[:, None, :]
    h1_sq = float(np.dot(weights, (difference**2).sum(axis=2).reshape(-1)))

    u_exact_nodal = np.asarray(u_exact_fn(nodes), dtype=float).ravel()
    linf = float(np.max(np.abs(u_h_nodal - u_exact_nodal))) if u_h_nodal.size else 0.0

    return {
        "L2": np.sqrt(max(l2_sq, 0.0)),
        "H1_semi": np.sqrt(max(h1_sq, 0.0)),
        "V": np.sqrt(max(l2_sq + h1_sq, 0.0)),
        "Linf_nodal": linf,
    }


def interpolate_p1(
    source_nodes: np.ndarray,
    source_elements: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    context: str = "",
) -> np.ndarray:
    """
    Evaluate a P1 function given on the source mesh at `target_points`, using the
    mesh connectivity.

    Points outside the source mesh, which boundary round-off can produce, are
    filled with the nearest source node value and reported. More than
    `MASKED_FRACTION_LIMIT` of them means the two meshes do not cover the same
    domain.
    """
    source_nodes = np.asarray(source_nodes, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    source_values = np.asarray(source_values, dtype=float).ravel()

    triangulation = mtri.Triangulation(
        source_nodes[:, 0], source_nodes[:, 1], np.asarray(source_elements, dtype=np.int64)
    )
    interpolator = mtri.LinearTriInterpolator(triangulation, source_values)
    values = np.ma.filled(interpolator(target_points[:, 0], target_points[:, 1]), np.nan)
    masked = ~np.isfinite(values)
    n_masked = int(masked.sum())

    if n_masked:
        fraction = n_masked / len(values)
        message = (
            f"interpolate_p1{f' [{context}]' if context else ''}: "
            f"{n_masked} of {len(values)} target points "
            f"({100 * fraction:.4f}%) fell outside the source mesh"
        )
        if fraction > MASKED_FRACTION_LIMIT:
            raise ValueError(
                message + f" - above the {100 * MASKED_FRACTION_LIMIT:.2f}% limit; "
                "the two meshes do not describe the same domain"
            )
        warnings.warn(message + " - filled with nearest-neighbour values", RuntimeWarning)
        print("WARNING: " + message + " - filled with nearest-neighbour values")
        nearest = np.argmin(
            ((target_points[masked][:, None, :] - source_nodes[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        values[masked] = source_values[nearest]

    return values


def evaluate_p1(
    source_nodes: np.ndarray,
    source_elements: np.ndarray,
    source_values: np.ndarray,
    points: np.ndarray,
):
    """
    Value and gradient of a P1 function at arbitrary points.

    Points must be interior to elements, as quadrature points are; on an edge
    the gradient is ambiguous and either side may be returned.
    """
    source_nodes = np.asarray(source_nodes, dtype=float)
    points = np.asarray(points, dtype=float)
    triangulation = mtri.Triangulation(
        source_nodes[:, 0], source_nodes[:, 1], np.asarray(source_elements, dtype=np.int64)
    )
    interpolator = mtri.LinearTriInterpolator(
        triangulation, np.asarray(source_values, dtype=float).ravel()
    )
    values = np.ma.filled(interpolator(points[:, 0], points[:, 1]), np.nan)
    grad_x, grad_y = interpolator.gradient(points[:, 0], points[:, 1])
    gradients = np.stack((np.ma.filled(grad_x, np.nan), np.ma.filled(grad_y, np.nan)), axis=1)
    return values, gradients


def interpolation_consistency_error(
    coarse_nodes: np.ndarray,
    coarse_elements: np.ndarray,
    coarse_values: np.ndarray,
    fine_nodes: np.ndarray,
    fine_elements: np.ndarray,
) -> Dict:
    """
    How much the coarse-to-fine interpolation perturbs the function it carries.

    Measures `u_coarse - I_fine u_coarse` over the fine mesh. For nested meshes
    the coarse function already belongs to the fine space and this is zero, so
    comparisons across such a pair of meshes carry no interpolation error. For
    meshes that are not nested it is not zero, and it enters every error
    computed through the interpolation path.
    """
    fine_values = interpolate_p1(
        coarse_nodes, coarse_elements, coarse_values, fine_nodes, context="consistency"
    )
    points, weights = quadrature_points(fine_nodes, fine_elements)
    u_fine = p1_at_quadrature(fine_elements, fine_values)
    u_coarse, grad_coarse = evaluate_p1(coarse_nodes, coarse_elements, coarse_values, points)
    finite = np.isfinite(u_coarse) & np.isfinite(grad_coarse).all(axis=1)

    grad_fine = np.repeat(p1_gradients(fine_nodes, fine_elements, fine_values), len(_TRI_W), axis=0)
    l2_sq = float(np.dot(weights[finite], (u_fine - u_coarse)[finite] ** 2))
    h1_sq = float(np.dot(weights[finite], ((grad_fine - grad_coarse)[finite] ** 2).sum(axis=1)))
    return {
        "L2": np.sqrt(max(l2_sq, 0.0)),
        "H1_semi": np.sqrt(max(h1_sq, 0.0)),
        "V": np.sqrt(max(l2_sq + h1_sq, 0.0)),
    }


def _mesh_of(state) -> tuple:
    return (
        np.asarray(state.body.mesh.nodes, dtype=float),
        np.asarray(state.body.mesh.elements, dtype=np.int64),
        np.asarray(state.temperature, dtype=float).ravel(),
    )


def errors_between_states(state_a, state_b, context: str = "") -> Dict:
    """
    Norms of `u_a - u_b` for two discrete solutions.

    On identical meshes the difference is nodal, with no interpolation at all.
    On different meshes the coarser solution is carried onto the finer mesh
    through the element connectivity; `interpolation_consistency_error` measures
    what that step costs for a given pair of meshes.
    """
    nodes_a, elements_a, values_a = _mesh_of(state_a)
    nodes_b, elements_b, values_b = _mesh_of(state_b)

    if np.array_equal(nodes_a, nodes_b) and np.array_equal(elements_a, elements_b):
        return norms_of_nodal_field(nodes_a, elements_a, values_a - values_b)

    if len(nodes_a) >= len(nodes_b):
        fine, coarse = (nodes_a, elements_a, values_a), (nodes_b, elements_b, values_b)
    else:
        fine, coarse = (nodes_b, elements_b, values_b), (nodes_a, elements_a, values_a)

    projected = interpolate_p1(coarse[0], coarse[1], coarse[2], fine[0], context=context)
    return norms_of_nodal_field(fine[0], fine[1], fine[2] - projected)


def rate_h(e_prev: float, e_curr: float, h_prev: float, h_curr: float) -> float:
    """`log(e_prev / e_curr) / log(h_prev / h_curr)`; `nan` if undefined."""
    if not (e_prev > 0 and e_curr > 0 and h_prev > 0 and h_curr > 0) or h_prev == h_curr:
        return float("nan")
    return float(np.log(e_prev / e_curr) / np.log(h_prev / h_curr))


def rate_alpha(e_prev: float, e_curr: float, alpha_prev: float, alpha_curr: float) -> float:
    """`log(e_prev / e_curr) / log(alpha_curr / alpha_prev)`; `nan` if undefined."""
    if not (e_prev > 0 and e_curr > 0 and alpha_prev > 0 and alpha_curr > 0):
        return float("nan")
    if not np.isfinite(alpha_prev) or not np.isfinite(alpha_curr) or alpha_prev == alpha_curr:
        return float("nan")
    return float(np.log(e_prev / e_curr) / np.log(alpha_curr / alpha_prev))


def rates_h(errors: Sequence[float], hs: Sequence[float]) -> list:
    """Consecutive `rate_h` values; the first entry is `nan`."""
    return [float("nan")] + [
        rate_h(errors[i - 1], errors[i], hs[i - 1], hs[i]) for i in range(1, len(errors))
    ]


def rates_alpha(errors: Sequence[float], alphas: Sequence[float]) -> list:
    """Consecutive `rate_alpha` values; the first entry is `nan`."""
    return [float("nan")] + [
        rate_alpha(errors[i - 1], errors[i], alphas[i - 1], alphas[i])
        for i in range(1, len(errors))
    ]
