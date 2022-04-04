"""
Created at 16.02.2022
"""
from typing import Tuple, Callable, List

import numba
import numpy as np


class Boundaries:
    """
    Rules:
    - Each boundary must have at least ONE EDGE (two vertices).
    - We indicate only dirichlet and contact boundaries, rest of them are assumed to be neumann.
    - Creating one-edge neumann boundaries shows warning. (not implemented, TODO #35 )
    """

    # CDN: Contact, Dirichlet, Neumann
    def __init__(self, all_, contact, dirichlet, neumann):
        self.all = all_
        self.contact = np.asarray(contact, dtype=np.int32)
        self.dirichlet = np.asarray(dirichlet, dtype=np.int32)
        self.neumann = np.asarray(neumann, dtype=np.int32)

    @staticmethod
    def identify_boundaries_and_reorder_vertices(
            vertices, elements, is_contact, is_dirichlet
    ) -> Tuple["Boundaries", np.ndarray, np.ndarray]:
        boundaries = identify_surfaces(elements, len(vertices))
        # move contact vertices to the beginning
        vertices, elements, boundaries = reorder(
            vertices, elements, boundaries, is_contact, to_end=False)
        # move dirichlet vertices to the end
        vertices, elements, boundaries = reorder(
            vertices, elements, boundaries, is_dirichlet, to_end=True)

        return Boundaries(
            boundaries,
            *get_boundaries(is_contact, is_dirichlet, boundaries, vertices)
        ), vertices, elements


def get_boundaries(
        is_contact: Callable[[np.ndarray], bool],
        is_dirichlet: Callable[[np.ndarray], bool],
        boundaries: List[np.ndarray],
        vertices: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    contact_boundaries = get_condition_boundaries(
        is_contact, boundaries, vertices)
    dirichlet_boundaries = get_condition_boundaries(
        is_dirichlet, boundaries, vertices)
    neumann_boundaries = get_condition_boundaries_neumann(
        is_contact, is_dirichlet, boundaries, vertices)

    # TODO #35

    return contact_boundaries, dirichlet_boundaries, neumann_boundaries


def get_condition_boundaries(
        predicate: Callable,
        boundaries: List[np.ndarray],
        vertices: np.ndarray
) -> List[np.ndarray]:
    condition_boundaries = []
    for boundary in boundaries:
        first_id = None
        i = 0
        while i < len(boundary):
            if predicate(vertices[boundary[i]]):
                condition_boundary = []
                while i < len(boundary) and predicate(vertices[boundary[i]]):
                    condition_boundary.append(boundary[i])
                    i += 1

                if first_id is None:
                    first_id = len(condition_boundaries)

                condition_boundaries.append(np.asarray(condition_boundary))
            i += 1
        merge_first_and_last(first_id, boundary, condition_boundaries)

    single_vertex_boundaries = []
    for condition_boundary in condition_boundaries:
        if len(condition_boundary) < 2:
            single_vertex_boundaries.append(condition_boundaries)
    if single_vertex_boundaries:
        raise AssertionError(
            "Following boundaries do not contain even one edge (two vertices):\n" +
            str(single_vertex_boundaries)
        )

    return condition_boundaries


def merge_first_and_last(first_id, boundary, condition_boundaries):
    if first_id is not None and first_id != len(condition_boundaries) - 1 \
            and condition_boundaries[first_id][0] == boundary[0] \
            and condition_boundaries[-1][-1] == boundary[-1]:
        condition_boundaries[-1] = np.concatenate(
            (condition_boundaries[-1], condition_boundaries[first_id]))
        if first_id != len(condition_boundaries) - 1:
            del condition_boundaries[first_id]


def get_condition_boundaries_neumann(
        predicate_0: Callable,
        predicate_1: Callable,
        boundaries: List[np.ndarray],
        vertices: np.ndarray
) -> List[np.ndarray]:
    def boundary_change(prev: int, curr: int):
        return predicate_0(vertices[prev]) and not predicate_0(vertices[curr]) \
               and not predicate_1(vertices[prev]) \
               or predicate_1(vertices[prev]) and not predicate_1(vertices[curr]) \
               and not predicate_0(vertices[prev])

    def no_conditions(curr: int):
        return not predicate_0(vertices[curr]) and not predicate_1(vertices[curr])

    condition_boundaries = []
    for boundary in boundaries:
        first_id = None
        i = 1
        while i < len(boundary):
            if boundary_change(boundary[i - 1], boundary[i]) \
                    or i == 1 and no_conditions(boundary[i - 1]):
                condition_boundary = [boundary[i - 1]]  # greedy
                while i < len(boundary) and no_conditions(boundary[i]):
                    condition_boundary.append(boundary[i])
                    i += 1
                if i < len(boundary):  # greedy
                    condition_boundary.append(boundary[i])

                if first_id is None:
                    first_id = len(condition_boundaries)
                condition_boundaries.append(np.asarray(condition_boundary))
            i += 1
        merge_first_and_last(first_id, boundary, condition_boundaries)

    return condition_boundaries


def identify_surfaces(elements, vertex_num):
    # Performance TIP: we need only sparse, triangular matrix
    edges = np.zeros((vertex_num, vertex_num))
    for element in elements:
        edges[element[0], element[1]] += 1
        edges[element[1], element[0]] += 1
        edges[element[1], element[2]] += 1
        edges[element[2], element[1]] += 1
        edges[element[0], element[2]] += 1
        edges[element[2], element[0]] += 1

    surface_edges = np.zeros((vertex_num, 2), dtype=int) - 1
    for i in range(vertex_num):
        first_founded = 0
        for j in range(vertex_num):
            if edges[i, j] == 1:
                surface_edges[i, first_founded] = j
                first_founded = 1

    surfaces = []
    for i in range(vertex_num):
        if surface_edges[i, 0] != -1:
            surface = np.zeros(vertex_num + 1, dtype=int) - 1
            curr = 0
            surface[curr] = i
            v = surface_edges[i, 0]
            v_next = surface_edges[v]
            surface_edges[i, 0] = -1

            while True:
                curr += 1
                surface[curr] = v

                if v_next[0] == surface[curr - 1]:
                    v = v_next[1]
                else:
                    v = v_next[0]

                v_next[0] = -1
                v_next[1] = -1

                if v == -1:
                    break

                v_next = surface_edges[v]

            surfaces.append(surface[:curr + 1].copy())

    return surfaces


def reorder(vertices: np.ndarray, elements: np.ndarray, boundaries, predicate, to_end: bool):
    # Possible improvement: keep condition boundary vertices together
    boundary_vertices = np.concatenate(boundaries)

    predicate_ = numba.njit(predicate)

    indicator = apply_predicates(vertices, boundary_vertices, predicate_, to_end)
    order = np.argsort(indicator, kind="stable")
    map_ = np.argsort(order, kind="stable")

    @numba.njit()
    def mapping(i):
        return map_[i]

    reordered_vertices = vertices[order]
    reordered_elements = np.fromiter(map(mapping, elements.ravel()), dtype=int).reshape(-1, 3)
    reordered_boundaries = []
    for boundary in boundaries:
        reordered_boundaries.append(np.fromiter(map(mapping, boundary), dtype=int))

    return reordered_vertices, reordered_elements, reordered_boundaries


def apply_predicates(vertices, boundary_vertices, value_pred, ascending):
    result = np.full(len(vertices), not ascending, dtype=bool)
    for i in boundary_vertices:
        result[i] = ascending == value_pred(vertices[i])
    return result
