"""
Created at 16.02.2022
"""
from dataclasses import dataclass
from typing import Callable, Tuple

import numba
import numpy as np


@numba.njit
def identify_surfaces_numba(sorted_elements):
    elements_count, element_size = sorted_elements.shape
    dim = element_size - 1
    surfaces = np.zeros((element_size * elements_count, dim), dtype=np.int64)
    opposing_indices = np.zeros((element_size * elements_count), dtype=np.int64)
    i = 0
    for j in range(element_size):
        # exclude each node from sorted elements and get all combinations to obtain surfaces
        surfaces[i : i + elements_count, :j] = sorted_elements[:, :j]
        surfaces[i : i + elements_count, j:dim] = sorted_elements[
            :, j + 1 : element_size
        ]
        opposing_indices[i : i + elements_count] = sorted_elements[:, j]
        i += elements_count
    return surfaces, opposing_indices


def get_boundary_surfaces(elements):
    elements.sort(axis=1)
    surfaces, opposing_indices = identify_surfaces_numba(sorted_elements=elements)
    # boundaries are created by unique surfaces
    boundary_surfaces, boundary_internal_indices = extract_unique_elements(
        surfaces, opposing_indices
    )
    boundary_indices = extract_unique_indices(boundary_surfaces)
    return boundary_surfaces, boundary_internal_indices, boundary_indices


def extract_unique_indices(surfaces):
    return np.unique(surfaces.flatten(), axis=0)


def extract_unique_elements(elements, opposing_indices):
    _, indices, count = np.unique(
        elements, axis=0, return_index=True, return_counts=True
    )
    unique_indices = indices[count == 1]
    return elements[unique_indices], opposing_indices[unique_indices]


def apply_predicate_to_surfaces(surfaces, nodes, predicate: Callable):
    mask = [
        predicate(m) for m in np.mean(nodes[surfaces], axis=1)
    ]  # TODO: #65 Use numba (?)
    return surfaces[mask]


def apply_predicate_to_boundary_nodes(elements, nodes, predicate: Callable):
    *_, boundary_indices = get_boundary_surfaces(elements)
    mask = [predicate(n) for n in nodes[boundary_indices]]  # TODO: #65 Use numba (?)
    return boundary_indices[mask]


def reorder_boundary_nodes(nodes, elements, is_contact, is_dirichlet):
    # move boundary nodes to the top
    nodes, elements, boundary_nodes_count = reorder(
        nodes, elements, lambda _: True, to_top=True
    )
    # then move contact nodes to the top
    nodes, elements, contact_nodes_count = reorder(
        nodes, elements, is_contact, to_top=True
    )
    # finally move dirichlet nodes to the bottom
    nodes, elements, dirichlet_nodes_count = reorder(
        nodes, elements, is_dirichlet, to_top=False
    )
    return (
        nodes,
        elements,
        boundary_nodes_count,
        contact_nodes_count,
        dirichlet_nodes_count,
    )


def reorder(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    predicate: Callable,
    to_top: bool,
):
    selected_indices = apply_predicate_to_boundary_nodes(
        unordered_elements, unordered_nodes, predicate
    )
    return reorder_numba(unordered_nodes, unordered_elements, selected_indices, to_top)


@numba.njit
def reorder_numba(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    selected_indices: np.ndarray,
    to_top: bool,
):
    nodes_count = len(unordered_nodes)
    last_index = nodes_count - 1

    nodes = np.zeros((nodes_count, unordered_nodes.shape[1]))
    # initially encode all indices to negative values minus one
    elements = -unordered_elements.copy() - 1

    selected_index = 0 if to_top else last_index
    other_index = last_index if to_top else 0
    index_change = 1 if to_top else -1

    # fill array with selected nodes from top and other from bottom (or vice versa)
    for old_index in range(nodes_count):
        if old_index in selected_indices:
            new_index = selected_index
            selected_index += index_change
        else:
            new_index = other_index
            other_index -= index_change

        nodes[new_index] = unordered_nodes[old_index]
        # change encoded old index to new one
        elements = np.where((elements == -old_index - 1), new_index, elements)

    return nodes, elements, len(selected_indices)


@dataclass
class Boundaries:
    contact_boundary: np.ndarray
    neumann_boundary: np.ndarray
    dirichlet_boundary: np.ndarray

    contact_nodes_count: int
    neumann_nodes_count: int
    dirichlet_nodes_count: int

    boundary_internal_indices: np.ndarray

    @property
    def boundary_surfaces(self):
        return np.unique(
            np.vstack(
                (self.contact_boundary, self.neumann_boundary, self.dirichlet_boundary)
            ),
            axis=1,
        )

    @property
    def boundary_nodes_count(self):
        return (
            self.contact_nodes_count
            + self.neumann_nodes_count
            + self.dirichlet_nodes_count
        )


class BoundariesFactory:
    """
    Rules:
    - We indicate only dirichlet and contact boundaries, rest of them are assumed to be neumann.
    - Indices of contact boundary nodes are placed first, then neumann nodes, and indices of
      dirichlet nodes are at the end
    """

    @staticmethod
    def identify_boundaries_and_reorder_nodes(
            unordered_nodes, unordered_elements, is_dirichlet, is_contact
    ) -> Tuple[np.ndarray, np.ndarray, Boundaries]:
        (
            initial_nodes,
            elements,
            boundary_nodes_count,
            contact_nodes_count,
            dirichlet_nodes_count,
        ) = reorder_boundary_nodes(
            unordered_nodes,
            unordered_elements,
            is_contact=is_contact,
            is_dirichlet=is_dirichlet,
        )

        neumann_nodes_count = (
            boundary_nodes_count - contact_nodes_count - dirichlet_nodes_count
        )

        (boundary_surfaces, boundary_internal_indices, *_) = get_boundary_surfaces(
            elements
        )

        contact_boundary = apply_predicate_to_surfaces(
            boundary_surfaces, initial_nodes, is_contact
        )
        dirichlet_boundary = apply_predicate_to_surfaces(
            boundary_surfaces, initial_nodes, is_dirichlet
        )
        neumann_boundary = apply_predicate_to_surfaces(
            boundary_surfaces,
            initial_nodes,
            lambda n: not is_contact(n) and not is_dirichlet(n),
        )

        contact_boundary = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, is_contact)
        dirichlet_boundary = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes,
                                                         is_dirichlet)
        neumann_boundary = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes,
                                                       lambda n: not is_contact(
                                                           n) and not is_dirichlet(n))

        boundaries_data = Boundaries(contact_boundary=contact_boundary,
                                     neumann_boundary=neumann_boundary,
                                     dirichlet_boundary=dirichlet_boundary,
                                     contact_nodes_count=contact_nodes_count,
                                     neumann_nodes_count=neumann_nodes_count,
                                     dirichlet_nodes_count=dirichlet_nodes_count,
                                     boundary_internal_indices=boundary_internal_indices)

        return initial_nodes, elements, boundaries_data


# For tests


def extract_boundary_paths_from_elements(elements):
    boundary_surfaces, *_ = get_boundary_surfaces(elements)
    boundary_indices_to_visit = extract_unique_indices(boundary_surfaces)

    boundary_paths = []
    while len(boundary_indices_to_visit) > 0:
        start_node = boundary_indices_to_visit[0]
        visited_path = extract_boundary_path(boundary_surfaces, start_node=start_node)
        visited_path = np.append(visited_path, visited_path[0])
        boundary_paths.append(visited_path)
        boundary_indices_to_visit = list(
            set(boundary_indices_to_visit) - set(visited_path)
        )

    return boundary_paths


def extract_boundary_path(boundary_edges, start_node=0):
    visited_path = []

    def get_neighbours(node):
        node_edges = boundary_edges[np.any(boundary_edges == node, axis=1)]
        node_edges_flatten = node_edges.flatten()
        neighbours = node_edges_flatten[node_edges_flatten != node]
        return neighbours

    def dfs(node):
        if node not in visited_path:
            visited_path.append(node)
            for neighbour in get_neighbours(node):
                dfs(neighbour)

    dfs(start_node)

    return np.array(visited_path)
