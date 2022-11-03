"""
Created at 16.02.2022
"""
from typing import Callable, Tuple

import numba
import numpy as np

from conmech.mesh.boundaries import Boundaries
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.boundary import Boundary


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
        surfaces[i : i + elements_count, j:dim] = sorted_elements[:, j + 1 : element_size]
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


def extract_unique_elements(elements: np.ndarray, opposing_indices: np.ndarray):
    _, indices, count = np.unique(elements, axis=0, return_index=True, return_counts=True)
    unique_indices = indices[count == 1]
    return elements[unique_indices], opposing_indices[unique_indices]


@numba.njit
def get_nodes_mask_numba(nodes: np.ndarray, predicate_numba: Callable):
    return [predicate_numba(n) for n in nodes]


def get_surface_centers(surfaces: np.ndarray, nodes: np.ndarray):
    return np.mean(nodes[surfaces], axis=1)


is_all_numba = numba.njit(lambda _: True)


def reorder_boundary_nodes(
    nodes: np.ndarray,
    elements: np.ndarray,
    is_dirichlet_numba: Callable,
    is_contact_numba: Callable,
):
    # move boundary nodes to the top
    nodes, elements, boundary_nodes_count = reorder(nodes, elements, is_all_numba, to_top=True)
    if is_contact_numba is None:
        # is_contact_numba is None - assuming all contact
        contact_nodes_count = boundary_nodes_count
        dirichlet_nodes_count = 0
    else:
        # then move contact nodes to the top
        nodes, elements, contact_nodes_count = reorder(
            nodes, elements, is_contact_numba, to_top=True
        )
        # finally move dirichlet nodes to the bottom
        nodes, elements, dirichlet_nodes_count = reorder(
            nodes, elements, is_dirichlet_numba, to_top=False
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
    predicate_numba: Callable,
    to_top: bool,
):
    *_, boundary_indices = get_boundary_surfaces(unordered_elements)
    unordered_boundary_nodes = unordered_nodes[boundary_indices]
    mask = get_nodes_mask_numba(nodes=unordered_boundary_nodes, predicate_numba=predicate_numba)
    selected_indices = boundary_indices[mask]
    return reorder_numba(unordered_nodes, unordered_elements, selected_indices, to_top)


@numba.njit
def reorder_numba(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    selected_indices: np.ndarray,
    to_top: bool,
):
    if len(selected_indices) == 0:
        return unordered_nodes, unordered_elements, 0

    nodes_count = len(unordered_nodes)
    last_index = nodes_count - 1

    nodes = np.zeros_like(unordered_nodes)
    # initially encode all indices to negative values minus one
    # elements_old = -unordered_elements.copy() - 1

    selected_index = 0 if to_top else last_index
    other_index = last_index if to_top else 0
    index_change = 1 if to_top else -1

    index_pairing = np.zeros(nodes_count, dtype=np.int64)
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
        # elements_old = np.where((elements_old == -old_index - 1), new_index, elements_old)

        index_pairing[old_index] = new_index

    elements = unordered_elements.copy()
    for i, element in enumerate(elements):
        for j, value in enumerate(element):
            elements[i][j] = index_pairing[value]

    # assert np.allclose(elements_old, elements)
    return nodes, elements, len(selected_indices)


class BoundariesFactory:
    """
    Rules:
    - We indicate only dirichlet and contact boundaries, rest of them are assumed to be neumann.
    - Indices of contact boundary nodes are placed first, then neumann nodes, and indices of
      dirichlet nodes are at the end
    """

    @staticmethod
    def get_other_boundaries(boundaries_description, boundary_surfaces, boundary_surface_centers):
        other_boundaries = {}
        for name, indicator in boundaries_description.boundaries.items():
            if name not in ("contact", "dirichlet"):
                indicator_numba = numba.njit(indicator)
                mask = get_nodes_mask_numba(
                    nodes=boundary_surface_centers,
                    predicate_numba=indicator_numba,
                )
                surfaces = boundary_surfaces[mask]
                node_indices = np.unique(surfaces).sort()
                if node_indices:
                    other_boundaries[name] = Boundary(
                        surfaces=surfaces, node_indices=node_indices, node_count=node_indices.size
                    )
        return other_boundaries

    @staticmethod
    def identify_boundaries_and_reorder_nodes(
        unordered_nodes: np.ndarray,
        unordered_elements: np.ndarray,
        boundaries_description: BoundariesDescription,
    ) -> Tuple[np.ndarray, np.ndarray, Boundaries]:

        is_dirichlet = boundaries_description["dirichlet"]
        is_contact = boundaries_description["contact"]
        is_dirichlet_numba = None if is_dirichlet is None else numba.njit(is_dirichlet)
        is_contact_numba = None if is_contact is None else numba.njit(is_contact)

        (
            initial_nodes,
            elements,
            boundary_nodes_count,
            contact_nodes_count,
            dirichlet_nodes_count,
        ) = reorder_boundary_nodes(
            nodes=unordered_nodes,
            elements=unordered_elements,
            is_dirichlet_numba=is_dirichlet_numba,
            is_contact_numba=is_contact_numba,
        )

        neumann_nodes_count = boundary_nodes_count - contact_nodes_count - dirichlet_nodes_count
        boundary_surfaces, boundary_internal_indices, *_ = get_boundary_surfaces(elements)
        if is_contact_numba is None:
            # is_contact_numba is None - assuming all contact
            contact_boundary_surfaces = boundary_surfaces
            dirichlet_boundary_surfaces = np.empty(
                shape=(0, boundary_surfaces.shape[-1]), dtype=np.int64
            )
            neumann_boundary_surfaces = dirichlet_boundary_surfaces.copy()
            other_boundaries = {}
        else:
            boundary_surface_centers = get_surface_centers(
                surfaces=boundary_surfaces, nodes=initial_nodes
            )
            dirichlet_mask = get_nodes_mask_numba(
                nodes=boundary_surface_centers,
                predicate_numba=is_dirichlet_numba,
            )
            contact_mask = get_nodes_mask_numba(
                nodes=boundary_surface_centers,
                predicate_numba=is_contact_numba,
            )
            neumann_mask = np.logical_and(
                np.logical_not(dirichlet_mask), np.logical_not(contact_mask)
            )

            dirichlet_boundary_surfaces = boundary_surfaces[dirichlet_mask]
            contact_boundary_surfaces = boundary_surfaces[contact_mask]
            neumann_boundary_surfaces = boundary_surfaces[neumann_mask]

            other_boundaries = BoundariesFactory.get_other_boundaries(
                boundaries_description, boundary_surfaces, boundary_surface_centers
            )

        boundaries = Boundaries(
            boundary_internal_indices=boundary_internal_indices,
            contact=Boundary(
                surfaces=contact_boundary_surfaces,
                node_indices=slice(0, contact_nodes_count),
                node_count=contact_nodes_count,
            ),
            neumann=Boundary(
                surfaces=neumann_boundary_surfaces,
                node_indices=slice(contact_nodes_count, contact_nodes_count + neumann_nodes_count),
                node_count=neumann_nodes_count,
            ),
            dirichlet=Boundary(
                surfaces=dirichlet_boundary_surfaces,
                node_indices=slice(contact_nodes_count + neumann_nodes_count, boundary_nodes_count),
                node_count=dirichlet_nodes_count,
            ),
            **other_boundaries,
        )

        return initial_nodes, elements, boundaries


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
        boundary_indices_to_visit = list(set(boundary_indices_to_visit) - set(visited_path))

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
