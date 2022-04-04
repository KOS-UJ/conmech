"""
Created at 16.02.2022
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numba
import numpy as np
from conmech.solvers.solver_methods import njit


@njit
def identify_surfaces_numba(sorted_elements):
    elements_count, element_size = sorted_elements.shape
    dim = element_size - 1
    faces = np.zeros((element_size * elements_count, dim), dtype=np.int64)
    opposing_indices = np.zeros((element_size * elements_count), dtype=np.int64)
    i = 0
    for j in range(element_size):
        # exclude each node from sorted elements and get all combinations to obtain surfaces
        faces[i: i + elements_count, :j] = sorted_elements[:, :j]
        faces[i: i + elements_count, j:dim] = sorted_elements[:, j + 1: element_size]
        opposing_indices[i: i + elements_count] = sorted_elements[:, j]
        i += elements_count
    return faces, opposing_indices


    
def extract_unique_elements(elements, opposing_indices):
    _, indices, count = np.unique(
        elements, axis=0, return_index=True, return_counts=True
    )
    unique_indices = indices[count == 1]
    return elements[unique_indices], opposing_indices[unique_indices]






def get_boundary_surfaces(elements):
    elements.sort(axis=1)
    faces, opposing_indices = identify_surfaces_numba(sorted_elements=elements)
    boundary_surfaces, boundary_internal_indices = extract_unique_elements(
        faces, opposing_indices
    )
    boundary_indices = np.unique(boundary_surfaces.flatten(), axis=0)
    return boundary_surfaces, boundary_internal_indices, boundary_indices



def apply_predicate_to_surfaces(surfaces, nodes, predicate:Callable):
    mask = [predicate(m) for m in np.mean(nodes[surfaces], axis=1)] #TODO: Use numba (?)
    return surfaces[mask]

def apply_predicate_to_nodes(indices, nodes, predicate:Callable):
    mask = [predicate(n) for n in nodes[indices]] #TODO: Use numba (?)
    return indices[mask]

    

def reorder_boundary_nodes(nodes, elements, is_contact, is_dirichlet):
    *_, boundary_indices = get_boundary_surfaces(elements)

    contact_indices = apply_predicate_to_nodes(boundary_indices, nodes, is_contact)
    dirichlet_indices = apply_predicate_to_nodes(boundary_indices, nodes, is_dirichlet)
    
    nodes, elements, boundary_nodes_count = reorder(
        nodes, elements, boundary_indices, to_start=True
    )

    contact_nodes_count = len(contact_indices)
    dirichlet_nodes_count = len(dirichlet_indices)
    neumann_nodes_count = boundary_nodes_count - contact_nodes_count - dirichlet_nodes_count
    return nodes, elements, contact_nodes_count, neumann_nodes_count, dirichlet_nodes_count

@njit
def reorder(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    selected_indices: np.ndarray,
    to_start: bool
):
    # move boundary nodes to the beginning starting with contact nodes and  dirichlet nodes to the end
    nodes_count = len(unordered_nodes)
    last_index = nodes_count - 1
    selected_nodes_count = len(selected_indices)

    nodes = np.zeros((nodes_count, unordered_nodes.shape[1]))
    # initially encode all indices to negative values minus one
    elements = -unordered_elements.copy() - 1

    selected_index = 0 if to_start else last_index
    other_index = last_index if to_start else 0
    index_change = 1 if to_start else -1

    #fill array with selected nodes from top and other from bottom (or vice versa)
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

    return nodes, elements, selected_nodes_count




@dataclass
class BoundariesData:
    contact_surfaces: np.ndarray
    neumann_surfaces: np.ndarray
    dirichlet_surfaces: np.ndarray

    contact_nodes_count:int
    neumann_nodes_count:int
    dirichlet_nodes_count:int

    boundary_internal_indices: np.ndarray

    @property
    def boundary_surfaces(self):
        return np.unique(np.vstack((self.contact_surfaces, self.neumann_surfaces, self.dirichlet_surfaces)), axis=1)

    @property
    def boundary_nodes_count(self):
        return self.contact_nodes_count + self.neumann_nodes_count + self.dirichlet_nodes_count


class BoundariesBuilder:
    """
    Rules:
    - We indicate only dirichlet and contact boundaries, rest of them are assumed to be neumann.
    - Indicies of contact boundary nodes are first, then neumann nodes, and indices of dirichlet nodes are at the end
    """

    @staticmethod
    def identify_boundaries_and_reorder_nodes(
        unordered_nodes, unordered_elements, is_dirichlet, is_contact
    ) -> Tuple[np.ndarray, np.ndarray, BoundariesData]:
        (
            initial_nodes,
            elements,
            contact_nodes_count,
            neumann_nodes_count,
            dirichlet_nodes_count
        ) = reorder_boundary_nodes(unordered_nodes, unordered_elements, is_contact, is_dirichlet)

        (
            boundary_surfaces,
            boundary_internal_indices,
            *_
        ) = get_boundary_surfaces(elements)

        contact_surfaces = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, is_contact)
        dirichlet_surfaces = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, is_dirichlet)
        neumann_surfaces = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, lambda n: not is_contact(n) and not is_dirichlet(n))

        return initial_nodes, elements, BoundariesData(contact_surfaces=contact_surfaces, neumann_surfaces=neumann_surfaces, dirichlet_surfaces=dirichlet_surfaces, contact_nodes_count=contact_nodes_count, neumann_nodes_count=neumann_nodes_count, dirichlet_nodes_count=dirichlet_nodes_count,boundary_internal_indices=boundary_internal_indices )
