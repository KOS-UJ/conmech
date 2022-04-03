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




def make_mask_nodes_numba(is_contact):
    is_contact_numba = njit(is_contact)
    @njit
    def mask_nodes_numba(nodes, boundary_faces):
        return np.array(
            [
                i
                for i in range(len(boundary_faces))
                if np.all(
                np.array([is_contact_numba(node) for node in nodes[boundary_faces[i]]])
            )
            ]
        )
    return mask_nodes_numba


def get_boundary_surfaces(elements):
    elements.sort(axis=1)
    faces, opposing_indices = identify_surfaces_numba(sorted_elements=elements)
    boundary_surfaces, boundary_internal_indices = extract_unique_elements(
        faces, opposing_indices
    )
    boundary_indices = np.unique(boundary_surfaces.flatten(), axis=0)
    return boundary_surfaces, boundary_internal_indices, boundary_indices



def apply_predicate(nodes, elements, predicate:Callable):
    surfaces, *_ = get_boundary_surfaces(elements)
    mask = make_mask_nodes_numba(predicate)(nodes, surfaces)
    selected_surfaces = surfaces[mask]
    selected_indices = np.unique(selected_surfaces.flatten(), axis=0)
    return selected_indices

    

def reorder_boundary_nodes(nodes, elements, is_contact, is_dirichlet):
    # move boundary nodes to the beginning
    *_, boundary_indices = get_boundary_surfaces(elements)
    nodes, elements, boundary_nodes_count = reorder(
        nodes, elements, boundary_indices, to_start=True
    )
    # move contact nodes to the beginning
    contact_indices = apply_predicate(nodes, elements, is_contact)
    nodes, elements, contact_nodes_count = reorder(
        nodes, elements, contact_indices, to_start=True
    )
    # move dirichlet nodes to the end
    dirichlet_indices = apply_predicate(nodes, elements, is_dirichlet)
    nodes, elements, dirichlet_nodes_count = reorder(
        nodes, elements, dirichlet_indices, to_start=False
    )
    return nodes, elements, boundary_nodes_count, contact_nodes_count, dirichlet_nodes_count

@njit
def reorder(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    selected_indices: np.ndarray,
    to_start: bool
):
    nodes_count = len(unordered_nodes)
    last_index = nodes_count - 1
    selected_nodes_count = len(selected_indices)

    nodes = np.zeros((nodes_count, unordered_nodes.shape[1]))
    # initially encode all indices to negative values minus one
    elements = -unordered_elements.copy() - 1

    selected_index = 0 if to_start else last_index
    other_index = last_index if to_start else 0
    #fill array with selected nodes from top and other from bottom (or vice versa)
    for old_index in range(nodes_count):
        if old_index in selected_indices:
            new_index = selected_index
            selected_index += 1 if to_start else -1
        else:
            new_index = other_index
            other_index += -1 if to_start else 1

        nodes[new_index] = unordered_nodes[old_index]
        # change encoded old index to new one
        elements = np.where((elements == -old_index - 1), new_index, elements)

    return nodes, elements, selected_nodes_count







@dataclass
class BoundariesData:
    boundary_faces: np.ndarray
    boundary_internal_indices: np.ndarray

    boundary_nodes_count:int
    contact_nodes_count:int
    dirichlet_nodes_count:int

    @property
    def all(self):
        return self.boundary_faces

    @property
    def contact(self):
        return self.boundary_faces

    @property
    def dirichlet(self):
        return self.boundary_faces

    @property
    def neumann(self):
        return self.boundary_faces
    


class BoundariesBuilder:
    """
    Rules:
    - Each boundary must have at least ONE EDGE (two vertices).
    - We indicate only dirichlet and contact boundaries, rest of them are assumed to be neumann.
    - Creating one-edge neumann boundaries shows warning. (not implemented, TODO #35 )
    """

    # CDN: Contact, Dirichlet, Neumann
    @staticmethod
    def identify_boundaries_and_reorder_nodes(
        unordered_nodes, unordered_elements, is_dirichlet, is_contact
    ) -> Tuple[np.ndarray, np.ndarray, BoundariesData]:
        (
            initial_nodes,
            elements,
            boundary_nodes_count,
            contact_nodes_count,
            dirichlet_nodes_count
        ) = reorder_boundary_nodes(unordered_nodes, unordered_elements, is_contact, is_dirichlet)

        (
            boundary_surfaces,
            boundary_internal_indices,
            *_
        ) = get_boundary_surfaces(elements)

        return initial_nodes, elements, BoundariesData(boundary_surfaces, boundary_internal_indices, boundary_nodes_count, contact_nodes_count, dirichlet_nodes_count)
