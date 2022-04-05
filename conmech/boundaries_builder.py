"""
Created at 16.02.2022
"""
from dataclasses import dataclass
from platform import node
from typing import Callable, List, Tuple

import numba
import numpy as np
from conmech.solvers.solver_methods import njit



class Boundaries:
    def __init__(self, contact, dirichlet, neumann):
        self.contact = np.asarray(contact, dtype=np.int32)
        self.dirichlet = np.asarray(dirichlet, dtype=np.int32)
        self.neumann = np.asarray(neumann, dtype=np.int32)




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
    boundary_indices = extract_boundary_indices(boundary_surfaces)
    #boundary_indices = extract_ordered_boundary_indices_2d(boundary_surfaces)
    return boundary_surfaces, boundary_internal_indices, boundary_indices

def extract_boundary_indices(boundary_surfaces):
    return np.unique(boundary_surfaces.flatten(), axis=0)



def apply_predicate_to_surfaces(surfaces, nodes, predicate:Callable):
    mask = [predicate(m) for m in np.mean(nodes[surfaces], axis=1)] #TODO: Use numba (?)
    return surfaces[mask]

def apply_predicate_to_boundary_nodes(elements, nodes, predicate:Callable):
    *_, boundary_indices = get_boundary_surfaces(elements)
    mask = [predicate(n) for n in nodes[boundary_indices]] #TODO: Use numba (?)
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
    return nodes, elements, boundary_nodes_count, contact_nodes_count, dirichlet_nodes_count



def reorder(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    predicate: Callable,
    to_top: bool
):
    selected_indices = apply_predicate_to_boundary_nodes(unordered_elements, unordered_nodes, predicate)
    return reorder_numba(unordered_nodes, unordered_elements, selected_indices, to_top)


@njit
def reorder_numba(
    unordered_nodes: np.ndarray,
    unordered_elements: np.ndarray,
    selected_indices: np.ndarray,
    to_top: bool
):
    nodes_count = len(unordered_nodes)
    last_index = nodes_count - 1

    nodes = np.zeros((nodes_count, unordered_nodes.shape[1]))
    # initially encode all indices to negative values minus one
    elements = -unordered_elements.copy() - 1

    selected_index = 0 if to_top else last_index
    other_index = last_index if to_top else 0
    index_change = 1 if to_top else -1

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

    return nodes, elements, len(selected_indices)




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

    boundaries:Boundaries

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
            boundary_nodes_count,
            contact_nodes_count,
            dirichlet_nodes_count
        ) = reorder_boundary_nodes(unordered_nodes, unordered_elements, is_contact=is_contact, is_dirichlet=is_dirichlet)
        
        neumann_nodes_count = boundary_nodes_count - contact_nodes_count - dirichlet_nodes_count
        
        (
            boundary_surfaces,
            boundary_internal_indices,
            *_
        ) = get_boundary_surfaces(elements)


        contact_surfaces = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, is_contact)
        dirichlet_surfaces = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, is_dirichlet)
        neumann_surfaces = apply_predicate_to_surfaces(boundary_surfaces, initial_nodes, lambda n: not is_contact(n) and not is_dirichlet(n))
        
        
        #boundaries = identify_boundaries(vertices=initial_nodes, elements=elements, boundary_surfaces=boundary_surfaces, is_contact=is_contact, is_dirichlet=is_dirichlet)

        boundaries_new = identify_boundaries_new(contact_surfaces, dirichlet_surfaces, neumann_surfaces)

        bd = BoundariesData(contact_surfaces=contact_surfaces, neumann_surfaces=neumann_surfaces, dirichlet_surfaces=dirichlet_surfaces, 
            contact_nodes_count=contact_nodes_count, neumann_nodes_count=neumann_nodes_count, dirichlet_nodes_count=dirichlet_nodes_count, 
            boundary_internal_indices=boundary_internal_indices, boundaries=boundaries_new)

        return initial_nodes, elements, bd


##############################################################
#for legacy tests
def extract_boundary_path_2d(boundary_edges, start_node = 0):
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


def extract_boundary_paths(elements):
    boundary_surfaces, *_ = get_boundary_surfaces(elements)
    return extract_boundary_paths_new(boundary_surfaces=boundary_surfaces, loop_paths=True)

def extract_boundary_paths_new(boundary_surfaces, loop_paths):
    boundary_indices_to_visit = extract_boundary_indices(boundary_surfaces)

    boundary_paths = []
    while len(boundary_indices_to_visit) > 0:
        start_node = boundary_indices_to_visit[0]
        visited_path = extract_boundary_path_2d(boundary_surfaces, start_node=start_node)
        if loop_paths:
            visited_path = np.append(visited_path, visited_path[0])
        boundary_paths.append(visited_path)
        boundary_indices_to_visit =  list(set(boundary_indices_to_visit) - set(visited_path))

    return boundary_paths


###########


def identify_boundaries_new(contact_surfaces, dirichlet_surfaces, neumann_surfaces):
    contact = extract_boundary_paths_new(contact_surfaces, loop_paths=False)
    dirichlet = extract_boundary_paths_new(dirichlet_surfaces, loop_paths=False)
    neumann = extract_boundary_paths_new(neumann_surfaces, loop_paths=False)

    return Boundaries(
        contact=contact, dirichlet=dirichlet, neumann=neumann
    )
        


def identify_boundaries(
        vertices, elements, boundary_surfaces, is_contact, is_dirichlet
) -> Tuple["Boundaries", np.ndarray, np.ndarray]:

    boundaries = extract_boundary_paths(elements)

    return Boundaries(
        *get_boundaries(is_contact=is_contact, is_dirichlet=is_dirichlet, boundaries=boundaries, vertices=vertices)
    )


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

    # TODO: TEMPORARY FIX
    return fix_boundaries(contact_boundaries), fix_boundaries(dirichlet_boundaries), fix_boundaries(neumann_boundaries)


def fix_boundaries(boundaries):
    boundaries_fixed = []
    for boundary in boundaries:
        boundary_fixed = []
        for index in boundary:
            if index not in boundary_fixed:
                boundary_fixed.append(index)
        boundaries_fixed.append(np.array(boundary_fixed))
    return boundaries_fixed



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
