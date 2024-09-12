import numpy as np

from conmech.mesh.boundaries_factory import (
    get_boundary_surfaces,
    extract_unique_indices,
)


def standard_boundary_nodes(nodes, elements):
    r"""
    Return nodes indices counter-clockwise for standard body (rectangle) with first node in (0, 0).

    For body:

    id1 ------ id4
     |  \     / |
     |    id2   |
     |  /    \  |
    id5 ------ id3

    result is [id5, id3, id4, id1]
    """
    boundaries = extract_boundary_paths_from_elements(elements)
    assert len(boundaries) == 1
    boundary = boundaries[0][:-1]  # without closure
    standard_boundary = []
    x = 0
    y = 1
    for i, node_id in enumerate(boundary):
        if nodes[node_id][x] == 0 and nodes[node_id][y] == 0:
            next_node_id = boundary[(i + 1) % len(boundary)]
            prev_node_id = boundary[(i - 1) % len(boundary)]
            if nodes[next_node_id][y] == 0:
                direction = 1
            elif nodes[prev_node_id][y] == 0:
                direction = -1
            else:
                raise AssertionError("Non standard body!")
            start_id = i
            break
    else:
        raise AssertionError("Non standard body!")

    standard_boundary.append(boundary[start_id])
    curr_id = (start_id + direction) % len(boundary)
    while curr_id != start_id:
        standard_boundary.append(boundary[curr_id])
        curr_id = (curr_id + direction) % len(boundary)

    return standard_boundary


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
