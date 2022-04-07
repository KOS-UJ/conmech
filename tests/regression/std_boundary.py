from conmech.mesh.boundaries_factory import extract_boundary_paths_from_elements


def standard_boundary_nodes(nodes, elements):
    """
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
