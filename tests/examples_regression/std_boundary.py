def standard_boundary_nodes(grid):
    """
    Return nodes indices counter-clockwise for standard bod (rectangle) with first node in (0, 0).

    For body:

    id1 ------ id4
     |  \     / |
     |    id2   |
     |  /    \  |
    id5 ------ id3

    result is [id5, id3, id4, id1]
    """
    nodes = grid.Points
    standard_boundary = []
    x = 0
    y = 1

    base = sorted(filter(lambda node: node[y] == 0, nodes), key=lambda node: node[x])
    right = sorted(filter(lambda node: node[x] == base[-1][x], nodes), key=lambda node: node[y])
    top = sorted(filter(lambda node: node[y] == right[-1][y], nodes), key=lambda node: -node[x])
    left = sorted(filter(lambda node: node[x] == 0, nodes), key=lambda node: -node[y])

    for n in base:
        node_id = grid.getPoint(*n[:2])
        assert node_id != -1
        standard_boundary.append(node_id)

    skip_first = True
    for n in right:
        if skip_first:
            skip_first = False
            continue
        node_id = grid.getPoint(*n[:2])
        assert node_id != -1
        standard_boundary.append(node_id)

    skip_first = True
    for n in top:
        if skip_first:
            skip_first = False
            continue
        node_id = grid.getPoint(*n[:2])
        assert node_id != -1
        standard_boundary.append(node_id)

    skip_first = True
    for n in left:
        if skip_first:
            skip_first = False
            continue
        node_id = grid.getPoint(*n[:2])
        assert node_id != -1
        standard_boundary.append(node_id)

    return standard_boundary[:-1]  # without closure
