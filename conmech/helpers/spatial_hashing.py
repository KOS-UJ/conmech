import numba
import numpy as np

# class Hasher:
#     def __init__(self, nodes, spacing=0.05, table_size_proportion=2):

#         initialize_numba(
#             nodes=nodes,
#             cell_starts=self.cell_starts,
#             cell_entries=self.cell_entries,
#             spacing=self.spacing,
#             table_size=self.table_size,
#         )

#     def query(self, query_node, max_dist):
#         result = query_numba(
#             query_node=query_node,
#             max_dist=max_dist,
#             cell_starts=self.cell_starts,
#             cell_entries=self.cell_entries,
#             table_size=self.table_size,
#             spacing=self.spacing,
#         )
#         return result


@numba.njit(inline="always")
def cell_coord_numba(node, spacing):
    return np.floor(node / spacing).astype(np.int64)


@numba.njit(inline="always")
def cell_hash_numba(node, spacing, table_size):
    return custom_hash_numba(cell_coord_numba(node, spacing), table_size)


@numba.njit(inline="always")
def custom_hash_numba(cell, table_size):
    return (
        abs(int(cell[0] * 92837111) ^ int(cell[1] * 689287499) ^ int(cell[2] * 283923481))
        % table_size
    )


@numba.njit()
def initialize_hasher_numba(
    nodes, spacing=0.05, table_size_proportion=2
):  # TODO: reuse starts and entries
    assert nodes.shape[1] == 3
    # self.max_objects_count = max_objects_count
    objects_count = len(nodes)
    table_size = table_size_proportion * objects_count
    cell_starts = np.zeros(table_size + 1, dtype=np.int64)
    cell_entries = np.zeros(objects_count, dtype=np.int64)

    # query_ids = np.zeros(objects_count, dtype=np.int64)

    for node in nodes:
        h = cell_hash_numba(node=node, spacing=spacing, table_size=table_size)
        cell_starts[h] += 1

    start = 0
    for i, cs in enumerate(cell_starts):
        start += cs
        cell_starts[i] = start
    cell_starts[table_size] = start  # guard

    for i, node in enumerate(nodes):
        h = cell_hash_numba(node=node, spacing=spacing, table_size=table_size)
        cell_starts[h] -= 1
        cell_entries[cell_starts[h]] = i
    return cell_starts, cell_entries, spacing


@numba.njit
def query_hasher_numba(query_node, max_dist, cell_starts, cell_entries, spacing):
    table_size = len(cell_starts) - 1
    cell_min = cell_coord_numba(node=query_node - max_dist, spacing=spacing)
    cell_max = cell_coord_numba(node=query_node + max_dist, spacing=spacing)

    # get all points from cells
    # query_size = 0
    matching_node_ids = set()
    for c1 in range(cell_min[0], cell_max[0] + 1):
        for c2 in range(cell_min[1], cell_max[1] + 1):
            for c3 in range(cell_min[2], cell_max[2] + 1):
                # get cell
                cell = np.array([c1, c2, c3])
                h = custom_hash_numba(cell=cell, table_size=table_size)
                start = cell_starts[h]
                end = cell_starts[h + 1]

                for i in range(start, end):
                    matching_node_ids.add(cell_entries[i])
                    # query_size += 1 # if all nodes - break
                    # if(query_size == len(self.query_ids)):
                    #     return self.query_ids[:query_size]

    return np.array(list(matching_node_ids))
