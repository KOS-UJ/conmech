import time

import jax
import jax.numpy as jnp
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


@numba.njit()  # "i8[3](f8[3], f8)")  # inline="always")
def cell_coord_numba(node, spacing):
    return np.floor(node / spacing).astype(np.int64)


@numba.njit()  # inline="always")
def cell_hash_numba(node, spacing, table_size):
    cell = cell_coord_numba(node, spacing)
    return custom_hash_numba(cell[0], cell[1], cell[2], table_size)


@numba.njit()  # inline="always")
def custom_hash_numba(c1, c2, c3, table_size):
    return abs((c1 * 92837111) ^ (c2 * 689287499) ^ (c3 * 283923481)) % table_size


@numba.njit()
def initialize_hasher_numba(
    nodes,
    spacing,
    cell_starts,
    node_cell,
):  # TODO: reuse starts and entries
    assert nodes.shape[1] == 3
    # self.max_objects_count = max_objects_count
    table_size = len(cell_starts) - 1

    # query_ids = np.zeros(objects_count, dtype=np.int64)
    cell_starts[:] = 0
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
        node_cell[cell_starts[h]] = i
    return cell_starts, node_cell, spacing


@numba.njit()
def query_hasher_numba(
    nodes_query, ready_nodes_mask, query_node, max_dist, cell_starts, node_cell, spacing
):
    # nodes_mask[:] = False
    table_size = len(cell_starts) - 1
    # cell_min = cell_coord_numba(node=query_node - max_dist, spacing=spacing)
    # cell_max = cell_coord_numba(node=query_node + max_dist, spacing=spacing)

    # get all points from cells
    query_size = 0
    for c1 in range(
        int((query_node[0] - max_dist) / spacing), int((query_node[0] + max_dist) / spacing) + 1
    ):

        for c2 in range(
            int((query_node[1] - max_dist) / spacing), int((query_node[1] + max_dist) / spacing) + 1
        ):

            for c3 in range(
                int((query_node[2] - max_dist) / spacing),
                int((query_node[2] + max_dist) / spacing) + 1,
            ):

                h = custom_hash_numba(c1=c1, c2=c2, c3=c3, table_size=table_size)
                start = cell_starts[h]
                end = cell_starts[h + 1]
                for i in range(start, end):
                    node_id = node_cell[i]
                    if not ready_nodes_mask[node_id]:
                        nodes_query[query_size] = node_id
                        query_size += 1
    return query_size


###


def cell_coord(node, spacing):
    return jnp.floor(node / spacing).astype(np.int64)


def custom_hash(cell, table_size):
    return (
        abs((cell[..., 0] * 92837111) ^ (cell[..., 1] * 689287499) ^ (cell[..., 2] * 283923481))
        % table_size
    )


# @numba.njit()
def query_hasher_jax(nodes_mask, query_node, max_dist, cell_starts, node_cell, spacing):
    nodes_mask = jnp.array(nodes_mask)
    query_node = jnp.array(query_node)
    # cell_starts = jnp.array(cell_starts)
    node_cell = jnp.array(node_cell)

    nodes_mask.at[:].set(False)
    table_size = len(cell_starts) - 1
    cell_min = jax.jit(cell_coord)(node=query_node - max_dist, spacing=spacing)
    cell_max = jax.jit(cell_coord)(node=query_node + max_dist, spacing=spacing)
    cell_max += 1

    def body_fun(c3, cell):
        cell.at[2].set(c3)
        h = jax.jit(custom_hash)(cell=cell, table_size=table_size)
        nodes_mask.at[node_cell[cell_starts[h] : cell_starts[h + 1]]].set(True)
        return cell

    cell = jnp.zeros(3, dtype=np.int64)
    for c1 in range(cell_min[0], cell_max[0]):
        cell.at[0].set(c1)
        for c2 in range(cell_min[1], cell_max[1]):
            cell.at[1].set(c2)
            jax.lax.fori_loop(cell_min[2], cell_max[2], body_fun, cell)
