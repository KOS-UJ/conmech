import numba
import numpy as np

from conmech.helpers import nph
from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.properties.mesh_description import CrossMeshDescription


class CrossMesh(RawMesh):
    def __init__(self, mesh_descr: CrossMeshDescription):
        super().__init__(*CrossMesh._get_cross_rectangle(mesh_descr))

    @staticmethod
    def _get_cross_rectangle(mesh_descr: CrossMeshDescription):
        scale_x, scale_y = mesh_descr.scale
        size_x, size_y = [
            int(np.ceil(scale / mesh_descr.max_element_perimeter))
            for scale in mesh_descr.scale
        ]

        min_ = np.array((0.0, 0.0))
        edge_len_x = scale_x / size_x
        edge_len_y = scale_y / size_y

        nodes_count = 2 * (size_x * size_y) + (size_x + size_y) + 1
        nodes = np.zeros((nodes_count, 2), dtype="float")

        elements_count = 4 * (size_x * size_y)
        elements = np.zeros((elements_count, 3), dtype="long")

        CrossMesh._set_cross_nodes_ordered_numba(
            nodes, size_x, size_y, edge_len_x, edge_len_y, min_
        )
        CrossMesh._set_cross_elements_numba(
            nodes, elements, size_x, size_y, edge_len_x, edge_len_y, min_
        )
        return nodes, elements

    @staticmethod
    @numba.njit
    def _set_cross_elements_numba(
        nodes, elements, size_x, size_y, edge_len_x, edge_len_y, left_bottom_node
    ):
        index = 0
        for i in range(size_x):
            for j in range(size_y):
                left_bottom = (
                    np.array((i * edge_len_x, j * edge_len_y)) + left_bottom_node
                )

                lb = nph.get_node_index_numba(left_bottom, nodes)
                rb = nph.get_node_index_numba(
                    left_bottom + np.array((edge_len_x, 0.0)), nodes
                )
                c = nph.get_node_index_numba(
                    left_bottom + np.array((0.5 * edge_len_x, 0.5 * edge_len_y)),
                    nodes,
                )
                lt = nph.get_node_index_numba(
                    left_bottom + np.array((0.0, edge_len_y)), nodes
                )
                rt = nph.get_node_index_numba(
                    left_bottom + np.array((edge_len_x, edge_len_y)), nodes
                )

                elements[index] = np.array((lb, rb, c))
                index += 1
                elements[index] = np.array((rb, rt, c))
                index += 1
                elements[index] = np.array((rt, lt, c))
                index += 1
                elements[index] = np.array((lt, lb, c))
                index += 1

    @staticmethod
    @numba.njit
    def _set_cross_nodes_ordered_numba(
        nodes, size_x, size_y, edge_len_x, edge_len_y, left_bottom_node
    ):
        index = 0
        for j in range(size_y - 1, -1, -1):
            for i in range(size_x - 1, -1, -1):
                nodes[index] = np.array(
                    ((i + 0.5) * edge_len_x, (j + 0.5) * edge_len_y)
                )
                index += 1

        for j in range(size_y - 1, 0, -1):
            for i in range(size_x - 1, 0, -1):
                nodes[index] = np.array((i * edge_len_x, j * edge_len_y))
                index += 1

        for i in range(1, size_x + 1):
            nodes[index] = np.array((i * edge_len_x, size_y * edge_len_y))
            index += 1

        for j in range(size_y - 1, -1, -1):
            nodes[index] = np.array((size_x * edge_len_x, j * edge_len_y))
            index += 1

        for i in range(size_x - 1, -1, -1):
            nodes[index] = np.array((i * edge_len_x, 0.0))
            index += 1

        for j in range(1, size_y + 1):
            nodes[index] = np.array((0.0, j * edge_len_y))
            index += 1

        nodes += left_bottom_node
