import numpy as np
from numba import njit

from conmech.helpers import nph


@njit
def get_cross_points_legacy_ordered_numba(
        points, size_x, size_y, edge_len_x, edge_len_y, left_bottom_point
):
    index = 0
    for j in range(size_y - 1, -1, -1):
        for i in range(size_x - 1, -1, -1):
            points[index] = np.array(((i + 0.5) * edge_len_x, (j + 0.5) * edge_len_y))
            index += 1

    for j in range(size_y - 1, 0, -1):
        for i in range(size_x - 1, 0, -1):
            points[index] = np.array((i * edge_len_x, j * edge_len_y))
            index += 1

    for i in range(1, size_x + 1):
        points[index] = np.array((i * edge_len_x, size_y * edge_len_y))
        index += 1

    for j in range(size_y - 1, -1, -1):
        points[index] = np.array((size_x * edge_len_x, j * edge_len_y))
        index += 1

    for i in range(size_x - 1, -1, -1):
        points[index] = np.array((i * edge_len_x, 0.0))
        index += 1

    for j in range(1, size_y + 1):
        points[index] = np.array((0.0, j * edge_len_y))
        index += 1

    points += left_bottom_point


@njit
def get_cross_elements_numba(
        points, elements, size_x, size_y, edge_len_x, edge_len_y, left_bottom_point
):
    index = 0
    for i in range(size_x):
        for j in range(size_y):
            left_bottom = np.array((i * edge_len_x, j * edge_len_y)) + left_bottom_point

            lb = nph.get_point_index_numba(left_bottom, points)
            rb = nph.get_point_index_numba(
                left_bottom + np.array((edge_len_x, 0.0)), points
            )
            c = nph.get_point_index_numba(
                left_bottom + np.array((0.5 * edge_len_x, 0.5 * edge_len_y)), points,
            )
            lt = nph.get_point_index_numba(
                left_bottom + np.array((0.0, edge_len_y)), points
            )
            rt = nph.get_point_index_numba(
                left_bottom + np.array((edge_len_x, edge_len_y)), points
            )

            elements[index] = np.array((lb, rb, c))
            index += 1
            elements[index] = np.array((rb, rt, c))
            index += 1
            elements[index] = np.array((rt, lt, c))
            index += 1
            elements[index] = np.array((lt, lb, c))
            index += 1


def get_cross_rectangle(mesh_density_x, mesh_density_y, scale_x, scale_y):
    min = np.array((0.0, 0.0))
    size_x = int(mesh_density_x)
    size_y = int(mesh_density_y)
    edge_len_x = scale_x / size_x
    edge_len_y = scale_y / size_y

    points_count = 2 * (size_x * size_y) + (size_x + size_y) + 1
    points = np.zeros((points_count, 2), dtype="float")

    elements_count = 4 * (size_x * size_y)
    elements = np.zeros((elements_count, 3), dtype="long")

    get_cross_points_legacy_ordered_numba(
        points, size_x, size_y, edge_len_x, edge_len_y, min
    )
    get_cross_elements_numba(
        points, elements, size_x, size_y, edge_len_x, edge_len_y, min
    )
    return points, elements
