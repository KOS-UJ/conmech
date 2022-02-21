# import dmsh
# import optimesh
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# import meshio
# import meshzoo
# import numba
# import dmsh as dmsh
import numpy as np
# import optimesh as optimesh
# import pygmsh
# import torch
# from numba import cuda, jit, njit, prange

# import config
# import graph.helpers
import graph.mesh as helpers


def get_cross_points_legacy_ordered(
    points, size_x, size_y, edge_len_x, edge_len_y, left_bottom_point
):

    index = 0
    for j in range(size_y - 1, -1, -1):
        for i in range(size_x - 1, -1, -1):
            points[index] = np.array([(i + 0.5) * edge_len_x, (j + 0.5) * edge_len_y])
            index += 1

    for j in range(size_y - 1, 0, -1):
        for i in range(size_x - 1, 0, -1):
            points[index] = np.array([i * edge_len_x, j * edge_len_y])
            index += 1

    for i in range(1, size_x + 1):
        points[index] = np.array([i * edge_len_x, size_y * edge_len_y])
        index += 1

    for j in range(size_y - 1, -1, -1):
        points[index] = np.array([size_x * edge_len_x, j * edge_len_y])
        index += 1

    for i in range(size_x - 1, -1, -1):
        points[index] = np.array([i * edge_len_x, 0.0])
        index += 1

    for j in range(1, size_y + 1):
        points[index] = np.array([0.0, j * edge_len_y])
        index += 1

    points += np.array(left_bottom_point)


# @njit
def get_cross_cells(points, cells, size_x, size_y, edge_len_x, edge_len_y, left_bottom_point):

    index = 0
    for i in range(size_x):
        for j in range(size_y):
            left_bottom = np.array([i * edge_len_x, j * edge_len_y]) + np.array(
                left_bottom_point
            )

            lb = helpers.get_point_index(left_bottom, points)
            rb = helpers.get_point_index(
                left_bottom + np.array([edge_len_x, 0.0]), points
            )
            c = helpers.get_point_index(
                left_bottom + np.array([0.5 * edge_len_x, 0.5 * edge_len_y]), points,
            )
            lt = helpers.get_point_index(
                left_bottom + np.array([0.0, edge_len_y]), points
            )
            rt = helpers.get_point_index(
                left_bottom + np.array([edge_len_x, edge_len_y]), points
            )

            cells[index] = np.array([lb, rb, c])
            index += 1
            cells[index] = np.array([rb, rt, c])
            index += 1
            cells[index] = np.array([rt, lt, c])
            index += 1
            cells[index] = np.array([lt, lb, c])
            index += 1


############################


def get_meshzoo_rectangle(mesh_size, corners):
    pass  # TODO
    # min = helpers.min(corners)
    # max = helpers.max(corners)
    # return meshzoo.rectangle_tri(
    #     np.linspace(min[0], max[0], int(mesh_size) + 1),
    #     np.linspace(min[1], max[1], int(mesh_size) + 1),
    #     variant="zigzag",
    # )


def get_dmsh_rectangle(mesh_size, corners):
    pass  # TODO
    # min = helpers.min(corners)
    # max = helpers.max(corners)
    # geo = dmsh.Rectangle(min[0], max[0], min[1], max[1])
    # # path = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])
    #
    # def edge_size(x):
    #     return mesh_size  # + 0.1 * path.dist(x)
    #
    # points_initial, cells_initial = dmsh.generate(geo, edge_size, tol=1.0e-10)
    # return optimesh.optimize_points_cells(
    #     points_initial, cells_initial, "CVT (full)", 1.0e-10, 100
    # )


def len_x(corners):
    return corners[2] - corners[0]


def get_base_density(mesh_size, corners):
    return len_x(corners) / mesh_size


def mesh_corner_data(base_density):
    scale = base_density * 0.3  # 0.4
    corner_data = np.random.uniform(low=-scale, high=scale, size=4)
    return corner_data


# @numba.njit
def get_adaptive_mesh_density(x, y, base_density, corner_data):
    correction_left = x * corner_data[0] + (1 - x) * (corner_data[1] - corner_data[0])
    correction_right = x * corner_data[2] + (1 - x) * (corner_data[3] - corner_data[2])
    correction = y * correction_left + (1 - y) * (correction_right - correction_left)
    mesh_density = base_density + correction
    return mesh_density
    #z = np.sin(np.sqrt(x**2 + y**2))
    #z = 2*(6.0e-2) + 2*(2.0e-1) * ((x+0.5) ** 2 + y ** 2)
    #return z


def get_pygmsh_rectangle(mesh_size, corners, is_adaptive):
    pass  # TODO
    # min = helpers.corner_min(corners)
    # max = helpers.corner_min(corners)
    # with pygmsh.geo.Geometry() as geom:
    #     geom.add_polygon(
    #         [[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]],]
    #     )
    #
    #     base_density = get_base_density(mesh_size, corners)
    #     # if(is_adaptive):
    #     #     corner_data = mesh_corner_data(base_density)
    #     #     geom.set_mesh_size_callback(
    #     #         lambda dim, tag, x, y, z: get_adaptive_mesh_density(x, y, base_density, corner_data)
    #     #     )
    #     # else:
    #     geom.set_mesh_size_callback(
    #         lambda dim, tag, x, y, z: base_density)
    #
    #     geom_mesh = geom.generate_mesh()
    #
    #     points = geom_mesh.points[:, 0:2].copy()
    #     #print("Nodes number: " + str(len(points)))
    #     cells = geom_mesh.cells[1].data.astype("long").copy()
    #
    # return points, cells
    #     # mesh.write("out.vtk")


def get_cross_rectangle(mesh_size_x, mesh_size_y, corners):
    size_x = int(mesh_size_x)
    size_y = int(mesh_size_y)
    min = helpers.corner_min(corners)
    edge_len_x = helpers.len_x(corners) / size_x
    edge_len_y = helpers.len_y(corners) / size_y

    points_count = 2 * (size_x * size_y) + (size_x + size_y)+ 1
    points = np.zeros([points_count, 2], dtype="float")

    cells_count = 4 * (size_x * size_y)
    cells = np.zeros([cells_count, 3], dtype="long")

    get_cross_points_legacy_ordered(points, size_x, size_y, edge_len_x, edge_len_y, min)
    get_cross_cells(points, cells, size_x, size_y, edge_len_x, edge_len_y, min)  # TODO size_y
    return points, cells
