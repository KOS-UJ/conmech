import numpy as np
from numba import njit

from conmech.mesh.mesh_properties import MeshProperties
from deep_conmech.graph.data.interpolation_helpers import \
    interpolate_point_numba


@njit
def random_corner_mesh_size(mesh_density):
    scale = mesh_density * 0.8
    random_data = np.random.rand(4)
    # random_data = np.zeros(4) #random_data[1] = 1.
    corner_data = (random_data * 2.0 * scale) - scale
    return 1.0 / (mesh_density + corner_data)

    # z = np.sin(np.sqrt(x**2 + y**2))
    # z = 2*(6.0e-2) + 2*(2.0e-1) * ((x+0.5) ** 2 + y ** 2)
    # return z


# CORNERS left, bottom, right, top
def set_mesh_size(geom, mesh_data: MeshProperties):
    if mesh_data.is_adaptive:
        corner_mesh_size = random_corner_mesh_size(mesh_data.mesh_density_x)
        callback = lambda dim, tag, x, y, z, *_args: interpolate_point_numba(
            np.array([x, y]), corner_mesh_size, mesh_data.scale_x, mesh_data.scale_y)
    else:
        callback = lambda dim, tag, x, y, z, *_args: 1.0 / mesh_data.mesh_density_x

    geom.set_mesh_size_callback(callback)


def normalize_nodes(nodes):
    nodes = nodes - np.min(nodes, axis=0)
    nodes = nodes / np.max(nodes, axis=0)
    return nodes


def get_nodes_and_elements(geom, dim):
    geom_mesh = geom.generate_mesh()
    nodes = geom_mesh.points.copy()
    elements = geom_mesh.cells[-2].data.astype("long").copy()
    return nodes[:, :dim], elements
