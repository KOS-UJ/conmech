import numba
import numpy as np

from conmech.properties.mesh_properties import MeshProperties
from deep_conmech.data.interpolation_helpers import interpolate_nodes


@numba.njit
def random_corner_mesh_size(mesh_density):
    scale = mesh_density * 0.8
    random_data = np.random.rand(4).reshape(-1, 1)
    # random_data = np.zeros(4) #random_data[1] = 1.
    corner_data = (random_data * 2.0 * scale) - scale
    return 1.0 / (mesh_density + corner_data)


# CORNERS left, bottom, right, top
def set_mesh_size(geom, mesh_prop: MeshProperties):
    if mesh_prop.is_adaptive:
        if mesh_prop.dimension != 2:
            raise NotImplementedError
        corner_mesh_size = random_corner_mesh_size(mesh_prop.mesh_density_x)
        callback = lambda dim, tag, x, y, z, *_: interpolate_nodes(
            scaled_nodes=np.array([[x / mesh_prop.scale_x, y / mesh_prop.scale_y]]),
            corner_vectors=corner_mesh_size,
        ).item()
    else:
        callback = lambda dim, tag, x, y, z, *_: 1.0 / mesh_prop.mesh_density_x

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
