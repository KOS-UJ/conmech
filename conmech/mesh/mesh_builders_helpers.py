import numpy as np

from conmech.helpers import nph
from conmech.properties.mesh_properties import MeshProperties
from deep_conmech.data.interpolation_helpers import interpolate_nodes


def get_random_corner_data(dimension: int, scale: float):
    random_vector = nph.generate_normal(rows=dimension * 2, columns=1, sigma=scale / 3)
    clipped_vector = np.maximum(-scale, np.minimum(random_vector, scale))
    normalized_cliped_vector = clipped_vector - np.mean(clipped_vector)
    return 1 + normalized_cliped_vector


def get_random_corner_mesh_size(mesh_prop: MeshProperties):
    return 1.0 / (mesh_prop.corner_mesh_data * mesh_prop.mesh_density_x)


def get_mesh_size_callback(mesh_prop: MeshProperties):
    if mesh_prop.corner_mesh_data is None:
        return lambda dim, tag, x, y, z, *_: 1.0 / mesh_prop.mesh_density_x

    corner_vectors = get_random_corner_mesh_size(mesh_prop=mesh_prop)
    if mesh_prop.dimension == 2:
        return lambda dim, tag, x, y, z, *_: interpolate_nodes(
            scaled_nodes=np.array([[x / mesh_prop.scale_x, y / mesh_prop.scale_y]]),
            corner_vectors=corner_vectors,
        ).item()
    else:
        return lambda dim, tag, x, y, z, *_: interpolate_nodes(
            scaled_nodes=np.array(
                [[x / mesh_prop.scale_x, y / mesh_prop.scale_y, z / mesh_prop.scale_z]]
            ),
            corner_vectors=corner_vectors,
        ).item()


def set_mesh_size(geom, mesh_prop: MeshProperties):
    geom.set_mesh_size_callback(get_mesh_size_callback(mesh_prop))


def normalize_nodes(nodes):
    nodes = nodes - np.min(nodes, axis=0)
    nodes = nodes / np.max(nodes, axis=0)
    return nodes


def get_nodes_and_elements(geom, dim):
    geom_mesh = geom.generate_mesh()
    nodes = geom_mesh.points.copy()
    elements = geom_mesh.cells[-2].data.astype("long").copy()
    return nodes[:, :dim], elements
