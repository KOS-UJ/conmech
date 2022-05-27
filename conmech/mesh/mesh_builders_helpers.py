import numpy as np

from conmech.properties.mesh_properties import MeshProperties
from deep_conmech.data.interpolation_helpers import interpolate_scaled_nodes


def get_mesh_corner_vectors(mesh_prop: MeshProperties):
    return 1.0 / (mesh_prop.mesh_corner_scalars * mesh_prop.mesh_density_x)


def get_mesh_size_callback(mesh_prop: MeshProperties):
    if mesh_prop.mesh_corner_scalars is None:
        return lambda dim, tag, x, y, z, *_: 1.0 / mesh_prop.mesh_density_x

    corner_vectors = get_mesh_corner_vectors(mesh_prop=mesh_prop)
    if mesh_prop.dimension == 2:
        return lambda dim, tag, x, y, z, *_: interpolate_scaled_nodes(
            scaled_nodes=np.array([[x / mesh_prop.scale_x, y / mesh_prop.scale_y]]),
            corner_vectors=corner_vectors,
        ).item()
    else:
        return lambda dim, tag, x, y, z, *_: interpolate_scaled_nodes(
            scaled_nodes=np.array(
                [[x / mesh_prop.scale_x, y / mesh_prop.scale_y, z / mesh_prop.scale_z]]
            ),
            corner_vectors=corner_vectors,
        ).item()


def normalize_nodes(nodes):
    nodes = nodes - np.min(nodes, axis=0)
    nodes = nodes / np.max(nodes, axis=0)
    return nodes


def get_nodes_and_elements(geom, dim):
    geom_mesh = geom.generate_mesh()
    nodes = geom_mesh.points.copy()
    elements = geom_mesh.cells[-2].data.astype("long").copy()
    return nodes[:, :dim], elements
