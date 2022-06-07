from ctypes import ArgumentError

import numpy as np

from conmech.properties.mesh_properties import MeshProperties
from deep_conmech.data import interpolation_helpers


def get_mesh_corner_vectors(mesh_prop: MeshProperties):
    return 1.0 / (mesh_prop.mesh_corner_scalars * mesh_prop.mesh_density_x)


def get_mesh_size_callback(mesh_prop: MeshProperties):
    if mesh_prop.mesh_corner_scalars is None:
        mesh_size = 1.0 / mesh_prop.mesh_density_x
        return lambda dim, tag, x, y, z, *_: mesh_size

    if mesh_prop.scale_x != 1 or mesh_prop.scale_y != 1 or mesh_prop.scale_z != 1:
        raise NotImplemented

    corner_vectors = get_mesh_corner_vectors(mesh_prop=mesh_prop)
    return interpolation_helpers.get_mesh_callback(corner_vectors)


def normalize(nodes: np.ndarray):
    v_min = np.min(nodes)
    v_max = np.max(nodes)
    return (nodes - v_min) / (v_max - v_min)


def get_normalized_nodes_and_elements(geom, dimension: int):
    geom_mesh = geom.generate_mesh()
    points = geom_mesh.points.copy()
    initial_nodes = points[:, :dimension]
    elements = geom_mesh.cells[-2].data.astype("long").copy()
    nodes = normalize(initial_nodes)
    return nodes, elements
