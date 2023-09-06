from typing import Tuple

import numpy as np

from conmech.helpers import nph
from conmech.mesh.zoo import MeshZOO
from conmech.mesh.zoo.raw_mesh import RawMesh
from conmech.properties.mesh_properties import MeshProperties
from conmech.mesh import interpolators


def build_mesh(
    mesh_prop: MeshProperties
) -> Tuple[np.ndarray, np.ndarray]:
    raw_mesh = build_initial_mesh(mesh_prop=mesh_prop)
    nodes = translate_nodes(nodes=raw_mesh.nodes, mesh_prop=mesh_prop)
    return nodes, raw_mesh.elements


def translate_nodes(nodes: np.ndarray, mesh_prop: MeshProperties):
    if mesh_prop.mean_at_origin:
        nodes -= np.mean(nodes, axis=0)
    if mesh_prop.initial_base is not None:
        nodes = nph.get_in_base(nodes, mesh_prop.initial_base)
    # TODO #65: Check if works with all combinations of options
    if mesh_prop.corners_vector is not None:
        nodes_interpolation = interpolators.get_nodes_interpolation(
            nodes=nodes,
            base=mesh_prop.initial_base,
            corner_vectors=mesh_prop.corners_vector,
        )
        nodes += nodes_interpolation
    if mesh_prop.initial_position is not None:
        nodes += mesh_prop.initial_position
    return nodes


def build_initial_mesh(
    mesh_prop: MeshProperties,
) -> RawMesh:
    return MeshZOO.get_by_name(mesh_prop.mesh_type)(mesh_prop)
