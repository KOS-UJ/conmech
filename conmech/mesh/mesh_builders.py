from typing import Tuple

import numpy as np

from conmech.properties.mesh_description import MeshDescription


def build_mesh(mesh_descr: MeshDescription) -> Tuple[np.ndarray, np.ndarray]:
    raw_mesh = mesh_descr.build()
    nodes = translate_nodes(nodes=raw_mesh.nodes, mesh_descr=mesh_descr)
    return nodes, raw_mesh.elements


def translate_nodes(nodes: np.ndarray, mesh_descr: MeshDescription):
    if mesh_descr.initial_position is not None:
        origin = np.min(nodes, axis=0)
        nodes -= origin
        nodes += mesh_descr.initial_position
    return nodes
