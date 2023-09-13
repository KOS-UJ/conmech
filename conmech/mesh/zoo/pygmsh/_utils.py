import numpy as np
from pygmsh.geo import Geometry

from conmech.properties.mesh_description import GeneratedMeshDescription


# CORNERS left, bottom, right, top
def set_mesh_size(geom: Geometry, mesh_descr: GeneratedMeshDescription):
    def callback(*_):
        return mesh_descr.max_element_perimeter

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
