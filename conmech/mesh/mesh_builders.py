from typing import Tuple

import numpy as np

from conmech.helpers import mph, nph
from conmech.mesh import mesh_builders_2d, mesh_builders_3d, mesh_builders_legacy
from conmech.properties.mesh_properties import MeshProperties


def build_mesh(
    mesh_prop: MeshProperties,
    create_in_subprocess=False,
) -> Tuple[np.ndarray, np.ndarray]:
    initial_nodes, elements = build_initial_mesh(
        mesh_prop=mesh_prop, create_in_subprocess=create_in_subprocess
    )
    nodes = translate_nodes(nodes=initial_nodes, mesh_prop=mesh_prop)
    return nodes, elements


def translate_nodes(nodes: np.ndarray, mesh_prop: MeshProperties):
    if mesh_prop.initial_base is not None:
        nodes = nph.get_in_base(nodes, mesh_prop.initial_base)
    if mesh_prop.initial_position is not None:
        nodes += mesh_prop.initial_position
    return nodes


def build_initial_mesh(
    mesh_prop: MeshProperties,
    create_in_subprocess=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if "cross" in mesh_prop.mesh_type:
        return mesh_builders_legacy.get_cross_rectangle(mesh_prop)

    if "meshzoo" in mesh_prop.mesh_type:
        if "3d" in mesh_prop.mesh_type:
            if "cube" in mesh_prop.mesh_type:
                return mesh_builders_3d.get_meshzoo_cube(mesh_prop)
            if "ball" in mesh_prop.mesh_type:
                return mesh_builders_3d.get_meshzoo_ball(mesh_prop)
        else:
            return mesh_builders_2d.get_meshzoo_rectangle(mesh_prop)

    if "pygmsh" in mesh_prop.mesh_type:
        if "3d" in mesh_prop.mesh_type:
            if "polygon" in mesh_prop.mesh_type:
                inner_function = lambda: mesh_builders_3d.get_pygmsh_polygon(mesh_prop)
            if "twist" in mesh_prop.mesh_type:
                inner_function = lambda: mesh_builders_3d.get_pygmsh_twist(mesh_prop)
        else:
            inner_function = lambda: mesh_builders_2d.get_pygmsh_elements_and_nodes(mesh_prop)

        return mph.run_process(inner_function) if create_in_subprocess else inner_function()

    raise NotImplementedError(f"Not implemented mesh type: {mesh_prop.mesh_type}")
