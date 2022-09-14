from ctypes import ArgumentError
from typing import Tuple

import numpy as np

from conmech.helpers import lnh, mph
from conmech.mesh import mesh_builders_2d, mesh_builders_3d, mesh_builders_legacy
from conmech.properties.mesh_properties import MeshProperties
from deep_conmech.data import interpolation_helpers


def build_mesh(
    mesh_prop: MeshProperties,
    create_in_subprocess=False,
) -> Tuple[np.ndarray, np.ndarray]:
    initial_nodes, elements = build_initial_mesh(
        mesh_prop=mesh_prop, create_in_subprocess=create_in_subprocess
    )
    nodes = initial_nodes
    # nodes = translate_nodes(nodes=initial_nodes, mesh_prop=mesh_prop)
    return nodes, elements


def translate_nodes(nodes: np.ndarray, mesh_prop: MeshProperties):
    # TODO #65: Check if all combinations of options work
    if mesh_prop.initial_nodes_corner_vectors is not None:
        if np.min(nodes) < -0 or np.max(nodes) > 1:
            raise ArgumentError
        nodes_interpolation = interpolation_helpers.interpolate_corner_vectors(
            nodes=nodes,
            base=mesh_prop.initial_base,
            corner_vectors=mesh_prop.initial_nodes_corner_vectors,
        )
        nodes += nodes_interpolation
    if mesh_prop.switch_orientation:
        nodes[:, 0] *= -1
    if mesh_prop.mean_at_origin:
        nodes -= np.mean(nodes, axis=0)
    if mesh_prop.initial_base is not None:
        nodes = lnh.get_in_base(nodes, mesh_prop.initial_base)
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

        def inner_function():
            if "3d" in mesh_prop.mesh_type:
                if "polygon" in mesh_prop.mesh_type:
                    return mesh_builders_3d.get_pygmsh_polygon(mesh_prop)
                if "twist" in mesh_prop.mesh_type:
                    return mesh_builders_3d.get_pygmsh_twist(mesh_prop)
                if "bunny" in mesh_prop.mesh_type:
                    return mesh_builders_3d.get_pygmsh_bunny(mesh_prop)
                if "armadillo" in mesh_prop.mesh_type:
                    return mesh_builders_3d.get_pygmsh_armadillo()
            else:
                return mesh_builders_2d.get_pygmsh_elements_and_nodes(mesh_prop)

        return mph.run_process(inner_function) if create_in_subprocess else inner_function()

    raise NotImplementedError(f"Mesh type not implemented: {mesh_prop.mesh_type}")
