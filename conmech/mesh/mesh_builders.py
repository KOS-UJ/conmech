from ctypes import ArgumentError
from typing import Tuple

import dmsh
import numpy as np
import pygmsh

from conmech.helpers import interpolation_helpers, lnh, mph
from conmech.mesh import (
    mesh_builders_2d,
    mesh_builders_3d,
    mesh_builders_helpers,
    mesh_builders_legacy,
)
from conmech.properties.mesh_properties import MeshProperties


def build_mesh(
    mesh_prop: MeshProperties,
    create_in_subprocess=False,
) -> Tuple[np.ndarray, np.ndarray]:
    initial_nodes, elements = build_initial_mesh(
        mesh_prop=mesh_prop, create_in_subprocess=create_in_subprocess
    )
    assert initial_nodes.shape[1] == mesh_prop.dimension
    nodes = initial_nodes
    nodes = translate_nodes(nodes=initial_nodes, mesh_prop=mesh_prop)
    return nodes, elements


def translate_nodes(nodes: np.ndarray, mesh_prop: MeshProperties):
    # TODO #65: Check if all combinations of options work
    if mesh_prop.initial_nodes_corner_vectors is not None:
        if np.min(nodes) < -0 or np.max(nodes) > 1:
            raise ArgumentError
        nodes_interpolation = interpolation_helpers.interpolate_3d_corner_vectors(
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
    # pylint: disable=too-many-return-statements,too-many-branches
    if "cross" in mesh_prop.mesh_type:
        return mesh_builders_legacy.get_cross_rectangle(mesh_prop)

    if "Barboteu2008" in mesh_prop.mesh_type:  # TODO # 85
        return special_mesh(mesh_prop)
    if "bow" in mesh_prop.mesh_type:  # TODO # 85
        return special_mesh_bow(mesh_prop)

    if "meshzoo" in mesh_prop.mesh_type:
        print("Using saved meshes from meshzoo, ignoring mesh properties")
        if "3d" in mesh_prop.mesh_type:
            if "cube" in mesh_prop.mesh_type:
                # return mesh_builders_3d.get_meshzoo_cube(mesh_prop)
                return mesh_builders_3d.get_test_cube(mesh_prop)
            if "ball" in mesh_prop.mesh_type:
                # return mesh_builders_3d.get_meshzoo_ball(mesh_prop)
                return mesh_builders_3d.get_test_ball(mesh_prop)
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
            return mesh_builders_2d.get_pygmsh_elements_and_nodes(mesh_prop)

        return mph.run_process(inner_function) if create_in_subprocess else inner_function()

    if "slide" in mesh_prop.mesh_type:
        return mesh_builders_3d.get_pygmsh_slide(mesh_prop)

    raise NotImplementedError(f"Mesh type not implemented: {mesh_prop.mesh_type}")


def special_mesh(mesh_prop):
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [1.5, 1.0],
                [1.0, 1.5],
                [1.0, 4.0],
                [0.0, 4.0],
            ],
            mesh_size=0.1,
        )
        geom.set_mesh_size_callback(mesh_builders_helpers.get_mesh_size_callback(mesh_prop))
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 2)
    return nodes, elements


def special_mesh_bow(mesh_prop):
    # pylint: disable=no-member  # for dmsh
    geo = dmsh.Polygon(
        [
            [0.0, 0.0],
            [1.2, 0.0],
            [1.2, 0.6],
            [0.0, 0.6],
        ]
    )
    x1 = 0.15
    x2 = 1.05
    y1 = 0.15
    y2 = 0.45
    r = 0.05
    geo = geo - dmsh.Circle([0.6, 0.0], 0.3)
    geo = geo - dmsh.Circle([x1, y1], r)
    geo = geo - dmsh.Circle([x2, y1], r)
    geo = geo - dmsh.Circle([x1, y2], r)
    geo = geo - dmsh.Circle([x2, y2], r)
    nodes, elements = dmsh.generate(geo, 1 / mesh_prop.mesh_density[0])

    return nodes, elements
