import meshzoo
import numpy as np
import pygmsh

from conmech.mesh import mesh_builders_helpers
from conmech.properties.mesh_properties import MeshProperties


def get_meshzoo_cube(mesh_prop: MeshProperties):
    # pylint: disable=no-member
    nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_prop.mesh_density_x),
        np.linspace(0.0, 1.0, mesh_prop.mesh_density_x),
        np.linspace(0.0, 1.0, mesh_prop.mesh_density_x),
    )
    return mesh_builders_helpers.normalize_nodes(nodes), elements


def get_test_cube(_mesh_prop: MeshProperties):
    # TODO: meshzoo seems to be not free for now, we should remove it from dependencies # 87
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 1.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.5, 0.5, 1.0],
            [1.0, 0.5, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    elements = np.asarray(
        [
            [0, 1, 3, 9],
            [12, 13, 15, 21],
            [10, 11, 13, 19],
            [4, 5, 7, 13],
            [1, 4, 3, 13],
            [13, 16, 15, 25],
            [11, 14, 13, 23],
            [5, 8, 7, 17],
            [1, 3, 9, 13],
            [13, 15, 21, 25],
            [11, 13, 19, 23],
            [5, 7, 13, 17],
            [1, 9, 10, 13],
            [13, 21, 22, 25],
            [11, 19, 20, 23],
            [5, 13, 14, 17],
            [3, 9, 13, 12],
            [15, 21, 25, 24],
            [13, 19, 23, 22],
            [7, 13, 17, 16],
            [18, 19, 9, 21],
            [12, 13, 3, 15],
            [10, 11, 1, 13],
            [22, 23, 13, 25],
            [19, 22, 13, 21],
            [13, 16, 7, 15],
            [11, 14, 5, 13],
            [23, 26, 17, 25],
            [19, 21, 13, 9],
            [13, 15, 7, 3],
            [11, 13, 5, 1],
            [23, 25, 17, 13],
            [19, 9, 13, 10],
            [13, 3, 7, 4],
            [11, 1, 5, 2],
            [23, 13, 17, 14],
            [21, 9, 12, 13],
            [15, 3, 6, 7],
            [13, 1, 4, 5],
            [25, 13, 16, 17],
        ]
    )

    return nodes, elements


def get_meshzoo_ball(mesh_prop: MeshProperties):
    # pylint: disable=no-member
    nodes, elements = meshzoo.ball_tetra(mesh_prop.mesh_density_x)
    return mesh_builders_helpers.normalize_nodes(nodes), elements


def get_pygmsh_polygon(mesh_prop: MeshProperties):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            [
                [0.0, 0.0],
                [1.0, -0.2],
                [1.1, 1.2],
                [0.1, 0.7],
            ]
        )
        geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)

        mesh_builders_helpers.set_mesh_size(geom, mesh_prop)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 3)
    return mesh_builders_helpers.normalize_nodes(nodes), elements


def get_pygmsh_twist(mesh_prop: MeshProperties):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            [
                [+0.0, +0.5],
                [-0.1, +0.1],
                [-0.5, +0.0],
                [-0.1, -0.1],
                [+0.0, -0.5],
                [+0.1, -0.1],
                [+0.5, +0.0],
                [+0.1, +0.1],
            ]
        )

        geom.twist(
            poly,
            translation_axis=[0, 0, 1],
            rotation_axis=[0, 0, 1],
            point_on_axis=[0, 0, 0],
            angle=np.pi / 3,
        )
        mesh_builders_helpers.set_mesh_size(geom, mesh_prop)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 3)
    return mesh_builders_helpers.normalize_nodes(nodes), elements
