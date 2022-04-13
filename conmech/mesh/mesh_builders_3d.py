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
