import meshzoo
import numpy as np
import pygmsh

from conmech.mesh.mesh_properties import MeshProperties
from deep_conmech.simulator.mesh import mesh_builders_helpers


def get_meshzoo_cube(mesh_data: MeshProperties):
    nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_data.mesh_density_x),
        np.linspace(0.0, 1.0, mesh_data.mesh_density_x),
        np.linspace(0.0, 1.0, mesh_data.mesh_density_x),
    )
    return mesh_builders_helpers.normalize_nodes(nodes), elements


def get_meshzoo_ball(mesh_data: MeshProperties):
    nodes, elements = meshzoo.ball_tetra(mesh_data.mesh_density_x)
    return mesh_builders_helpers.normalize_nodes(nodes), elements


def get_pygmsh_polygon(mesh_data: MeshProperties):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            [[0.0, 0.0], [1.0, -0.2], [1.1, 1.2], [0.1, 0.7], ]
        )
        geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)

        mesh_builders_helpers.set_mesh_size(geom, mesh_data)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 3)
    return mesh_builders_helpers.normalize_nodes(nodes), elements


def get_pygmsh_twist(mesh_data: MeshProperties):
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
        mesh_builders_helpers.set_mesh_size(geom, mesh_data)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 3)
    return mesh_builders_helpers.normalize_nodes(nodes), elements
