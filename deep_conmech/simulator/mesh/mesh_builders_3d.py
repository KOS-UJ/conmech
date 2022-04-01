import meshzoo
import pygmsh

from deep_conmech.simulator.mesh.mesh_builders_helpers import *


def get_meshzoo_cube(mesh_density):
    nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_density),
        np.linspace(0.0, 1.0, mesh_density),
        np.linspace(0.0, 1.0, mesh_density),
    )
    return normalize_nodes(nodes), elements


def get_meshzoo_ball(mesh_density):
    nodes, elements = meshzoo.ball_tetra(mesh_density)
    return normalize_nodes(nodes), elements


def get_pygmsh_polygon(mesh_density):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            [[0.0, 0.0], [1.0, -0.2], [1.1, 1.2], [0.1, 0.7], ]
        )
        geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)
        geom.set_mesh_size_callback(lambda dim, tag, x, y, z, _: 1.0 / mesh_density)
        nodes, elements = get_nodes_and_elements(geom, 3)
    return normalize_nodes(nodes), elements


def get_pygmsh_twist(mesh_density):
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
        geom.set_mesh_size_callback(lambda dim, tag, x, y, z, _: 1.0 / mesh_density)
        nodes, elements = get_nodes_and_elements(geom, 3)
    return normalize_nodes(nodes), elements
