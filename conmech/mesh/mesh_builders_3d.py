from ctypes import ArgumentError

import meshio
import meshzoo
import numpy as np
import pygmsh

from conmech.helpers import nph
from conmech.mesh import mesh_builders_helpers
from conmech.properties.mesh_properties import MeshProperties


def get_meshzoo_cube(mesh_prop: MeshProperties):
    # pylint: disable=no-member
    initial_nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_prop.mesh_density_x),
        np.linspace(0.0, 1.0, mesh_prop.mesh_density_x),
        np.linspace(0.0, 1.0, mesh_prop.mesh_density_x),
    )
    return mesh_builders_helpers.normalize(initial_nodes), elements


def get_meshzoo_ball(mesh_prop: MeshProperties):
    # pylint: disable=no-member
    initial_nodes, elements = meshzoo.ball_tetra(mesh_prop.mesh_density_x)
    return mesh_builders_helpers.normalize(initial_nodes), elements


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

        geom.set_mesh_size_callback(mesh_builders_helpers.get_mesh_size_callback(mesh_prop))
        nodes, elements = mesh_builders_helpers.get_normalized_nodes_and_elements(geom, 3)
    return nodes, elements


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
        geom.set_mesh_size_callback(mesh_builders_helpers.get_mesh_size_callback(mesh_prop))
        nodes, elements = mesh_builders_helpers.get_normalized_nodes_and_elements(geom, 3)
    return nodes, elements



def get_relative_ideal_edge_length(mesh_id):
    mesh = meshio.read(f"models/bunny/bun_zipper_res{mesh_id}.ply")
    nodes = mesh.points
    diag_of_bbox = nph.euclidean_norm_numba(np.max(nodes, axis=0) - np.min(nodes, axis=0))
    
    surfaces = mesh.cells_dict['triangle']
    edges = np.array([[[s[0],s[1]], [s[1],s[2]], [s[2],s[0]]] for s in surfaces]).reshape(-1,2)
    edge_nodes = nodes[edges]
    edge_lengths = nph.euclidean_norm_numba(edge_nodes[:,0] - edge_nodes[:,1])
    mean_length = np.mean(edge_lengths)
    return mean_length / diag_of_bbox


def get_pygmsh_bunny(mesh_prop):
    if mesh_prop.mesh_density_x == 16:
        mesh_id = 2#3
    elif mesh_prop.mesh_density_x == 8:
        mesh_id = 3#4
    else:
        raise ArgumentError

    #relative_ideal_edge_length = get_relative_ideal_edge_length(mesh_id)

    mesh = meshio.read(f"models/bunny/bun_zipper_res{mesh_id}_.msh")
    scale = 3.0 #1.0
    # mesh_builders_helpers.normalize(
    nodes, elements = mesh.points, mesh.cells_dict["tetra"].astype("long")
    nodes += 0.1
    nodes *= 3
    nodes[:, [1, 2]] = nodes[:, [2, 1]]
    nodes *=  scale
    return nodes, elements


def get_pygmsh_armadillo():
    mesh = meshio.read("models/armadillo/armadillo.msh")
    scale = 3.0 #1.0
    nodes, elements = mesh_builders_helpers.normalize(mesh.points), mesh.cells_dict["tetra"].astype("long")
    nodes[:, [1, 2]] = nodes[:, [2, 1]]
    nodes *= scale
    return nodes, elements
