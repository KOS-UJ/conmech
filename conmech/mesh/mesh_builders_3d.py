from ctypes import ArgumentError

import meshio
import meshzoo
import numpy as np
import pygmsh

from conmech.helpers import cmh, nph
from conmech.mesh import mesh_builders_helpers
from conmech.properties.mesh_properties import MeshProperties


def read_mesh(path):
    with cmh.HiddenPrints():
        mesh = meshio.read(path)
    if mesh is None:
        raise ArgumentError
    return mesh


def get_edges_from_surfaces(surfaces):
    return np.array([[[s[0], s[1]], [s[1], s[2]], [s[2], s[0]]] for s in surfaces]).reshape(-1, 2)


def get_relative_ideal_edge_length(mesh_id):
    mesh = read_mesh(f"models/bunny/bun_zipper_res{mesh_id}.ply")
    nodes = mesh.points
    diag_of_bbox = nph.euclidean_norm_numba(np.max(nodes, axis=0) - np.min(nodes, axis=0))

    surfaces = mesh.cells_dict["triangle"]
    edges = get_edges_from_surfaces(surfaces)
    edge_nodes = nodes[edges]
    edge_lengths = nph.euclidean_norm_numba(edge_nodes[:, 0] - edge_nodes[:, 1])
    mean_length = np.mean(edge_lengths)
    return mean_length / diag_of_bbox


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


def get_pygmsh_slide(mesh_prop):
    with pygmsh.geo.Geometry() as geom:
        if "left" in mesh_prop.mesh_type:
            poly = geom.add_polygon(
                [
                    [-4.0, -1.0, -0.6],
                    [4.0, -1.0, -0.6],
                    [4.0, 1.0, 0.6],
                    [-4.0, 1.0, 0.6],
                ]
            )
        elif "right" in mesh_prop.mesh_type:
            poly = geom.add_polygon(
                [
                    [-4.0, -1.0, 0.6],
                    [4.0, -1.0, 0.6],
                    [4.0, 1.0, -0.6],
                    [-4.0, 1.0, -0.6],
                ]
            )
        else:
            raise ArgumentError
        geom.extrude(poly, [0.0, 0.0, 0.4], num_layers=3)

        geom.set_mesh_size_callback(mesh_builders_helpers.get_mesh_size_callback(mesh_prop))
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, dimension=3)
    return nodes, elements


def get_pygmsh_bunny(mesh_prop, lifted=False):
    if mesh_prop.mesh_density_x == 64:
        mesh_id = 1
    elif mesh_prop.mesh_density_x == 32:
        mesh_id = 2
    elif mesh_prop.mesh_density_x == 16:
        mesh_id = 3
    elif mesh_prop.mesh_density_x in [8, 4]:
        mesh_id = 4
    else:
        raise ArgumentError

    # relative_ideal_edge_length = get_relative_ideal_edge_length(mesh_id)

    mesh = read_mesh(f"models/bunny/bun_zipper_res{mesh_id}_.msh")
    nodes, elements = mesh.points, mesh.cells_dict["tetra"].astype("long")
    if lifted:
        nodes += 0.1
        nodes *= 6
    else:
        nodes *= 6
        nodes -= np.mean(nodes, axis=0)
    nodes[:, [1, 2]] = nodes[:, [2, 1]]
    return nodes, elements


def get_pygmsh_armadillo():
    mesh = read_mesh("models/armadillo/armadillo.msh")
    # scale = 3.0  # 1.0
    scale = 1.0
    nodes, elements = mesh_builders_helpers.normalize(mesh.points), mesh.cells_dict["tetra"].astype(
        "long"
    )
    nodes += 0.5
    nodes[:, [1, 2]] = nodes[:, [2, 1]]
    nodes *= scale
    return nodes, elements


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


def get_test_ball(_mesh_prop: MeshProperties):
    # TODO: meshzoo seems to be not free for now, we should remove it from dependencies # 87
    nodes = np.asarray(
        [
            [-0.33333333, -0.33333333, -0.33333333],
            [-0.13245324, -0.39735971, -0.39735971],
            [0.13245324, -0.39735971, -0.39735971],
            [0.33333333, -0.33333333, -0.33333333],
            [-0.39735971, -0.13245324, -0.39735971],
            [-0.17407766, -0.17407766, -0.52223297],
            [0.17407766, -0.17407766, -0.52223297],
            [0.39735971, -0.13245324, -0.39735971],
            [-0.39735971, 0.13245324, -0.39735971],
            [-0.17407766, 0.17407766, -0.52223297],
            [0.17407766, 0.17407766, -0.52223297],
            [0.39735971, 0.13245324, -0.39735971],
            [-0.33333333, 0.33333333, -0.33333333],
            [-0.13245324, 0.39735971, -0.39735971],
            [0.13245324, 0.39735971, -0.39735971],
            [0.33333333, 0.33333333, -0.33333333],
            [-0.39735971, -0.39735971, -0.13245324],
            [-0.17407766, -0.52223297, -0.17407766],
            [0.17407766, -0.52223297, -0.17407766],
            [0.39735971, -0.39735971, -0.13245324],
            [-0.52223297, -0.17407766, -0.17407766],
            [-0.11111111, -0.11111111, -0.11111111],
            [0.11111111, -0.11111111, -0.11111111],
            [0.52223297, -0.17407766, -0.17407766],
            [-0.52223297, 0.17407766, -0.17407766],
            [-0.11111111, 0.11111111, -0.11111111],
            [0.11111111, 0.11111111, -0.11111111],
            [0.52223297, 0.17407766, -0.17407766],
            [-0.39735971, 0.39735971, -0.13245324],
            [-0.17407766, 0.52223297, -0.17407766],
            [0.17407766, 0.52223297, -0.17407766],
            [0.39735971, 0.39735971, -0.13245324],
            [-0.39735971, -0.39735971, 0.13245324],
            [-0.17407766, -0.52223297, 0.17407766],
            [0.17407766, -0.52223297, 0.17407766],
            [0.39735971, -0.39735971, 0.13245324],
            [-0.52223297, -0.17407766, 0.17407766],
            [-0.11111111, -0.11111111, 0.11111111],
            [0.11111111, -0.11111111, 0.11111111],
            [0.52223297, -0.17407766, 0.17407766],
            [-0.52223297, 0.17407766, 0.17407766],
            [-0.11111111, 0.11111111, 0.11111111],
            [0.11111111, 0.11111111, 0.11111111],
            [0.52223297, 0.17407766, 0.17407766],
            [-0.39735971, 0.39735971, 0.13245324],
            [-0.17407766, 0.52223297, 0.17407766],
            [0.17407766, 0.52223297, 0.17407766],
            [0.39735971, 0.39735971, 0.13245324],
            [-0.33333333, -0.33333333, 0.33333333],
            [-0.13245324, -0.39735971, 0.39735971],
            [0.13245324, -0.39735971, 0.39735971],
            [0.33333333, -0.33333333, 0.33333333],
            [-0.39735971, -0.13245324, 0.39735971],
            [-0.17407766, -0.17407766, 0.52223297],
            [0.17407766, -0.17407766, 0.52223297],
            [0.39735971, -0.13245324, 0.39735971],
            [-0.39735971, 0.13245324, 0.39735971],
            [-0.17407766, 0.17407766, 0.52223297],
            [0.17407766, 0.17407766, 0.52223297],
            [0.39735971, 0.13245324, 0.39735971],
            [-0.33333333, 0.33333333, 0.33333333],
            [-0.13245324, 0.39735971, 0.39735971],
            [0.13245324, 0.39735971, 0.39735971],
            [0.33333333, 0.33333333, 0.33333333],
        ]
    )
    elements = np.asarray(
        [
            [0, 1, 4, 16],
            [32, 33, 36, 48],
            [20, 21, 24, 36],
            [8, 9, 12, 24],
            [40, 41, 44, 56],
            [17, 18, 21, 33],
            [5, 6, 9, 21],
            [37, 38, 41, 53],
            [25, 26, 29, 41],
            [2, 3, 6, 18],
            [34, 35, 38, 50],
            [22, 23, 26, 38],
            [10, 11, 14, 26],
            [42, 43, 46, 58],
            [1, 5, 4, 21],
            [33, 37, 36, 53],
            [21, 25, 24, 41],
            [9, 13, 12, 29],
            [41, 45, 44, 61],
            [18, 22, 21, 38],
            [6, 10, 9, 26],
            [38, 42, 41, 58],
            [26, 30, 29, 46],
            [3, 7, 6, 23],
            [35, 39, 38, 55],
            [23, 27, 26, 43],
            [11, 15, 14, 31],
            [43, 47, 46, 63],
            [1, 4, 16, 21],
            [33, 36, 48, 53],
            [21, 24, 36, 41],
            [9, 12, 24, 29],
            [41, 44, 56, 61],
            [18, 21, 33, 38],
            [6, 9, 21, 26],
            [38, 41, 53, 58],
            [26, 29, 41, 46],
            [3, 6, 18, 23],
            [35, 38, 50, 55],
            [23, 26, 38, 43],
            [11, 14, 26, 31],
            [43, 46, 58, 63],
            [1, 16, 17, 21],
            [33, 48, 49, 53],
            [21, 36, 37, 41],
            [9, 24, 25, 29],
            [41, 56, 57, 61],
            [18, 33, 34, 38],
            [6, 21, 22, 26],
            [38, 53, 54, 58],
            [26, 41, 42, 46],
            [3, 18, 19, 23],
            [35, 50, 51, 55],
            [23, 38, 39, 43],
            [11, 26, 27, 31],
            [43, 58, 59, 63],
            [4, 16, 21, 20],
            [36, 48, 53, 52],
            [24, 36, 41, 40],
            [12, 24, 29, 28],
            [44, 56, 61, 60],
            [21, 33, 38, 37],
            [9, 21, 26, 25],
            [41, 53, 58, 57],
            [29, 41, 46, 45],
            [6, 18, 23, 22],
            [38, 50, 55, 54],
            [26, 38, 43, 42],
            [14, 26, 31, 30],
            [46, 58, 63, 62],
            [32, 33, 16, 36],
            [20, 21, 4, 24],
            [52, 53, 36, 56],
            [40, 41, 24, 44],
            [17, 18, 1, 21],
            [49, 50, 33, 53],
            [37, 38, 21, 41],
            [25, 26, 9, 29],
            [57, 58, 41, 61],
            [34, 35, 18, 38],
            [22, 23, 6, 26],
            [54, 55, 38, 58],
            [42, 43, 26, 46],
            [33, 37, 21, 36],
            [21, 25, 9, 24],
            [53, 57, 41, 56],
            [41, 45, 29, 44],
            [18, 22, 6, 21],
            [50, 54, 38, 53],
            [38, 42, 26, 41],
            [26, 30, 14, 29],
            [58, 62, 46, 61],
            [35, 39, 23, 38],
            [23, 27, 11, 26],
            [55, 59, 43, 58],
            [43, 47, 31, 46],
            [33, 36, 21, 16],
            [21, 24, 9, 4],
            [53, 56, 41, 36],
            [41, 44, 29, 24],
            [18, 21, 6, 1],
            [50, 53, 38, 33],
            [38, 41, 26, 21],
            [26, 29, 14, 9],
            [58, 61, 46, 41],
            [35, 38, 23, 18],
            [23, 26, 11, 6],
            [55, 58, 43, 38],
            [43, 46, 31, 26],
            [33, 16, 21, 17],
            [21, 4, 9, 5],
            [53, 36, 41, 37],
            [41, 24, 29, 25],
            [18, 1, 6, 2],
            [50, 33, 38, 34],
            [38, 21, 26, 22],
            [26, 9, 14, 10],
            [58, 41, 46, 42],
            [35, 18, 23, 19],
            [23, 6, 11, 7],
            [55, 38, 43, 39],
            [43, 26, 31, 27],
            [36, 16, 20, 21],
            [24, 4, 8, 9],
            [56, 36, 40, 41],
            [44, 24, 28, 29],
            [21, 1, 5, 6],
            [53, 33, 37, 38],
            [41, 21, 25, 26],
            [29, 9, 13, 14],
            [61, 41, 45, 46],
            [38, 18, 22, 23],
            [26, 6, 10, 11],
            [58, 38, 42, 43],
            [46, 26, 30, 31],
        ]
    )
    return nodes, elements
