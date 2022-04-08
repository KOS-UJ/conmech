from ctypes import ArgumentError

import meshzoo
import numpy as np
import pygmsh

from conmech.mesh import mesh_builders_helpers


def get_meshzoo_rectangle(mesh_data):
    # pylint: disable=no-member
    points, elements = meshzoo.rectangle_tri(
        np.linspace(0.0, mesh_data.scale_x, int(mesh_data.mesh_density_x) + 1),
        np.linspace(0.0, mesh_data.scale_y, int(mesh_data.mesh_density_y) + 1),
        variant="zigzag",
    )
    return points, elements


def get_pygmsh_elements_and_nodes(mesh_data):
    with pygmsh.geo.Geometry() as geom:
        if "rectangle" in mesh_data.mesh_type:
            geom.add_polygon(
                [
                    [0.0, 0.0],
                    [0.0, mesh_data.scale_y],
                    [mesh_data.scale_x, mesh_data.scale_y],
                    [mesh_data.scale_x, 0.0],
                ]
            )
        elif "circle" in mesh_data.mesh_type:
            geom.add_circle(
                [mesh_data.scale_x / 2.0, mesh_data.scale_y / 2.0],
                mesh_data.scale_x / 2.0,
            )
        elif "polygon" in mesh_data.mesh_type:
            geom.add_polygon(
                [
                    [mesh_data.scale_x * 0.0 / 1.4, mesh_data.scale_y * 0.2 / 1.4],
                    [mesh_data.scale_x * 1.0 / 1.4, mesh_data.scale_y * 0.0 / 1.4],
                    [mesh_data.scale_x * 1.1 / 1.4, mesh_data.scale_y * 1.4 / 1.4],
                    [mesh_data.scale_x * 0.1 / 1.4, mesh_data.scale_y * 0.9 / 1.4],
                ]
            )
        elif "spline" in mesh_data.mesh_type:
            p_1 = geom.add_point([0.0, 0.0])
            p_2 = geom.add_point([mesh_data.scale_x, 0.0])
            p_3 = geom.add_point([mesh_data.scale_x, mesh_data.scale_y / 2.0])
            p_4 = geom.add_point([mesh_data.scale_x, mesh_data.scale_y])
            s_1 = geom.add_bspline([p_1, p_2, p_3, p_4])

            p_2 = geom.add_point([0.0, mesh_data.scale_y])
            p_3 = geom.add_point([mesh_data.scale_x / 2.0, mesh_data.scale_y])
            s_2 = geom.add_spline([p_4, p_3, p_2, p_1])

            curve_loop = geom.add_curve_loop([s_1, s_2])
            geom.add_plane_surface(curve_loop)

        else:
            raise ArgumentError

        mesh_builders_helpers.set_mesh_size(geom, mesh_data)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 2)
        # boundary_surfaces = geom_mesh.cells[0].data.astype("long").copy()

    return nodes, elements
