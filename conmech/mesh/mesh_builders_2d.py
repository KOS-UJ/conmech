from ctypes import ArgumentError

# import meshzoo
import numpy as np
import pygmsh

from conmech.mesh import mesh_builders_helpers
from conmech.properties.mesh_properties import MeshProperties


def get_meshzoo_rectangle(mesh_prop: MeshProperties):
    # pylint: disable=no-member
    nodes, elements = meshzoo.rectangle_tri(
        np.linspace(0.0, mesh_prop.scale_x, int(mesh_prop.mesh_density_x) + 1),
        np.linspace(0.0, mesh_prop.scale_y, int(mesh_prop.mesh_density_y) + 1),
        variant="zigzag",
    )
    return nodes, elements


def get_pygmsh_elements_and_nodes(mesh_prop):
    with pygmsh.geo.Geometry() as geom:
        if "rectangle" in mesh_prop.mesh_type:
            geom.add_polygon(
                [
                    [0.0, 0.0],
                    [0.0, mesh_prop.scale_y],
                    [mesh_prop.scale_x, mesh_prop.scale_y],
                    [mesh_prop.scale_x, 0.0],
                ]
            )
        elif "circle" in mesh_prop.mesh_type:
            geom.add_circle(
                [mesh_prop.scale_x / 2.0, mesh_prop.scale_y / 2.0],
                mesh_prop.scale_x / 2.0,
            )
        elif "polygon" in mesh_prop.mesh_type:
            geom.add_polygon(
                [
                    [mesh_prop.scale_x * 0.0 / 1.4, mesh_prop.scale_y * 0.2 / 1.4],
                    [mesh_prop.scale_x * 1.0 / 1.4, mesh_prop.scale_y * 0.0 / 1.4],
                    [mesh_prop.scale_x * 1.1 / 1.4, mesh_prop.scale_y * 1.4 / 1.4],
                    [mesh_prop.scale_x * 0.1 / 1.4, mesh_prop.scale_y * 0.9 / 1.4],
                ]
            )
        elif "spline" in mesh_prop.mesh_type:
            p_1 = geom.add_point([0.0, 0.0])
            p_2 = geom.add_point([mesh_prop.scale_x, 0.0])
            p_3 = geom.add_point([mesh_prop.scale_x, mesh_prop.scale_y / 2.0])
            p_4 = geom.add_point([mesh_prop.scale_x, mesh_prop.scale_y])
            s_1 = geom.add_bspline([p_1, p_2, p_3, p_4])

            p_2 = geom.add_point([0.0, mesh_prop.scale_y])
            p_3 = geom.add_point([mesh_prop.scale_x / 2.0, mesh_prop.scale_y])
            s_2 = geom.add_spline([p_4, p_3, p_2, p_1])

            curve_loop = geom.add_curve_loop([s_1, s_2])
            geom.add_plane_surface(curve_loop)

        else:
            raise ArgumentError

        mesh_builders_helpers.set_mesh_size(geom, mesh_prop)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 2)
        # boundary_surfaces = geom_mesh.cells[0].data.astype("long").copy()

    return nodes, elements
