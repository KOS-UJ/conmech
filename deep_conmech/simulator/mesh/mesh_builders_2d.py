from ctypes import ArgumentError

import meshzoo
import numpy as np
import pygmsh

from deep_conmech.simulator.mesh import mesh_builders_helpers


def get_meshzoo_rectangle(mesh_data):
    points, elements = meshzoo.rectangle_tri(
        np.linspace(0.0, mesh_data.scale_x, int(mesh_data.mesh_density_x) + 1),
        np.linspace(0.0, mesh_data.scale_y, int(mesh_data.mesh_density_y) + 1),
        variant="zigzag",
    )
    return points, elements



###############################


def get_pygmsh_elements_and_nodes(mesh_data):
    with pygmsh.geo.Geometry() as geom:
        if "rectangle" in mesh_data.mesh_type:
            poly = geom.add_polygon(
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
            )  # add elipsoid
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
            # lcar = 0.1
            p1 = geom.add_point([0.0, 0.0])
            p2 = geom.add_point([mesh_data.scale_x, 0.0])
            p3 = geom.add_point([mesh_data.scale_x, mesh_data.scale_y / 2.0])
            p4 = geom.add_point([mesh_data.scale_x, mesh_data.scale_y])
            s1 = geom.add_bspline([p1, p2, p3, p4])

            p2 = geom.add_point([0.0, mesh_data.scale_y])
            p3 = geom.add_point([mesh_data.scale_x / 2.0, mesh_data.scale_y])
            s2 = geom.add_spline([p4, p3, p2, p1])

            ll = geom.add_curve_loop([s1, s2])
            pl = geom.add_plane_surface(ll)

        else:
            raise ArgumentError

        mesh_builders_helpers.set_mesh_size(geom, mesh_data)
        nodes, elements = mesh_builders_helpers.get_nodes_and_elements(geom, 2)
        # boundary_faces = geom_mesh.cells[0].data.astype("long").copy()

    return nodes, elements
    # mesh.write("out.vtk")
