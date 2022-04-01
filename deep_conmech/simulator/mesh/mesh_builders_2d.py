from ctypes import ArgumentError

import meshzoo
import pygmsh

from conmech.helpers import nph
from deep_conmech.simulator.mesh.mesh_builders_helpers import *


def get_meshzoo_rectangle(mesh_data):
    points, elements = meshzoo.rectangle_tri(
        np.linspace(0.0, mesh_data.scale_x, int(mesh_data.mesh_density_x) + 1),
        np.linspace(0.0, mesh_data.scale_y, int(mesh_data.mesh_density_y) + 1),
        variant="zigzag",
    )
    return points, elements


def get_dmsh_rectangle(mesh_density, corners):
    min = nph.min(corners)
    max = nph.max(corners)
    geo = dmsh.Rectangle(min[0], max[0], min[1], max[1])

    # path = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])

    def edge_size(x):
        return mesh_density  # + 0.1 * path.dist(x)

    points_initial, elements_initial = dmsh.generate(geo, edge_size, tol=1.0e-10)
    return optimesh.optimize_points_elements(
        points_initial, elements_initial, "CVT (full)", 1.0e-10, 100
    )


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

        set_mesh_size(geom, mesh_data)
        nodes, elements = get_nodes_and_elements(geom, 2)
        # boundary_faces = geom_mesh.cells[0].data.astype("long").copy()

    # present_nodes = [i in elements for i in range(len(nodes))]

    return nodes, elements  # , boundary_faces  # not in with
    # mesh.write("out.vtk")
