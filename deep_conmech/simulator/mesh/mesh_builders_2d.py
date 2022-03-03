from ctypes import ArgumentError
# import dmsh
# import optimesh
import meshzoo
import numpy as np
import pygmsh
from deep_conmech.graph.data.data_interpolation import interpolate_point_numba
from conmech.helpers import nph
from deep_conmech.simulator.mesh import mesh_builders_legacy
from deep_conmech.simulator.mesh.mesh_builders_helpers import *



def get_meshzoo_rectangle(mesh_density, corners):
    min = nph.min(corners)
    max = nph.max(corners)
    points, cells = meshzoo.rectangle_tri(
        np.linspace(min[0], max[0], int(mesh_density) + 1),
        np.linspace(min[1], max[1], int(mesh_density) + 1),
        variant="zigzag",
    )
    return points, cells


def get_dmsh_rectangle(mesh_density, corners):
    min = nph.min(corners)
    max = nph.max(corners)
    geo = dmsh.Rectangle(min[0], max[0], min[1], max[1])
    # path = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])

    def edge_size(x):
        return mesh_density  # + 0.1 * path.dist(x)

    points_initial, cells_initial = dmsh.generate(geo, edge_size, tol=1.0e-10)
    return optimesh.optimize_points_cells(
        points_initial, cells_initial, "CVT (full)", 1.0e-10, 100
    )

###############################

def get_pygmsh_elements_and_nodes(mesh_type, mesh_density, scale_x, scale_y, is_adaptive):
    with pygmsh.geo.Geometry() as geom:
        if "rectangle" in mesh_type:
            poly = geom.add_polygon(
                [[0.0, 0.0], [0.0, scale_y], [scale_x, scale_y], [scale_x, 0.0]]
            )
        elif "circle" in mesh_type:
            geom.add_circle(
                [scale_x / 2.0, scale_y / 2.0], scale_x / 2.0
            )  # add elipsoid
        elif "polygon" in mesh_type:
            geom.add_polygon(
                [
                    [scale_x * 0.0 / 1.4, scale_y * 0.2 / 1.4],
                    [scale_x * 1.0 / 1.4, scale_y * 0.0 / 1.4],
                    [scale_x * 1.1 / 1.4, scale_y * 1.4 / 1.4],
                    [scale_x * 0.1 / 1.4, scale_y * 0.9 / 1.4],
                ]
            )
        elif "spline" in mesh_type:
            # lcar = 0.1
            p1 = geom.add_point([0.0, 0.0])
            p2 = geom.add_point([scale_x, 0.0])
            p3 = geom.add_point([scale_x, scale_y / 2.0])
            p4 = geom.add_point([scale_x, scale_y])
            s1 = geom.add_bspline([p1, p2, p3, p4])

            p2 = geom.add_point([0.0, scale_y])
            p3 = geom.add_point([scale_x / 2.0, scale_y])
            s2 = geom.add_spline([p4, p3, p2, p1])

            ll = geom.add_curve_loop([s1, s2])
            pl = geom.add_plane_surface(ll)

        else:
            raise ArgumentError

        set_mesh_size(geom, mesh_density, scale_x, scale_y, is_adaptive)
        nodes, elements = get_nodes_and_elements(geom)
        # boundary_edges = geom_mesh.cells[0].data.astype("long").copy()

    return nodes, elements  # , boundary_edges  # not in with
    # mesh.write("out.vtk")



