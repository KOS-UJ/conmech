from ctypes import ArgumentError

# import dmsh
# import optimesh
# import meshzoo
import numpy as np
import pygmsh
from deep_conmech.graph.data.data_interpolation import interpolate_point_numba
from numba import njit
from conmech.helpers import mph, nph
from deep_conmech.simulator.mesh import legacy_mesh



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


@njit
def random_corner_mesh_size(mesh_density):
    scale = mesh_density * 0.8
    random_data = np.random.rand(4)
    # random_data = np.zeros(4) #random_data[1] = 1.
    corner_data = (random_data * 2.0 * scale) - scale
    return 1.0 / (mesh_density + corner_data)

    # z = np.sin(np.sqrt(x**2 + y**2))
    # z = 2*(6.0e-2) + 2*(2.0e-1) * ((x+0.5) ** 2 + y ** 2)
    # return z


###############################


def build_mesh(
    mesh_type,
    mesh_density_x,
    mesh_density_y,
    scale_x,
    scale_y,
    is_adaptive,
    create_in_subprocess,
):
    if mesh_type == "cross":
        function = lambda: legacy_mesh.get_cross_rectangle(
            mesh_density_x, mesh_density_y, scale_x, scale_y
        )
    if mesh_type == "meshzoo":
        function = lambda: get_meshzoo_rectangle(mesh_density_x, scale_x, scale_y)
    elif mesh_type == "dmsh":
        function = lambda: get_dmsh_rectangle(mesh_density_x, scale_x, scale_y)
    elif "pygmsh" in mesh_type:
        inner_function = lambda: get_pygmsh(
            mesh_type, mesh_density_x, scale_x, scale_y, is_adaptive
        )
        function = (
            (lambda: mph.run_process(inner_function))
            if create_in_subprocess
            else inner_function
        )
    else:
        raise ArgumentError

    unordered_points, unordered_cells = function()
    return unordered_points, unordered_cells


def get_pygmsh(type, mesh_density, scale_x, scale_y, is_adaptive):

    with pygmsh.geo.Geometry() as geom:
        if "rectangle" in type:
            poly = geom.add_polygon(
                [[0.0, 0.0], [0.0, scale_y], [scale_x, scale_y], [scale_x, 0.0]]
            )
        elif "circle" in type:
            geom.add_circle(
                [scale_x / 2.0, scale_y / 2.0], scale_x / 2.0
            )  # add elipsoid
        elif "polygon" in type:
            geom.add_polygon(
                [
                    [scale_x * 0.0 / 1.4, scale_y * 0.2 / 1.4],
                    [scale_x * 1.0 / 1.4, scale_y * 0.0 / 1.4],
                    [scale_x * 1.1 / 1.4, scale_y * 1.4 / 1.4],
                    [scale_x * 0.1 / 1.4, scale_y * 0.9 / 1.4],
                ]
            )
        elif "spline" in type:
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

        mesh = geom.generate_mesh()
        points = mesh.points[:, 0:2].copy()
        # print("Nodes number: " + str(len(points)))
        cells = mesh.cells[1].data.astype("long").copy()
        # boundary_edges = geom_mesh.cells[0].data.astype("long").copy()

    return points, cells  # , boundary_edges  # not in with
    # mesh.write("out.vtk")


# CORNERS left, bottom, right, top
def set_mesh_size(geom, mesh_density, scale_x, scale_y, is_adaptive):
    if is_adaptive:
        corner_mesh_size = random_corner_mesh_size(mesh_density)
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z: interpolate_point_numba(
                np.array([x, y]), corner_mesh_size, scale_x, scale_y
            )
        )
    else:
        geom.set_mesh_size_callback(lambda dim, tag, x, y, z: 1.0 / mesh_density)


