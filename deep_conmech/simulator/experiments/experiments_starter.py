# %%
import os

import imageio
import matplotlib.pyplot as plt
import meshzoo
import numba
import numpy as np
import pygmsh
from conmech.helpers import nph
from deep_conmech.common.plotter.plotter_basic import Plotter
from deep_conmech.graph.helpers import thh
from deep_conmech.simulator.experiments.matrices import *
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numba import njit
from scipy.spatial import Delaunay
from torch import arange

DIM = 3
EDIM = 4


def normalize_nodes(nodes):
    nodes = nodes - np.min(nodes, axis=0)
    nodes = nodes / np.max(nodes, axis=0)
    return nodes


def get_from_pygmsh(mesh):
    nodes = mesh.points.copy()
    elements = mesh.cells[2].data.astype("long").copy()
    return nodes, elements


def get_meshzoo_cube(mesh_size):
    initial_nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_size),
        np.linspace(0.0, 1.0, mesh_size),
        np.linspace(0.0, 1.0, mesh_size),
    )
    return initial_nodes, elements


def get_meshzoo_ball(mesh_size):
    nodes, elements = meshzoo.ball_tetra(mesh_size)
    return normalize_nodes(nodes), elements


def get_extrude(mesh_size):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            [[0.0, 0.0], [1.0, -0.2], [1.1, 1.2], [0.1, 0.7],],
            mesh_size=1.0 / mesh_size,
        )
        geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)
        mesh = geom.generate_mesh()

    nodes, elements = get_from_pygmsh(mesh)
    return normalize_nodes(nodes), elements


def get_twist(mesh_size):
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
            ],
            mesh_size=1.0 / mesh_size,
        )

        geom.twist(
            poly,
            translation_axis=[0, 0, 1],
            rotation_axis=[0, 0, 1],
            point_on_axis=[0, 0, 0],
            angle=np.pi / 3,
        )

        mesh = geom.generate_mesh()
    nodes, elements = get_from_pygmsh(mesh)
    return normalize_nodes(nodes), elements


def list_all_faces(elements):
    elements.sort(axis=1)
    elements_count, element_size = elements.shape
    dim = element_size - 1
    faces = np.zeros((element_size * elements_count, dim), dtype=np.int64)
    i = 0
    for j in range(element_size):
        faces[i : i + elements_count, :j] = elements[:, :j]  # ignoring j-th column
        faces[i : i + elements_count, j:dim] = elements[:, j + 1 : element_size]
        i += elements_count
    return faces


def extract_unique_elements(elements):
    _, indices, count = np.unique(
        elements, axis=0, return_index=True, return_counts=True
    )
    return elements[indices[count == 1]]


def get_boundary_faces(elements):
    faces = list_all_faces(elements)
    boundary_faces = extract_unique_elements(faces)
    return boundary_faces


######################################################


def plot_mesh(ax, nodes, boundary_faces, nodes_indices, elements, color):
    boundary_faces_nodes = nodes[boundary_faces]
    ax.add_collection3d(
        Poly3DCollection(
            boundary_faces_nodes,
            edgecolors=color,
            linewidths=0.1,
            facecolors=color,
            alpha=0.08,
        )
    )

    """
    for e in elements:
        pts = nodes[e, :]
        for i in range(EDIM):
            for j in range(EDIM):
                if i < j:
                    ax.plot3D(
                        pts[[i, j], 0], pts[[i, j], 1], pts[[i, j], 2], color=color, lw=0.4
                    )
    
    """
    # boundary_nodes = nodes[nodes_indices]
    # ax.scatter(
    #    boundary_nodes[:, 0], boundary_nodes[:, 1], boundary_nodes[:, 2], color="b"
    # )


######################################################


# @njit
def get_furthest_apart_numba(nodes, variable):
    max_dist = 0.0
    max_i, max_j = 0, 0
    nodes_count = len(nodes)
    for i in range(nodes_count):
        for j in range(i, nodes_count):
            dist = np.abs(nodes[i, variable] - nodes[j, variable])
            error = nph.euclidean_norm_numba(np.delete(nodes[i], variable) - np.delete(nodes[j], variable))
            if dist > max_dist and error < 0.02:
                max_dist = dist
                max_i, max_j = i, j

    if nodes[max_i, variable] < nodes[max_j, variable]:
        return [max_i, max_j]
    else:
        return [max_j, max_i]


def get_base_seed_indices(nodes):
    dim = nodes.shape[1]
    base_seed_indices = np.array(
        [get_furthest_apart_numba(nodes, i) for i in range(1, dim)]
    )
    return base_seed_indices


def get_base(nodes, base_seed_indices):
    base_seed_initial_nodes = nodes[base_seed_indices]
    base_seed = base_seed_initial_nodes[..., 1, :] - base_seed_initial_nodes[..., 0, :]
    return nph.complete_base(base_seed)


######################################################


def correct_base(moved_base):
    
    vx, vy, vz = moved_base
    ix, iy, iz = initial_base
    e1, e2, e3 = np.eye(3)
    base = nph.normalize_euclidean_numba(
        np.array(
            [
                vx - e2 * vy * (ix / iy) - e3 * vz * (ix / iz),
                vy - e1 * vx * (iy / ix) - e3 * vz * (iy / iz),
                vz - e1 * vx * (iz / ix) - e2 * vy * (iz / iy),
            ]
        )
    )
    return base

# przedstawić siły w initial base (-)
def common_base(moved_base):
    return moved_base
    #return correct_base(moved_base)
    # a = nph.get_in_base(moved_base, np.linalg.inv(initial_base.T))
    # a = nph.get_in_base(correcting_base, moved_base)
    # return a


def normalize_rotate(vectors, moved_base):
    return nph.get_in_base(
        vectors,
        common_base(
            moved_base
        ),  # normalized_base_seed(moved_base_seed, initial_base_seed)
    )


def denormalize_rotate(vectors, moved_base):
    reverse_base = np.linalg.inv(common_base(moved_base))
    return nph.get_in_base(
        vectors,
        # nph.get_in_base(initial_base, moved_base)
        reverse_base,  # denormalized_base_seed(moved_base_seed, initial_base_seed)
    )


######################################################

mesh_size = 3
#initial_nodes, elements = get_meshzoo_cube(mesh_size)
#initial_nodes, elements = get_meshzoo_ball(mesh_size)
# initial_nodes, elements = get_twist(mesh_size)
initial_nodes, elements = get_extrude(mesh_size)

boundary_faces = get_boundary_faces(elements)
boundary_nodes_indices = np.unique(boundary_faces.flatten(), axis=0)

base_seed_indices = get_base_seed_indices(initial_nodes)
#base_seed_indices[0, 1] += 1
#base_seed_indices[1, 1] -= 1  #################
initial_base = get_base(initial_nodes, base_seed_indices)


mean_initial_nodes = np.mean(initial_nodes, axis=0)
normalized_initial_nodes = normalize_rotate(
    initial_nodes - mean_initial_nodes, initial_base
)


edges_features_matrix, element_initial_volume = get_edges_features_matrix_numba(
    elements, initial_nodes
)
# TODO: To tests - sum off slice for area and u == 1
# np.moveaxis(edges_features_matrix, -1,0)[i].sum() == 0
# np.moveaxis(edges_features_matrix, -1,0)[0].sum() == 1
# TODO: switch to dictionary
# TODO: reshape so that AREA = edges_features_matrix[..., 0] is   AREA = edges_features_matrix[0]
# np.moveaxis(edges_features_matrix, -1,0)[0].sum()
# rollaxis -> moveaxis


mu = 0.01
la = 0.01
th = 0.01
ze = 0.01
density = 0.01
time_step = 0.01
nodes_count = len(initial_nodes)
independent_nodes_count = nodes_count
slice_ind = slice(0, nodes_count)


C, B, AREA, A_plus_B_times_ts = get_matrices(
    edges_features_matrix, mu, la, th, ze, density, time_step, slice_ind
)


def unstack(data):
    return data.reshape(-1, DIM, order="F")


def get_E(forces, u_old, v_old):
    F_vector = nph.stack_column(AREA @ forces)
    u_old_vector = nph.stack_column(u_old)
    v_old_vector = nph.stack_column(v_old)

    E = F_vector - A_plus_B_times_ts @ v_old_vector - B @ u_old_vector
    return E


def plt_save(path, extension):
    plt.savefig(
        path,
        transparent=False,
        bbox_inches="tight",
        format=extension,
        pad_inches=0.1,
        dpi=800,  # 1200,
    )
    plt.close()


def get_fig(ax_count):
    # plt.style.use('dark_background')

    background_color = "24292E"  # '1F2428'
    plt.rcParams["axes.facecolor"] = background_color
    plt.rcParams["figure.facecolor"] = background_color
    plt.tight_layout()

    fig = plt.figure(figsize=plt.figaspect(1.0 / ax_count))
    return fig


def prepare_ax(ax):
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    # ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.set_box_aspect((12, 4, 3))

    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 3)
    ax.set_zlim(-1, 2)

    ax.set_xlabel("x", color="w")
    ax.set_ylabel("y", color="w")
    ax.set_zlabel("z", color="w")

    ticks = np.arange(0, 2, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.tick_params(axis="x", colors="w")
    ax.tick_params(axis="y", colors="w")
    ax.tick_params(axis="z", colors="w")

    ax.w_xaxis.line.set_color("w")
    ax.w_yaxis.line.set_color("w")
    ax.w_zaxis.line.set_color("w")


def print_frame(
    moved_nodes,
    normalized_nodes,
    normalized_data,
    a,
    forces,
    elements,
    path,
    extension,
    all_images_paths,
    moved_base,
):
    angles = [[0, 270], [0, 0], [90, 0]]
    angles_count = len(angles)
    fig = get_fig(angles_count)
    # ax = fig.add_subplot(projection="3d")
    for i in range(angles_count):
        ax = fig.add_subplot(1, angles_count, i + 1, projection="3d")
        ax.view_init(elev=angles[i][0], azim=angles[i][1])
        prepare_ax(ax)
        print_frame_internal(
            moved_nodes=moved_nodes,
            normalized_nodes=normalized_nodes,
            normalized_data=normalized_data,
            a=a,
            forces=forces,
            elements=elements,
            moved_base=moved_base,
            ax=ax,
        )

    plt.show()
    plt_save(path, extension)
    all_images_paths.append(path)


def print_frame_internal(
    moved_nodes, normalized_nodes, normalized_data, a, forces, elements, moved_base, ax
):
    """
    reverse_base = np.linalg.inv(moved_base)
    for v in reverse_base: #initial_base:
        z = np.array([0, 0., 0.])
        nv = nph.normalize_euclidean_numba(v)
        ax.quiver(*z, *nv, color="tab:blue")
    """

    base = common_base(moved_base)
    z = np.array([0, 0, 1.5])
    ax.quiver(*z, *(base[0]), color="r")
    ax.quiver(*z, *(base[1]), color="y")
    ax.quiver(*z, *(base[2]), color="g")

    base2 = moved_base
    z = np.array([0, 1.5, 1.5])
    ax.quiver(*z, *(base2[0]), color="r")
    ax.quiver(*z, *(base2[1]), color="y")
    ax.quiver(*z, *(base2[2]), color="g")

    n = normalized_nodes[base_seed_indices[0]]
    ax.scatter(n[:, 0], n[:, 1], n[:, 2], color="y")
    n = normalized_nodes[base_seed_indices[1]]
    ax.scatter(n[:, 0], n[:, 1], n[:, 2], color="g")
    """
    n = moved_nodes[base_seed_indices[0]]
    ax.scatter(n[:,0],n[:,1],n[:,2])
    n = moved_nodes[base_seed_indices[1]]
    ax.scatter(n[:,0],n[:,1],n[:,2])
    """
    """
    for i in range(len(moved_nodes)):
        m = moved_nodes[i]
        f = forces[i] * 40.
        ax.quiver(*m, *f, color="w")
    """

    normalized_nodes2 = normalized_nodes
    for data in normalized_data:
        normalized_nodes2 = normalized_nodes2 + np.array([1.5, 0, 0])
        for i in boundary_nodes_indices:
            m = normalized_nodes2[i]
            d = data[i]
            ax.quiver(*m, *d, color="w", lw=0.2)

        plot_mesh(
            ax,
            normalized_nodes2,
            boundary_faces,
            boundary_nodes_indices,
            elements,
            "tab:blue",
        )  # boundary_nodes_indices

    """
    for i in range(len(moved_nodes)):
        m = moved_nodes[i]
        data = a[i]
        ax.quiver(*m, *data, color="w")
    """
    plot_mesh(
        ax, moved_nodes, boundary_faces, boundary_nodes_indices, elements, "tab:orange",
    )  # boundary_nodes_indices

    plot_mesh(
        ax,
        normalized_nodes,
        boundary_faces,
        boundary_nodes_indices,
        elements,
        "tab:blue",
    )  # boundary_nodes_indices


catalog = f"output/3D {thh.get_timestamp()}"


def f_push(ip, t):
    return np.array([0.05, 0.05, 0.05])
    # return np.repeat(np.array([f0]), nodes_count, axis=0)


def f_rotate(ip, t):
    if t <= 0.5:
        scale = ip[2]  # * ip[2]
        return scale * np.array([0.1, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])


def get_forces_by_function(forces_function, initial_nodes, current_time):
    nodes_count = len(initial_nodes)
    forces = np.zeros((nodes_count, 3), dtype=np.double)
    for i in range(nodes_count):
        forces[i] = forces_function(initial_nodes[i], current_time)
    return forces


def print_one_dynamic():
    all_images_paths = []
    extension = "png"  # pdf
    thh.create_folders(catalog)

    u_old = np.zeros((nodes_count, DIM), dtype=np.double)
    v_old = np.zeros((nodes_count, DIM), dtype=np.double)

    scenario_length = 400
    moved_nodes = initial_nodes

    for i in range(1, scenario_length + 1):
        current_time = i * time_step
        print(f"time: {current_time}")

        moved_nodes = initial_nodes + u_old
        moved_base = get_base(moved_nodes, base_seed_indices)

        mean_moved_nodes = np.mean(moved_nodes, axis=0)
        normalized_nodes = normalize_rotate(moved_nodes - mean_moved_nodes, moved_base)

        forces = get_forces_by_function(f_rotate, initial_nodes, current_time)
        normalized_forces = normalize_rotate(forces, moved_base)
        normalized_u_old = normalized_nodes - normalized_initial_nodes
        normalized_v_old = normalize_rotate(v_old - np.mean(v_old, axis=0), moved_base)

        normalized_E = get_E(normalized_forces, normalized_u_old, normalized_v_old)
        normalized_a = unstack(np.linalg.solve(C, normalized_E))
        a = denormalize_rotate(normalized_a, moved_base)

        if i % 10 == 0:
            print_frame(
                moved_nodes=moved_nodes,
                normalized_nodes=normalized_nodes,
                normalized_data=[
                    normalized_forces * 20,
                    normalized_a / 10,
                    normalized_u_old / 2,
                    normalized_v_old / 2,
                ],
                a=a,
                forces=forces,
                elements=elements,
                path=f"{catalog}/{int(thh.get_timestamp() * 100)}.{extension}",
                extension=extension,
                all_images_paths=all_images_paths,
                moved_base=moved_base,
            )

        v_old = v_old + time_step * a
        u_old = u_old + time_step * v_old

    path = f"{catalog}/ANIMATION.gif"

    images = []
    for image_path in all_images_paths:
        images.append(imageio.imread(image_path))

    duration = 0.1
    args = {"duration": duration}
    imageio.mimsave(path, images, **args)

    for image_path in all_images_paths:
        os.remove(image_path)


print_one_dynamic()


# %%
