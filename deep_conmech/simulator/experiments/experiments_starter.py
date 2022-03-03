# %%
import os

import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import meshzoo
import numba
import numpy as np
import pygmsh
from conmech.helpers import nph
from deep_conmech.graph.helpers import thh
from deep_conmech.simulator.experiments.matrices import *
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from numba import njit

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
            alpha=0.2,
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


@njit
def get_closest_to_axis(nodes, variable):
    min_error = 1.0
    final_i, final_j = 0, 0
    nodes_count = len(nodes)
    for i in range(nodes_count):
        for j in range(i + 1, nodes_count):
            # dist = np.abs(nodes[i, variable] - nodes[j, variable])
            error = nph.euclidean_norm_numba(
                np.delete(nodes[i], variable) - np.delete(nodes[j], variable)
            )
            if error < min_error:
                min_error, final_i, final_j = error, i, j

    correct_order = nodes[final_i, variable] < nodes[final_j, variable]
    indices = (final_i, final_j) if correct_order else (final_j, final_i)
    return np.array([error, *indices])


def get_base_seed_indices(nodes):
    dim = nodes.shape[1]
    base_seed_indices = np.zeros((dim, 2), dtype=np.int64)
    errors = np.zeros(3)
    for i in range(dim):
        result = get_closest_to_axis(nodes, i)
        errors[i] = result[0]
        base_seed_indices[i] = result[1:].astype(np.int64)
        # print(f"MIN ERROR for variable {i}: {errors[i]}")
    return base_seed_indices, np.argmin(errors)


def get_base(nodes, base_seed_indices, closest_seed_index):
    base_seed_initial_nodes = nodes[base_seed_indices]
    base_seed = base_seed_initial_nodes[..., 1, :] - base_seed_initial_nodes[..., 0, :]
    return nph.complete_base(base_seed, closest_seed_index)


######################################################


def normalize_rotate(vectors, moved_base):
    return nph.get_in_base(vectors, moved_base)


def denormalize_rotate(vectors, moved_base):
    return nph.get_in_base(vectors, np.linalg.inv(moved_base),)


######################################################

mesh_size = 10
# initial_nodes, elements = get_meshzoo_cube(mesh_size)
# initial_nodes, elements = get_meshzoo_ball(mesh_size)
# initial_nodes, elements = get_twist(mesh_size)
initial_nodes, elements = get_extrude(mesh_size)

boundary_faces = get_boundary_faces(elements)
boundary_nodes_indices = np.unique(boundary_faces.flatten(), axis=0)

base_seed_indices, closest_seed_index = get_base_seed_indices(initial_nodes)
initial_base = get_base(initial_nodes, base_seed_indices, closest_seed_index)


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


def get_ax(fig, grid, angle, distance):
    ax = fig.add_subplot(grid, projection="3d", facecolor="none")  # none") #000000
    ax.set_proj_type("ortho")
    ax.view_init(elev=angle[0], azim=angle[1])  # , vertical_axis='y')
    ax.dist = distance

    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    aspect = (12, 4, 4)
    ax.set_box_aspect(aspect)

    ax.set_xlim(-1, aspect[0] - 1)
    ax.set_ylim(-1, aspect[1] - 1)
    ax.set_zlim(-1, aspect[2] - 1)

    # ax.set_xlabel("x", color="w")
    # ax.set_ylabel("y", color="w")
    # ax.set_zlabel("z", color="w")

    ticks = []  # np.arange(0, 2, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.tick_params(axis="x", colors="w")
    ax.tick_params(axis="y", colors="w")
    ax.tick_params(axis="z", colors="w")

    ax.w_xaxis.line.set_color("w")
    ax.w_yaxis.line.set_color("w")
    ax.w_zaxis.line.set_color("w")

    return ax


def print_frame(
    moved_nodes,
    normalized_nodes,
    normalized_data,
    elements,
    path,
    extension,
    all_images_paths,
    moved_base,
):
    print = lambda ax: print_subframe(
        moved_nodes=moved_nodes,
        normalized_nodes=normalized_nodes,
        normalized_data=normalized_data,
        elements=elements,
        moved_base=moved_base,
        ax=ax,
    )

    angles = np.array([[[0, -90], [0, 0]], [[30, -60], [90, 0]]])
    distances = np.array([[10, 10], [11, 10]])
    rows, columns, _ = angles.shape

    fig = plt.figure()  # constrained_layout=True)
    grid = fig.add_gridspec(nrows=rows, ncols=columns)
    # , width_ratios=[1, 1.], height_ratios=[1., 1.])
    # fig.subplots_adjust(left=-0.2, bottom=0., right=1., top=1.)#, wspace=-0.4, hspace=-0.4)

    ax1 = get_ax(fig, grid[0, 0], angles[0, 0], distances[0, 0])
    ax1.set_position([0.6, 0.8, 0.4, 0.4])
    print(ax1)

    ax2 = get_ax(fig, grid[0, 1], angles[0, 1], distances[0, 1])
    ax2.set_position([1.0, 0.8, 0.4, 0.4])
    print(ax2)

    ax3 = get_ax(fig, grid[1, 0], angles[1, 0], distances[1, 0])
    ax3.set_position([0.5, 0.3, 0.7, 0.7])
    print(ax3)

    ax4 = get_ax(fig, grid[1, 1], angles[1, 1], distances[1, 1])
    ax4.set_position([1.0, 0.5, 0.4, 0.4])
    print(ax4)

    plt.show()
    plt_save(path, extension)
    all_images_paths.append(path)


def print_subframe(
    moved_nodes, normalized_nodes, normalized_data, elements, moved_base, ax
):
    base = moved_base
    z = np.array([0, 2.0, 2.0])
    ax.quiver(*z, *(base[0]), arrow_length_ratio=0.1, color="r")
    ax.quiver(*z, *(base[1]), arrow_length_ratio=0.1, color="y")
    ax.quiver(*z, *(base[2]), arrow_length_ratio=0.1, color="g")

    plot_mesh(
        ax, moved_nodes, boundary_faces, boundary_nodes_indices, elements, "tab:orange",
    )

    shifted_normalized_nodes = normalized_nodes + np.array([0, 2.0, 0])
    for data in normalized_data:
        for i in boundary_nodes_indices:
            m = shifted_normalized_nodes[i]
            d = data[i]
            ax.quiver(*m, *d, arrow_length_ratio=0.1, color="w", lw=0.2)

        plot_mesh(
            ax,
            shifted_normalized_nodes,
            boundary_faces,
            boundary_nodes_indices,
            elements,
            "tab:blue",
        )

        shifted_normalized_nodes = shifted_normalized_nodes + np.array([2.5, 0, 0])


def plt_save(path, extension):
    plt.savefig(
        path,
        transparent=False,
        facecolor="#24292E",  # AAAAAA',
        bbox_inches="tight",
        pad_inches=0.0,
        format=extension,
        dpi=1000,  # 800 1200,
    )
    plt.close()


catalog = f"output/3D {thh.get_timestamp()}"


def f_push(ip, t):
    return np.array([0.05, 0.05, 0.05])
    # return np.repeat(np.array([f0]), nodes_count, axis=0)


def f_rotate(ip, t):
    if t <= 0.5:
        scale = ip[1] * ip[2]
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
        moved_base = get_base(moved_nodes, base_seed_indices, closest_seed_index)

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
                    normalized_u_old,
                    normalized_v_old,
                    normalized_a,
                ],
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
