# %%
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import nph
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

from deep_conmech.common.plotter import plotter_3d
from deep_conmech.graph.helpers import thh
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.setting_mesh import *

DIM = 3
EDIM = 4

catalog = f"output/3D - {thh.CURRENT_TIME}"



def normalize_rotate(vectors, moved_base):
    return nph.get_in_base(vectors, moved_base)


def denormalize_rotate(vectors, moved_base):
    return nph.get_in_base(vectors, np.linalg.inv(moved_base))


######################################################

def get_boundary_normals(moved_nodes):
    faces_nodes = moved_nodes[setting.boundary_faces]
    internal_nodes = moved_nodes[setting.boundary_internal_indices]

    tail_nodes, head_nodes1, head_nodes2 = [
        faces_nodes[:, i, :] for i in range(3)
    ]

    unoriented_normals = nph.normalize_euclidean_numba(
        np.cross(head_nodes1 - tail_nodes, head_nodes2 - tail_nodes)
    )

    external_orientation = (-1) * np.sign(
        nph.elementwise_dot(internal_nodes - tail_nodes, unoriented_normals, keepdims=True)
    )

    return unoriented_normals * external_orientation




############################

setting = SettingMesh(mesh_type="meshzoo_cube_3d", mesh_density_x=2)

initial_base = get_base(setting.initial_nodes, setting.base_seed_indices, setting.closest_seed_index)


edges_features_matrix, element_initial_volume = get_edges_features_matrix_3d_numba(
    setting.cells, setting.initial_nodes
)
# TODO: To tests - sum of slice for area and u == 1
# edges_features_matrix[i].sum() == 0
# edges_features_matrix[0].sum() == 1
# TODO: switch to dictionary
# rollaxis -> moveaxis


mu = 0.01
la = 0.01
th = 0.01
ze = 0.01
density = 0.01
time_step = 0.01
nodes_count = len(setting.initial_nodes)
independent_nodes_count = nodes_count
slice_ind = slice(0, nodes_count)


C, B, AREA, A_plus_B_times_ts = get_matrices(
    edges_features_matrix, mu, la, th, ze, density, time_step, slice_ind
)


def get_E(forces, u_old, v_old):
    F_vector = nph.stack_column(AREA @ forces)
    u_old_vector = nph.stack_column(u_old)
    v_old_vector = nph.stack_column(v_old)

    E = F_vector - A_plus_B_times_ts @ v_old_vector - B @ u_old_vector
    return E


######


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

    scenario_length = 400
    moved_nodes = setting.initial_nodes

    for i in range(1, scenario_length + 1):
        current_time = i * time_step
        print(f"time: {current_time}")


        forces = get_forces_by_function(f_rotate, setting.initial_nodes, current_time)
        normalized_forces = setting.normalize_rotate(forces)

        normalized_E = get_E(normalized_forces, setting.normalized_u_old, setting.normalized_v_old)
        normalized_a = nph.unstack(np.linalg.solve(C, normalized_E), dim=DIM)
        a = setting.denormalize_rotate(normalized_a)

        if i % 10 == 0:
            plotter_3d.print_frame(
                moved_nodes=moved_nodes,
                normalized_nodes=setting.normalized_points,
                normalized_data=[
                    normalized_forces * 20,
                    setting.normalized_u_old,
                    setting.normalized_v_old,
                    normalized_a,
                ],
                elements=setting.cells,
                path=f"{catalog}/{int(thh.get_timestamp() * 100)}.{extension}",
                extension=extension,
                all_images_paths=all_images_paths,
                moved_base=setting.moved_base,
                boundary_nodes_indices=setting.boundary_nodes_indices,
                boundary_faces=setting.boundary_faces,
                boundary_normals=get_boundary_normals(moved_nodes),
                boundary_internal_indices=setting.boundary_internal_indices,
            )

        setting.set_v_old(setting.v_old + time_step * a)
        setting.set_u_old(setting.u_old + time_step * setting.v_old)

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
