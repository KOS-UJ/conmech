# %%
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import nph
from matplotlib.gridspec import GridSpec

from deep_conmech.common.plotter import plotter_3d
from deep_conmech.graph.helpers import thh
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.setting_mesh import *



catalog = f"output/3D - {thh.CURRENT_TIME}"

######################################################
mesh_density_x = 5
setting = SettingMesh(mesh_type="meshzoo_cube_3d", mesh_density_x=mesh_density_x)

initial_base = get_base(setting.initial_nodes, setting.base_seed_indices, setting.closest_seed_index)


edges_features_matrix, element_initial_volume = get_edges_features_matrix_3d_numba(
    setting.cells, setting.initial_nodes
)
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

    for i in range(1, scenario_length + 1):
        current_time = i * time_step
        print(f"time: {current_time}")


        forces = get_forces_by_function(f_rotate, setting.initial_nodes, current_time)
        normalized_forces = setting.normalize_rotate(forces)
        setting.prepare() #normalized_forces

        normalized_E = get_E(normalized_forces, setting.normalized_u_old, setting.normalized_v_old)
        normalized_a = nph.unstack(np.linalg.solve(C, normalized_E), dim=3)
        a = setting.denormalize_rotate(normalized_a)

        if i % 10 == 0:
            plotter_3d.print_frame(
                moved_nodes=setting.moved_points,
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
                boundary_faces_normals=setting.boundary_faces_normals,
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
