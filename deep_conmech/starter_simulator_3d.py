# %%
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import nph
from deep_conmech.common.plotter import plotter_3d
from deep_conmech.graph.helpers import thh
from deep_conmech.simulator.setting.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.setting_mesh import *
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

DIM = 3
EDIM = 4

catalog = f"output/SIMULATOR 3D - {thh.CURRENT_TIME}"


def list_all_faces(elements):
    elements.sort(axis=1)
    elements_count, element_size = elements.shape
    dim = element_size - 1
    faces = np.zeros((element_size * elements_count, dim), dtype=np.int64)
    opposing_indices = np.zeros((element_size * elements_count), dtype=np.int64)
    i = 0
    for j in range(element_size):
        faces[i : i + elements_count, :j] = elements[:, :j]  # ignoring j-th column
        faces[i : i + elements_count, j:dim] = elements[:, j + 1 : element_size]
        opposing_indices[i : i + elements_count] = elements[:, j]
        i += elements_count
    return faces, opposing_indices


def extract_unique_elements(elements, opposing_indices):
    _, indices, count = np.unique(
        elements, axis=0, return_index=True, return_counts=True
    )
    unique_indices = indices[count == 1]
    return elements[unique_indices], opposing_indices[unique_indices]


def get_boundary_faces(elements):
    faces, opposing_indices = list_all_faces(elements)
    boundary_faces, boundary_internal_indices = extract_unique_elements(
        faces, opposing_indices
    )
    return boundary_faces, boundary_internal_indices


######################################################


def normalize_rotate(vectors, moved_base):
    return nph.get_in_base(vectors, moved_base)


def denormalize_rotate(vectors, moved_base):
    return nph.get_in_base(vectors, np.linalg.inv(moved_base))


######################################################

initial_nodes, elements = mesh_builders.build_mesh(
    mesh_type="pygmsh_3d", mesh_density_x=3
)

boundary_faces, boundary_internal_indices = get_boundary_faces(elements)


def get_boundary_normals(moved_nodes):
    boundary_faces_nodes = moved_nodes[boundary_faces]
    boundary_internal_nodes = moved_nodes[boundary_internal_indices]

    va = boundary_faces_nodes[...,1]-boundary_faces_nodes[...,0]
    vb = boundary_faces_nodes[...,2]-boundary_faces_nodes[...,0]
    vc = np.cross(va,vb)

    boundary_normals = nph.normalize_euclidean_numba(va)
    return boundary_normals

#nph.elementwise_dot(vc, boundary_internal_nodes) > 0



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
        normalized_a = nph.unstack(np.linalg.solve(C, normalized_E), dim=DIM)
        a = denormalize_rotate(normalized_a, moved_base)

        if i % 10 == 0:
            plotter_3d.print_frame(
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
                boundary_nodes_indices=boundary_nodes_indices,
                boundary_faces=boundary_faces,
                boundary_normals=get_boundary_normals(moved_nodes)
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
