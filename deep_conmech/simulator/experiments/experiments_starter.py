# %%
import os

import imageio
import matplotlib.pyplot as plt
import meshzoo
import numba
import numpy as np
from conmech.helpers import nph
from deep_conmech.common.plotter.plotter_basic import Plotter
from deep_conmech.graph.helpers import thh
from deep_conmech.simulator.experiments.matrices import *
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
from scipy.spatial import Delaunay

DIM = 3
EDIM = 4


def get_meshzoo_cube(mesh_size):
    initial_nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_size),
        np.linspace(0.0, 1.0, mesh_size),
        np.linspace(0.0, 1.0, mesh_size),
    )
    return initial_nodes, elements


def list_all_faces(elements):
    elements.sort(axis=1)
    elements_count, element_size = elements.shape
    dim = element_size - 1
    faces = np.zeros((element_size * elements_count, dim), dtype=int)
    i = 0
    for j in range(element_size):
        faces[i : i + elements_count, :j] = elements[:, :j]  # ignoring j-th column
        faces[i : i + elements_count, j:dim] = elements[:, j + 1 : element_size]
        i += elements_count
    return faces


def extract_unique_elements(elements):
    _, indxs, count = np.unique(elements, axis=0, return_index=True, return_counts=True)
    return elements[indxs[count == 1]]


def get_boundary_faces(elements):
    faces = list_all_faces(elements)
    boundary_faces = extract_unique_elements(faces)
    return boundary_faces


######################################################


def plot_mesh(ax, nodes, nodes_indices, elements):
    boundary_nodes = nodes[nodes_indices]
    
    for e in elements:
        pts = nodes[e, :]
        lw = 0.4
        for i in range(EDIM):
            for j in range(EDIM):
                if i < j:
                    ax.plot3D(
                        pts[[i, j], 0], pts[[i, j], 1], pts[[i, j], 2], color="b", lw=lw
                    )
    
    # ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="b")
    ax.scatter(
        boundary_nodes[:, 0], boundary_nodes[:, 1], boundary_nodes[:, 2], color="y"
    )


######################################################


@njit
def get_furthest_apart_numba(nodes, variable):
    max_dist = 0.0
    max_i, max_j = 0, 0
    nodes_count = len(nodes)
    for i in range(nodes_count):
        for j in range(i, nodes_count):
            dist = np.abs(nodes[i, variable] - nodes[j, variable])
            if dist > max_dist:
                max_dist = dist
                max_i, max_j = i, j

    if nodes[max_i, variable] < nodes[max_j, variable]:
        return [max_i, max_j]
    else:
        return [max_j, max_i]

def get_base_seed_indices(nodes):
    dim = nodes.shape[1]
    base_seed_indices = np.array([ get_furthest_apart_numba(nodes, i)  for i in range(1, dim)])
    return base_seed_indices


def get_base_seed(nodes, base_seed_indices):
    base_seed_initial_nodes = nodes[base_seed_indices]
    return base_seed_initial_nodes[...,1,:] - base_seed_initial_nodes[...,0,:]



######################################################

mesh_size = 5
initial_nodes, elements = get_meshzoo_cube(mesh_size)
boundary_faces = get_boundary_faces(elements)
boundary_nodes_indices = np.unique(boundary_faces.flatten(), axis=0)


base_seed_indices = get_base_seed_indices(initial_nodes)

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


def print_frame(moved_nodes, normalized_nodes, forces, elements, path, extension, all_images_paths):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((10,4,3))
    #ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 3)
    ax.set_zlim(-1, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    initial_base_seed = get_base_seed(initial_nodes, base_seed_indices)
    moved_base_seed = get_base_seed(moved_nodes, base_seed_indices)

    initial_base = nph.complete_base(initial_base_seed)
    moved_base = nph.complete_base(moved_base_seed)

    for v in initial_base:
        z = np.zeros_like(v)
        nv = nph.normalize_euclidean_numba(v)
        ax.quiver(*z,*nv)
    for v in moved_base:
        z = np.zeros_like(v)
        nv = nph.normalize_euclidean_numba(v)
        ax.quiver(*z,*nv,color='g')
    for i in range(len(forces)):
        m = moved_nodes[i]
        f = forces[i] * 50
        ax.quiver(*m,*f,color='g')

    plot_mesh(ax, moved_nodes, boundary_nodes_indices, elements) # boundary_nodes_indices
    plot_mesh(ax, normalized_nodes, boundary_nodes_indices, elements) # boundary_nodes_indices

    plt.show()
    plt_save(path, extension)
    all_images_paths.append(path)


catalog = f"output/3D {thh.get_timestamp()}"



def f_push(ip, t):
    return np.array([0.05, -0.05, 0.05])
    #return np.repeat(np.array([f0]), nodes_count, axis=0)

def f_rotate(ip, t):
    if t <= 0.5:
        scale = ip[1] * ip[2]
        return scale * np.array([0.05, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])
        

def get_forces_by_function(
    forces_function, initial_nodes, current_time
):
    nodes_count = len(initial_nodes)
    forces = np.zeros((nodes_count, 3), dtype=np.double)
    for i in range(nodes_count):
        forces[i] = forces_function(initial_nodes[i], current_time)
    return forces



def to_rotated_base_seed(moved_base_seed, initial_base_seed):
    return nph.get_in_base(moved_base_seed, initial_base_seed)
    
def from_rotated_base_seed(moved_base_seed, initial_base_seed):
    return nph.get_in_base(initial_base_seed, moved_base_seed)

def rotate_to_upward(vectors, moved_base_seed, initial_base_seed):
    return nph.get_in_base(vectors, to_rotated_base_seed(moved_base_seed, initial_base_seed))

def rotate_from_upward(vectors, moved_base_seed, initial_base_seed):
    return nph.get_in_base(vectors, from_rotated_base_seed(moved_base_seed, initial_base_seed))


def print_one_dynamic():
    all_images_paths = []
    extension = "png"  # pdf
    thh.create_folders(catalog)

    u_old = np.zeros((nodes_count, DIM), dtype=np.double)
    v_old = np.zeros((nodes_count, DIM), dtype=np.double)

    scenario_length = 400
    moved_nodes = initial_nodes
    
    mean_initial_nodes = np.mean(initial_nodes, axis=0)
    normalized_initial_nodes = initial_nodes - mean_initial_nodes
    initial_base_seed = get_base_seed(initial_nodes, base_seed_indices)
    for i in range(1, scenario_length + 1):
        current_time = (i + 1) * time_step
        print(f"time: {current_time}")
        
        moved_base_seed = get_base_seed(moved_nodes, base_seed_indices)

        mean_moved_nodes =  np.mean(moved_nodes, axis=0)
        normalized_nodes = rotate_to_upward(moved_nodes - mean_moved_nodes, moved_base_seed, initial_base_seed)

        forces = get_forces_by_function(f_rotate, initial_nodes, current_time)
        normalized_forces = rotate_to_upward(forces, moved_base_seed, initial_base_seed)
        normalized_u_old = normalized_nodes - normalized_initial_nodes
        normalized_v_old = rotate_to_upward(v_old-np.mean(v_old, axis=0), moved_base_seed, initial_base_seed)

        normalized_E = get_E(normalized_forces, normalized_u_old, normalized_v_old)
        normalized_a = unstack(np.linalg.solve(C, normalized_E))
        a = rotate_from_upward(normalized_a, moved_base_seed, initial_base_seed)

        v_old = v_old + time_step * a
        u_old = u_old + time_step * v_old

        moved_nodes = initial_nodes + u_old

        if i % 10 == 0:
            print_frame(
                moved_nodes,
                normalized_nodes,
                forces,
                elements,
                f"{catalog}/{int(thh.get_timestamp() * 100)}.{extension}",
                extension,
                all_images_paths,
            )

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
