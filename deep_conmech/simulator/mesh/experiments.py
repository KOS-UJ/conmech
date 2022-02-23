#%%
import matplotlib.pyplot as plt
import meshzoo
import numpy as np
from deep_conmech.graph.data.data_interpolation import interpolate_point
from mpl_toolkits import mplot3d
import numba
from numba import njit
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deep_conmech.graph.helpers import thh
from conmech.helpers import nph


def get_meshzoo_cube(mesh_size):
    nodes, elements = meshzoo.cube_tetra(
        np.linspace(0.0, 1.0, mesh_size),
        np.linspace(0.0, 1.0, mesh_size),
        np.linspace(0.0, 1.0, mesh_size),
    )
    return nodes, elements


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


######################################


@njit  # (parallel=True)
def get_edges_features_matrix(elements, nodes):
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape
    dim=element_size-1

    edges_features_matrix = np.zeros((nodes_count, nodes_count, 8), dtype=np.double)
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: prange?
        element = elements[element_index]
        element_points = nodes[element]

        # TODO: Get rid of repetition (?)
        for i in range(element_size):
            i_dPhX, i_dPhY, element_volume = get_integral_parts(element_points, i)
            # TODO: Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_dPhX, j_dPhY, _ = get_integral_parts(element_points, j)

                area = (i != j) / 6.0
                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                u = (1 + (i == j)) / 12.0

                u1 = i_dPhX / 3.0
                u2 = i_dPhY / 3.0

                edges_features_matrix[element[i], element[j]] += element_volume * np.array(
                    [area, w11, w12, w21, w22, u1, u2, u]
                )

    return edges_features_matrix, element_initial_volume


@njit
def get_integral_parts(element_nodes, element_index):
    x_i = element_nodes[element_index % 3]
    x_j1 = element_nodes[(element_index + 1) % 3]
    x_j2 = element_nodes[(element_index + 2) % 3]

    dm = denominator(x_i, x_j1, x_j2)
    triangle_area = np.abs(dm) / 2.0 # = np.abs(dm) / 2.0 = shoelace_area

    y_sub = x_j2[1] - x_j1[1]
    x_sub = x_j1[0] - x_j2[0]

    dPhX = y_sub / dm
    dPhY = x_sub / dm

    return dPhX, dPhY, triangle_area


@njit
def shoelace_area(points):
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


@njit
def denominator(x_i, x_j1, x_j2):
    return (
        x_i[1] * x_j1[0]
        + x_j1[1] * x_j2[0]
        + x_i[0] * x_j2[1]
        - x_i[1] * x_j2[0]
        - x_j2[1] * x_j1[0]
        - x_i[0] * x_j1[1]
    )


#######################################


def plot_mesh(ax, nodes, elements):
    for e in elements:
        pts = nodes[e, :]
        lw = 0.4
        for i in range(4):
            for j in range(4):
                if i < j:
                    ax.plot3D(
                        pts[[i, j], 0], pts[[i, j], 1], pts[[i, j], 2], color="b", lw=lw
                    )

    # ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="b")


mesh_size = 3
nodes, elements = get_meshzoo_cube(mesh_size)
boundary_faces = get_boundary_faces(elements)
boundary_nodes = np.unique(nodes[boundary_faces].reshape(-1, 3), axis=0)


edges_features = get_edges_features_matrix(elements, nodes)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_mesh(ax, nodes, elements)
# ax.scatter(boundary_nodes[:, 0], boundary_nodes[:, 1], boundary_nodes[:, 2], color="r")


ax.set_xlim(-1.1, 2.1)
ax.set_ylim(-1.1, 2.1)
ax.set_zlim(-1.1, 2.1)

plt.show()


# %%
