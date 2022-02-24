import matplotlib.pyplot as plt
import meshzoo
import numpy as np
from mpl_toolkits import mplot3d
import numba
from numba import njit
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deep_conmech.graph.helpers import thh
from conmech.helpers import nph

from deep_conmech.simulator.experiments.matrices import *


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


mesh_size = 8
nodes, elements = get_meshzoo_cube(mesh_size)
boundary_faces = get_boundary_faces(elements)
boundary_nodes = np.unique(nodes[boundary_faces].reshape(-1, 3), axis=0)

edges_features_matrix, element_initial_volume = get_edges_features_matrix_numba(elements, nodes)
# TODO: To tests - sum off slice for area and u == 1
# np.moveaxis(edges_features_matrix, -1,0)[i].sum() == 0
# np.moveaxis(edges_features_matrix, -1,0)[0].sum() == 1
# TODO: switch to dictionary
# TODO: simplify get_integral_parts_numba
# TODO: reshape so that AREA = edges_features_matrix[..., 0] is   AREA = edges_features_matrix[0]
# np.moveaxis(edges_features_matrix, -1,0)[0].sum()
#rollaxis -> moveaxis



fig = plt.figure()
ax = fig.add_subplot(projection="3d")
plot_mesh(ax, nodes, elements)
# ax.scatter(boundary_nodes[:, 0], boundary_nodes[:, 1], boundary_nodes[:, 2], color="r")


ax.set_xlim(-1.1, 2.1)
ax.set_ylim(-1.1, 2.1)
ax.set_zlim(-1.1, 2.1)

plt.show()


# %%
