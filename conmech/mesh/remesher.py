import numba
import numpy as np

from conmech.helpers import nph


@numba.njit
def get_alpha_numba(x, p_1, p_2, p_3):
    return ((p_2[1] - p_3[1]) * (x[0] - p_3[0]) + (p_3[0] - p_2[0]) * (x[1] - p_3[1])) / (
        (p_2[1] - p_3[1]) * (p_1[0] - p_3[0]) + (p_3[0] - p_2[0]) * (p_1[1] - p_3[1])
    )


@numba.njit
def get_beta_numba(x, p_1, p_2, p_3):
    return ((p_3[1] - p_1[1]) * (x[0] - p_3[0]) + (p_1[0] - p_3[0]) * (x[1] - p_3[1])) / (
        (p_2[1] - p_3[1]) * (p_1[0] - p_3[0]) + (p_3[0] - p_2[0]) * (p_1[1] - p_3[1])
    )


@numba.njit
def bigger_or_zero(data):
    return data > -1e-05


# @numba.njit
def approximate_one_numba(new_node, old_nodes, old_values, old_elements):
    closest_element = 0
    min_penality = None

    for element in old_elements:
        p_1, p_2, p_3 = old_nodes[element]

        alpha = get_alpha_numba(new_node, p_1, p_2, p_3)
        beta = get_alpha_numba(new_node, p_1, p_2, p_3)
        gamma = 1.0 - alpha - beta

        if alpha > 0 and beta > 0 and gamma > 0:
            closest_element = element
            break

        penality = -(alpha * (alpha < 0) + beta * (beta < 0) + gamma * (gamma < 0))
        if min_penality is None or penality < min_penality:
            min_penality = penality
            closest_element = element

    v1, v2, v3 = old_values[closest_element]
    return alpha * v1 + beta * v2 + gamma * v3


# @numba.njit
def approximate_all_numba_old(old_values, old_elements, old_nodes, new_nodes):
    new_values = np.zeros_like(new_nodes)

    for i, new_node in enumerate(new_nodes):
        new_value = approximate_one_numba(new_node, old_nodes, old_values, old_elements)
        new_values[i] = new_value

    return new_values


@numba.njit
def get_edges_numba(nodes, node_degree):
    dimenasion = nodes.shape[1]
    edges = np.zeros((len(nodes) * node_degree, dimenasion), dtype=numba.int64)
    i = 0
    for index, node in enumerate(nodes):
        distances = nph.euclidean_norm_numba(node - nodes)
        closest_nodes = distances.argsort()[1 : node_degree + 1]
        for j in closest_nodes:
            edges[i, :] = [index, j]
            i += 1
    assert i == len(edges)
    return edges
