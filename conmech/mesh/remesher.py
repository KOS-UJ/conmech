import numpy as np
from numba import njit


@njit
def get_alpha(x, p1, p2, p3):
    return ((p2[1] - p3[1]) * (x[0] - p3[0]) + (p3[0] - p2[0]) * (x[1] - p3[1])) / (
            (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    )


@njit
def get_beta(x, p1, p2, p3):
    return ((p3[1] - p1[1]) * (x[0] - p3[0]) + (p1[0] - p3[0]) * (x[1] - p3[1])) / (
            (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    )


@njit
def bigger_or_zero(data):
    return data > -1e-05


# @njit
def approximate_one(new_point, old_points, old_values, old_elements):
    closest_element = 0
    min_penality = None

    for element in old_elements:
        p1, p2, p3 = old_points[element]

        alpha = get_alpha(new_point, p1, p2, p3)
        beta = get_beta(new_point, p1, p2, p3)
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


@njit
def approximate_all_numba(new_points, old_points, old_values, old_elements):
    new_values = np.zeros_like(new_points)

    for i in range(len(new_points)):
        new_point = new_points[i]
        new_value = approximate_one(new_point, old_points, old_values, old_elements)
        new_values[i] = new_value

    return new_values
