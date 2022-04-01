"""
numpy helpers
"""
from ctypes import ArgumentError

import numba
import numpy as np
from numba import njit


def stack(data):
    return data.T.flatten()


def stack_column(data):
    return data.T.flatten().reshape(-1, 1)


stack_column_numba = numba.njit(stack_column)


def unstack(vector, dim):
    return vector.reshape(-1, dim, order="F")


def unstack_and_sum_columns(data, dim, keepdims=False):
    return np.sum(unstack(data, dim), axis=1, keepdims=keepdims)


def elementwise_dot(x, y, keepdims=False):
    return (x * y).sum(axis=1, keepdims=keepdims)


def get_occurances(data):
    return np.array(list(set(data.flatten())))


def close_modulo(value, divider):
    if divider is None:
        return True
    return np.allclose(value % divider, 0.0) or np.allclose(value % divider, divider)


###################


def euclidean_norm(vector, keepdims=False):
    data = (vector ** 2).sum(axis=-1, keepdims=keepdims)
    if isinstance(data, np.ndarray):
        return np.sqrt(data)
    return data.sqrt()
    # return np.linalg.norm(vector, axis=-1)
    # return np.sqrt(np.sum(vector ** 2, axis=-1))[..., np.newaxis]


@njit
def euclidean_norm_numba(vector):
    data = (vector ** 2).sum(axis=-1)
    return np.sqrt(data)


@njit
def normalize_euclidean_numba(data):
    norm = euclidean_norm_numba(data)
    reshaped_norm = norm if data.ndim == 1 else norm.reshape(-1, 1)
    return data / reshaped_norm


def get_normal(vector, normal):
    return elementwise_dot(vector, normal, keepdims=True)


def get_normal_tangential(vector, normal):
    normal_vector = get_normal(vector, normal)
    tangential_vector = vector - (normal_vector * normal)
    return normal_vector, tangential_vector


def get_tangential(vector, normal):
    _, tangential_vector = get_normal_tangential(vector, normal)
    return tangential_vector


@njit
def get_tangential_numba(vector, normal):
    normal_vector = vector @ normal
    tangential_vector = vector - (normal_vector * normal)
    return tangential_vector


###################


def get_tangential_2d(normal):
    return np.array((normal[..., 1], -normal[..., 0])).T


def complete_base(base_seed, closest_seed_index=0):
    dim = base_seed.shape[-1]
    normalized_base_seed = normalize_euclidean_numba(base_seed)
    if dim == 2:
        unnormalized_base = orthogonalize_gram_schmidt(normalized_base_seed)
    elif dim == 3:
        rolled_base_seed = np.roll(normalized_base_seed, -closest_seed_index, axis=0)
        unnormalized_rolled_base = orthogonalize_gram_schmidt(rolled_base_seed)
        unnormalized_base = np.roll(
            unnormalized_rolled_base, closest_seed_index, axis=0
        )
    else:
        raise ArgumentError()
    base = normalize_euclidean_numba(unnormalized_base)
    return base


def orthogonalize_gram_schmidt(vectors):
    # Gramm-schmidt orthog.
    b0 = vectors[0]
    if len(vectors) == 1:
        return np.array((b0))

    b1 = vectors[1] - (vectors[1] @ b0) * b0
    if len(vectors) == 2:
        return np.array((b0, b1))

    # MGS for stability
    w2 = vectors[2] - (vectors[2] @ b0) * b0
    b2 = w2 - (w2 @ b1) * b1
    # nx = np.cross(ny,nz)
    return np.array((b0, b1, b2))


def get_in_base(vectors, base):
    return vectors @ base.T


###################


@njit
def len_x_numba(corners):
    return corners[2] - corners[0]


@njit
def len_y_numba(corners):
    return corners[3] - corners[1]


@njit
def min_numba(corners):
    return [corners[0], corners[1]]


@njit
def max_numba(corners):
    return [corners[2], corners[3]]


# TODO: @numba.njit(inline='always') - when using small function inside other numba
# TODO: Use numba.njit(...)
# TODO : use slice instead of int


@njit
def get_point_index_numba(point, points):
    for i in range(len(points)):
        if np.sum(np.abs(point - points[i])) < 0.0001:
            return i
    raise ArgumentError


def get_random_normal(dim, nodes_count, scale):
    # noise = np.random.uniform(low=-scale, high=scale, size=shape)
    noise = np.random.normal(loc=0.0, scale=scale * 0.5, size=[nodes_count, dim])
    return noise


@njit
def get_random_normal_circle_numba(dim, nodes_count, scale):
    result = np.zeros((nodes_count, dim))
    for i in range(nodes_count):
        alpha = 2 * np.pi * np.random.uniform(0, 1)  # low=0, high=1)
        r = np.abs(np.random.normal(loc=0.0, scale=scale * 0.5))
        result[i] = [r * np.cos(alpha), r * np.sin(alpha)]
    return result


"""
@njit
def calculate_angle_numba(new_up_vector):
    old_up_vector = np.array([0., 1.])
    angle = (2 * (new_up_vector[0] >= 0) - 1) * np.arccos(np.dot(new_up_vector, old_up_vector))
    return angle

@njit
def rotate_numba(vectors, angle):
    s = np.sin(angle)
    c = np.cos(angle)

    rotated_vectors = np.zeros_like(vectors)
    rotated_vectors[:, 0] = vectors[:, 0] * c - vectors[:, 1] * s
    rotated_vectors[:, 1] = vectors[:, 0] * s + vectors[:, 1] * c
    
    return rotated_vectors
"""
