# NumPy Helpers
import math
from ctypes import ArgumentError

import numba
import numpy as np
from numba import njit

DIM = 2



def stack(data):
    return data.T.flatten()

def stack_column(data):
    return data.T.flatten().reshape(-1, 1)

def unstack(data):
    return data.reshape(-1, DIM, order="F")


def get_occurances(data):
    return np.array(list(set(data.flatten())))


def norm(data):
    return np.sqrt((data ** 2).sum(-1))[..., np.newaxis] #.reshape(-1,1)

def normalize(data):
    #return np.divide(data, np.linalg.norm(data, axis=-1))
    return data / norm(data)

def elementwise_dot(x, y):
    return (x * y).sum(axis=1)













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
def stack_column_numba(data):
    return data.T.flatten().reshape(-1, 1)


@njit
def euclidean_norm_numba(vector):
    return np.sqrt(np.sum(vector ** 2, axis=-1))

@njit
def get_point_index_numba(point, points):
    for i in range(len(points)):
        if np.sum(np.abs(point - points[i])) < 0.0001:
            return i
    raise ArgumentError


def get_random_normal(count, scale):
    # noise = np.random.uniform(low=-scale, high=scale, size=shape)
    noise = np.random.normal(loc=0.0, scale=scale * 0.5, size=[count, DIM])
    return noise


@njit
def get_random_normal_circle_numba(count, scale):
    result = np.zeros((count, DIM))
    for i in range(count):
        alpha = 2 * math.pi * np.random.uniform(low=0, high=1)
        r = np.abs(np.random.normal(loc=0.0, scale=scale * 0.5))
        result[i] = [r * math.cos(alpha), r * math.sin(alpha)]
    return result


@njit
def internal_tuple_to_array_numba(tuple, argument):
    return np.array(tuple) if argument.ndim == 1 else np.vstack(tuple).T

    
@njit
def get_oriented_tangential_numba(normal):
    tuple = (normal[...,1], -normal[...,0])
    result = internal_tuple_to_array_numba(tuple, normal)
    return result

@njit
def rotate_up_numba(old_vectors, up_vector):
    tangential = get_oriented_tangential_numba(up_vector)
    tuple = (old_vectors @ tangential, old_vectors @ up_vector)
    result = internal_tuple_to_array_numba(tuple, old_vectors)
    return result


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
