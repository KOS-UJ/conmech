"""
Created at 16.02.2022
"""
import numba
import numpy as np


@numba.njit(inline="always")
def length(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
