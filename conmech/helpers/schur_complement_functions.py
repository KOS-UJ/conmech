from typing import Tuple

import numpy as np

from conmech.helpers import nph


def calculate_schur_complement_vector(
    vector: np.ndarray,
    dimension: int,
    contact_indices: slice,
    free_indices: slice,
    free_x_free_inverted: np.ndarray,
    contact_x_free: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vector_split = nph.unstack(vector, dimension)
    vector_contact = nph.stack_column(vector_split[contact_indices, :])
    vector_free = nph.stack_column(vector_split[free_indices, :])
    vector_boundary = vector_contact - (contact_x_free @ (free_x_free_inverted @ vector_free))
    return vector_boundary, vector_free


def calculate_schur_complement_matrices(
    matrix: np.ndarray, dimension: int, contact_indices: slice, free_indices: slice
):
    def get_sliced(matrix_split, indices_height, indices_width):
        matrix = np.moveaxis(matrix_split[..., indices_height, indices_width], 1, 2)
        dim, height, _, width = matrix.shape
        return matrix.reshape(dim * height, dim * width)

    matrix_split = np.array(
        np.split(np.array(np.split(matrix, dimension, axis=-1)), dimension, axis=1)
    )
    free_x_free = get_sliced(matrix_split, free_indices, free_indices)
    free_x_contact = get_sliced(matrix_split, free_indices, contact_indices)
    contact_x_free = get_sliced(matrix_split, contact_indices, free_indices)
    contact_x_contact = get_sliced(matrix_split, contact_indices, contact_indices)

    free_x_free_inverted = np.linalg.inv(free_x_free)
    matrix_boundary = contact_x_contact - contact_x_free @ (free_x_free_inverted @ free_x_contact)

    return matrix_boundary, free_x_contact, contact_x_free, free_x_free_inverted
