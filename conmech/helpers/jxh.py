"""
jax helpers
"""
import jax.experimental.sparse
import jax.numpy as jnp
import jax.scipy
import numpy as np
import scipy.sparse


def euclidean_norm(vector, keepdims=False):
    data = (vector**2).sum(axis=-1, keepdims=keepdims)
    return jnp.sqrt(data)


def normalize_euclidean(data):
    norm = euclidean_norm(data)
    reshaped_norm = norm if data.ndim == 1 else norm.reshape(-1, 1)
    return data / reshaped_norm


def get_tangential_2d(normal):
    return jnp.array((normal[..., 1], -normal[..., 0])).T


def append_euclidean_norm(data):
    return jnp.hstack((data, euclidean_norm(data, keepdims=True)))


def to_inverse_diagonal(martix):
    return scipy.sparse.diags(1.0 / martix.diagonal(), shape=martix.shape)


def solve_linear_jax(matrix, vector):
    # M = to_diagonal(martix)
    result, _ = jax.scipy.sparse.linalg.cg(A=matrix, b=vector)  # , M=M)
    return result


def to_jax_sparse(matrix):
    coo_matrix = matrix.tocoo()
    if coo_matrix is None:
        return None
    indices = np.block([[coo_matrix.row], [coo_matrix.col]]).T
    result = jax.experimental.sparse.BCOO((coo_matrix.data, indices), shape=coo_matrix.shape)
    return result.sort_indices()


def to_dense_np(array):
    return np.array(array.todense(), dtype=np.float64)


def complete_data_with_zeros(data: np.ndarray, nodes_count):
    return jnp.pad(data, ((0, nodes_count - len(data)), (0, 0)), "constant")
