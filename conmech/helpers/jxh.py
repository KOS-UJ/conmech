"""
jax helpers
"""
import cupyx.scipy.sparse.linalg
import jax.experimental.sparse
import jax.scipy
import numpy as np
import scipy.sparse


def slice_matrix(matrix, indices):
    return matrix.tocsr()[indices, indices].tocoo()


def to_diagonal(martix):
    return to_cupy_sparse(
        scipy.sparse.diags(to_scipy_sparse(martix).diagonal(), shape=martix.shape).tocoo()
    )


def solve_linear_jax(matrix, vector):
    # M = to_diagonal(martix)
    result, _ = jax.scipy.sparse.linalg.cg(A=matrix, b=vector)  # , M=M)
    return result


def to_cupy_sparse(coo_matrix):
    if coo_matrix is None:
        return None
    result = cupyx.scipy.sparse.coo_matrix(coo_matrix)
    return result


def to_jax_sparse(coo_matrix):
    if coo_matrix is None:
        return None
    indices = np.block([[coo_matrix.row], [coo_matrix.col]]).T
    result = jax.experimental.sparse.BCOO((coo_matrix.data, indices), shape=coo_matrix.shape)
    return result


def to_scipy_sparse(coo_matrix):
    if coo_matrix is None:
        return None
    return coo_matrix.get()
    data = np.array(coo_matrix.data)
    indices = np.array(coo_matrix.indices).T
    row = indices[0]
    col = indices[1]
    result = scipy.sparse.coo_matrix((data, (row, col)), shape=coo_matrix.shape, dtype=np.float64)
    return result


def to_dense_np(array):
    return np.array(array.todense(), dtype=np.float64)
