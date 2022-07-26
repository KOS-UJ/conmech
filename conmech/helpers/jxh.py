"""
jax helpers
"""
import cupyx.scipy.sparse.linalg
import jax.experimental.sparse
import jax.scipy
import numpy as np
import scipy.sparse


def to_diagonal(martix):
    return to_cupy_csr(scipy.sparse.diags(to_scipy_sparse(martix).diagonal(), shape=martix.shape))


def to_inverse_diagonal(martix):
    return scipy.sparse.diags(1.0 / martix.diagonal(), shape=martix.shape)



def solve_linear_jax(matrix, vector):
    # M = to_diagonal(martix)
    result, _ = jax.scipy.sparse.linalg.cg(A=matrix, b=vector)  # , M=M)
    return result


def to_cupy_csr(matrix):
    if matrix is None:
        return None
    result = cupyx.scipy.sparse.csr_matrix(matrix)
    return result


def to_jax_sparse(matrix):
    coo_matrix = matrix.tocoo()
    if coo_matrix is None:
        return None
    indices = np.block([[coo_matrix.row], [coo_matrix.col]]).T
    result = jax.experimental.sparse.BCOO((coo_matrix.data, indices), shape=coo_matrix.shape)
    return result.sort_indices()


def to_scipy_sparse(matrix):
    if matrix is None:
        return None
    return matrix.get()
    data = np.array(coo_matrix.data)
    indices = np.array(coo_matrix.indices).T
    row = indices[0]
    col = indices[1]
    result = scipy.sparse.coo_matrix((data, (row, col)), shape=coo_matrix.shape, dtype=np.float64)
    return result


def to_dense_np(array):
    return np.array(array.todense(), dtype=np.float64)
