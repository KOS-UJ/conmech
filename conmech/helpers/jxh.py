"""
jax helpers
"""
import jax.experimental.sparse
import jax.scipy
import numpy as np
import scipy.sparse


def to_diagonal(A):
    return to_jax_sparse(scipy.sparse.diags(to_scipy_sparse(A).diagonal(), shape=A.shape).tocoo())


def solve_linear(A, b):
    # M = to_diagonal(A)
    result, _ = jax.scipy.sparse.linalg.cg(A=A, b=b)  # , M=M)
    return result


def to_jax_sparse(coo_matrix):
    if coo_matrix is None:
        return None
    indices = np.block([[coo_matrix.row], [coo_matrix.col]]).T
    result = jax.experimental.sparse.BCOO((coo_matrix.data, indices), shape=coo_matrix.shape)
    return result


def to_scipy_sparse(bcoo_matrix):
    if bcoo_matrix is None:
        return None
    data = np.array(bcoo_matrix.data)
    indices = np.array(bcoo_matrix.indices).T
    row = indices[0]
    col = indices[1]
    result = scipy.sparse.coo_matrix((data, (row, col)), shape=bcoo_matrix.shape, dtype=np.float64)
    return result


def to_dense_np(array):
    return np.array(array.todense(), dtype=np.float64)


def slice(matrix, indices):
    return matrix.tocsr()[indices, indices].tocoo()
