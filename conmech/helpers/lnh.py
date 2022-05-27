"""
linear algebra helpers
"""

import numpy as np

from conmech.helpers import nph


def move_vector(vectors, index):
    return np.roll(vectors, -index, axis=0)


def complete_base(base_seed, best_seed_index):
    prioritized_base_seed = move_vector(vectors=base_seed, index=best_seed_index)
    prioritized_base = __orthonormalize_gram_schmidt(base_seed=prioritized_base_seed)
    base = move_vector(vectors=prioritized_base, index=-best_seed_index)

    assert correct_base(base)
    assert np.allclose(
        nph.normalize_euclidean_numba(base_seed[best_seed_index]), base[best_seed_index]
    )
    return base


def __orthonormalize_gram_schmidt(base_seed):
    normalized_base_seed = nph.normalize_euclidean_numba(base_seed)
    unnormalized_base = __orthogonalize_gram_schmidt(normalized_base_seed)
    base = nph.normalize_euclidean_numba(unnormalized_base)
    return base


def __orthogonalize_gram_schmidt(vectors):
    # Gramm-schmidt orthog.
    b0 = vectors[0]
    if len(vectors) == 1:
        return np.array((b0))

    b1 = vectors[1] - (vectors[1] @ b0) * b0
    if len(vectors) == 2:
        return np.array((b0, b1))

    b2 = np.cross(b0, b1)
    return np.array((b0, b1, b2))


def generate_base(dimension):
    while True:
        vectors = nph.generate_normal(rows=dimension, columns=dimension, sigma=1)
        base = __get_orthonormalized(vectors)
        if correct_base(base):
            return base
        print("Base generation error")


def __get_orthonormalized(vectors):
    vectors = nph.normalize_euclidean_numba(vectors)
    base = np.linalg.qr(vectors)[0]
    if len(base) == 2:
        base[0] *= np.cross(*base)  # keep right orientetion
    return base


def correct_base(base):
    dim = len(base)
    for i in range(dim):
        for j in range(i + 1, dim):
            if not np.allclose(base[i] @ base[j], 0):
                return False

    if not np.allclose(nph.euclidean_norm(base), np.ones(dim)):
        return False

    if len(base) == 2 and not np.allclose(np.cross(*base), 1):
        return False
    if len(base) == 3 and not np.allclose(np.cross(*base[:2]), base[2]):
        return False
    return True


def get_in_base(vectors, base):
    return vectors @ base.T
