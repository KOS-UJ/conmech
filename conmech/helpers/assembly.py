# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
"""Helpers bridging numba COO output and ``scipy.sparse`` operators.

numba cannot build ``scipy.sparse`` matrices, so the assembly kernels emit COO
triplets and the actual sparse matrices are constructed here, outside the
``numba``.  The block helpers let the operator factories compose block operators
without caring whether the blocks are dense ``ndarray``s or sparse matrices.
"""

import scipy.sparse

from conmech.struct.types import FeatureVector, FeatureMatrix


def coo_features_to_csr(rows, cols, data, nodes_count):
    """Turn per-feature COO ``data`` (shape ``(F, nnz)``) into ``F`` CSR blocks.

    ``scipy.sparse.csr_matrix`` sums duplicate ``(row, col)`` entries, which is
    exactly the accumulation the (former) dense kernel performed explicitly.
    """
    feature_count = data.shape[0]
    return [
        scipy.sparse.csr_matrix((data[f], (rows, cols)), shape=(nodes_count, nodes_count))
        for f in range(feature_count)
    ]


def split_features(features, dimension):
    """Split assembled sparse features into ``(volume_at_nodes, U, V, W)``.

    ``V`` / ``W`` become :class:`FeatureVector` / :class:`FeatureMatrix`
    wrappers so the operator factories can index them as ``V[i]`` / ``W[i, j]``
    while the field kind stays explicit.
    """
    volume_at_nodes = features[0]
    u_matrix = features[1]
    v_vector = FeatureVector([features[2 + j] for j in range(dimension)])
    w_matrix = FeatureMatrix(
        [
            [features[2 + dimension * (k + 1) + j] for j in range(dimension)]
            for k in range(dimension)
        ]
    )
    return volume_at_nodes, u_matrix, v_vector, w_matrix


def block(rows):
    """Compose a block operator from sparse blocks (``scipy.sparse.bmat``).

    ``None`` entries denote zero blocks, which ``bmat`` understands natively.
    """
    return scipy.sparse.bmat(rows, format="csr")


def hstack_blocks(blocks):
    """Horizontal stack of sparse ``(N, N)`` blocks."""
    return scipy.sparse.hstack(blocks, format="csr")
