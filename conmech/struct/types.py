# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2024-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

from numba import types
from numba import float64 as f64
from numba import int64 as i64
from numba.types import Tuple

ci64_vec = types.Array(types.int64, 1, "A", readonly=True)
ci64_mat = types.Array(types.int64, 2, "A", readonly=True)
cf64_vec = types.Array(types.float64, 1, "A", readonly=True)
cf64_mat = types.Array(types.float64, 2, "A", readonly=True)


class Tci64:
    def __getitem__(self, item):
        if item == slice(None, None, None):
            return ci64_vec
        if item == (slice(None, None, None), slice(None, None, None)):
            return ci64_mat
        raise TypeError()


class Tcf64:
    def __getitem__(self, item):
        if item == slice(None, None, None):
            return cf64_vec
        if item == (slice(None, None, None), slice(None, None, None)):
            return cf64_mat
        raise TypeError()


ci64 = Tci64()
cf64 = Tcf64()


# Backend-aware wrappers for assembled node-feature arrays.
#
# The FEM assembly (see ``dynamics/factory``) produces, per mesh feature, an
# ``N x N`` node-to-node matrix.  Grouped together these form the building
# blocks of every operator:
#   * ``U``  -- a single ``(N, N)`` mass-like matrix,
#   * ``V``  -- a ``dim``-vector of ``(N, N)`` matrices (1D field, e.g. thermal
#               expansion coupling),
#   * ``W``  -- a ``dim x dim`` matrix of ``(N, N)`` matrices (gradient/gradient
#               products, the core of every stiffness operator).
class FeatureVector:
    """A ``dim``-long vector of assembled ``(N, N)`` blocks."""

    FIELD = 1

    def __init__(self, blocks):
        self.blocks = list(blocks)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        return self.blocks[item]


class FeatureMatrix:
    """A ``dim x dim`` grid of assembled ``(N, N)`` blocks."""

    FIELD = 2

    def __init__(self, blocks):
        # ``blocks`` is a list of lists (``dim x dim``) of (N, N) matrices.
        self.blocks = [list(row) for row in blocks]
        self.dim = len(self.blocks)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            i, j = item
            return self.blocks[i][j]
        return self.blocks[item]

    def diagonal_sum(self):
        result = self.blocks[0][0]
        for i in range(1, self.dim):
            result = result + self.blocks[i][i]
        return result


__all__ = [
    "ci64",
    "cf64",
    "i64",
    "f64",
    "Tuple",
    "FeatureVector",
    "FeatureMatrix",
]
