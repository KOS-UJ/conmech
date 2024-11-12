# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

const_i64_vec = types.Array(types.int64, 1, 'A', readonly=True)
const_i64_mat = types.Array(types.int64, 2, 'A', readonly=True)
const_f64_vec = types.Array(types.float64, 1, 'A', readonly=True)
const_f64_mat = types.Array(types.float64, 2, 'A', readonly=True)

__all__ = ['const_i64_vec', 'const_i64_mat', 'const_f64_vec', 'const_f64_mat', 'i64', 'f64', 'Tuple']
