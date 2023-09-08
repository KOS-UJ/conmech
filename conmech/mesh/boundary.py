# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np


@dataclass
class Boundary:
    """
    Represents a boundary condition for a finite element analysis.

    This class defines a boundary condition with information about the boundary's
    surfaces, node indices (either as a slice or direct SORTED indices), the total
    number of nodes affected by the boundary condition, and an optional node
    condition values.

    Parameters:
    ----------
    surfaces : np.ndarray
        An array containing the surfaces associated with the boundary condition.

    node_indices : Union[slice, np.ndarray]
        Either a slice object specifying a range of node indices or a direct
        sorted array of node indices. These indices define the nodes affected by
        the boundary condition.

    node_count : int
        The total number of nodes affected by the boundary condition.

    node_condition : Optional[np.ndarray], optional
        An optional array specifying the conditions applied to the affected nodes.
        If provided, it should have the same length as `node_count`.

    Examples:
    --------
    To create a Boundary instance for fixing nodes 0 to 9 on a surface:
    ```
    surface = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bc = Boundary(surfaces=surface, node_indices=slice(0, 10), node_count=10)
    ```

    To create a Boundary instance for applying a specific condition to nodes:
    ```
    surface = np.array([0, 1, 2])
    indices = np.array([2, 0, 1])
    conditions = np.array([0.0, 1.0, -1.0])
    bc = Boundary(surfaces=surface, node_indices=indices, node_count=3, node_condition=conditions)
    ```
    """

    surfaces: np.ndarray
    node_indices: Union[slice, np.ndarray]
    node_count: int
    node_condition: Optional[np.ndarray] = None
