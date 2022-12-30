from dataclasses import dataclass
from typing import Union, Optional

import numpy as np


@dataclass
class Boundary:
    surfaces: np.ndarray
    node_indices: Union[slice, np.ndarray]  # slice or direct SORTED! indices
    node_count: int
    node_condition: Optional[np.ndarray] = None
