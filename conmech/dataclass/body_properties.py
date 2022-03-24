from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BodyProperties:
    mass_density: float
    mu: float
    lambda_: float
    theta: Optional[float] = None
    zeta: Optional[float] = None

    C_coeff = [[0.5, 0.0], [0.0, 0.5]]
    K_coeff = [[0.1, 0.0], [0.0, 0.1]]
