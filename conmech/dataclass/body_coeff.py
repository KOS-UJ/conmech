from dataclasses import dataclass
from typing import Optional


@dataclass
class BodyCoeff:
    mu: float
    lambda_: float
    theta: Optional[float] = None
    zeta: Optional[float] = None
    mass_density: float = 1.0
