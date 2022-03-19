from dataclasses import dataclass
from typing import Optional


@dataclass
class BodyCoeff:
    mass_density: float
    mu: float
    lambda_: float
    theta: Optional[float] = None
    zeta: Optional[float] = None
