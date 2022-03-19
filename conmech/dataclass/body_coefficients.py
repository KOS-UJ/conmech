from dataclasses import dataclass
from typing import Optional


@dataclass
class BodyCoefficients:
    mu: float
    lambda_: float
    theta: Optional[float] = None
    zeta: Optional[float] = None
    density: float = 1.
