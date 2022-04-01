"""
Created at 18.02.2021
"""

__all__ = ["Solver", "Solvers"]

from conmech.solvers._solvers import Solvers
from conmech.solvers.direct import Direct
from conmech.solvers.optimization.global_optimization import Global
from conmech.solvers.optimization.schur_complement import SchurComplement
from conmech.solvers.solver import Solver
