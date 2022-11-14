"""
Created at 18.02.2021
"""

__all__ = ["Solver", "SolversRegistry"]

from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.solver import Solver
from conmech.solvers.direct import Direct
from conmech.solvers.optimization.global_optimization import GlobalOptimization
from conmech.solvers.optimization.schur_complement import SchurComplementOptimization
