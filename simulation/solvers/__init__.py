"""
Created at 18.02.2021
"""

from simulation.solvers.direct import Direct
from simulation.solvers.optimization.global_optimization import Global
from simulation.solvers.optimization.schur_complement import SchurComplement


def get_solver_class(method: str) -> type:
    if method == 'direct':
        solver_class = Direct
    elif method == 'schur':
        solver_class = SchurComplement
    elif method == 'global optimization':
        solver_class = Global
    else:
        raise ValueError()
    return solver_class
