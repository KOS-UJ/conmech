"""
Created at 18.02.2021
"""

from conmech.solvers.direct import Direct
from conmech.solvers.optimization.global_optimization import Global
from conmech.solvers.optimization.schur_complement import SchurComplement


def get_solver_class(method: str, time_dependent: str) -> type:  # TODO
    if method == 'direct':
        solver_class = Direct
    elif method == 'schur':
        solver_class = SchurComplement
    elif method == 'global optimization':
        solver_class = Global
    else:
        raise ValueError()
    return solver_class
