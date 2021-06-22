"""
Created at 18.02.2021
"""

from conmech.solvers.direct import Direct
from conmech.solvers.optimization.global_optimization import Global
from conmech.problems import Problem, \
                             Static as StaticProblem, \
                             Quasistatic as QuasistaticProblem, \
                             Dynamic as DynamicProblem
from conmech.solvers.optimization.schur_complement import Static as StaticSchur, \
                                                          Quasistatic as QuasistaticSchur, \
                                                          Dynamic as DynamicSchur


def get_solver_class(method: str, problem: Problem) -> type:  # TODO
    if method.lower() == 'direct':
        solver_class = Direct
    elif method.lower() == 'schur':
        if isinstance(problem, StaticProblem):
            solver_class = StaticSchur
        elif isinstance(problem, QuasistaticProblem):
            solver_class = QuasistaticSchur
        elif isinstance(problem, DynamicProblem):
            solver_class = DynamicSchur
        else:
            raise ValueError("Unknown problem class.")
    elif method.lower() == 'global optimization':
        solver_class = Global
    else:
        raise ValueError("Unknown method.")
    return solver_class
