from typing import Type, Callable, Dict

from conmech.solvers.solver import Solver
from conmech.scenarios.problems import (
    StaticProblem, QuasistaticProblem, DynamicProblem, Problem,
)


class SolversRegistry:
    solvers: Dict[str, Dict[str, Type[Solver]]] = {"static": {}, "quasistatic": {}, "dynamic": {}}

    @staticmethod
    def register(dynamism: str, *names: str) -> Callable[[Type[Solver]], Type[Solver]]:
        if dynamism == "*":
            dynamism_types = ("static", "quasistatic", "dynamic")
        else:
            dynamism_types = (dynamism,)

        def add_to_dict(solver: Type[Solver]) -> Type[Solver]:
            for dynamism_type in dynamism_types:
                dyn: Dict[str, Type[Solver]] = SolversRegistry.solvers[dynamism_type]
                for name in names:
                    lower_name = name.lower()
                    assert lower_name not in dyn  # name already taken
                    dyn[lower_name] = solver
            return solver

        return add_to_dict

    @staticmethod
    def get_by_name(solver_name: str, problem: Problem) -> Type[Solver]:
        dynamism_type: str
        if isinstance(problem, StaticProblem):
            dynamism_type = "static"
        elif isinstance(problem, QuasistaticProblem):
            dynamism_type = "quasistatic"
        elif isinstance(problem, DynamicProblem):
            dynamism_type = "dynamic"
        else:
            raise ValueError(f"Unsupported class: {problem.__class__.__name__}")
        dyn: Dict[str, Type[Solver]] = SolversRegistry.solvers[dynamism_type]
        return dyn[solver_name.lower()]
