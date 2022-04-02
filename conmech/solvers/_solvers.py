from conmech.problems import Problem, \
    Static as StaticProblem, \
    Quasistatic as QuasistaticProblem, \
    Dynamic as DynamicProblem


class Solvers:
    solvers = {"static": {},
               "quasistatic": {},
               "dynamic": {}}

    @staticmethod
    def register(dynamism: str, *names):
        if dynamism == "*":
            dynamism_types = ("static", "quasistatic", "dynamic")
        else:
            dynamism_types = (dynamism,)

        def add_to_dict(solver):
            for dynamism_type in dynamism_types:
                dyn = Solvers.solvers[dynamism_type]
                for name in names:
                    lower_name = name.lower()
                    assert lower_name not in dyn  # name already taken
                    dyn[lower_name] = solver
            return solver

        return add_to_dict

    @staticmethod
    def get_by_name(solver_name: str, problem: Problem) -> type:
        if isinstance(problem, StaticProblem):
            dynamism_type = "static"
        elif isinstance(problem, QuasistaticProblem):
            dynamism_type = "quasistatic"
        elif isinstance(problem, DynamicProblem):
            dynamism_type = "dynamic"
        else:
            raise ValueError(f"Unknown problem class: {problem.__class__.__name__}")
        dyn = Solvers.solvers[dynamism_type]
        return dyn[solver_name.lower()]
