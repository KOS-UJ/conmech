import time
from typing import Callable, Optional

from conmech.helpers import nph
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.solver import Solver
from deep_conmech.simulator.setting.setting_iterable import SettingIterable
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature


def simulate(
    compare_with_base_setting,
    solve_function,
    scenario: Scenario,
    get_setting_function: Callable[[Scenario, bool, bool], SettingIterable],
    simulate_dirty_data: bool,
    operation: Optional[Callable] = None,
    time_skip: Optional[float] = None,
) -> None:
    setting = get_setting_function(
        scenario, randomize=simulate_dirty_data, create_in_subprocess=True
    )
    with_temperature = isinstance(setting, SettingTemperature)
    if compare_with_base_setting:
        base_setting = get_setting_function(
            scenario, randomize=False, create_in_subprocess=True
        )
    else:
        base_setting = None
        base_a = None
    # max_data = thh.MaxData(scenario.id, episode_steps)

    solver_time = 0
    comparison_time = 0

    time_tqdm = scenario.get_tqdm("Simulating")
    a = None
    t = None
    for time_step in time_tqdm:
        current_time = (time_step + 1) * setting.time_step

        forces = scenario.get_forces_by_function(setting, current_time)
        if with_temperature:
            heat = scenario.get_heat_by_function(setting, current_time)
            setting.prepare(forces, heat)
        else:
            setting.prepare(forces)

        # max_data.set(setting, time_step)

        start_time = time.time()
        if with_temperature:
            a, t = solve_function(setting, initial_a=a, initial_t=t)
        else:
            a = solve_function(setting, initial_a=a)

        solver_time += time.time() - start_time

        if simulate_dirty_data:
            setting.make_dirty()

        if compare_with_base_setting:
            scenario.base_setting.set_forces_from_function(
                scenario.forces_function, current_time
            )

            start_time = time.time()
            base_a = Solver.solve(base_setting)  ## save in setting
            comparison_time += time.time() - start_time

        if time_skip is None or nph.close_modulo(current_time, time_skip):
            if operation is not None:
                operation(current_time, setting, base_setting, a, base_a)

        if with_temperature:
            setting.iterate_self(a, t)
        else:
            setting.iterate_self(a, randomized_inputs=simulate_dirty_data)

        if compare_with_base_setting:
            base_setting.iterate_self(base_a)

        # setting.remesh_self() ####################################################

    comparison_str = (
        f" | Comparison time: {comparison_time}" if compare_with_base_setting else ""
    )
    print(f"    Solver time : {solver_time}{comparison_str}")

    # max_data.print()
