import copy
import time
from typing import Callable, Optional

from conmech.helpers import cmh
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature


def simulate(
    compare_with_base_setting,
    solve_function,
    scenario: Scenario,
    get_setting_function,
    simulate_dirty_data,
    description,
    operation: Optional[Callable] = None
):
    all_settings = []
    all_base_settings = [] if compare_with_base_setting else None

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

    time_tqdm = scenario.get_tqdm(description)
    a = None
    t = None
    for time_step in time_tqdm:
        current_time = (time_step + 1) * setting.time_step

        forces = setting.get_forces_by_function(scenario.forces_function, current_time)
        setting.prepare(forces)
        # max_data.set(setting, time_step)
        all_settings.append(copy.deepcopy(setting))
        if compare_with_base_setting:
            all_base_settings.append(copy.deepcopy(setting))

        start_time = time.time()
        if with_temperature:
            a, t = solve_function(setting, initial_a=a, initial_t=t)
        else:
            a = solve_function(setting, initial_a=a)


        solver_time += time.time() - start_time

        if simulate_dirty_data:
            setting.make_dirty()

        if compare_with_base_setting:
            base_setting.set_forces_from_function(
                scenario.forces_function, current_time
            )

            start_time = time.time()
            base_a = Calculator.solve(base_setting)  ## save in setting
            comparison_time += time.time() - start_time

        if operation is not None:
            operation(current_time, setting, base_setting, a, base_a)

        if with_temperature:
            setting.iterate_self(a, t, randomized_inputs=simulate_dirty_data)
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
    return all_settings, all_base_settings
