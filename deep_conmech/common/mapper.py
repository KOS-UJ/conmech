import time
from simulator.calculator import Calculator

import common.config as config
from deep_conmech.common import config, basic_helpers
from deep_conmech.common.basic_helpers import *
from graph.model import *
from graph.setting.setting_input import *


def get_setting(scenario, simulate_dirty_data):
    setting = SettingInput(
        scenario.mesh_density,
        scenario.mesh_type,
        scenario.scale,
        scenario.is_adaptive,
        create_in_subprocess=True
    )
    setting.set_randomization(simulate_dirty_data)
    setting.set_obstacles(scenario.obstacles)
    return setting


def get_base_setting(scenario):
    setting = SettingInput(
        scenario.mesh_density,
        scenario.mesh_type,
        scenario.scale,
        scenario.is_adaptive,
        create_in_subprocess=True
    )
    setting.set_obstacles(scenario.obstacles)
    return setting


def map_time(
    compare_with_base_setting,
    operation,
    episode_steps,
    solve_function,
    scenario,
    simulate_dirty_data,
    description
):
    print("-----")
    setting = get_setting(scenario, simulate_dirty_data)
    if compare_with_base_setting:
        base_setting = get_base_setting(scenario)
    else:
        base_setting = None
        base_a = None
    max_data = MaxData(scenario.id, episode_steps)

    solver_time = 0
    comparison_time = 0

    time_tqdm = basic_helpers.get_tqdm(range(episode_steps), f"{description} - {scenario.id}")
    for timestep in time_tqdm:
        current_time = (timestep + 1) * config.TIMESTEP

        forces =basic_helpers.get_forces_by_function(scenario.forces_function, setting, current_time)
        setting.prepare(forces)
        max_data.set(setting, timestep)

        start_time = time.time()
        a = solve_function(setting)
        solver_time += time.time() - start_time

        if simulate_dirty_data:
            setting.make_dirty()

        if compare_with_base_setting:
            base_setting.set_forces_from_function(scenario.forces_function, current_time)

            start_time = time.time()
            base_a = Calculator.solve(base_setting)  ## save in setting
            comparison_time += time.time() - start_time

        operation(current_time, setting, base_setting, a, base_a)

        setting.iterate_self(a, randomized_inputs=simulate_dirty_data)
        if compare_with_base_setting:
            base_setting.iterate_self(base_a)

        #setting.remesh_self() ####################################################

    comparison_str = (
        f" | Comparison {Calculator.mode()} time: {comparison_time}"
        if compare_with_base_setting
        else ""
    )
    print(f"    Solver time : {solver_time}{comparison_str}")

    # max_data.print()
