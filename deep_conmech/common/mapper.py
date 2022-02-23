import time

from deep_conmech.common import config
from conmech.helpers import nph
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.model import *
from deep_conmech.graph.setting.setting_input import *
from deep_conmech.simulator.calculator import Calculator

import deep_conmech.common.config as config


def get_setting(scenario, simulate_dirty_data):
    setting = SettingInput(
        mesh_type=scenario.mesh_type,
        mesh_density_x=scenario.mesh_density,
        mesh_density_y=scenario.mesh_density,
        scale_x=scenario.scale,
        scale_y=scenario.scale,
        is_adaptive=scenario.is_adaptive,
        create_in_subprocess=True,
    )
    setting.set_randomization(simulate_dirty_data)
    setting.set_obstacles(scenario.obstacles)
    return setting


def get_base_setting(scenario):
    setting = SettingInput(
        mesh_type=scenario.mesh_type,
        mesh_density_x=scenario.mesh_density,
        mesh_density_y=scenario.mesh_density,
        scale_x=scenario.scale,
        scale_y=scenario.scale,
        is_adaptive=scenario.is_adaptive,
        create_in_subprocess=True,
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
    description,
):
    print("-----")
    setting = get_setting(scenario, simulate_dirty_data)
    if compare_with_base_setting:
        base_setting = get_base_setting(scenario)
    else:
        base_setting = None
        base_a = None
    max_data = thh.MaxData(scenario.id, episode_steps)

    solver_time = 0
    comparison_time = 0

    time_tqdm = thh.get_tqdm(range(episode_steps), f"{description} - {scenario.id}")
    for time_step in time_tqdm:
        current_time = (time_step + 1) * config.TIMESTEP

        forces = setting.get_forces_by_function(scenario.forces_function, current_time)
        setting.prepare(forces)
        max_data.set(setting, time_step)

        start_time = time.time()
        a = solve_function(setting)
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

        operation(current_time, setting, base_setting, a, base_a)

        setting.iterate_self(a, randomized_inputs=simulate_dirty_data)
        if compare_with_base_setting:
            base_setting.iterate_self(base_a)

        # setting.remesh_self() ####################################################

    comparison_str = (
        f" | Comparison {Calculator.mode()} time: {comparison_time}"
        if compare_with_base_setting
        else ""
    )
    print(f"    Solver time : {solver_time}{comparison_str}")

    # max_data.print()
