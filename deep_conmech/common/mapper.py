import time

from deep_conmech.common import config
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.simulator.calculator import Calculator
from conmech.helpers import cmh, nph
from deep_conmech.common import config




def get_randomized_setting(scenario, randomize=False, create_in_subprocess=False):
    setting = SettingRandomized(
        mesh_data=scenario.mesh_data,
        body_coeff=scenario.body_coeff,
        obstacle_coeff=scenario.obstacle_coeff,
        time_data=scenario.time_data,
        create_in_subprocess=create_in_subprocess,
    )
    setting.set_randomization(randomize)
    setting.set_obstacles(scenario.obstacles)
    return setting



def map_time(
    compare_with_base_setting,
    operation,
    solve_function,
    scenario,
    simulate_dirty_data,
    description,
):
    setting = get_randomized_setting(scenario, randomize=simulate_dirty_data, create_in_subprocess=True)
    if compare_with_base_setting:
        base_setting = get_randomized_setting(scenario, randomize=False, create_in_subprocess=True)
    else:
        base_setting = None
        base_a = None
    #max_data = thh.MaxData(scenario.id, episode_steps)

    solver_time = 0
    comparison_time = 0

    time_tqdm = cmh.get_tqdm(range(scenario.time_data.episode_steps), f"{description} - {scenario.id} scale_{scenario.mesh_data.scale_x}")
    a = None
    for time_step in time_tqdm:
        current_time = (time_step + 1) * setting.time_step

        forces = setting.get_forces_by_function(scenario.forces_function, current_time)
        setting.prepare(forces)
        #max_data.set(setting, time_step)

        start_time = time.time()
        a = solve_function(setting, initial_vector=a)
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
