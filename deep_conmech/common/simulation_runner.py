import os
import time
from typing import Callable, List, Optional

from conmech.helpers import cmh
from conmech.helpers.config import Config
from deep_conmech.common import training_config
from deep_conmech.common.plotter import plotter_2d, plotter_3d, plotter_common
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_iterable import SettingIterable
from deep_conmech.simulator.setting.setting_temperature import \
    SettingTemperature


def run_examples(
    all_scenarios,
    file,
    plot_animation,
    config: Config,
    simulate_dirty_data=False,
    get_setting_function: Optional[Callable] = None,
):
    for i, scenario in enumerate(all_scenarios):
        print(f"-----EXAMPLE {i+1}/{len(all_scenarios)}-----")
        catalog = os.path.splitext(os.path.basename(file))[0].upper()
        plot_scenario(
            solve_function=scenario.get_solve_function(),
            scenario=scenario,
            catalog=catalog,
            simulate_dirty_data=simulate_dirty_data,
            plot_animation=plot_animation,
            config=config,
            get_setting_function=get_setting_function,
        )
        print()
    print("DONE")


def plot_scenario(
    solve_function,
    scenario: Scenario,
    catalog,
    config: Config,
    simulate_dirty_data=False,
    plot_animation=True,
    save_all=False,
    get_setting_function: Optional[Callable] = None,
):
    final_catalog = f"output/{config.CURRENT_TIME} - {catalog}"
    all_settings_path = f"{final_catalog}/{scenario.id}_DATA"
    cmh.create_folders(final_catalog)

    time_skip = config.PRINT_SKIP
    ts = int(time_skip / scenario.time_step)
    index_skip = ts if save_all else 1
    plot_settings_count = [0]

    settings_file, file_meta = SettingIterable.open_files_append_pickle(all_settings_path)
    with settings_file, file_meta:
        step = [0] #TODO: Clean
        def operation_save(current_time: float, setting : SettingIterable, base_setting, a, base_a):
            step[0] += 1
            if save_all or step[0] % ts == 0: 
                setting.append_pickle(settings_file=settings_file, file_meta=file_meta)
                plot_settings_count[0]+=1

        simulate(
            compare_with_base_setting=False,
            solve_function=solve_function,
            scenario=scenario,
            simulate_dirty_data=simulate_dirty_data,
            config=config,
            operation=operation_save if plot_animation else None,
            get_setting_function=get_setting_function,
        )

    if plot_animation:
        plot_scenario_animation(scenario, config, final_catalog, time_skip, index_skip, plot_settings_count[0], all_settings_path)

    return all_settings_path


def plot_scenario_animation(
    scenario:Scenario,
    config: Config,
    final_catalog: str,
    time_skip: float,
    index_skip: int,
    plot_settings_count: int,
    all_settings_path: str
):
    #t_scale = plotter_common.get_t_scale(scenario, plot_setting_paths)
    t_scale = None
    plot_function = (
        plotter_2d.plot_animation
        if scenario.dimension == 2
        else plotter_3d.plot_animation
    )
    plot_function(
        save_path=f"{final_catalog}/{scenario.id}.gif",
        config=config,
        time_skip=time_skip,
        index_skip=index_skip,
        plot_settings_count=plot_settings_count,
        all_settings_path=all_settings_path,
        t_scale=t_scale
    )


########


def simulate(
    compare_with_base_setting,
    solve_function,
    scenario: Scenario,
    simulate_dirty_data: bool,
    config: Config,
    operation: Optional[Callable] = None,
    get_setting_function: Optional[Callable] = None,
) -> None:
    _get_setting_function = (
        scenario.get_setting
        if get_setting_function is None
        else lambda randomize, create_in_subprocess: get_setting_function(
            config=config,
            scenario=scenario,
            randomize=randomize,
            create_in_subprocess=create_in_subprocess,
        )
    )

    setting = _get_setting_function(
        randomize=simulate_dirty_data, create_in_subprocess=True
    )
    with_temperature = isinstance(setting, SettingTemperature)
    if compare_with_base_setting:
        base_setting = _get_setting_function(randomize=False, create_in_subprocess=True)
    else:
        base_setting = None
        base_a = None

    solver_time = 0
    comparison_time = 0

    time_tqdm = scenario.get_tqdm(desc="Simulating", config=config)
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

        if operation is not None:
            operation(current_time, setting, base_setting, a, base_a)

        if with_temperature:
            setting.iterate_self(a, t)
        else:
            setting.iterate_self(a, randomized_inputs=simulate_dirty_data)

        if compare_with_base_setting:
            base_setting.iterate_self(base_a)

        # setting.remesh_self() ####################################################

    # comparison_str = (
    #    f" | Comparison time: {comparison_time}" if compare_with_base_setting else ""
    # )
    # print(f"    Solver time : {solver_time}{comparison_str}")


########


def plot_setting(
    current_time, setting, path, base_setting, draw_detailed, extension,
):
    if setting.dimension == 2:
        fig = plotter_2d.get_fig()
        axs = plotter_2d.get_axs(fig)
        plotter_2d.plot_frame(
            fig=fig,
            axs=axs,
            setting=setting,
            current_time=current_time,
            draw_detailed=draw_detailed,
            base_setting=base_setting,
        )
        plotter_common.plt_save(path, extension)
    else:
        fig = plotter_3d.get_fig()
        axs = plotter_3d.get_axs(fig)
        plotter_3d.plot_frame(
            fig=fig, axs=axs, setting=setting, current_time=current_time
        )
        plotter_common.plt_save(path, extension)


############################
