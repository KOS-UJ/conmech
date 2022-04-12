import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.plotting import plotter_2d, plotter_3d, plotter_common
from conmech.scenarios.scenarios import Scenario
from conmech.solvers.calculator import Calculator
from deep_conmech.simulator.setting.scene import Scene
from deep_conmech.simulator.setting.scene_temperature import SceneTemperature


def run_examples(
    all_scenarios,
    file,
    plot_animation,
    config: Config,
    simulate_dirty_data=False,
    get_setting_function: Optional[Callable] = None,
):
    for i, scenario in enumerate(all_scenarios):
        print(f"-----EXAMPLE {i + 1}/{len(all_scenarios)}-----")
        catalog = os.path.splitext(os.path.basename(file))[0].upper()
        run_scenario(
            solve_function=scenario.get_solve_function(),
            scenario=scenario,
            config=config,
            run_config=RunScenarioConfig(
                catalog=catalog,
                simulate_dirty_data=simulate_dirty_data,
                plot_animation=plot_animation,
            ),
            get_setting_function=get_setting_function,
        )
        print()
    print("DONE")


@dataclass
class RunScenarioConfig:
    catalog: str
    simulate_dirty_data: bool = False
    compare_with_base_setting: bool = False
    plot_animation: bool = True
    save_all: bool = False


def run_scenario(
    solve_function: Callable,
    scenario: Scenario,
    config: Config,
    run_config: RunScenarioConfig,
    get_setting_function: Optional[Callable] = None,
) -> Tuple[Scene, str]:
    time_skip = config.print_skip
    ts = int(time_skip / scenario.time_step)
    index_skip = ts if run_config.save_all else 1
    plot_settings_count = [0]

    save_files = run_config.plot_animation or run_config.save_all
    if save_files:
        final_catalog = f"{config.output_catalog}/{config.current_time} - {run_config.catalog}"
        cmh.create_folders(f"{final_catalog}/scenarios")
        data_path = f"{final_catalog}/scenarios/{scenario.name}_DATA"
        if run_config.compare_with_base_setting:
            cmh.create_folders(f"{final_catalog}/scenarios_calculator")
            calculator_data_path = f"{final_catalog}/scenarios_calculator/{scenario.name}_DATA"
    else:
        final_catalog = ""
        data_path = ""
        calculator_data_path = ""

    def save_setting(setting: Scene, data_path: str):
        settings_file, file_meta = pkh.open_files_append_pickle(data_path)
        with settings_file, file_meta:
            pkh.append_pickle(setting=setting, settings_file=settings_file, file_meta=file_meta)

    step = [0]  # TODO: #65 Clean

    def operation_save(setting: Scene, base_setting: Optional[Scene] = None):
        step[0] += 1
        plot_index = step[0] % ts == 0
        if run_config.save_all or plot_index:
            save_setting(setting=setting, data_path=data_path)
            if base_setting is not None:
                save_setting(setting=base_setting, data_path=calculator_data_path)
        if plot_index:
            plot_settings_count[0] += 1

    setting = simulate(
        solve_function=solve_function,
        scenario=scenario,
        simulate_dirty_data=run_config.simulate_dirty_data,
        compare_with_base_setting=run_config.compare_with_base_setting,
        config=config,
        operation=operation_save if save_files else None,
        get_setting_function=get_setting_function,
    )

    if run_config.plot_animation:
        animation_path = f"{final_catalog}/{scenario.name}.gif"
        plot_scenario_animation(
            scenario,
            config,
            animation_path,
            time_skip,
            index_skip,
            plot_settings_count[0],
            data_path,
            calculator_data_path if run_config.compare_with_base_setting else None,
        )

    return setting, data_path


def plot_scenario_animation(
    scenario: Scenario,
    config: Config,
    animation_path: str,
    time_skip: float,
    index_skip: int,
    plot_settings_count: int,
    all_settings_path: str,
    all_calc_settings_path: Optional[str],
):
    t_scale = plotter_common.get_t_scale(
        scenario, index_skip, plot_settings_count, all_settings_path
    )
    plot_function = (
        plotter_2d.plot_animation if scenario.dimension == 2 else plotter_3d.plot_animation
    )
    plot_function(
        save_path=animation_path,
        config=config,
        time_skip=time_skip,
        index_skip=index_skip,
        plot_settings_count=plot_settings_count,
        all_settings_path=all_settings_path,
        all_calc_settings_path=all_calc_settings_path,
        t_scale=t_scale,
    )


def prepare(scenario, setting, current_time, forces, with_temperature):
    if with_temperature:
        heat = scenario.get_heat_by_function(setting, current_time)
        setting.prepare(forces, heat)
    else:
        setting.prepare(forces)


def iterate(acceleration, temperature, setting, simulate_dirty_data, with_temperature):
    if with_temperature:
        setting.iterate_self(acceleration, temperature)
    else:
        setting.iterate_self(acceleration, randomized_inputs=simulate_dirty_data)


def simulate(
    solve_function,
    scenario: Scenario,
    simulate_dirty_data: bool,
    compare_with_base_setting: bool,
    config: Config,
    operation: Optional[Callable] = None,
    get_setting_function: Optional[Callable] = None,
) -> Scene:
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

    setting = _get_setting_function(randomize=simulate_dirty_data, create_in_subprocess=True)
    with_temperature = isinstance(setting, SceneTemperature)
    if compare_with_base_setting:
        base_setting = _get_setting_function(randomize=False, create_in_subprocess=True)
    else:
        base_setting = None
        base_a = None

    solver_time = 0.0
    calculator_time = 0.0

    time_tqdm = scenario.get_tqdm(desc="Simulating", config=config)
    acceleration = None
    temperature = None
    for time_step in time_tqdm:
        current_time = (time_step + 1) * setting.time_step

        forces = scenario.get_forces_by_function(setting, current_time)
        prepare(scenario, setting, current_time, forces, with_temperature)

        start_time = time.time()
        if with_temperature:
            acceleration, temperature = solve_function(
                setting, initial_a=acceleration, initial_t=temperature
            )
        else:
            acceleration = solve_function(setting, initial_a=acceleration)
        solver_time += time.time() - start_time

        if simulate_dirty_data:
            setting.make_dirty()

        if compare_with_base_setting:
            base_forces = scenario.get_forces_by_function(base_setting, current_time)
            base_setting.prepare(base_forces)

            start_time = time.time()
            base_a = Calculator.solve(base_setting)  # TODO #65: save in setting
            calculator_time += time.time() - start_time

        if operation is not None:
            operation(setting, base_setting)  # (current_time, setting, base_setting, a, base_a)

        iterate(acceleration, temperature, setting, simulate_dirty_data, with_temperature)

        if compare_with_base_setting:
            base_setting.iterate_self(base_a)

        # setting.remesh_self() # TODO #65

    comparison_str = f" | Calculator time: {calculator_time}" if compare_with_base_setting else ""
    print(f"    Solver time : {solver_time}{comparison_str}")
    return setting


def plot_setting(
    current_time,
    setting,
    path,
    base_setting,
    draw_detailed,
    extension,
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
        plotter_3d.plot_frame(fig=fig, axs=axs, setting=setting, current_time=current_time)
        plotter_common.plt_save(path, extension)
