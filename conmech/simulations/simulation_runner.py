import copy
from ctypes import ArgumentError
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.helpers.tmh import Timer
from conmech.plotting import plotter_2d, plotter_3d, plotter_common
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature


def run_examples(
    all_scenarios,
    file,
    plot_animation,
    config: Config,
    simulate_dirty_data=False,
    get_scene_function: Optional[Callable] = None,
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
            get_scene_function=get_scene_function,
        )
        print()
    print("DONE")


@dataclass
class RunScenarioConfig:
    catalog: Optional[str] = None
    simulate_dirty_data: bool = False
    plot_animation: bool = False
    save_all: bool = False


def save_scene(scene: Scene, scenes_path: str, save_animation: bool):
    scene_copy = copy.copy(scene)
    scene_copy.prepare_to_save()

    arrays_path = scenes_path + "_data"
    nodes = scene.boundary_nodes
    elements = scene.boundaries.boundary_surfaces
    arrays = (nodes, elements)
    if isinstance(scene, SceneTemperature):
        temperatue = scene.t_old
        arrays += (temperatue,)
    scenes_file, indices_file = pkh.open_files_append(arrays_path)
    with scenes_file, indices_file:
        pkh.append_data(data=arrays, data_path=arrays_path, lock=None)

    if save_animation:
        scenes_file, indices_file = pkh.open_files_append(scenes_path)
        with scenes_file, indices_file:
            pkh.append_data(data=scene_copy, data_path=scenes_path, lock=None)


def run_scenario(
    solve_function: Callable,
    scenario: Scenario,
    config: Config,
    run_config: RunScenarioConfig,
    get_scene_function: Optional[Callable] = None,
) -> Tuple[Scene, str, float]:

    print("Creating scene...")
    create_in_subprocess = False

    if get_scene_function is None:

        def _get_scene_function(randomize):
            return scenario.get_scene(
                randomize=randomize,
                create_in_subprocess=create_in_subprocess,
            )

    else:

        def _get_scene_function(randomize):
            return get_scene_function(
                config=config,
                scenario=scenario,
                randomize=randomize,
                create_in_subprocess=create_in_subprocess,
            )

    scene = cmh.profile(
        lambda: _get_scene_function(randomize=run_config.simulate_dirty_data), baypass=True
    )

    # np.save("./pt-jax/bunny_boundary_nodes2.npy", scene.boundary_nodes)
    # np.save("./pt-jax/contact_boundary2.npy", scene.boundaries.contact_boundary)

    time_skip = config.print_skip
    ts = int(time_skip / scenario.time_step)
    plot_scenes_count = [0]
    with_reduced = hasattr(scene, "reduced")
    save_files = run_config.plot_animation or run_config.save_all
    save_animation = run_config.plot_animation

    if save_files:
        final_catalog = f"{config.output_catalog}/{config.current_time} - {run_config.catalog}"
        cmh.create_folders(f"{final_catalog}/scenarios")
        if with_reduced:
            cmh.create_folders(f"{final_catalog}/scenarios_reduced")
        scenes_path = f"{final_catalog}/scenarios/{scenario.name}_DATA.scenes"
        scenes_path_reduced = f"{final_catalog}/scenarios_reduced/{scenario.name}_DATA.scenes"
    else:
        final_catalog = ""
        scenes_path = ""
        scenes_path_reduced = ""

    step = [0]  # TODO: #65 Clean

    def operation_save(scene: Scene):
        plot_index = step[0] % ts == 0
        if run_config.save_all or plot_index:
            save_scene(scene=scene, scenes_path=scenes_path, save_animation=save_animation)
            if with_reduced:
                save_scene(
                    scene=scene.reduced,
                    scenes_path=scenes_path_reduced,
                    save_animation=save_animation,
                )
        if plot_index:
            plot_scenes_count[0] += 1
        step[0] += 1

    def fun_sim():
        return simulate(
            scene=scene,
            solve_function=solve_function,
            scenario=scenario,
            simulate_dirty_data=run_config.simulate_dirty_data,
            config=config,
            operation=operation_save if save_files else None,
        )

    # cmh.profile(fun_sim)
    setting = fun_sim()

    if run_config.plot_animation:
        if config.animation_backend == "blender":
            plot_using_blender()
        elif config.animation_backend == "matplotlib":
            plot_scenario_animation(
                scenario=scenario,
                config=config,
                animation_path=f"{final_catalog}/{scenario.name}.gif",
                time_skip=time_skip,
                index_skip=ts if run_config.save_all else 1,
                plot_scenes_count=plot_scenes_count[0],
                all_scenes_path=scenes_path,
            )
        else:
            raise ArgumentError

    return setting, scenes_path


def plot_using_blender():
    path = "~/Desktop/Blender/blender-3.2.0-linux-x64/blender"
    args = " --background --python ~/Desktop/conmech/blender/load.py"
    print("Plotting using Blender...")
    subprocess.call(path + args, shell=True, stdout=subprocess.DEVNULL)
    print("Blender done")


def plot_scenario_animation(
    scenario: Scenario,
    config: Config,
    animation_path: str,
    time_skip: float,
    index_skip: int,
    plot_scenes_count: int,
    all_scenes_path: str,
):
    t_scale = plotter_common.get_t_scale(scenario, index_skip, plot_scenes_count, all_scenes_path)
    plot_function = (
        plotter_2d.plot_animation if scenario.dimension == 2 else plotter_3d.plot_animation
    )
    plot_function(
        save_path=animation_path,
        config=config,
        time_skip=time_skip,
        index_skip=index_skip,
        plot_scenes_count=plot_scenes_count,
        all_scenes_path=all_scenes_path,
        t_scale=t_scale,
    )


def prepare(scenario, scene: Scene, current_time, with_temperature):
    forces = scenario.get_forces_by_function(scene, current_time)
    if with_temperature:
        heat = scenario.get_heat_by_function(scene, current_time)
        scene.prepare_tmp(forces, heat)
    else:
        scene.prepare(forces)


def print_mesh_data(scene):
    print(f"Mesh type: {scene.mesh_prop.mesh_type}")
    print(
        f" Elements: {scene.elements_count}\
 | Boundary surfaces: {scene.boundary_surfaces_count} | Nodes: {scene.nodes_count}"
    )
    print()


def prepare_energy_functions(scenario, scene, solve_function, with_temperature):
    energy_functions = EnergyFunctions(
        scene.use_green_strain,
        scene.use_nonconvex_friction_law,
        scene.use_constant_contact_integral,
    )

    print("Precompiling...")
    with cmh.HiddenPrints():
        prepare(scenario, scene, 0, with_temperature)
        print("Prepared")
        for mode in EnergyFunctions.get_manual_modes():
            energy_functions.set_manual_mode(mode)
            try:
                _ = solve_function(
                    scene=scene,
                    energy_functions=energy_functions,
                    initial_a=None,
                    initial_t=None,
                )
            except Exception:
                pass

        if hasattr(scene, "reduced"):
            scene.reduced.exact_acceleration = None
            scene.reduced.lifted_acceleration = None

        energy_functions.set_automatic_mode()
    return energy_functions


def simulate(
    scene,
    solve_function,
    scenario: Scenario,
    simulate_dirty_data: bool,
    config: Config,
    operation: Optional[Callable] = None,
) -> Tuple[Scene, float]:
    with_temperature = isinstance(scene, SceneTemperature)

    print_mesh_data(scene)
    energy_functions = prepare_energy_functions(scenario, scene, solve_function, with_temperature)

    acceleration, temperature = (None,) * 2
    time_tqdm = scenario.get_tqdm(desc="Simulating", config=config)
    steps = len(time_tqdm)
    timer = Timer()

    for time_step in time_tqdm:
        current_time = (time_step) * scene.time_step

        with timer["all_prepare"]:
            prepare(scenario, scene, current_time, with_temperature)

        with timer["all_operation"]:
            if operation is not None:
                operation(scene)  # (current_time, scene, a, base_a)

        with timer["all_solver"]:
            acceleration, temperature = solve_function(
                scene=scene,
                energy_functions=energy_functions,
                initial_a=acceleration,
                initial_t=temperature,
                timer=timer,
            )

        if simulate_dirty_data:
            scene.make_dirty()

        with timer["all_iterate"]:
            scene.iterate_self(acceleration, temperature=temperature)
            scene.exact_acceleration = acceleration

    for key in timer:
        all_time = timer.dt[key].sum()
        print(f" {key}: {all_time:.2f}s | {(steps/all_time):.2f}it/s")

    print("Saving timings")
    fig = timer.dt["all_solver"].plot.hist(bins=100).get_figure()
    fig.savefig(f"log/timing-{cmh.get_timestamp(config)}.png")

    return scene


def plot_setting(
    current_time,
    scene,
    path,
    draw_detailed,
    extension,
):
    if scene.dimension == 2:
        fig = plotter_2d.get_fig()
        axs = plotter_2d.get_axs(fig)
        plotter_2d.plot_frame(
            fig=fig,
            axs=axs,
            scene=scene,
            current_time=current_time,
            draw_detailed=draw_detailed,
        )
        plotter_common.plt_save(path, extension)
    else:
        fig = plotter_3d.get_fig()
        axs = plotter_3d.get_axs(fig)
        plotter_3d.plot_frame(fig=fig, axs=axs, scene=scene, current_time=current_time)
        plotter_common.plt_save(path, extension)
