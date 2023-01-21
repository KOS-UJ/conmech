import subprocess
import sys

from conmech.helpers.config import Config
from conmech.plotting import plotter_2d, plotter_3d
from conmech.plotting.plotter_common import get_t_scale, plt_save
from conmech.scenarios.scenarios import Scenario


def plot_using_blender(output: bool = True):
    path = "~/Desktop/Blender/blender-3.2.0-linux-x64/blender"
    args = " --background --python ~/Desktop/conmech/blender/load.py --render"
    print("Plotting using Blender...")
    stdout = sys.stdout if output else subprocess.DEVNULL
    subprocess.call(path + args, shell=True, stdout=stdout)
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
    t_scale = get_t_scale(scenario, index_skip, plot_scenes_count, all_scenes_path)
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
        plt_save(path, extension)
    else:
        fig = plotter_3d.get_fig()
        axs = plotter_3d.get_axs(fig)
        plotter_3d.plot_frame(fig=fig, axs=axs, scene=scene, current_time=current_time)
        plt_save(path, extension)
