import json
import os
import subprocess
import sys

from conmech.helpers import cmh
from conmech.helpers.config import Config
from conmech.mesh.mesh_builders_3d import get_edges_from_surfaces
from conmech.plotting import plotter_2d, plotter_3d
from conmech.plotting.plotter_common import get_t_scale, plt_save
from conmech.scenarios.scenarios import Scenario


def save_three(scene, step, label, folder, skip=5):
    # Three.js
    if step % skip != 0:
        return

    simulation_folder = f"{folder}/{label}"
    cmh.create_folder(simulation_folder)
    # if step == 0:
    #     remove(file_path)
    #     remove(file_path_tmp)

    file_path = f"{simulation_folder}/{step}.json"

    def convert_to_list(array):
        return list(array.reshape(-1))

    def get_data(scene, get_edges):
        nodes = convert_to_list(scene.boundary_nodes)
        if get_edges:
            boundary_data = get_edges_from_surfaces(scene.boundaries.boundary_surfaces)
        else:
            boundary_data = scene.boundaries.boundary_surfaces

        return nodes, [int(i) for i in boundary_data.reshape(-1)]

    nodes, boundary_surfaces = get_data(scene, get_edges=False)
    if hasattr(scene, "reduced"):
        nodes_reduced, boundary_edges_reduced = get_data(scene.reduced, get_edges=True)
    else:
        nodes_reduced, boundary_edges_reduced = [], []

    highlighted_nodes = convert_to_list(scene.boundary_nodes[scene.self_collisions_mask])

    sn1 = (scene.initial_nodes + scene.norm_exact_new_displacement)[scene.boundary_indices]
    sn2 = (scene.initial_nodes + scene.to_normalized_displacement(0 * scene.exact_acceleration))[
        scene.boundary_indices
    ]

    sn1r = (scene.reduced.initial_nodes + scene.reduced.norm_exact_new_displacement)[
        scene.reduced.boundary_indices
    ]

    # sn2 = (
    #     scene.initial_nodes + scene.to_normalized_displacement_rotated(scene.exact_acceleration)
    # )[scene.boundary_indices]
    # sn3 = (
    #     scene.initial_nodes
    #     + scene.to_normalized_displacement_rotated_displaced(scene.exact_acceleration)
    # )[scene.boundary_indices]

    nodes_list = [
        nodes,
        convert_to_list(sn1),
        convert_to_list(sn2),
    ]  # , convert_to_list(sn2), convert_to_list(sn3)]
    nodes_reduced_list = [nodes_reduced, convert_to_list(sn1r)]

    json_dict = {
        "skip": skip,
        "step": step,
        "nodes_list": nodes_list,
        "nodes_reduced_list": nodes_reduced_list,
        "highlighted_nodes": highlighted_nodes,
        "linear_obstacles": convert_to_list(scene.linear_obstacles),
    }
    if step == 0:
        json_dict["boundary_surfaces"] = boundary_surfaces
        json_dict["boundary_edges_reduced"] = boundary_edges_reduced

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(json_dict, file)

    # if step == 0:
    list_path = f"{folder}/list.json"
    cmh.clear_file(list_path)
    # all_folders = os.walk(folder)
    # all_folders.sort(reverse=True)
    # folder_list = [os.path.basename(folder[0]) for folder in all_folders]
    folder_list = [f[0] for f in os.walk(folder)][1:]
    folder_list.sort(reverse=True)
    simulations_list, step_list = [], []
    for simulation in folder_list:
        steps = [int(os.path.splitext(file)[0]) for file in os.listdir(simulation)]
        if len(steps) > 1:
            simulations_list.append(os.path.basename(simulation))
            step_list.append(max(steps))

    # simulations_list = [label]
    with open(list_path, "w", encoding="utf-8") as file:
        json.dump({"simulations": simulations_list, "steps": step_list}, file)


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
