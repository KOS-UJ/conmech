from conmech.helpers import cmh
from deep_conmech.common import config, mapper
from deep_conmech.common.plotter import plotter_2d, plotter_3d, plotter_common
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.setting.setting_forces import *


def print_one_dynamic(
    solve_function,
    scenario: Scenario,
    get_setting_function,
    catalog,
    simulate_dirty_data,
    draw_base,
    draw_detailed,
    description,
    print_images=False
):
    extension = "png"  # pdf
    final_catalog = f"{cmh.CURRENT_TIME}- {catalog}"
    cmh.create_folders(f"output/{final_catalog}")

    all_settings, all_base_settings = mapper.map_time(
        compare_with_base_setting=draw_base,
        solve_function=solve_function,
        scenario=scenario,
        get_setting_function=get_setting_function,
        simulate_dirty_data=simulate_dirty_data,
        description=description,
    )

    if print_images:
        time_tqdm = scenario.get_tqdm(f"Printing {description}")
        for i in time_tqdm:
            current_time = (i + 1) * scenario.time_step
            plot_at_interval(
                current_time=current_time,
                setting=all_settings[i],
                path=f"output/{final_catalog}/{scenario.id} {int(current_time * 100)}.{extension}",
                base_setting=all_base_settings[i]
                if all_base_settings is not None
                else None,
                draw_detailed=draw_detailed,
                extension=extension,
            )

    print("Generating animation...")
    animation_path = f"output/{final_catalog}/{scenario.id} scale_{scenario.mesh_data.scale_x} ANIMATION.gif"
    if scenario.dimension == 2:
        plotter_2d.plot_animation(scenario, all_settings, animation_path)
    else:
        plotter_3d.plot_animation(scenario, all_settings, animation_path)


def plot_at_interval(
    current_time,
    setting,
    path,
    base_setting,
    draw_detailed,
    extension,
    skip=config.PRINT_SKIP,
):
    if skip is not None and nph.close_modulo(current_time, skip) == False:
        return

    if setting.dim == 2:
        fig = plotter_2d.get_fig()
        axs = plotter_2d.get_axs(fig)
        plotter_2d.plot_frame(setting, current_time, axs, draw_detailed, base_setting)
        plotter_common.plt_save(path, extension)
    else:
        fig = plotter_3d.get_fig()
        axs = plotter_3d.get_axs(fig)
        plotter_3d.plot_frame(setting=setting, current_time=current_time, axs=axs)
        plotter_common.plt_save(path, extension)


def print_setting_test(setting):
    fig = plotter_2d.get_fig()
    axs = plotter_2d.get_axs(fig)
    plotter_2d.set_perspective(scale=1, axs=axs)
    plotter_2d.draw_displaced(setting, [0.0, 0.0], "tab:orange", axs)
    plotter_common.plt_save("./output/1.png", "png")


def print_simple_data(elements, nodes, path):
    fig = plotter_2d.get_fig()
    axs = plotter_2d.get_axs(fig)
    plotter_2d.set_perspective(scale=1, axs=axs)
    plotter_2d.triplot(nodes, elements, "tab:orange", axs)
    extension = path.split(".")[-1]
    plotter_common.plt_save(path, extension)


############################

def print_setting(setting, filename, catalog):
    cmh.create_folders(catalog)
    extension = "png"  # pdf
    path = f"{catalog}/{filename}.{extension}"
    plot_at_interval(0, setting, path, None, True, extension)

