from conmech.helpers import cmh
from deep_conmech.common import config, simulator
from deep_conmech.common.plotter import plotter_2d, plotter_3d, plotter_common
from deep_conmech.scenarios import Scenario, TemperatureScenario
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_iterable import SettingIterable


def print_one_dynamic(
    solve_function,
    scenario: Scenario,
    get_setting_function,
    catalog,
    simulate_dirty_data,
    plot_base,
    plot_detailed,
    plot_images=False,
    plot_animation=True,
):
    extension = "png"  # pdf
    final_catalog = f"output/{cmh.CURRENT_TIME} - {catalog}"
    setting_catalog = f"{final_catalog}/settings"
    cmh.create_folders(setting_catalog)
    time_skip = config.PRINT_SKIP
    all_setting_paths = []

    def operation_plot(current_time, setting, base_setting, a, base_a):
        plot_setting(
            current_time=current_time,
            setting=setting,
            path=f"{final_catalog}/{scenario.id} {int(current_time * 100)}.{extension}",
            base_setting=base_setting,
            draw_detailed=plot_detailed,
            extension=extension,
        )

    def operation_save(current_time, setting, base_setting, a, base_a):
        path = f"{setting_catalog}/time_{current_time:.4f}"
        setting_path = f"{path}"
        setting.save_pickle(setting_path)
        all_setting_paths.append(setting_path)
        # if draw_base:
        #    setting.save_pickle(f"{path}_base_setting")

    simulator.simulate(
        compare_with_base_setting=plot_base,
        solve_function=solve_function,
        scenario=scenario,
        get_setting_function=get_setting_function,
        simulate_dirty_data=simulate_dirty_data,
        operation=operation_save
        if plot_animation
        else None,  # plot_at_interval if plot_images else None,
        time_skip=time_skip,
    )

    """
    if plot_images:
        time_tqdm = scenario.get_tqdm(f"Plotting images {description}")
        for i in time_tqdm:
            current_time = (i + 1) * scenario.time_step
            base_setting = (
                all_base_settings[i] if all_base_settings is not None else None
            )
            operation_plot(current_time, all_settings[i], base_setting, None, None)
    """

    t_scale = get_t_scale(scenario, all_setting_paths)
    
    if plot_animation:
        animation_path = f"{final_catalog}/{scenario.id} scale_{scenario.mesh_data.scale_x} ANIMATION.gif"
        if scenario.dimension == 2:
            plotter_2d.plot_animation(
                all_setting_paths, time_skip, animation_path, t_scale
            )
        else:
            plotter_3d.plot_animation(
                all_setting_paths, time_skip, animation_path, t_scale
            )

    cmh.clear_folder(setting_catalog)


def get_t_scale(scenario, all_setting_paths):
    if isinstance(scenario, TemperatureScenario) == False:
        return None

    temperatures = np.array(
        [SettingIterable.load_pickle(path).t_old for path in all_setting_paths]
    )
    return [np.min(temperatures), np.max(temperatures)]


def plot_setting(
    current_time, setting, path, base_setting, draw_detailed, extension,
):
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
    plotter_common.plt_save("output/1.png", "png")


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
