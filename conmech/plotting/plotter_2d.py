from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections
from matplotlib.patches import Rectangle

from conmech.helpers.config import Config
from conmech.plotting import plotter_common
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.simulator.setting.setting_temperature import \
    SettingTemperature


def get_fig():
    return plt.figure(figsize=(4, 2))


def get_axs(fig):
    # axs = fig.add_subplot(1, 1, 1, facecolor="none")
    axs = fig.add_axes([0.075, 0.15, 0.9, 0.8], facecolor="none")
    return axs


def set_perspective(scale, axes):
    axes.set_aspect("equal", "box")
    padding = 6
    axes.set_xlim(-padding * scale, 18 * scale)
    axes.set_ylim(-padding * scale, padding * scale)
    plotter_common.set_ax(axes)


def plot_animation(
        save_path: str,
        config: Config,
        time_skip: float,
        index_skip: int,
        plot_settings_count: int,
        all_settings_path: str,
        t_scale: Optional[np.ndarray] = None,
):
    plotter_common.plot_animation(
        get_axs=get_axs,
        plot_frame=plot_frame,
        fig=get_fig(),
        save_path=save_path,
        config=config,
        time_skip=time_skip,
        index_skip=index_skip,
        plot_settings_count=plot_settings_count,
        all_settings_path=all_settings_path,
        t_scale=t_scale
    )


def plot_frame(
        fig,
        axs,
        setting: SettingRandomized,
        current_time: float,
        draw_detailed: bool = True,
        base_setting: Optional[SettingRandomized] = None,
        t_scale: Optional[np.ndarray] = None,
):
    axes = axs
    scale = setting.mesh_data.scale_x
    set_perspective(scale, axes=axes)

    if isinstance(setting, SettingTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale)
        plotter_common.plot_colorbar(fig, axs=[axes], cbar_settings=cbar_settings)
        draw_main_temperature(
            axes=axes, setting=setting, cbar_settings=cbar_settings
        )
    else:
        draw_main_displaced(setting, axes=axes)
    if base_setting is not None:
        draw_base_displaced(base_setting, scale, axes=axes)

    draw_parameters(current_time, setting, scale, axes=axes)
    # draw_angles(setting, axes)

    position = np.array([-4.2, -4.2]) * scale
    shift = 2.5 * scale
    draw_forces(setting, position, axes=axes)
    if draw_detailed:  # detailed:
        position[0] += shift
        if setting.obstacles is not None:
            draw_obstacle_resistance_normalized(setting, position, axes=axes)
            position[0] += shift
        # draw_boundary_surfaces_normals(setting, position, axes)
        # position[0] += shift
        # draw_boundary_normals(setting, position, axes)
        # position[0] += shift

        draw_boundary_resistance_normal(setting, position, axes=axes)
        position[0] += shift
        draw_boundary_resistance_tangential(setting, position, axes=axes)
        position[0] += shift
        draw_boundary_v_tangential(setting, position, axes=axes)
        position[0] += shift

        draw_input_u(setting, position, axes=axes)
        position[0] += shift
        draw_input_v(setting, position, axes=axes)
        position[0] += shift
        draw_a(setting, position, axes=axes)

        position[0] += shift
        if isinstance(setting, SettingTemperature):
            plot_temperature(
                axes=axes, setting=setting, position=position, cbar_settings=cbar_settings
            )

        # draw_edges_data(setting, position, axes)
        # draw_vertices_data(setting, position, axes)


def plot_temperature(
        axes,
        setting: SettingTemperature,
        position,
        cbar_settings: plotter_common.ColorbarSettings,
):
    add_annotation("TEMP", setting, position, axes)
    points = (setting.normalized_nodes + position).T
    axes.scatter(
        *points,
        c=setting.t_old,
        cmap=cbar_settings.cmap,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        s=1,
        marker=".",
        linewidths=0.1,
    )


def draw_main_temperature(axes, setting, cbar_settings):
    draw_main_obstacles(setting, axes)
    axes.tricontourf(
        *setting.moved_nodes.T,
        setting.elements,
        setting.t_old.reshape(-1),
        cmap=cbar_settings.cmap,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        antialiased=True,
    )


def draw_obstacles(obstacle_origins, obstacle_normals, position, color, axes):
    obstacles_tangient = np.hstack(
        (-obstacle_normals[:, 1, None], obstacle_normals[:, 0, None])
    )
    for i, obstacle_origin in enumerate(obstacle_origins):
        bias = obstacle_origin + position
        axes.arrow(
            *bias,
            *obstacle_normals[i],
            color=f"tab:{color}",
            alpha=0.4,
            width=0.00002,
            length_includes_head=True,
            head_width=0.01,
        )
        axes.axline(
            bias,
            obstacles_tangient[i] + bias,
            color=f"tab:{color}",
            alpha=0.4,
            linewidth=0.4,
        )

def plot_arrows(starts, vectors, axes):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(
        starts, vectors
    )
    axes.quiver(
        *prepared_starts,
        *prepared_vectors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        color="w",
        zorder=2,
    )


def draw_main_obstacles(setting, axes):
    draw_obstacles(
        setting.obstacle_origins, setting.obstacle_normals, [0, 0], "orange", axes
    )


def draw_normalized_obstacles(setting, position, axes):
    draw_obstacles(
        setting.normalized_obstacle_origins,
        setting.normalized_obstacle_normals,
        position,
        "blue",
        axes,
    )


def draw_obstacle_resistance_normalized(setting, position, axes):
    draw_additional_setting("P", setting, position, axes)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_penetration,
        axes,
    )


def draw_boundary_normals(setting, position, axes):
    draw_additional_setting("N", setting, position, axes)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.get_normalized_boundary_normals(),
        axes,
    )


def draw_boundary_v_tangential(setting, position, axes):
    draw_additional_setting("V_TNG", setting, position, axes)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.get_normalized_boundary_v_tangential(),
        axes,
    )


def draw_boundary_resistance_normal(setting, position, axes):
    draw_additional_setting("RES_N", setting, position, axes)
    data = setting.get_normalized_boundary_normals() * setting.resistance_normal / 100
    plot_arrows(
        setting.normalized_boundary_nodes + position, data, axes,
    )


def draw_boundary_resistance_tangential(setting, position, axes):
    draw_additional_setting("RES_T", setting, position, axes)
    data = setting.get_normalized_boundary_normals() * setting.get_resistance_tangential() / 100
    plot_arrows(
        setting.normalized_boundary_nodes + position, data, axes,
    )


def draw_rectangle(axes, position, scale_x, scale_y):
    axes.add_patch(
        Rectangle(
            (position[0], position[1],),
            scale_x,
            scale_y,
            fill=None,
            alpha=1.0,
            color="w",
            lw=0.2,
            zorder=3,
        )
    )


def draw_main_displaced(setting, axes):
    position = np.array([0.0, 0.0])
    draw_displaced(setting, position, "orange", axes)
    if setting.obstacles is not None:
        draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, position, "orange", axes,
        )


def draw_base_displaced(setting, scale, axes):
    position = np.array([0.0, 1.5]) * scale
    draw_displaced(setting, position, "purple", axes)
    if setting.obstacles is not None:
        draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, position, "orange", axes,
        )


def draw_displaced(setting, position, color, axes):
    # draw_rectangle(axes, position, setting.mesh_data.scale_x, setting.mesh_data.scale_y)
    draw_triplot(setting.moved_nodes + position, setting, f"tab:{color}", axes)
    # draw_data("P", obstacle_forces, setting, [7.5, -1.5], axes)


def draw_points(points, position, color, axes):
    moved_nodes = points + position
    axes.scatter(moved_nodes[:, 0], moved_nodes[:, 1], s=0.1, c=f"tab:{color}")


def draw_forces(setting, position, axes):
    return draw_data("F", setting.normalized_forces, setting, position, axes)


def draw_input_u(setting, position, axes):
    return draw_data("U", setting.input_u_old, setting, position, axes)


def draw_input_v(setting, position, axes):
    return draw_data("V", setting.input_v_old, setting, position, axes)


def draw_a(setting, position, axes):
    return draw_data(
        "A * ts", setting.normalized_a_old * setting.time_step, setting, position, axes,
    )


def draw_data(annotation, data, setting, position, axes):
    draw_additional_setting(annotation, setting, position, axes)
    plot_arrows(setting.normalized_nodes + position, data, axes)


def draw_additional_setting(annotation, setting, position, axes):
    draw_triplot(setting.normalized_nodes + position, setting, "tab:blue", axes)
    add_annotation(annotation, setting, position, axes)


def add_annotation(annotation, setting, position, axes):
    scale = setting.mesh_data.scale_x
    description_offset = np.array([-0.5, -1.1]) * scale
    axes.annotate(annotation, xy=position + description_offset, color="w", fontsize=5)


def draw_parameters(current_time, setting, scale, axes):
    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    args = dict(color="w", fontsize=5, )

    annotation = plotter_common.get_frame_annotation(
        current_time=current_time, setting=setting
    )
    axes.text(x_max - 4.0 * scale, y_max - 2.0 * scale, s=annotation, **args)


def draw_triplot(nodes, setting, color, axes):
    boundary_nodes = nodes[setting.boundary_surfaces]
    axes.add_collection(
        collections.LineCollection(
            boundary_nodes,
            colors=[color for _ in range(boundary_nodes.shape[0])],
            linewidths=0.3,
        )
    )
    triplot(nodes, setting.elements, color, axes)


def triplot(nodes, elements, color, axes):
    axes.triplot(nodes[:, 0], nodes[:, 1], elements, color=color, linewidth=0.1)


# TODO #66


def draw_edges_data(position, setting, axes):
    draw_data_at_edges(setting, setting.edges_data[:, 2:4], position, axes)


def draw_vertices_data(position, setting, axes):
    draw_data_at_vertices(setting, setting.normalized_u_old, position, axes)


def draw_data_at_edges(setting, features, position, axes):
    draw_triplot(setting.normalized_nodes + position, setting, "tab:orange", axes)

    centers = np.sum(setting.edges_normalized_nodes + position, axis=1) / 2.0
    vertices = setting.edges_normalized_nodes[:, 0] + position
    points = (centers + vertices) / 2.0

    for i in range(len(setting.edges_normalized_nodes)):
        feature = np.around(features[i], 2)
        # np.set_printoptions(precision=3)
        axes.text(
            points[i, 0] - 0.04,
            points[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


def draw_data_at_vertices(setting, features, position, axes):
    draw_triplot(setting.normalized_nodes + position, setting, "tab:orange", axes)

    points = setting.normalized_nodes + position
    for i in range(len(setting.normalized_nodes)):
        feature = np.around(features[i], 2)
        axes.text(
            points[i, 0] - 0.04,
            points[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


def plot_simple_data(elements, nodes, path):
    fig = get_fig()
    axs = get_axs(fig)
    set_perspective(scale=1, axes=axs)
    triplot(nodes, elements, "tab:orange", axs)
    extension = path.split(".")[-1]
    plotter_common.plt_save(path, extension)
