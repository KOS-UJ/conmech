from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections
from matplotlib.patches import Rectangle

from conmech.helpers.config import Config
from deep_conmech.common.plotter import plotter_common
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature


def get_fig():
    return plt.figure(figsize=(4, 2))


def get_axs(fig):
    # axs = fig.add_subplot(1, 1, 1, facecolor="none")
    axs = fig.add_axes([0.075, 0.15, 0.9, 0.8], facecolor="none")
    return axs


def set_perspective(scale, ax):
    ax.set_aspect("equal", "box")
    padding = 6
    ax.set_xlim(-padding * scale, 18 * scale)
    ax.set_ylim(-padding * scale, padding * scale)
    plotter_common.set_ax(ax)


def plot_animation(
        plot_setting_paths: List[str],
        time_skip: float,
        save_path: str,
        config: Config,
        t_scale: Optional[np.ndarray] = None,
):
    plotter_common.plot_animation(
        plot_setting_paths=plot_setting_paths,
        time_skip=time_skip,
        save_path=save_path,
        get_axs=get_axs,
        plot_frame=plot_frame,
        fig=get_fig(),
        config=config,
        t_scale=t_scale,
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
    ax = axs
    scale = setting.mesh_data.scale_x
    set_perspective(scale, ax=ax)

    if isinstance(setting, SettingTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale)
        plotter_common.plot_colorbar(fig, axs=[ax], cbar_settings=cbar_settings)
        draw_main_temperature(
            fig=fig, ax=ax, setting=setting, cbar_settings=cbar_settings
        )
    else:
        draw_main_displaced(setting, ax=ax)
    if base_setting is not None:
        draw_base_displaced(base_setting, scale, ax=ax)

    draw_parameters(current_time, setting, scale, ax=ax)
    # draw_angles(setting, ax)

    position = np.array([-4.2, -4.2]) * scale
    shift = 2.5 * scale
    draw_forces(setting, position, ax=ax)
    if draw_detailed:  # detailed:
        position[0] += shift
        if setting.obstacles is not None:
            draw_obstacle_resistance_normalized(setting, position, ax=ax)
            position[0] += shift
        # draw_boundary_faces_normals(setting, position, ax)
        # position[0] += shift
        # draw_boundary_normals(setting, position, ax)
        # position[0] += shift

        draw_boundary_resistance_normal(setting, position, ax=ax)
        position[0] += shift
        draw_boundary_resistance_tangential(setting, position, ax=ax)
        position[0] += shift
        draw_boundary_v_tangential(setting, position, ax=ax)
        position[0] += shift

        draw_input_u(setting, position, ax=ax)
        position[0] += shift
        draw_input_v(setting, position, ax=ax)
        position[0] += shift
        draw_a(setting, position, ax=ax)

        position[0] += shift
        if isinstance(setting, SettingTemperature):
            plot_temperature(
                ax=ax, setting=setting, position=position, cbar_settings=cbar_settings
            )

        # draw_edges_data(setting, position, ax)
        # draw_vertices_data(setting, position, ax)


def plot_temperature(
        ax,
        setting: SettingTemperature,
        position,
        cbar_settings: plotter_common.ColorbarSettings,
):
    add_annotation("TEMP", setting, position, ax)
    # vmin, vmax, cmap = plotter_common.get_t_data(t_scale)
    points = (setting.normalized_points + position).T
    ax.scatter(
        *points,
        c=setting.t_old,
        cmap=cbar_settings.cmap,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        s=1,
        marker=".",
        linewidths=0.1,
    )


def draw_main_temperature(fig, ax, setting, cbar_settings):
    draw_main_obstacles(setting, ax)
    values = ax.tricontourf(
        *(setting.moved_nodes.T),
        setting.elements,
        setting.t_old.reshape(-1),
        cmap=cbar_settings.cmap,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        antialiased=True,
    )


def draw_obstacles(obstacle_origins, obstacle_normals, position, color, ax):
    obstacles_tangient = np.hstack(
        (-obstacle_normals[:, 1, None], obstacle_normals[:, 0, None])
    )
    for i in range(len(obstacle_origins)):
        bias = obstacle_origins[i] + position
        ax.arrow(
            *bias,
            *obstacle_normals[i],
            color=f"tab:{color}",
            alpha=0.4,
            width=0.00002,
            length_includes_head=True,
            head_width=0.01,
        )
        line = ax.axline(
            bias,
            obstacles_tangient[i] + bias,
            color=f"tab:{color}",
            alpha=0.4,
            linewidth=0.4,
        )
        # ax.fill_between(bias, obstacles_tangient[i] + bias)


def plot_arrows(starts, vectors, ax):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(
        starts, vectors
    )
    ax.quiver(
        *prepared_starts,
        *prepared_vectors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.001,
        color="w",
        zorder=2,
    )


def draw_main_obstacles(setting, ax):
    draw_obstacles(
        setting.obstacle_origins, setting.obstacle_normals, [0, 0], "orange", ax
    )


def draw_normalized_obstacles(setting, position, ax):
    draw_obstacles(
        setting.normalized_obstacle_origins,
        setting.normalized_obstacle_normals,
        position,
        "blue",
        ax,
    )


def draw_obstacle_resistance_normalized(setting, position, ax):
    # draw_normalized_obstacles(setting, position, ax)
    draw_additional_setting("P", setting, position, ax)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_penetration,
        ax,
    )


def draw_boundary_normals(setting, position, ax):
    draw_additional_setting("N", setting, position, ax)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_normals,
        ax,
    )


def draw_boundary_v_tangential(setting, position, ax):
    draw_additional_setting("V_TNG", setting, position, ax)
    plot_arrows(
        setting.normalized_boundary_nodes + position,
        setting.normalized_boundary_v_tangential,
        ax,
    )


def draw_boundary_resistance_normal(setting, position, ax):
    draw_additional_setting("RES_N", setting, position, ax)
    data = setting.normalized_boundary_normals * setting.resistance_normal / 100
    plot_arrows(
        setting.normalized_boundary_nodes + position, data, ax,
    )


def draw_boundary_resistance_tangential(setting, position, ax):
    draw_additional_setting("RES_T", setting, position, ax)
    data = setting.normalized_boundary_normals * setting.resistance_tangential / 100
    plot_arrows(
        setting.normalized_boundary_nodes + position, data, ax,
    )


def draw_rectangle(ax, position, scale_x, scale_y):
    ax.add_patch(
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


def draw_main_displaced(setting, ax):
    position = np.array([0.0, 0.0])
    draw_displaced(setting, position, "orange", ax)
    # draw_points(setting.moved_reference_points, position, "orange", ax)
    if setting.obstacles is not None:
        draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, position, "orange", ax,
        )


def draw_base_displaced(setting, scale, ax):
    position = np.array([0.0, 1.5]) * scale
    draw_displaced(setting, position, "purple", ax)
    if setting.obstacles is not None:
        draw_obstacles(
            setting.obstacle_origins, setting.obstacle_normals, position, "orange", ax,
        )


def draw_displaced(setting, position, color, ax):
    # draw_rectangle(ax, position, setting.mesh_data.scale_x, setting.mesh_data.scale_y)
    draw_triplot(setting.moved_nodes + position, setting, f"tab:{color}", ax)
    # draw_data("P", obstacle_forces, setting, [7.5, -1.5], ax)


def draw_points(points, position, color, ax):
    moved_nodes = points + position
    ax.scatter(moved_nodes[:, 0], moved_nodes[:, 1], s=0.1, c=f"tab:{color}")


def draw_forces(setting, position, ax):
    return draw_data("F", setting.normalized_forces, setting, position, ax)


def draw_input_u(setting, position, ax):
    return draw_data("U", setting.input_u_old, setting, position, ax)


def draw_input_v(setting, position, ax):
    return draw_data("V", setting.input_v_old, setting, position, ax)


def draw_a(setting, position, ax):
    return draw_data(
        "A * ts", setting.normalized_a_old * setting.time_step, setting, position, ax,
    )


def draw_data(annotation, data, setting, position, ax):
    draw_additional_setting(annotation, setting, position, ax)
    plot_arrows(setting.normalized_points + position, data, ax)


def draw_additional_setting(annotation, setting, position, ax):
    draw_triplot(setting.normalized_points + position, setting, "tab:blue", ax)
    add_annotation(annotation, setting, position, ax)


def add_annotation(annotation, setting, position, ax):
    scale = setting.mesh_data.scale_x
    description_offset = np.array([-0.5, -1.1]) * scale
    ax.annotate(annotation, xy=position + description_offset, color="w", fontsize=5)


def draw_parameters(current_time, setting, scale, ax):
    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]
    args = dict(color="w", fontsize=5, )

    annotation = plotter_common.get_frame_annotation(
        current_time=current_time, setting=setting
    )
    ax.text(x_max - 4.0 * scale, y_max - 2.0 * scale, s=annotation, **args)


def draw_triplot(nodes, setting, color, ax):
    boundary_nodes = nodes[setting.boundary_faces]
    ax.add_collection(
        collections.LineCollection(
            boundary_nodes,
            colors=[color for _ in range(boundary_nodes.shape[0])],
            linewidths=0.3,
        )
    )
    triplot(nodes, setting.elements, color, ax)


def triplot(nodes, elements, color, ax):
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color=color, linewidth=0.1)


##############


def draw_edges_data(position, setting, ax):
    draw_data_at_edges(setting, setting.edges_data[:, 2:4], position, ax)


def draw_vertices_data(position, setting, ax):
    draw_data_at_vertices(setting, setting.normalized_u_old, position, ax)


def draw_data_at_edges(setting, features, position, ax):
    draw_triplot(setting.normalized_points + position, setting, "tab:orange", ax)

    centers = np.sum(setting.edges_normalized_points + position, axis=1) / 2.0
    vertices = setting.edges_normalized_points[:, 0] + position
    points = (centers + vertices) / 2.0

    for i in range(len(setting.edges_normalized_points)):
        feature = np.around(features[i], 2)
        # np.set_printoptions(precision=3)
        ax.text(
            points[i, 0] - 0.04,
            points[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


def draw_data_at_vertices(setting, features, position, ax):
    draw_triplot(setting.normalized_points + position, setting, "tab:orange", ax)

    points = setting.normalized_points + position
    for i in range(len(setting.normalized_points)):
        feature = np.around(features[i], 2)
        ax.text(
            points[i, 0] - 0.04,
            points[i, 1],
            str(feature),
            # xy=(0.5, 0.5),
            # xycoords="axes fraction",
            fontsize=0.01,
            rotation=30,
            color="w",
        )


###################


def plot_simple_data(elements, nodes, path):
    fig = get_fig()
    axs = get_axs(fig)
    set_perspective(scale=1, ax=axs)
    triplot(nodes, elements, "tab:orange", axs)
    extension = path.split(".")[-1]
    plotter_common.plt_save(path, extension)
