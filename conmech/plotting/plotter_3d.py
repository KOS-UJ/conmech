from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from conmech.helpers.config import Config
from conmech.plotting import plotter_common
from conmech.plotting.plotter_common import make_animation
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature


def get_fig():
    return plt.figure(figsize=(3, 2))


def get_one_ax(fig, rect, angle, distance):
    # axes = fig.add_subplot(1, 1, 1, projection="3d", facecolor="none")
    axes = fig.add_axes(rect, projection="3d", facecolor="none")
    axes.set_proj_type("ortho")
    axes.view_init(elev=angle[0], azim=angle[1])  # , vertical_axis='y')
    axes.dist = distance

    axes.grid(False)
    axes.w_xaxis.pane.fill = False
    axes.w_yaxis.pane.fill = False
    axes.w_zaxis.pane.fill = False

    aspect = (12, 4, 4)
    axes.set_box_aspect(aspect)

    axes.set_xlim(-1, aspect[0] - 1)
    axes.set_ylim(-1, aspect[1] - 1)
    axes.set_zlim(-1, aspect[2] - 1)

    # axes.set_xlabel("x", labelpad=0.05, color="w")
    # axes.set_ylabel("y", labelpad=0.05, color="w")
    # axes.set_zlabel("z", labelpad=0.05, color="w")

    ticks = []  # np.arange(0, 2, 1)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_zticks(ticks)

    # axes.tick_params(axis="x", colors="w")
    # axes.tick_params(axis="y", colors="w")
    # axes.tick_params(axis="z", colors="w")

    axes.w_xaxis.line.set_color("w")
    axes.w_yaxis.line.set_color("w")
    axes.w_zaxis.line.set_color("w")

    return axes


def get_axs(fig):
    angles = np.array([[[30, -60], [0, -90]], [[0, 0], [90, 0]]])
    distances = np.array([[9, 10], [10, 10]])

    ax0 = get_one_ax(fig, [0.0, 0.0, 0.7, 0.7], angles[0, 0], distances[0, 0])
    ax1 = get_one_ax(fig, [0.2, 0.5, 0.4, 0.4], angles[0, 1], distances[0, 0])
    ax2 = get_one_ax(fig, [0.6, 0.5, 0.4, 0.4], angles[1, 0], distances[0, 1])
    ax3 = get_one_ax(fig, [0.6, 0.2, 0.4, 0.4], angles[1, 1], distances[1, 1])

    return [ax0, ax1, ax2, ax3]


def plot_frame(fig, axs, setting, current_time, t_scale: Optional[List] = None):
    for axes in axs:
        plot_subframe(
            axes=axes,
            setting=setting,
            normalized_data=[
                setting.normalized_forces,
                setting.normalized_u_old,
                setting.normalized_v_old,
                setting.normalized_a_old,
            ],
            t_scale=t_scale,
        )
    draw_parameters(axes=axs[0], setting=setting, current_time=current_time)

    if isinstance(setting, SettingTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale=t_scale)
        plotter_common.plot_colorbar(fig, axs=axs, cbar_settings=cbar_settings)


def draw_parameters(axes, setting, current_time):
    annotation = plotter_common.get_frame_annotation(current_time=current_time, setting=setting)
    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    z_max = axes.get_zlim()[1]

    args = dict(color="w", fontsize=4)
    axes.text(x_max - 2.0, y_max - 0.5, z_max + 1.0, s=annotation, **args)  # zdir=None,


def plot_arrows(starts, vectors, axes):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(starts, vectors)
    axes.quiver(*prepared_starts, *prepared_vectors, arrow_length_ratio=0.1, color="w", lw=0.1)


def draw_base_arrows(axes, base):
    z = np.array([0, 2.0, 2.0])
    axes.quiver(*z, *(base[0]), arrow_length_ratio=0.1, color="r")
    axes.quiver(*z, *(base[1]), arrow_length_ratio=0.1, color="y")
    axes.quiver(*z, *(base[2]), arrow_length_ratio=0.1, color="g")


def plot_subframe(axes, setting, normalized_data, t_scale):
    # draw_base_arrows(axes, setting.moved_base)

    if isinstance(setting, SettingTemperature):
        cbar_settings = plotter_common.get_t_data(t_scale)
        plot_main_temperature(
            axes,
            nodes=setting.moved_nodes,
            setting=setting,
            cbar_settings=cbar_settings,
        )
    else:
        plot_mesh(nodes=setting.moved_nodes, setting=setting, color="tab:orange", axes=axes)
    plot_obstacles(axes, setting, "tab:orange")

    shifted_normalized_nodes = setting.normalized_nodes + np.array([0, 2.0, 0])
    for data in normalized_data:
        plot_arrows(starts=shifted_normalized_nodes, vectors=data, axes=axes)
        plot_mesh(nodes=shifted_normalized_nodes, setting=setting, color="tab:blue", axes=axes)
        shifted_normalized_nodes = shifted_normalized_nodes + np.array([2.5, 0, 0])

    if isinstance(setting, SettingTemperature):
        plot_temperature(
            axes=axes,
            nodes=shifted_normalized_nodes,
            setting=setting,
            cbar_settings=cbar_settings,
        )


def plot_temperature(axes, nodes, setting, cbar_settings: plotter_common.ColorbarSettings):
    points = nodes.T
    axes.scatter(
        *points,
        c=setting.t_old,
        vmin=cbar_settings.vmin,
        vmax=cbar_settings.vmax,
        cmap=cbar_settings.cmap,
        s=1,
        marker=".",
        linewidths=0.1,
    )


def plot_main_temperature(axes, nodes, setting, cbar_settings):
    boundary_surfaces_nodes = nodes[setting.boundary_surfaces]
    nodes_temperature = setting.t_old[setting.boundary_surfaces]
    faces_temperature = np.mean(nodes_temperature, axis=1)

    facecolors = cbar_settings.mappable.to_rgba(faces_temperature)
    axes.add_collection3d(
        Poly3DCollection(
            boundary_surfaces_nodes,
            linewidths=0.1,
            facecolors=facecolors,
            alpha=0.2,
        )
    )


def plot_mesh(nodes, setting, color, axes):
    boundary_surfaces_nodes = nodes[setting.boundary_surfaces]
    axes.add_collection3d(
        Poly3DCollection(
            boundary_surfaces_nodes,
            edgecolors=color,
            linewidths=0.1,
            facecolors=color,
            alpha=0.2,
        )
    )


def plot_obstacles(axes, setting, color):
    if setting.obstacles is None:
        return
    alpha = 0.3
    node = setting.obstacle_nodes[0]
    normal = setting.obstacle_normals[0]

    # a plane is a*x+b*y+c*z+d=0
    # z = (-d-axes-by) / c
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -node.dot(normal)

    x_rng = np.arange(-1.2, 11.2, 0.2)
    y_rng = np.arange(-1.2, 3.2, 0.2)

    X, Y = np.meshgrid(x_rng, y_rng)
    Z = (-normal[0] * X - normal[1] * Y - d) / normal[2]
    mask = (Z > -1.2) * (Z < 3.2)

    axes.plot_surface(X * mask, Y * mask, Z * mask, color=color, alpha=alpha)
    # axes.plot_surface(X[:,col], Y[:,col], Z[:,col], color=color, alpha=alpha)

    axes.quiver(*node, *normal, color=color, alpha=alpha)


def plot_animation(
    save_path: str,
    config: Config,
    time_skip: float,
    index_skip: int,
    plot_settings_count: int,
    all_settings_path: str,
    t_scale: Optional[np.ndarray] = None,
):
    animate = make_animation(get_axs, plot_frame, t_scale)
    plotter_common.plot_animation(
        animate=animate,
        fig=get_fig(),
        save_path=save_path,
        config=config,
        time_skip=time_skip,
        index_skip=index_skip,
        plot_settings_count=plot_settings_count,
        all_settings_path=all_settings_path,
    )
