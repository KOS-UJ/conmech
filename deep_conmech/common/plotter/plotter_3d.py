from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from deep_conmech.common.plotter import plotter_common
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.mesh import *
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_fig():
    return plt.figure(figsize=(5, 4))


def get_one_ax(fig, grid, angle, distance):
    ax = fig.add_subplot(grid, projection="3d", facecolor="none")
    ax.set_proj_type("ortho")
    ax.view_init(elev=angle[0], azim=angle[1])  # , vertical_axis='y')
    ax.dist = distance

    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    aspect = (12, 4, 4)
    ax.set_box_aspect(aspect)

    ax.set_xlim(-1, aspect[0] - 1)
    ax.set_ylim(-1, aspect[1] - 1)
    ax.set_zlim(-1, aspect[2] - 1)

    # ax.set_xlabel("x", labelpad=0.05, color="w")
    # ax.set_ylabel("y", labelpad=0.05, color="w")
    # ax.set_zlabel("z", labelpad=0.05, color="w")

    ticks = []  # np.arange(0, 2, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # ax.tick_params(axis="x", colors="w")
    # ax.tick_params(axis="y", colors="w")
    # ax.tick_params(axis="z", colors="w")

    ax.w_xaxis.line.set_color("w")
    ax.w_yaxis.line.set_color("w")
    ax.w_zaxis.line.set_color("w")

    return ax


def get_axs(fig):
    angles = np.array([[[0, -90], [0, 0]], [[30, -60], [90, 0]]])
    distances = np.array([[10, 10], [11, 10]])
    rows, columns, _ = angles.shape

    # fig = get_figure()  # constrained_layout=True)
    grid = fig.add_gridspec(nrows=rows, ncols=columns)
    # , width_ratios=[1, 1.], height_ratios=[1., 1.])
    # fig.subplots_adjust(left=-0.2, bottom=0., right=1., top=1.)#, wspace=-0.4, hspace=-0.4)

    ax1 = get_one_ax(fig, grid[0, 0], angles[0, 0], distances[0, 0])
    ax1.set_position([0.1, 0.5, 0.4, 0.4])

    ax2 = get_one_ax(fig, grid[0, 1], angles[0, 1], distances[0, 1])
    ax2.set_position([0.5, 0.5, 0.4, 0.4])

    ax3 = get_one_ax(fig, grid[1, 0], angles[1, 0], distances[1, 0])
    ax3.set_position([0.0, 0.0, 0.7, 0.7])

    ax4 = get_one_ax(fig, grid[1, 1], angles[1, 1], distances[1, 1])
    ax4.set_position([0.5, 0.2, 0.4, 0.4])
    return [ax1, ax2, ax3, ax4]


def plot_frame(setting, current_time, axs):
    return plot_frame_internal(
        setting=setting,
        normalized_data=[
            setting.normalized_forces,
            setting.normalized_u_old,
            setting.normalized_v_old,
            setting.normalized_a_old,
        ],
        axs=axs,
    )


def plot_frame_internal(setting, normalized_data, axs):
    for ax in axs:
        plot_subframe(
            setting, normalized_data=normalized_data, ax=ax,
        )


def plot_arrows(starts, vectors, ax):
    prepared_starts, prepared_vectors = plotter_common.prepare_for_arrows(
        starts, vectors
    )
    ax.quiver(
        *prepared_starts, *prepared_vectors, arrow_length_ratio=0.1, color="w", lw=0.1
    )


def draw_base_arrows(ax, base):
    z = np.array([0, 2.0, 2.0])
    ax.quiver(*z, *(base[0]), arrow_length_ratio=0.1, color="r")
    ax.quiver(*z, *(base[1]), arrow_length_ratio=0.1, color="y")
    ax.quiver(*z, *(base[2]), arrow_length_ratio=0.1, color="g")


def plot_subframe(
    setting, normalized_data, ax,
):
    draw_base_arrows(ax, setting.moved_base)

    plot_mesh(nodes=setting.moved_nodes, setting=setting, color="tab:orange", ax=ax)
    plot_obstacles(ax, setting, "tab:orange")

    shifted_normalized_nodes = setting.normalized_points + np.array([0, 2.0, 0])
    for data in normalized_data:
        plot_arrows(starts=shifted_normalized_nodes, vectors=data, ax=ax)
        plot_mesh(
            nodes=shifted_normalized_nodes, setting=setting, color="tab:blue", ax=ax
        )
        shifted_normalized_nodes = shifted_normalized_nodes + np.array([2.5, 0, 0])

    if isinstance(setting, SettingTemperature):
        draw_temperature(nodes=shifted_normalized_nodes, setting=setting, ax=ax)


def draw_temperature(nodes, setting, ax):
    points = nodes.T
    t_min = 0.0
    t_max = 0.1

    ax.scatter(
        *points,
        c=setting.t_old,
        vmin=t_min,
        vmax=t_max,
        cmap=plt.cm.plasma,
        s=1,
        marker=".",
        linewidths=0.1,
    )


def plot_mesh(nodes, setting, color, ax):
    boundary_faces_nodes = nodes[setting.boundary_faces]
    ax.add_collection3d(
        Poly3DCollection(
            boundary_faces_nodes,
            edgecolors=color,
            linewidths=0.1,
            facecolors=color,
            alpha=0.2,
        )
    )


def plot_obstacles(ax, setting, color):
    alpha = 0.3
    node = setting.obstacle_nodes[0]
    normal = setting.obstacle_normals[0]

    # a plane is a*x+b*y+c*z+d=0
    # z = (-d-ax-by) / c
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -node.dot(normal)

    x_rng = np.arange(-1.2, 11.2, 0.2)
    y_rng = np.arange(-1.2, 3.2, 0.2)

    X, Y = np.meshgrid(x_rng, y_rng)
    Z = (-normal[0] * X - normal[1] * Y - d) / normal[2]
    col = (Z[0, :] > -1.2) & (Z[0, :] < 3.2)
    mask = (Z > -1.2) * (Z < 3.2)

    ax.plot_surface(X * mask, Y * mask, Z * mask, color=color, alpha=alpha)
    # ax.plot_surface(X[:,col], Y[:,col], Z[:,col], color=color, alpha=alpha)

    ax.quiver(*node, *normal, color=color, alpha=alpha)


def plot_animation(all_setting_paths: List[str], time_skip: float, path: str):
    plotter_common.plot_animation(
        all_setting_paths, time_skip, path, get_axs, plot_frame, get_fig()
    )
